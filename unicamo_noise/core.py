"""Paper-audited NumPy implementation of UniCaMo noise (not official code). Arrays: noise [T,H,W,C], poses world->camera.
See docs/PAPER_AUDIT.md for reproduction assumptions.
"""
import numpy as np


def project(tracks, intrinsics, w2c):
    camera = np.einsum('tij,tnj->tni', w2c[:, :3, :3], tracks) + w2c[:, None, :3, 3]
    homogeneous = np.einsum('tij,tnj->tni', intrinsics, camera)
    xy = homogeneous[..., :2] / np.where(abs(homogeneous[..., 2:]) > 1e-12, homogeneous[..., 2:], np.nan)
    return xy, camera[..., 2]


def sphere_indices(intrinsics, w2c, height, width, radius, env_height, env_width):
    """Pixel centers are (x+.5,y+.5); intrinsics already match the output grid."""
    c2w = np.linalg.inv(w2c)
    center = c2w[0, :3, 3]
    origins = c2w[:, :3, 3]
    if radius <= np.max(np.linalg.norm(origins-center, axis=-1)):
        raise ValueError('Sphere must strictly contain every camera center')
    y, x = np.mgrid[:height, :width]
    pixels = np.stack([x+.5, y+.5, np.ones_like(x)], -1)
    rays = np.einsum('tij,hwj->thwi', np.linalg.inv(intrinsics), pixels)
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    rays = np.einsum('tij,thwj->thwi', c2w[:, :3, :3], rays)
    offset = origins-center
    b = np.einsum('thwi,ti->thw', rays, offset)
    c = np.sum(offset**2, axis=-1)-radius**2
    distance = -b + np.sqrt(np.maximum(b*b-c[:, None, None], 0))
    normals = (offset[:, None, None]+distance[..., None]*rays)/radius
    u = (np.arctan2(normals[..., 0], normals[..., 2])/(2*np.pi)+.5) % 1
    v = np.arccos(np.clip(normals[..., 1], -1, 1))/np.pi
    # Texel-center convention: nearest lookup = floor(normalized_coordinate * size).
    ix = np.floor(u*env_width).astype(int) % env_width
    iy = np.minimum(np.floor(v*env_height).astype(int), env_height-1)
    return iy*env_width+ix


def construct_noise(tracks, intrinsics, w2c, *, height, width, channels=16,
                    image_height, image_width, depth_bound,
                    sphere_downsample, first_frame_policy,
                    visibility=None, seed=0, patch_radius=2,
                    env_scale=4, degradation=.2, blend='linear'):
    """Build geometric noise; return raw/mixed noise, provenance, owner and depth.

    tracks [T,N,3] in world coordinates; K [T,3,3] in IMAGE pixels. Image and latent sizes are explicit.
    visibility [T,N] is externally supplied. Z-buffer resolves only supplied patches.
    Unspecified-paper decisions MUST be supplied: sphere_downsample='nearest',
    first_frame_policy='preserve_reference' or 'compose'. Default alpha blending
    is the ordinary linear interpretation, NOT an author-confirmed formula.
    Source IDs < H*W refer to first-frame noise; larger IDs refer to sphere texels.
    """
    tracks, intrinsics, w2c = map(lambda x: np.asarray(x, dtype=float), (tracks, intrinsics, w2c))
    if tracks.ndim != 3 or tracks.shape[-1] != 3:
        raise ValueError('tracks must have shape [T,N,3]')
    t, n, _ = tracks.shape
    if intrinsics.shape != (t,3,3) or w2c.shape != (t,4,4):
        raise ValueError('Camera sequence shapes must match tracks')
    if not all(np.isfinite(x).all() for x in (tracks, intrinsics, w2c)):
        raise ValueError('Inputs must be finite')
    if min(height,width,channels,patch_radius,env_scale) < 1 or depth_bound <= 0:
        raise ValueError('Grid sizes, radius and depth bound must be positive')
    if not 0 <= degradation <= .5 or blend != 'linear':
        raise ValueError('Invalid blending configuration')
    if sphere_downsample != 'nearest':
        raise ValueError('Only explicitly selected nearest downsampling is implemented')
    if first_frame_policy not in ('preserve_reference', 'compose'):
        raise ValueError('Explicit first-frame policy is required')
    if image_height < height or image_width < width:
        raise ValueError('Image dimensions must be at least the latent dimensions')
    visibility = np.ones((t,n),bool) if visibility is None else np.asarray(visibility,bool)
    if visibility.shape != (t,n): raise ValueError('visibility must have shape [T,N]')
    rng = np.random.default_rng(seed)
    reference = rng.standard_normal((height*width,channels)).astype(np.float32)
    eh, ew = image_height*env_scale, image_width*env_scale
    environment = rng.standard_normal((eh*ew,channels)).astype(np.float32)
    centers = np.linalg.inv(w2c)[:, :3, 3]
    radius = max(depth_bound, 1.2*np.linalg.norm(centers-centers[0],axis=-1).max())
    # A.3: image pixel rays -> sphere nearest lookup -> latent downsampling.
    # The latter operator is unspecified in the paper; this run chooses nearest.
    full_ids = sphere_indices(intrinsics,w2c,image_height,image_width,radius,eh,ew)
    rows = np.floor((np.arange(height)+.5)*image_height/height).astype(int)
    cols = np.floor((np.arange(width)+.5)*image_width/width).astype(int)
    sphere_id = full_ids[:, rows[:, None], cols[None, :]]
    sphere = environment[sphere_id]
    source = sphere_id+height*width
    owner = np.full((t,height,width),-1,np.int32)
    depth = np.full((t,height,width),np.inf)
    image_xy, z = project(tracks,intrinsics,w2c)
    xy = image_xy * np.array([width/image_width, height/image_height])
    finite = np.isfinite(xy).all(-1)
    # Grid centers lie at half-integers, so floor selects the nearest pixel center.
    q = np.floor(np.where(np.isfinite(xy),xy,0)).astype(int)
    raw = sphere.copy()
    for frame in range(t):
        for i in range(n):
            if not (visibility[0,i] and visibility[frame,i] and finite[0,i] and finite[frame,i] and z[0,i]>0 and z[frame,i]>0): continue
            sx,sy = q[0,i]; tx,ty = q[frame,i]
            if not (0<=sx<width and 0<=sy<height and 0<=tx<width and 0<=ty<height): continue
            for dy in range(1-patch_radius,patch_radius):
                for dx in range(1-patch_radius,patch_radius):
                    a,b,c,d = sx+dx,sy+dy,tx+dx,ty+dy
                    if 0<=a<width and 0<=b<height and 0<=c<width and 0<=d<height and z[frame,i]<depth[frame,d,c]:
                        idx=b*width+a
                        raw[frame,d,c]=reference[idx]
                        source[frame,d,c]=idx
                        owner[frame,d,c]=i
                        depth[frame,d,c]=z[frame,i]
    track_coverage = owner >= 0
    if first_frame_policy == 'preserve_reference':
        # A.2 Eq.(1): retain the entire initial iid frame, not only tracked patches.
        raw[0] = reference.reshape(height,width,channels)
        source[0] = np.arange(height*width).reshape(height,width)
        owner[0, ~track_coverage[0]] = -2  # untracked reference, not sphere
    independent = rng.standard_normal(raw.shape).astype(np.float32)
    a,b=1-degradation,degradation
    mixed=(a*raw+b*independent).astype(np.float32)
    return dict(raw=raw, noise=mixed, sphere=sphere, source=source, owner=owner,
                depth=depth, reference=reference.reshape(height,width,channels),
                projected=xy, image_projected=image_xy, track_coverage=track_coverage,
                radius=float(radius), expected_variance=float(a*a+b*b),
                config_json=__import__('json').dumps(dict(
                    version='paper_audit_v2', image_shape=[image_height,image_width],
                    latent_shape=[height,width], env_shape=[eh,ew], channels=channels,
                    depth_bound=depth_bound, env_scale=env_scale, patch_radius=patch_radius,
                    seed=seed, degradation=degradation, blend=blend,
                    sphere_downsample=sphere_downsample, first_frame_policy=first_frame_policy,
                    temporal_mapping='one_input_frame_per_noise_frame',
                    quantization='floor_half_pixel_centers',
                    source_coupling='independent_reference_and_environment')))

