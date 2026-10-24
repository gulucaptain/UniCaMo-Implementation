import sys, unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from unicamo_noise import construct_noise, sphere_indices

class GeometryTests(unittest.TestCase):
    def setup_scene(self,t=2):
        return np.tile([[8.,0,4],[0,8.,4],[0,0,1]],(t,1,1)),np.tile(np.eye(4),(t,1,1))
    def build(self,tracks,k,p,**kw):
        options=dict(image_height=8,image_width=8,depth_bound=8.,
            sphere_downsample='nearest',first_frame_policy='preserve_reference',degradation=0.)
        options.update(kw)
        return construct_noise(tracks,k,p,height=8,width=8,channels=128,**options)
    def test_translation_and_source_copy(self):
        k,p=self.setup_scene();tracks=np.array([[[0,0,4]],[[1,0,4]]])
        r=self.build(tracks,k,p)
        np.testing.assert_array_equal(r['raw'][1,3:6,5:8],r['reference'][3:6,3:6])
    def test_depth_collision_order_and_visibility(self):
        k,p=self.setup_scene();tracks=np.array([[[-1,0,4],[1,0,4]],[[0,0,3],[0,0,5]]])
        r=self.build(tracks,k,p)
        self.assertEqual(r['owner'][1,4,4],0)
        self.assertEqual(r['source'][1,4,4],4*8+2)
        swapped=self.build(tracks[:,::-1],k,p)
        np.testing.assert_array_equal(r['raw'],swapped['raw'])
        tracks[1,:,2]=[5,3]
        r=self.build(tracks,k,p);self.assertEqual(r['owner'][1,4,4],1)
        r=self.build(tracks,k,p,visibility=[[1,1],[1,0]])
        self.assertEqual(r['owner'][1,4,4],0)
    def test_static_sphere_and_reference_identity(self):
        k,p=self.setup_scene();r=self.build(np.empty((2,0,3)),k,p,first_frame_policy='compose')
        np.testing.assert_array_equal(r['raw'][0],r['raw'][1])
        ids=r['source'].ravel();values=r['raw'].reshape(-1,128)
        _,first,inverse=np.unique(ids,return_index=True,return_inverse=True)
        np.testing.assert_array_equal(values,values[first][inverse])
    def test_geometry_independent_reference(self):
        k,p=self.setup_scene();p[1,0,3]=-1
        idx=sphere_indices(k,p,8,8,8,32,32)
        # A hand-computed ray intersection for one translated camera pixel.
        ray=np.array([(.5-4)/8,(.5-4)/8,1]);ray/=np.linalg.norm(ray)
        o=np.array([1.,0,0]);dist=-np.dot(ray,o)+np.sqrt(np.dot(ray,o)**2-(np.dot(o,o)-64))
        n=(o+dist*ray)/8
        u=(np.arctan2(n[0],n[2])/(2*np.pi)+.5)%1;v=np.arccos(n[1])/np.pi
        self.assertEqual(idx[1,0,0],int(v*32)*32+int(u*32))
        self.assertFalse(np.array_equal(idx[0],idx[1]))
    def test_rotation_world_ray_lookup(self):
        k,p=self.setup_scene()
        # +90 degree camera yaw: camera z ray is world +x.
        c2w=np.eye(4); c2w[:3,:3]=[[0,0,1],[0,1,0],[-1,0,0]]
        p[1]=np.linalg.inv(c2w)
        # Place principal point exactly on the sampled center.
        k[:,0,2]=4.5;k[:,1,2]=4.5
        idx=sphere_indices(k,p,8,8,8,32,32)
        self.assertEqual(idx[0,4,4],16*32+16)
        self.assertEqual(idx[1,4,4],16*32+24)

    def test_invalid_and_behind_camera(self):
        k,p=self.setup_scene();r=self.build(np.array([[[0,0,-1]],[[0,0,2]]]),k,p)
        self.assertFalse(r['track_coverage'].any())
        self.assertTrue((r['owner'][1:]==-1).all())
        with self.assertRaises(ValueError):self.build(np.empty((2,0,3)),k,p,degradation=2)
    def test_ensemble_moments_and_blending(self):
        k,p=self.setup_scene();tracks=np.empty((2,0,3))
        for blend,expected in [('linear',.68)]:
            arrays=[self.build(tracks,k,p,seed=s,degradation=.2,blend=blend)['noise'] for s in range(20)]
            x=np.concatenate(arrays)
            self.assertLess(abs(x.mean()),.025)
            self.assertLess(abs(x.var()-expected),.035)

    def test_first_frame_is_full_iid_reference(self):
        k,p=self.setup_scene()
        r=self.build(np.empty((2,0,3)),k,p)
        np.testing.assert_array_equal(r['raw'][0],r['reference'])
        np.testing.assert_array_equal(r['source'][0],np.arange(64).reshape(8,8))
        self.assertTrue((r['owner'][0]==-2).all())

    def test_image_projection_and_downsampling(self):
        import json
        k,p=self.setup_scene();k[:,:2,:]*=2
        r=self.build(np.array([[[0,0,4]],[[1,0,4]]]),k,p,image_height=16,image_width=16)
        np.testing.assert_allclose(r['projected'],[[[4,4]],[[6,4]]])
        config=json.loads(r['config_json']);self.assertEqual(config['env_shape'],[64,64])
        full=sphere_indices(k,p,16,16,8.,64,64)
        # center-aligned nearest resize of the image-space source-ID grid.
        expected=full[:,1::2,1::2]+64
        uncovered=r['owner'][1]==-1
        np.testing.assert_array_equal(r['source'][1][uncovered],expected[1][uncovered])

    def test_no_improved_blend_default(self):
        k,p=self.setup_scene();tracks=np.empty((2,0,3))
        r=self.build(tracks,k,p,degradation=.2)
        self.assertAlmostEqual(r['expected_variance'],.68)
        with self.assertRaises(ValueError):self.build(tracks,k,p,blend='variance_preserving')
        with self.assertRaises(TypeError):
            construct_noise(tracks,k,p,height=8,width=8)

if __name__=='__main__':unittest.main()
