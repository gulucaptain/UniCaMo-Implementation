"""Run deterministic synthetic experiments, export GIFs/NPZ/statistics/report."""
from pathlib import Path
import sys, json, hashlib, platform
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image, ImageDraw
from unicamo_noise import construct_noise
OUT=ROOT/'results'
H,W,T,C=64,96,25,16
IH,IW=128,192


def build(tracks,k,pose,seed=42,degradation=.2,first_frame_policy='preserve_reference'):
    return construct_noise(tracks,k,pose,height=H,width=W,channels=C,
        image_height=IH,image_width=IW,depth_bound=8.,
        sphere_downsample='nearest',first_frame_policy=first_frame_policy,
        seed=seed,degradation=degradation,blend='linear')


def scene(name):
    k=np.tile([[75.,0,W/2],[0,75.,H/2],[0,0,1]],(T,1,1))
    pose=np.tile(np.eye(4),(T,1,1))
    times=np.linspace(0,1,T)
    gx,gy=np.meshgrid(np.linspace(-.45,.45,7),np.linspace(-.4,.4,7))
    base=np.stack([gx.ravel(),gy.ravel(),np.zeros(gx.size)],-1)
    if name=='object_translation':
        tracks=np.tile(base,(T,1,1)); tracks[:,:,2]=4
        tracks[:,:,0]+=(-1+2*times)[:,None]
    elif name=='occlusion':
        a=np.tile(base,(T,1,1)); b=a.copy()
        a[:,:,0]+=(-.9+1.8*times)[:,None]; b[:,:,0]+=(.9-1.8*times)[:,None]
        a[:,:,2]=(3.2+1.6*times)[:,None]; b[:,:,2]=(4.8-1.6*times)[:,None]
        tracks=np.concatenate([a,b],axis=1)
    else:
        tracks=np.empty((T,0,3))
        for j,s in enumerate(times):
            if name=='camera_rotation':
                angle=np.deg2rad(-15+30*s)
                rot=np.array([[np.cos(angle),0,np.sin(angle)],[0,1,0],[-np.sin(angle),0,np.cos(angle)]])
                c2w=np.eye(4); c2w[:3,:3]=rot
            else:
                c2w=np.eye(4); c2w[0,3]=2*s
            pose[j]=np.linalg.inv(c2w)
    # Controlled synthetic foregrounds plus static background tracks: N=1024.
    # Foreground counts are test fixtures (49/98), not a reimplementation of GUI masks.
    rng=np.random.default_rng(10)
    count=1024-tracks.shape[1]
    y,x=np.mgrid[:H,:W]
    available=np.ones((H,W),bool)
    if tracks.shape[1]:
        from unicamo_noise import project
        uv,_=project(tracks[:1],k[:1],pose[:1])
        for a,b in uv[0]:
            available[max(0,int(b)-2):min(H,int(b)+3),max(0,int(a)-2):min(W,int(a)+3)]=False
    selected=rng.choice(np.flatnonzero(available),count,replace=False)
    pixels=np.stack([x.ravel()[selected]+.5,y.ravel()[selected]+.5,np.ones(count)],-1)
    camera=pixels@np.linalg.inv(k[0]).T*8.
    c2w=np.linalg.inv(pose[0]); world=camera@c2w[:3,:3].T+c2w[:3,3]
    background=np.tile(world,(T,1,1))
    tracks=np.concatenate([tracks,background],axis=1)
    k=k.copy();k[:,:2,:]*=2
    return tracks,k,pose


def corr(a,b):
    a=a.ravel().astype(float);b=b.ravel().astype(float)
    return float(np.corrcoef(a,b)[0,1])


def statistics(result):
    x=result['noise']; raw=result['raw']; ids=result['source']
    # Pooling frames estimates marginal moments, not an iid significance test.
    return dict(mean=float(x.mean()),variance=float(x.var()),
        skewness=float(np.mean(((x-x.mean())/x.std())**3)),
        excess_kurtosis=float(np.mean(((x-x.mean())/x.std())**4)-3),
        per_frame_variance=x.var(axis=(1,2,3)).tolist(),
        horizontal_corr=corr(x[:,:,:-1],x[:,:,1:]),
        vertical_corr=corr(x[:,:-1],x[:,1:]),
        same_pixel_temporal_corr=corr(x[:-1],x[1:]),
        raw_horizontal_corr=corr(raw[:,:,:-1],raw[:,:,1:]),
        raw_horizontal_expected=float((ids[:,:,:-1]==ids[:,:,1:]).mean()),
        raw_temporal_corr=corr(raw[:-1],raw[1:]),
        raw_temporal_expected=float((ids[:-1]==ids[1:]).mean()),
        unique_source_fraction=float(np.mean([len(np.unique(i))/(H*W) for i in ids])),
        track_coverage=float((result['owner']>=0).mean()),
        expected_variance=result['expected_variance'],
        raw_variance=float(raw.var()),
        raw_per_frame_variance=raw.var(axis=(1,2,3)).tolist(),
        raw_temporal_after_first=corr(raw[1:-1],raw[2:]),
        raw_temporal_expected_after_first=float((ids[1:-1]==ids[2:]).mean()),
        expected_horizontal_corr=float((ids[:,:,:-1]==ids[:,:,1:]).mean())*.64/.68,
        config=json.loads(result['config_json']))


def render(name,result):
    images=[]
    for j in range(T):
        canvas=Image.new('RGB',(W*5*3, H*3+76),'#111827'); d=ImageDraw.Draw(canvas)
        d.text((12,8),f'{name} | frame {j+1:02}/{T} | channel 0 | raw vs linear alpha=0.2',fill='white')
        track=result['raw'][j,:,:,0].copy()
        mask=result['owner'][j]>=0
        panels=[track,result['sphere'][j,:,:,0],result['raw'][j,:,:,0],result['noise'][j,:,:,0]]
        for i,p in enumerate(panels):
            grey=np.uint8(np.clip((p+3)/6,0,1)*255)
            rgb=np.repeat(grey[...,None],3,-1)
            if i==0: rgb[~mask]=[40,47,60]
            im=Image.fromarray(rgb).resize((W*3,H*3),Image.Resampling.NEAREST)
            canvas.paste(im,(i*W*3,40))
        own=np.zeros((H,W,3),np.uint8); own[:]=[40,47,60]
        own[mask]=[90,120,160]  # static background
        own[result['owner'][j]==-2]=[120,75,140]
        if name in ('object_translation','occlusion'):
            own[(result['owner'][j]>=0)&(result['owner'][j]<49)]=[70,210,230]
        if name=='occlusion': own[(result['owner'][j]>=49)&(result['owner'][j]<98)]=[255,150,60]
        canvas.paste(Image.fromarray(own).resize((W*3,H*3),Image.Resampling.NEAREST),(4*W*3,40))
        for i,label in enumerate(['Track patches','Sphere sampling','Composed RAW','Final: linear blend','Owner: A / B / background']):
            d.text((i*W*3+8,H*3+49),label,fill='white')
        images.append(canvas)
    images[0].save(OUT/f'{name}.gif',save_all=True,append_images=images[1:],duration=100,loop=0)
    images[T//2].save(OUT/f'{name}.png')


def main():
    OUT.mkdir(exist_ok=True)
    manifest=dict(version='paper_audit_v2', python=sys.version, numpy=np.__version__,
        platform=platform.platform(), source='https://arxiv.org/html/2607.02798v1',
        files_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
          for p in [ROOT/'unicamo_noise/core.py',Path(__file__),
                    ROOT/'docs/PAPER_AUDIT.md']})
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
    summary={}
    validation={}
    for name in ['object_translation','occlusion','camera_rotation','camera_translation']:
        tracks,k,pose=scene(name)
        r=build(tracks,k,pose)
        render(name,r)
        np.savez_compressed(OUT/f'{name}.npz',**r,tracks=tracks,intrinsics=k,w2c=pose,visibility=np.ones(tracks.shape[:2],bool))
        summary[name]=statistics(r)
        ids=r['source'].ravel(); values=r['raw'].reshape(-1,C)
        _,first,inverse=np.unique(ids,return_index=True,return_inverse=True)
        matched_error=float(np.max(np.abs(values-values[first][inverse])))
        track_mask=ids<H*W
        copy_error=float(np.max(np.abs(values[track_mask]-r['reference'].reshape(-1,C)[ids[track_mask]]))) if track_mask.any() else 0.
        assert matched_error==0 and copy_error==0
        validation[name]=dict(same_source_max_error=matched_error,track_copy_max_error=copy_error,
                              marginal_variance_ok=abs(summary[name]['variance']-.68)<.03)
        assert validation[name]['marginal_variance_ok']
    # Independent seeds matter: within-video values are deliberately correlated.
    ensemble=[]
    tracks,k,pose=scene('occlusion')
    for seed in range(20):
        r=build(tracks,k,pose,seed=seed)
        ensemble.append(statistics(r))
    (OUT/'ensemble_samples.json').write_text(json.dumps(ensemble,indent=2))
    keys=['mean','variance','horizontal_corr','raw_horizontal_corr','raw_horizontal_expected','raw_temporal_corr','raw_temporal_expected']
    summary['ensemble_20_seeds']={key:dict(mean=float(np.mean([x[key] for x in ensemble])),std=float(np.std([x[key] for x in ensemble]))) for key in keys}
    # Explicit ambiguity sensitivity checks, not improved defaults.
    summary['ambiguity_sensitivity']={}
    for policy in ['preserve_reference','compose']:
        r=build(tracks,k,pose,first_frame_policy=policy)
        summary['ambiguity_sensitivity'][policy]=statistics(r)
    summary['sphere_only_diagnostics']={}
    for name in ['camera_rotation','camera_translation']:
        _,k,pose=scene(name)
        r=build(np.empty((T,0,3)),k,pose)
        summary['sphere_only_diagnostics'][name]=statistics(r)
    (OUT/'validation.json').write_text(json.dumps(validation,indent=2))
    (OUT/'statistics.json').write_text(json.dumps(summary,indent=2))
    names=list(summary)[:4]
    chart=Image.new('RGB',(1200,430),'white'); draw=ImageDraw.Draw(chart)
    colors=['#0072B2','#D55E00','#009E73','#CC79A7']
    titles=['Final variance (0.62 to 0.74)','Final density vs N(0,0.68)','Raw temporal correlation / ID prediction']
    for panel,title in enumerate(titles):
        left=panel*400+35
        draw.text((left,15),title,fill='black')
        draw.line([(left,45),(left,310),(left+340,310)],fill='black')
    for i,name in enumerate(names):
        color=colors[i]
        variances=summary[name]['per_frame_variance']
        draw.line([(35+j*340/(T-1),310-(v-.62)/.12*265) for j,v in enumerate(variances)],fill=color,width=2)
        a=np.load(OUT/f'{name}.npz')['noise']
        hist,edges=np.histogram(a.ravel(),bins=65,range=(-4,4),density=True)
        centers=(edges[:-1]+edges[1:])/2
        draw.line([(435+(x+4)/8*340,310-y/.55*265) for x,y in zip(centers,hist)],fill=color,width=2)
        x=835+i*82
        for j,key in enumerate(['raw_temporal_corr','raw_temporal_expected']):
            value=max(0,summary[name][key]);draw.rectangle((x+j*26,310-value*265,x+j*26+20,310),fill=color if j==0 else '#999999')
        draw.text((x,317),['Object','Overlap','Rotate','Translate'][i],fill='black')
        draw.text((35,345+i*18),name,fill=color)
    draw.line([(35,177.5),(375,177.5)],fill='#888888')
    z=np.linspace(-4,4,160)
    draw.line([(435+(x+4)/8*340,310-np.exp(-x*x/(2*.68))/np.sqrt(2*np.pi*.68)/.55*265) for x in z],fill='black',width=2)
    draw.text((435,345),'x: -4 to +4; black: Gaussian density',fill='black')
    draw.text((835,345),'Color: measured; gray: predicted',fill='black')
    chart.save(OUT/'statistics.png')
    lines=['# UniCaMo 论文核对后实验（v2）','',
        '正文 §3.1–3.2、§4.3，附录 A.2–A.3 和 B.3 逐项核对见 ../docs/PAPER_AUDIT.md。', '',
        '配置：图像 128×192 → 噪声网格 64×96；球面纹理 512×768（图像尺寸×4）；25 帧、16 通道、1024 条轨迹、seed=42。尺寸为合成实验配置，不冒充 Wan 配置。', '',
        '默认使用通常 alpha blending 的线性解释：0.8 raw + 0.2 iid，未归一化。公式原文未明确，属于有标注的解释；理论方差 0.68。首帧保持完整 iid raw（附录 A.2 Eq.1 的直接解释）；图像到 latent 采用显式选择的最近邻降采样。', '',
        '| 场景 | 最终均值 | raw 方差 | 最终方差 | 横向相关 | raw 时间相关 / 来源预测 |',
        '|---|---:|---:|---:|---:|---:|']
    for n in names:
        s=summary[n]; lines.append(f"| {n} | {s['mean']:.4f} | {s['raw_variance']:.4f} | {s['variance']:.4f} | {s['horizontal_corr']:.4f} | {s['raw_temporal_corr']:.4f} / {s['raw_temporal_expected']:.4f} |")
    lines+=['',
        '全部 raw 同源采样一致，所有实际轨迹覆盖位置均与首帧参考源一致，最大误差为 0；验证记录见 validation.json。', '',
        '20 个独立种子的遮挡场景统计见 statistics.json。相关系数与 source-ID 预测对照，不把相关的像素视为独立样本做显著性检验。', '',
        '另记录两种首帧策略的敏感性，以及两种相机运动的纯球面诊断。主场景现在都有背景轨迹；它们并非上一版的纯球面实验，不能把总相关性变化单独归因为纹理分辨率。', '',
        '首帧独立源与球面独立源的衔接仍未由论文充分定义；首帧附近时间相关会受该选择影响。逐帧方差和排除首帧的时间相关另存 JSON。', '',
        '动画依次显示轨迹噪声、球面噪声、raw 合成、线性混合后最终噪声、所有权（青=A，橙=B，蓝灰=背景，紫=首帧未覆盖参考源）。固定 [-3,3] 灰度映射；统计不截断。', '',
        '本版是可追溯的论文描述实现，未宣称与官方代码逐值一致；不包含 Wan 或视频质量复现。此前探索版本未包含在此独立仓库中。']
    (OUT/'REPORT.md').write_text('\n'.join(lines))
    print(json.dumps({n:summary[n] for n in names},indent=2))

if __name__=='__main__':main()
