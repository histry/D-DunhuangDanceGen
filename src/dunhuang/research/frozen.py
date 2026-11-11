"""Load the exact frozen research backbone without introducing a new model."""
import sys,zipfile,pickletools
from dunhuang.io import pinned,checked

def payload(assets):
    import torch
    root=pinned(assets['edge_root'],assets['edge_commit']);sys.path.insert(0,str(root))
    from dataset.preprocess import Normalizer
    from dataset.scaler import MinMaxScaler
    file=checked(assets['checkpoint'])
    with zipfile.ZipFile(file) as archive:
        data=archive.read(next(n for n in archive.namelist() if n.endswith('data.pkl')))
        names={arg for op,arg,_ in pickletools.genops(data) if op.name=='GLOBAL'}
    allowed={'collections OrderedDict','torch._utils _rebuild_tensor_v2','torch FloatStorage','torch LongStorage','torch DoubleStorage','torch HalfStorage','torch._utils _rebuild_parameter','dataset.preprocess Normalizer','dataset.scaler MinMaxScaler','numpy.core.multiarray _reconstruct','numpy ndarray','numpy dtype','_codecs encode','numpy.core.multiarray scalar'}
    if not names<=allowed:raise ValueError('Unexpected checkpoint globals')
    result=torch.load(file,map_location='cpu',weights_only=False)
    if not isinstance(result['normalizer'],Normalizer):raise ValueError('Frozen Normalizer type differs')
    return result

def denoiser(assets):
    import torch
    checkpoint=payload(assets)
    from model.model import DanceDecoder
    from model.diffusion import GaussianDiffusion
    from vis import SMPLSkeleton
    model=DanceDecoder(nfeats=151,seq_len=150,latent_dim=512,ff_size=1024,num_layers=8,num_heads=8,dropout=.1,cond_feature_dim=4800,activation=torch.nn.functional.gelu)
    model.load_state_dict(checkpoint['ema_state_dict'],strict=True)
    normalizer=checkpoint['normalizer'];del checkpoint
    for p in model.parameters():p.requires_grad_(False)
    diffusion=GaussianDiffusion(model,150,151,SMPLSkeleton('cuda'),schedule='cosine',n_timestep=1000,predict_epsilon=False,loss_type='l2',use_p2=False,cond_drop_prob=.25,guidance_weight=2).cuda().eval()
    return model,normalizer,diffusion
