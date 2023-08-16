

import os
import argparse

# third party
import numpy as np
import nibabel as nib
import torch
import glob
from data.PCA import do_PCA
from ddpm import script_utils

# import voxelmorph with pytorch backend
os.environ['NEURITE_BACKEND'] = 'pytorch'
os.environ['VXM_BACKEND'] = 'pytorch'
import voxelmorph_baseline as vxm   # nopep8
from data.filter import measure_filter as mf 

# parse commandline args

def create_argparser():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    defaults = dict(num_images=1, device=device)
    defaults.update(script_utils.diffusion_defaults())

    parser = argparse.ArgumentParser()
    parser.add_argument('--schedule_low',type=float,default=1e-4)
    parser.add_argument('--schedule_high',type=float,default=0.02)
    parser.add_argument("--t1_model_path", type=str,default='DDPM_Models/T1/')
    parser.add_argument("--t2_model_path", type=str,default='DDPM_Models/T2/')
    parser.add_argument("--save_dir", type=str,default='result/4.16/')
    parser.add_argument('--epochs', type=int, default=15000,
                    help='number of training epochs (default: 1500)')
    parser.add_argument('--steps-per-epoch', type=int, default=100,
                    help='frequency of model saves (default: 100)')
    

    parser.add_argument('--model-dir', default='models',
                        help='model output directory (default: models)')
    parser.add_argument('--multichannel', action='store_true',
                        help='specify that data has multiple channels')

    # training parameters
    parser.add_argument('--load-model', help='optional model file to initialize with')
    parser.add_argument('--initial-epoch', type=int, default=0,
                        help='initial epoch number (default: 0)')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-4)')
    parser.add_argument('--cudnn-nondet', action='store_true',
                        help='disable cudnn determinism - might slow down training')

    # network architecture parameters
    parser.add_argument('--enc', type=int, nargs='+',
                        help='list of unet encoder filters (default: 16 32 32 32)')
    parser.add_argument('--dec', type=int, nargs='+',
                        help='list of unet decorder filters (default: 32 32 32 32 32 16 16)')
    parser.add_argument('--int-steps', type=int, default=7,
                        help='number of integration steps (default: 7)')
    parser.add_argument('--int-downsize', type=int, default=2,
                        help='flow downsample factor for integration (default: 2)')
    parser.add_argument('--bidir', action='store_true', help='enable bidirectional cost function')

    # loss hyperparameters
    parser.add_argument('--image-loss', default='ncc',
                        help='image reconstruction loss - can be mse or ncc (default: mse)')
    parser.add_argument('--lambda', type=float, dest='weight', default=0.01,
                        help='weight of deformation loss (default: 0.01)')
    parser.add_argument('--moving',  help='moving image (source) filename',default='D:\\Desktop\\diffusion model code\\voxelmorph-dev\\data\\train\\T1\\OAS30001_MR_d0129_T1w.nii.gz')
    parser.add_argument('--fixed', help='fixed image (target) filename',default='D:\\Desktop\\diffusion model code\\voxelmorph-dev\\data\\train\\T2\\OAS30003_MR_d2669_T2w.nii.gz')
    parser.add_argument('--moved', help='warped image output filename',default='result_ncc_noddpm.nii.gz')
    parser.add_argument('--model',  help='pytorch model for nonlinear registration',default='D:\\Desktop\\diffusion model code\\voxelmorph-dev\model\models_ncc_noddpm\\1500.pt')
    parser.add_argument('--warp', help='output warp deformation filename',default='warped_ncc_noddpm.nii.gz')
    parser.add_argument('-g', '--gpu', help='GPU number(s) - if not supplied, CPU is used',default='1')

    script_utils.add_dict_to_argparser(parser, defaults)
    return parser

args = create_argparser().parse_args()

# device handling
if args.gpu and (args.gpu != '-1'):
    device = 'cuda'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
else:
    device = 'cpu'
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# load moving and fixed images
add_feat_axis = not args.multichannel



# load and set up model
model = vxm.networks.VxmDense.load(args.model, device)
model.to(device)
model.eval()
image_path_list =glob.glob(os.path.join('D:\Desktop\diffusion model code\\voxelmorph-dev\\result\DCT\\onlyDDPM\\',"*T2w.nii.gz"))
for filename in image_path_list[:700]:
    moving_image_path='data\\train\\T1\\'+filename[filename.find('OAS'):filename.find('_to_')]+'.nii.gz'
    fixed_image_path = 'data\\train\\T2\\'+filename[filename.find('_to_')+4:]
    save_path1 =  'result\\DCT\\prueVM\\'+filename[filename.find('OAS'):]
    save_path2 =  'result\\DCT\\prueVM\\'+filename[filename.find('OAS'):filename.find('.nii.gz')]+'_deformation.nii.gz'
    moving = vxm.py.utils.load_volfile(moving_image_path, add_batch_axis=True, add_feat_axis=add_feat_axis)
    fixed, fixed_affine = vxm.py.utils.load_volfile(
        fixed_image_path, add_batch_axis=True, add_feat_axis=add_feat_axis, ret_affine=True)
    moving = torch.from_numpy(moving).to(device).float().permute(0, 4, 1, 2, 3)
    fixed = torch.from_numpy(fixed).to(device).float().permute(0, 4, 1, 2, 3)

    input_moving = moving
    input_fixed = fixed
    input_moving=(input_moving-input_moving.min())/input_moving.max()
    input_fixed=(input_fixed-input_fixed.min())/input_fixed.max()
    moving_image=input_moving
    fixed_image = input_fixed

    moved, warp = model(moving_image, fixed_image, registration=True)

    if args.moved:
        moved = moved.detach().cpu().numpy().squeeze()
        vxm.py.utils.save_volfile(moved, save_path1, fixed_affine)

    if args.warp:
        warp = warp.permute(0,2,3,4,1).detach().cpu().numpy().squeeze()
        vxm.py.utils.save_volfile(warp, save_path2, fixed_affine)
    print('registration : {}'.format(save_path1))
