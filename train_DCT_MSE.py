'''
读取数据
输入数据到DDPM—model
输入数据到VM
Loss-ncc
'''
import argparse
import datetime
import torch
import wandb

from torch.utils.data import DataLoader
from torchvision import datasets
import sys
from ddpm import script_utils
import os
from data import data
from data.filter import measure_filter as mf 
import torch.distributed as dist 
from torch.nn.parallel import DistributedDataParallel as DDP
import time
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
import glob
from tqdm import tqdm
from data.PCA import do_PCA
from data.DCT import dct_3d,idct_3d
import numpy as np
# import voxelmorph with pytorch backend
os.environ['NEURITE_BACKEND'] = 'pytorch'
os.environ['VXM_BACKEND'] = 'pytorch'
import voxelmorph as vxm  # nopep8
import warnings
warnings.filterwarnings("ignore")


def main():
    #数据初始化
    batch_size=1
    train_dataset_t1 = data.create_dataset(dataroot='data/',type='nii.gz',T1orT2='T1',trainortest='train')
    train_loader_t1=script_utils.cycle(data.create_dataloader(train_dataset_t1,batch_size=1,phase='train'))
    train_dataset_t2 =  data.create_dataset(dataroot='data/',type='nii.gz',T1orT2='T2',trainortest='train')      
    train_loader_t2=script_utils.cycle(data.create_dataloader(train_dataset_t2,batch_size=1,phase='train'))
    # DDPM 初始化
    args = create_argparser().parse_args()
    device = args.device
    epochs = args.epochs
    steps_per_epoch = args.steps_per_epoch
    path_t1 = args.t1_model_path
    path_t2 = args.t2_model_path
    model_t1 = glob.glob(os.path.join(path_t1,"*-{}".format('model.pth')))
    model_t2 = glob.glob(os.path.join(path_t2,"*-{}".format('model.pth')))
    #VM初始化
    #VM
    bidir = args.bidir
    # no need to append an extra feature axis if data is multichannel
    add_feat_axis = not args.multichannel
    inshape = next(train_loader_t1).shape[-3:]
    # prepare model folder
    model_dir = args.model_dir
    os.makedirs(model_dir, exist_ok=True)
    # device handling
    gpus = args.gpu.split(',')
    nb_gpus = len(gpus)
    device = 'cuda:0'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    assert np.mod(1, nb_gpus) == 0, \
        'Batch size (%d) should be a multiple of the nr of gpus (%d)' % (args.batch_size, nb_gpus)
    # enabling cudnn determinism appears to speed up training by a lot
    torch.backends.cudnn.deterministic = not args.cudnn_nondet
    # unet architecture
    enc_nf = args.enc if args.enc else [16, 32, 32, 32]
    dec_nf = args.dec if args.dec else [32, 32, 32, 32, 32, 16, 16]
    if args.load_model:
        # load initial model (if specified)
        model = vxm.networks.VxmDense.load(args.load_model, device)
    else:
        # otherwise configure new model
        model = vxm.networks.VxmDense(
            inshape=inshape,
            nb_unet_features=[enc_nf, dec_nf],
            bidir=bidir,
            int_steps=args.int_steps,
            int_downsize=args.int_downsize
            )
    if nb_gpus > 1:
        # use multiple GPUs via DataParallel
        model = torch.nn.DataParallel(model)
        model.save = model.module.save
    # prepare the model for training and send to device
    model.to(device)
    model.train()
    # set optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # prepare image loss
    if args.image_loss == 'ncc':
        image_loss_func = vxm.losses.NCC().loss
    elif args.image_loss == 'mse':
        image_loss_func = vxm.losses.MSE().loss
    else:
        raise ValueError('Image loss should be "mse" or "ncc", but found "%s"' % args.image_loss)
            # need two image loss functions if bidirectional
    if bidir:
        losses = [image_loss_func, image_loss_func]
        weights = [0.5, 0.5]
    else:
        losses = [image_loss_func]
        weights = [1]
        # prepare deformation loss
    losses += [vxm.losses.Grad('l2', loss_mult=args.int_downsize).loss]
    weights += [args.weight]
    diffusion_t1 = script_utils.get_diffusion_from_args(args).to(device)
    diffusion_t1.load_state_dict(torch.load(model_t1[0],map_location=device))
    diffusion_t2 = script_utils.get_diffusion_from_args(args).to(device)
    diffusion_t2.load_state_dict(torch.load(model_t2[0],map_location=device))
    try:
        file = open('loss/train_detail_DCT.txt','w')
        for epoch in range(epochs):
            if epoch % 100 == 0:
                model.save(os.path.join(model_dir, '%04d.pt' % epoch))
            epoch_loss = []
            epoch_total_loss = []
            epoch_step_time = []   
            for step in tqdm(range(steps_per_epoch)):
                step_start_time = time.time()
                #只对T1 T2做一步加噪声、然后主成分分析后跟原图进行通道拼接，输入到配准网络里面
                #DDPM
                moving_image_origin=next(train_loader_t1).to(device)
                fixed_image_origin = next(train_loader_t2).to(device)
                #只对T1 T2做一步加噪声、然后主成分分析后跟原图进行通道拼接，输入到配准网络里面
                #DDPM
                t = torch.full((1,), 0, device=device, dtype=torch.long)
                diffusion1=diffusion_t1.module.get_feature_extraction(moving_image_origin,t)
                moving_image = moving_image_origin-diffusion1

                diffusion2=diffusion_t2.module.get_feature_extraction(fixed_image_origin,t)
                fixed_image = fixed_image_origin-diffusion2

                image = mf(moving_path=moving_image.to('cpu').detach().numpy(),fixed_path=fixed_image.to('cpu').detach().numpy())

                #对T1 T2一步加噪的内容、特征强化后的内容、再拼接到现有的数据里，infeat变成4



                moving_image_DCT = dct_3d(image[0])
                moving_image_iDCT=idct_3d(moving_image_DCT)


                fixed_image_DCT = dct_3d(image[1])
                fixed_image_iDCT = idct_3d(fixed_image_DCT)
                

                diffusion_moving_DCT=dct_3d(diffusion1.to('cpu').detach().numpy())
                diffusion_moving_DCT=idct_3d(diffusion_moving_DCT)


                diffusion_fixed_DCT = dct_3d(diffusion2.to('cpu').detach().numpy()) 
                diffusion_fixed_DCT = idct_3d(diffusion_fixed_DCT)


                moving_image=torch.from_numpy(image[0])
                fixed_image=torch.from_numpy(image[1])
                moving_image_iDCT = torch.from_numpy(moving_image_iDCT).unsqueeze(0).unsqueeze(0)
                fixed_image_iDCT = torch.from_numpy(fixed_image_iDCT).unsqueeze(0).unsqueeze(0)
                diffusion_moving_DCT=torch.from_numpy(diffusion_moving_DCT).unsqueeze(0).unsqueeze(0)
                diffusion_fixed_DCT=torch.from_numpy(diffusion_fixed_DCT).unsqueeze(0).unsqueeze(0)


                moving_image = torch.cat((moving_image_origin,moving_image.to(device),moving_image_iDCT.to(device),diffusion_moving_DCT.to(device)),dim=1).float()
                fixed_image = torch.cat((fixed_image_origin,fixed_image.to(device),fixed_image_iDCT.to(device),diffusion_fixed_DCT.to(device)),dim=1).float()
                predict_image=model(moving_image,fixed_image)


                compute_loss = [fixed_image_origin,moving_image_origin]
                loss=0
                loss_list=[]
                mse_loss = vxm.losses.MSE().loss
                for n,loss_function in enumerate(losses):
                    curr_loss = loss_function(compute_loss[n], predict_image[n]) * weights[n]
                    loss_list.append(curr_loss.item())
                    loss += curr_loss
                mse_dct_loss = (-mse_loss(torch.from_numpy(fixed_image_DCT).to(device),torch.from_numpy(moving_image_DCT).to(device))*1e-4).item()
                loss_list.append(mse_dct_loss)
                loss+=mse_dct_loss
                epoch_loss.append(loss_list)
                epoch_total_loss.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_step_time.append(time.time() - step_start_time)
            # print epoch info
            epoch_info = 'Epoch %d/%d' % (epoch, args.epochs)
            time_info = '%.4f sec/step' % np.mean(epoch_step_time)
            losses_info = ', '.join(['%.4e' % f for f in np.mean(epoch_loss, axis=0)])
            loss_info = 'loss: %.4e  (%s)' % (np.mean(epoch_total_loss), losses_info)
            print(' - '.join((epoch_info, time_info, loss_info)), flush=True)
            file.write(" - ".join((epoch_info, time_info, loss_info)) + "\n")
        print('Train success!')
        model.save(os.path.join(model_dir, '%04d.pt' % args.epochs))

        
    except KeyboardInterrupt:
        print("Keyboard interrupt, generation finished early")


def create_argparser():
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    defaults = dict(num_images=1, device=device)
    defaults.update(script_utils.diffusion_defaults())

    parser = argparse.ArgumentParser()
    parser.add_argument('--schedule_low',type=float,default=1e-4)
    parser.add_argument('--schedule_high',type=float,default=0.02)
    parser.add_argument("--t1_model_path", type=str,default='DDPM_Models/T1/')
    parser.add_argument("--t2_model_path", type=str,default='DDPM_Models/T2/')
    parser.add_argument("--save_dir", type=str,default='result/4.16/')
    parser.add_argument('--epochs', type=int, default=1500,
                    help='number of training epochs (default: 1500)')
    parser.add_argument('--steps-per-epoch', type=int, default=100,
                    help='frequency of model saves (default: 100)')
    

    parser.add_argument('--model-dir', default='model/models_DCT',
                        help='model output directory (default: models)')
    parser.add_argument('--multichannel', action='store_true',
                        help='specify that data has multiple channels')

    # training parameters
    parser.add_argument('--gpu', default='0', help='GPU ID number(s), comma-separated (default: 0)')
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
    script_utils.add_dict_to_argparser(parser, defaults)
    return parser

if __name__=="__main__":
    main()
