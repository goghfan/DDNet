import merics
import SimpleITK as sitk
import numpy as np
import glob
import os
import sys
from ddpm import script_utils 
import torch
import argparse


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
    

    parser.add_argument('--model-dir', default='model/models_ncc',
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



#首先获得每个部位的黑白mask矩阵
#针对输入的y_true（也就是fixed_image-T2图像）
#    输入的y_pred（也就是得到的配准结果）
#    形变场（计算雅克比行列式）
#计算metrics
class segment:
#将多结构，包括左右结构等等合到一起，形成一个合理的结构
    def filter_elements(matrix, elements):
        filtered_matrix = np.zeros_like(matrix)  # 创建一个和原始矩阵相同大小的全零矩阵
        for a in elements:
            indices=np.where((matrix>=a-0.5) & (matrix<=a+0.5))
            filtered_matrix[indices] = 255  # 将值为 a 的元素设置为 a
        return filtered_matrix

    def count_non_zero(matrix):
        non_zero_elements = matrix[matrix != 0]  # 使用布尔索引筛选非零元素
        count = non_zero_elements.size  # 获取非零元素的数量
        unique_values = np.unique(non_zero_elements)  # 获取非零元素的唯一值
        return count, unique_values


def get_struct(img_matrix):
    structs={}
    dict_struct = {'ce_nao_shi':"4-43-5-45-",
                "xiao_nao_bai_zhi":"7-46-",
                "xiao_nao_pi_zhi":"8-47-",
                "di_san_nao_shi":"14-",
                "di_si_nao_shi":"15-",
                "di_wu_nao_shi":"72-",
                "pian_zhi_ti":"251-252-253-254-255-",
                "hai_ma_ti":"17-23-",
                "qiu_nao":"10-49-",
                "xing_ren_he":"18-54-",
                "wei-zhuang-he":"11-50-"
                }
    for i,j in dict_struct.items():
        # print(i)
        num=[]
        temp=""
        for k in j:
            if k!='-' and k!=len(j)-1:
                temp+=k
            else:
                # print(temp)
                num.append(int(temp))
                temp=""
        img=segment.filter_elements(img_matrix,num)
        structs[i]=img
    return structs

def compute(y_true,y_pred,deformation_field,mask_matrix):
    structs=get_struct(mask_matrix)
    for i,j in structs.items():
        print("struct is:"+i)
        indices=np.where(j==1)
        temp=y_true.copy()
        temp[indices]=0
        temp_y_true=np.subtract(y_true,temp)
        temp_y_pred=np.subtract(y_pred,temp)
        print("DICE is:{}".format(merics.Dice.dice(temp_y_true,temp_y_pred)))
        print("JD is:{}".format(merics.Jacobians.Get_Jac(deformation_field)))
        print("ASSD is:{}".format(merics.ASSD.assd(temp_y_true,temp_y_pred)))
        print("SSIM is:{}".format(merics.SSIM.ssim(temp_y_true,temp_y_pred)))
        print("PCC is:{}".format(merics.PCC.pcc(temp_y_true,temp_y_pred)))
        print("HD is:{}".format(merics.HD.hd(temp_y_true,temp_y_pred)))
        print("HD95 is:{}".format(merics.HD.hd95(temp_y_true,temp_y_pred)))
def compute_all(y_true,y_pred,deformation_field,moving_mask_image,fixed_mask_image):
    print("DICE is:{}".format(merics.Dice.dice(fixed_mask_image, moving_mask_image,deformation_field)))
    print("JD is:{}".format(merics.JAC.calculate_jacobian_metrics(deformation_field)))
    print("SSIM is:{}".format(merics.SSIM.ssim( y_true, y_pred)))
    print("HD is:{}".format(merics.HD().hd(sitk.GetImageFromArray(y_true), sitk.GetImageFromArray(y_pred))))
    #print("HD95 is:{}".format(merics.HD().hd95(sitk.GetImageFromArray(y_true), sitk.GetImageFromArray(y_pred))))
    print("PCC is:{}".format(merics.PCC.pcc( y_true, y_pred)))
    print("ASSD is:{}".format(merics.ASSD.calculate_assd(y_true, y_pred)))



def compute_metric(fixed_image_path,pred_image_path,deformation_field_path,moving_mask_image,fixed_mask_image):   
    y_true=sitk.ReadImage(fixed_image_path)
    y_pred=sitk.ReadImage(pred_image_path)
    deformation_field_path=sitk.ReadImage(deformation_field_path)
    moving_mask_matrix=sitk.ReadImage(moving_mask_image)
    fixed_mask_matrix = sitk.ReadImage(fixed_mask_image)


    y_true=sitk.GetArrayFromImage(y_true)
    y_true=(y_true-y_true.min())/y_true.max()

    y_pred=sitk.GetArrayFromImage(y_pred)
    deformation_field=np.expand_dims(sitk.GetArrayFromImage(deformation_field_path),axis=0)
    moving_mask_matrix=sitk.GetArrayFromImage(moving_mask_matrix)
    fixed_mask_matrix=sitk.GetArrayFromImage(fixed_mask_matrix)
    compute_all(y_true,y_pred,deformation_field,moving_mask_matrix,fixed_mask_matrix)

fixed_image_path='D:\\Desktop\\diffusion model code\\voxelmorph-dev\\data\\train\\T2\\OAS30003_MR_d2669_T2w.nii.gz'
pred_image_path='D:\\Desktop\\diffusion model code\\voxelmorph-dev\\result\\DCT\\result_ncc_DCT.nii.gz'
deformation_field_path='D:\\Desktop\\diffusion model code\\voxelmorph-dev\\result\\DCT\\warped_ncc_DCT.nii.gz'
moving_mask_image_path='D:\\Desktop\\diffusion model code\\voxelmorph-dev\\metric\\aseg_struct\\OAS30001_MR_d0129_mask.nii.gz'
fixed_mask_image_path='D:\\Desktop\\diffusion model code\\voxelmorph-dev\\metric\\aseg_struct\\OAS30003_MR_d2669_mask.nii.gz'
compute_metric(fixed_image_path,pred_image_path,deformation_field_path,moving_mask_image_path,fixed_mask_image_path)
