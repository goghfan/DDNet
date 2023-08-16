import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt

import torch
from torch.nn import functional as F
import torch.nn.functional as nnf

def mk_grid_img_1(grid_step, line_thickness=1, grid_sz=(128, 128, 128)):
    grid_img = np.zeros(grid_sz)
    for j in range(0, grid_img.shape[1], grid_step):
        grid_img[:, j+line_thickness-1, :] = 1
    for i in range(0, grid_img.shape[2], grid_step):
        grid_img[:, :, i+line_thickness-1] = 1
    grid_img = grid_img[None, None, ...]
    grid_img = torch.from_numpy(grid_img).cuda()
    return grid_img

def mk_grid_img_2(grid_step, line_thickness=1, grid_sz=(128, 128, 128)):
    grid_img = np.zeros(grid_sz)
    for j in range(0, grid_img.shape[1], grid_step):
        grid_img[j+line_thickness-1, :, :] = 1
    for i in range(0, grid_img.shape[2], grid_step):
        grid_img[:, i+line_thickness-1, :] = 1
    grid_img = grid_img[None, None, ...]
    grid_img = torch.from_numpy(grid_img).cuda()
    return grid_img

def mk_grid_img_3(grid_step, line_thickness=1, grid_sz=(128, 128, 128)):
    grid_img = np.zeros(grid_sz)
    for j in range(0, grid_img.shape[1], grid_step):
        grid_img[j+line_thickness-1, :, :] = 1
    for i in range(0, grid_img.shape[2], grid_step):
        grid_img[:,:, i+line_thickness-1] = 1
    grid_img = grid_img[None, None, ...]
    grid_img = torch.from_numpy(grid_img).cuda()
    return grid_img


def warp(x, disp, interp_mode="bilinear"):
    """
    Spatially transform an image by sampling at transformed locations (2D and 3D)
    Args:
        x: (Tensor float, shape (N, ndim, *sizes)) input image
        disp: (Tensor float, shape (N, ndim, *sizes)) dense disp field in i-j-k order (NOT spatially normalised)
        interp_mode: (string) mode of interpolation in grid_sample()
    Returns:
        deformed x, Tensor of the same shape as input
    """
    ndim = x.ndim - 2
    size = x.size()[2:]
    disp = disp.type_as(x)

    # normalise disp to [-1, 1]
    disp = normalise_disp(disp)

    # generate standard mesh grid
    grid = torch.meshgrid([torch.linspace(-1, 1, size[i]).type_as(disp) for i in range(ndim)])
    grid = [grid[i].requires_grad_(False) for i in range(ndim)]

    # apply displacements to each direction (N, *size)
    warped_grid = [grid[i] + disp[:, i, ...] for i in range(ndim)]

    # swapping i-j-k order to x-y-z (k-j-i) order for grid_sample()
    warped_grid = [warped_grid[ndim - 1 - i] for i in range(ndim)]
    warped_grid = torch.stack(warped_grid, -1)  # (N, *size, dim)

    return F.grid_sample(x, warped_grid, mode=interp_mode, align_corners=False)

def normalise_disp(disp):
    """
    Spatially normalise DVF to [-1, 1] coordinate system used by Pytorch `grid_sample()`
    Assumes disp size is the same as the corresponding image.
    Args:
        disp: (numpy.ndarray or torch.Tensor, shape (N, ndim, *size)) Displacement field
    Returns:
        disp: (normalised disp)
    """

    ndim = disp.ndim - 2

    if type(disp) is np.ndarray:
        norm_factors = 2. / np.array(disp.shape[2:])
        norm_factors = norm_factors.reshape(1, ndim, *(1,) * ndim)

    elif type(disp) is torch.Tensor:
        norm_factors = torch.tensor(2.) / torch.tensor(disp.size()[2:], dtype=disp.dtype, device=disp.device)
        norm_factors = norm_factors.view(1, ndim, *(1,)*ndim)

    else:
        raise RuntimeError("Input data type not recognised, expect numpy.ndarray or torch.Tensor")
    return disp * norm_factors

def comput_fig(img1,img2,img3):
    img_list=[]
    img_list.append(img1.detach().cpu().numpy()[0, 0, 64, :, :])
    img_list.append(img2.detach().cpu().numpy()[0, 0, :, :, 64])
    img_list.append(img3.detach().cpu().numpy()[0, 0, :, 64, :])
    fig = plt.figure(figsize=(16,16), dpi=180)
    for i in range(len(img_list)):
        plt.subplot(1, 3, i + 1)
        plt.axis('off')
        plt.imshow(img_list[i], cmap='gray')
    fig.subplots_adjust(wspace=1, hspace=0)
    return fig

if __name__ == "__main__":
    # 读入形变场
    phi = sitk.ReadImage("result\\ablation_study\\warped_ncc_vm.nii.gz")  # [324,303,2]
    phi_arr = torch.from_numpy(sitk.GetArrayFromImage(phi)).float().unsqueeze(dim=0)
    # 产生网格图片

    grid_img1 = mk_grid_img_1(4, 1)
    grid_img2 = mk_grid_img_2(4, 1)
    grid_img3 = mk_grid_img_3(4, 1)
    def_grid1 = warp(grid_img1.float(),  phi_arr.cuda(), interp_mode='bilinear')
    def_grid2 = warp(grid_img2.float(),  phi_arr.cuda(), interp_mode='bilinear')
    def_grid3 = warp(grid_img3.float(),  phi_arr.cuda(), interp_mode='bilinear')

    grid_fig1 = comput_fig(def_grid1,def_grid2,def_grid3)
    # def_grid = def_grid.cpu().numpy()
    # def_grid = def_grid[0,0,:,:,:]
    # grid_img = grid_img.cpu().numpy()
    # grid_img = grid_img[0,0,:,:,:]
    # sitk.WriteImage(sitk.GetImageFromArray(def_grid),'result_grid.nii.gz')
    # sitk.WriteImage(sitk.GetImageFromArray(grid_img),'grid.nii.gz')

    plt.show()