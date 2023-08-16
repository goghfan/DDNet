"""
Created on Wed Apr 11 10:08:36 2018
@author: Dongyang
This script contains some utility functions for data visualization
"""

import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import SimpleITK as sitk
#==============================================================================
# 雅克比行列式可视化
#==============================================================================
class JAC:
    @staticmethod
    def calculate_jacobian_metrics(disp):
        for n in range(disp.shape[0]):
            disp_n = np.moveaxis(disp[n, ...], 0, -1)  
            jac_det_n = JAC.jacobian_det(disp_n)
        return jac_det_n
    @staticmethod
    def jacobian_det(disp):
        disp_img = sitk.GetImageFromArray(disp.astype('float32'), isVector=True)
        jac_det_img = sitk.DisplacementFieldJacobianDeterminant(disp_img)
        jac_det = sitk.GetArrayFromImage(jac_det_img)
        return jac_det
    
#==============================================================================
# Define a custom colormap for visualizing Jacobian
#==============================================================================
class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

def show_sample_slices(sample_list, name_list, Jac=False, cmap='bwr', attentionlist=None):
    num = len(sample_list)
    fig, ax = plt.subplots(1, num)
    
    for i in range(num):
        if Jac:
            im = ax[i].imshow(sample_list[i], cmap, norm=MidpointNormalize(midpoint=0))
        else:
            im = ax[i].imshow(sample_list[i], cmap)
        ax[i].set_title(name_list[i])
        ax[i].axis('off')
        if attentionlist:
            ax[i].add_artist(attentionlist[i])

        # Add colorbar for each plot
        cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
        cbar = plt.colorbar(im, cax=cax, cmap=cmap)
        cbar.ax.yaxis.set_ticks([])  # Hide colorbar ticks
        cbar.ax.set_ylabel('')  # Set an empty label for the colorbar
    plt.subplots_adjust(wspace=0)
    plt.show()

jd = sitk.ReadImage('result\\ablation_study\\warped_ncc_DDPM_DCT_MSE.nii.gz')
jd_img = sitk.GetArrayFromImage(jd)
jd_img = np.expand_dims(jd_img,axis=0)
jac = JAC.calculate_jacobian_metrics(jd_img)
jac=sitk.GetImageFromArray(jac)
moving = sitk.ReadImage('D:\\Desktop\\diffusion model code\\voxelmorph-dev\\data\\train\\T1\\OAS30001_MR_d0129_T1w.nii.gz')
fixed = sitk.ReadImage('D:\\Desktop\\diffusion model code\\voxelmorph-dev\\data\\train\\T2\\OAS30003_MR_d2669_T2w.nii.gz')

jac.SetDirection(fixed.GetDirection())
jac.SetOrigin(fixed.GetOrigin())
jac.SetSpacing(fixed.GetSpacing())
sitk.WriteImage(jac,'JAC_DDPM_DCT_MSE.nii.gz')

# moving = sitk.ReadImage('D:\\Desktop\\diffusion model code\\voxelmorph-dev\\data\\train\\T1\\OAS30001_MR_d0129_T1w.nii.gz')
# fixed = sitk.ReadImage('D:\\Desktop\\diffusion model code\\voxelmorph-dev\\data\\train\\T2\\OAS30003_MR_d2669_T2w.nii.gz')
# warped = sitk.ReadImage('D:\\Desktop\\diffusion model code\\voxelmorph-dev\\result\\DCT\\result_ncc_DCT.nii.gz')

# moving_matrix = sitk.GetArrayFromImage(moving)
# fixed_matrix = sitk.GetArrayViewFromImage(fixed)
# warped_matrix = sitk.GetArrayFromImage(warped)

# error_map = sitk.ReadImage("D:\\Desktop\\diffusion model code\\voxelmorph-dev\\error_map.nii.gz")
# error_map_matrix = sitk.GetArrayFromImage(error_map)

# show_sample_slices(sample_list=[jac[:,:,65],jac[:,65,:],jac[65,:,:]],name_list=['Jacobian','Jac','JAC'],Jac=True)
# show_sample_slices(sample_list=[moving_matrix[:,:,65],moving_matrix[:,65,:],moving_matrix[65,:,:]],cmap='gray' ,name_list=['A','B','C'])
# show_sample_slices(sample_list=[fixed_matrix[:,:,65],fixed_matrix[:,65,:],fixed_matrix[65,:,:]],cmap='gray' ,name_list=['A','B','C'])
# show_sample_slices(sample_list=[warped_matrix[:,:,65],warped_matrix[:,65,:],warped_matrix[65,:,:]],cmap='gray' ,name_list=['A','B','C'])
# show_sample_slices(sample_list=[error_map_matrix[:,:,65],error_map_matrix[:,65,:],error_map_matrix[65,:,:]] ,name_list=['A','B','C'])
