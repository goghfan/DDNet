import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk

from mpl_toolkits.mplot3d import Axes3D
def error_map1():
    # 假设您的矩阵为matrix
    # 这里只是一个例子，您需要将matrix替换为您的实际矩阵
    warped = sitk.ReadImage('result\\DCT\\DDPM_DCT\\OAS30226_MR_d0183_T1w_to_OAS30299_MR_d0119_T2w.nii.gz')
    fixed1 = sitk.ReadImage('D:\\Desktop\\diffusion model code\\voxelmorph-dev\\data\\train\\T2\\OAS30299_MR_d0119_T2w.nii.gz')

    warped = sitk.GetArrayFromImage(warped)
    fixed = sitk.GetArrayFromImage(fixed1)
    fixed = (fixed-fixed.min())/(fixed.max()-fixed.min())

    error_map = fixed-warped
    matrix = error_map

    # 创建一个3D的坐标网格
    # 创建一个3D图像对象
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 设置颜色映射，0为白色，大于0为红色，小于0为蓝色
    cmap = plt.cm.colors.ListedColormap(['blue', 'white', 'red'])
    bounds = [-1.5, -0.5, 0.5, 1.5]
    norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)
    # 绘制3D图像
    ax.voxels(matrix < 0, facecolors=cmap(norm(matrix)), edgecolor='k')

    # 显示图像
    plt.show()

def error_map():
    warped = sitk.ReadImage('result\\ablation_study\\result_ncc_vm.nii.gz')
    fixed1 = sitk.ReadImage('D:\\Desktop\\diffusion model code\\voxelmorph-dev\\data\\train\\T2\\OAS30003_MR_d2669_T2w.nii.gz')

    warped = sitk.GetArrayFromImage(warped)
    fixed = sitk.GetArrayFromImage(fixed1)
    fixed = (fixed-fixed.min())/(fixed.max()-fixed.min())

    error_map = fixed-warped

    error_map[np.where(error_map>0.3)]-=0.3
    # error_map[np.where((error_map>=-0.3) & (error_map<=0.3))]=100
    error_map[np.where(error_map<-0.3)]+=0.3
    error_map[np.where((error_map>=-0.1) & (error_map<=0.1))]=0
    error_map=error_map/4.0
    

    error_map = sitk.GetImageFromArray(error_map)

    error_map.SetDirection(fixed1.GetDirection())
    error_map.SetOrigin(fixed1.GetOrigin())
    error_map.SetSpacing(fixed1.GetSpacing())

    sitk.WriteImage(error_map,"error_map_vm.nii.gz")
    # # 绘制warped_image和fixed_image

    # plt.imshow(error_map[64,:,:], cmap='jet', interpolation='none')
    # plt.colorbar()
    # plt.title('Error Map')
    # plt.show()


def diejia():
    moving = sitk.ReadImage('D:\\Desktop\\diffusion model code\\voxelmorph-dev\\result\\DCT\\result_ncc_DCT.nii.gz')
    fix = sitk.ReadImage('D:\\Desktop\\diffusion model code\\voxelmorph-dev\\data\\train\\T2\\OAS30003_MR_d2669_T2w.nii.gz')

    warped = sitk.GetArrayFromImage(moving)
    fixed = sitk.GetArrayFromImage(fix)
    fixed = (fixed-fixed.min())/(fixed.max()-fixed.min())

    result = np.where((warped>0)&(fixed>0),warped, np.where((warped==0) & (fixed>0),fixed , warped))

    result = sitk.GetImageFromArray(result)

    result .SetDirection(fix.GetDirection())
    result .SetOrigin(fix.GetOrigin())
    result .SetSpacing(fix.GetSpacing())

    sitk.WriteImage(result,'result.nii.gz')

error_map()