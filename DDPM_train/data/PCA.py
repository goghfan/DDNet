
import numpy as np
from sklearn.decomposition import PCA
import SimpleITK as sitk
import torch

def resize_image_itk(itkimage, newSize, resamplemethod=sitk.sitkNearestNeighbor):
    resampler = sitk.ResampleImageFilter()

    originSize = itkimage.GetSize()  # 获取原图size
    originSpacing = itkimage.GetSpacing()  # 获取原图spacing
    newSize = np.array(newSize, dtype='uint32')
    factor = originSize / newSize
    newSpacing = originSpacing * factor

    resampler.SetReferenceImage(itkimage)   # 指定需要重新采样的目标图像
    resampler.SetSize(newSize.tolist())
    resampler.SetOutputSpacing(newSpacing.tolist())
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetInterpolator(resamplemethod)
    itkimgResampled = resampler.Execute(itkimage)  # 得到重新采样后的图像
    return itkimgResampled

class do_PCA:
    def __init__(self,matrix_moving,matrix_fix):
        self.matrix_moving = matrix_moving
        self.matrix_fix = matrix_fix
    def get_lambda(self):
        new_matrix_moving=self.matrix_moving.reshape(32,65536).to('cpu').detach().numpy()
        new_matrix_fix = self.matrix_fix.reshape(32,65536).to('cpu').detach().numpy()
        
        pca = PCA(n_components=0.80)
        sigma_moving = pca.fit_transform(new_matrix_moving)
        
        sigma_moving = pca.inverse_transform(sigma_moving).reshape(128,128,128)
        lambda_moving = self.matrix_moving.to('cpu').detach().numpy() - sigma_moving

        sigma_fix = pca.fit_transform(new_matrix_fix)
        sigma_fix = pca.inverse_transform(sigma_fix).reshape(128,128,128)
        lambda_fix = self.matrix_fix.to('cpu').detach().numpy() - sigma_fix
        del self.matrix_fix,self.matrix_moving,new_matrix_fix,new_matrix_moving,sigma_moving,sigma_fix
        return {'M':lambda_moving,'F':lambda_fix}

# path1='C:\\Users\\GAOFAN\Desktop\\my_diffusion\\toy_sample\\3d\\OAS30586_MR_d0070.nii.gz'
# path2='E:\\datas\\OAS30001_MR_d0129\\scans\\anat4-T2w\\resources\\NIFTI\\files\\sub-OAS30001_ses-d0129_T2w.nii.gz'
# path3 = 'C:\\Users\\GAOFAN\Desktop\\my_diffusion\\toy_sample\\3d\\test\\OAS30638_MR_d0071.nii.gz'
# t1o = sitk.ReadImage(path1)
# t2o = sitk.ReadImage(path3)
# t2 = sitk.GetArrayFromImage(t2o)


# t1 = sitk.GetArrayFromImage(t1o)  # 128, 128, 128
# t2 = sitk.GetArrayFromImage(t2o)    #  128, 128, 128

# # t1 = t1.reshape(1, 128, 128, 128)
# # t2 = t2.reshape(1, 128, 128, 128)


# pca = PCA(n_components=0.80)

# # t1_re = np.concatenate([t1, t2], axis=0).reshape(32,65536)
# t1_re = t1.reshape(32,65536)
# newt1=pca.fit_transform(t1_re) #降维后数据

# t1_rev=pca.inverse_transform(newt1)
# t1_rev=t1_rev.reshape(128,128,128) #原始数据

# t2_re=t2.reshape(2,1048576)
# newt2=pca.fit_transform(t2_re)
# t2_rev=pca.inverse_transform(newt2).reshape(128,128,128)

# t1_rev=t1-t1_rev

# t1_r = sitk.GetImageFromArray(t1_rev)
# t2_r = sitk.GetImageFromArray(t2_rev)

# t1_r.SetDirection(t1o.GetDirection())
# t1_r.SetSpacing(t1o.GetSpacing())
# t1_r.SetOrigin(t1o.GetOrigin())

# t1_r.SetDirection(t1o.GetDirection())
# t1_r.SetSpacing(t1o.GetSpacing())
# t1_r.SetOrigin(t1o.GetOrigin())

# sitk.WriteImage(t1_r,'C:\\Users\\GAOFAN\Desktop\\t10.nii.gz')
# # sitk.WriteImage(t2_r,'C:\\Users\\GAOFAN\Desktop\\t2.nii.gz')
