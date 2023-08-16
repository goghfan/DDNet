from nilearn.image import resample_to_img,load_img
import os
import SimpleITK as sitk
import glob
import numpy as np
def resize(path,saved_path):
    images = glob.glob(os.path.join(path,"*.nii.gz"))
    names=[]
    for i in os.listdir(path):
        names.append(i)
    i=0
    for image in images:
        data = sitk.ReadImage(image)
        data = sitk.GetArrayFromImage(data)
        data = data[14:166,14:198,0:152]
        print(data.shape)
        data = sitk.GetImageFromArray(data)
        data  = sitk.WriteImage(data,saved_path+names[i])
        print(saved_path+names[i])
        i+=1

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


 # 读入图片
img_path = 'C:\\Users\\GAOFAN\\Desktop\\OAS2_0001_MR12_affine.nii.gz'
img = sitk.ReadImage(img_path)
sample=sitk.GetArrayFromImage(img)
# re_img为放缩之后的图片大小，(256, 256, 256)为目标大小，sitkNeartestNeighbor是放缩用到的算法，默认为线性放缩，这里选择了最近邻
re_img = resize_image_itk(img, (128,128,128), resamplemethod=sitk.sitkNearestNeighbor)

# 对放缩后的图片进行一些操作
# 1.原图，2.空间，3.方向
re_img.SetOrigin(img.GetOrigin())
re_img.SetSpacing(img.GetSpacing())
re_img.SetDirection(img.GetDirection())

# 保存图片
sitk.WriteImage(re_img, 'C:\\Users\\GAOFAN\\Desktop\\re_img1.nii.gz')

path = 'data/deal/'
# template = load_img(r'C:\\Users\\GAOFAN\\Desktop\\DDPM-main\\data\\MNI152_T1_1mm.nii.gz')
saved_path = 'data/dealed/'
resize(path,saved_path)