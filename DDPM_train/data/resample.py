from nilearn.image import resample_to_img,load_img
import os
import glob
# template = load_img(r'C:\\Users\\GAOFAN\\Desktop\\DDPM-main\\data\\MNI152_T1_1mm.nii.gz')   # 参考图像
# data = load_img(r'C:\\Users\\GAOFAN\\Desktop\\DDPM-main\\data\\OAS2_0001_MR12_affine.nii.gz')
# resampled_img = resample_to_img(data, template)
# resampled_img.to_filename(r'C:\\Users\\GAOFAN\\Desktop\\DDPM-main\\data\\fined.nii.gz')  # 保存重采样后的图像

def resample(path,template,saved_path):
    images = glob.glob(os.path.join(path,"*.nii.gz"))
    names=[]
    for i in os.listdir(path):
        names.append(i)
    i=0
    for image in images:
        data = load_img(image)
        resampled_img = resample_to_img(data,template)
        resampled_img.to_filename(saved_path+names[i])
        i+=1

path = 'data\\train\\T1\\'
template = load_img(r'C:\\Users\\GAOFAN\\Desktop\\MNI152_T1_1mm_brain.nii.gz')
saved_path = 'data/dealed/'
resample(path,template,saved_path)
