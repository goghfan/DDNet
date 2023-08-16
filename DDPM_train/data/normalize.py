import os
import SimpleITK as sitk
import glob
import os
import nibabel as nib
import skimage.io as io
#cpath='dealed/*_affine.nii.gz'图像文件存放目录
#dpath=‘deals/’移动到某个目录
class Normalize1:
    def __init__(self,cpath,dpath):
        self.cpath=cpath
        self.dpath=dpath
    def normalize1(self):
        images=sorted(glob.glob(self.cpath))
        for image in images:
            vol = nib.load(image).get_data()
            vol = vol/255
            newvol = nib.Nifti1Image(vol, affine=None)
            pre_index=image.find('OAS2')
            name=image[pre_index:]
            nib.save(newvol, self.dpath+name)
            print('save success:{}'.format(self.dpath+name))

class Normalize:
    def __init__(self,cpath,dpath):
        self.cpath=cpath
        self.dpath=dpath
    def normalize(self):
        images=sorted(glob.glob(self.cpath))
        for image in images:
            img_obj = sitk.ReadImage(image)
            origin = img_obj.GetOrigin()        #读取图像origin, direction, space便于后面存储保留这些信息
            direction = img_obj.GetDirection()
            space = img_obj.GetSpacing()
            voxel_ndarray = sitk.GetArrayFromImage(img_obj)
            voxel_ndarray=voxel_ndarray/255
            savedImg = sitk.GetImageFromArray(voxel_ndarray)
            savedImg.SetOrigin(origin)
            savedImg.SetDirection(direction)
            savedImg.SetSpacing(space)
            pre_index=image.find('OAS2')
            name=image[pre_index:]
            sitk.WriteImage(savedImg,self.dpath+name)

n=Normalize(cpath='dealed/*_affine.nii.gz',dpath='')
n.normalize()
