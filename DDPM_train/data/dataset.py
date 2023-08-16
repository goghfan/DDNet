from torch.utils.data import Dataset
import os
import numpy as np
import SimpleITK as sitk
import scipy.io as sio
import glob
import data.util_3D as util_3D
#dataroot下存放train和test文件夹，
class Datasets(Dataset):
    def __init__(self,dataroot,type,trainortest,split='train'):
        self.imageNum = []
        self.dataroot = dataroot
        if trainortest=='train':
            datapath_t1 = glob.glob(os.path.join(dataroot+'train\\T1\\',"*.{}".format(type)))
            datapath_t2 = glob.glob(os.path.join(dataroot+'train\\T2\\',"*.{}".format(type)))
            self.trainortest = 'train'
        else:
            datapath_t1 = glob.glob(os.path.join(dataroot+'train\\t1\\',"*.{}".format(type)))
            datapath_t2 = glob.glob(os.path.join(dataroot+'train\\t2\\',"*.{}".format(type)))
            
            self.trainortest = 'test'
        datapath = datapath_t1+datapath_t2
        dataFiles = sorted(datapath)
        for isub,dataname in enumerate(dataFiles):
            self.imageNum.append(dataname)
        self.data_len = len(self.imageNum)
        if split=='test':
            self.nsample=10
        else:
            self.nsample = 10
    
    def __len__(self):
        return self.data_len
    
    def __getitem__(self, index):
        if self.trainortest=='train':
            datapath = self.imageNum[index]
            moving_image = sitk.ReadImage(datapath)
            moving_image = sitk.GetArrayFromImage(moving_image)
            moving_image = (moving_image - moving_image.min())/moving_image.max()
            [moving_image]=util_3D.transform_augment([moving_image],split=self.trainortest)
            return moving_image
        else:
            datapath = self.imageNum[index]
            fixed_detapath = 'C:\\Users\\GAOFAN\\Desktop\\my_diffusion\\toy_sample\\3d\\OAS30586_MR_d0070.nii.gz'
            moving_image = sitk.ReadImage(datapath)
            fixed_image = sitk.ReadImage(fixed_detapath)
            
            moving_image = sitk.GetArrayFromImage(moving_image)
            fixed_image = sitk.GetArrayFromImage(fixed_image)

            moving_image = (moving_image - moving_image.min())/moving_image.max()
            fixed_image = (fixed_image-fixed_image.min())/fixed_image.max()

            [moving_image,fixed_image]=util_3D.transform_augment([moving_image,fixed_image],split=self.trainortest)
            return moving_image,fixed_image
