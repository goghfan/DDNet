import os
import glob
import shutil

#    cpath='/home/gaofan/data/'
#    dpath='/home/gaofan/data/dealed/'

class move_to_dir:
    def __init__(self,cur_path,destination_path):
        self.cpath=cur_path
        self.dpath=destination_path
    def get_all_data(self):
        datalist=[]
        for root,dir,filename in os.walk(self.cpath):
            for d in dir:
                if root.find('mri')!=-1: 
                    for file in filename:
                        if file.endswith('_affine.nii.gz'):
                            datalist.append(root+'/'+file)
        return datalist
    def move_to_dir(self):
        datalist=self.get_all_data()
        datalist=list(set(datalist))
        for data in datalist:
            shutil.copy(data,self.dpath)
            print('move success：{} to {}'.format(data,self.dpath))
