import os
import glob
#去颅骨和仿射对齐操作，会在数据目录生成对应文件夹，文件夹下mri目录里面的*_affine.nii.gz文件即为处理完成后的文件
class skullboned:
      def __init__(self,path,number,environment):
      # path应该为一个路径字符串，存放有医学图像，参考路径如下：
      # path = r"/home/gaofan/data/datas"
            self.path=path
      # number为待处理图像的数量
            self.number=number
      # enviroment是export FREESURFER_HOME=路径
            self.environment=environment
      def skullbones(self):
            #读取目录下的.nii文件
            images = glob.glob(os.path.join(self.path,"*.nii"))
            #下面为freesurfer的环境配置命令,实例如下：
            #a="export FREESURFER_HOME=/home/gaofan/freesurfer;"
            a = self.environment
            b = "source $FREESURFER_HOME/SetUpFreeSurfer.sh;"
            #数据所在的目录
            c = "export SUBJECTS_DIR="+self.path+";"
            cur=0
            total=self.number
            #images=['/home/syzhou/zuzhiang/Dataset/MGH10/Heads/1127.img']
            for image in images:
            # 将文件路径和文件名分离
                  if cur<self.number:
                        cur+=1
                        filename = os.path.split(image)[1] # 将路径名和文件名分开
                        filename = os.path.splitext(filename)[0] #将文件名和扩展名分开，如果为.nii.gz，则认为扩展名是.gz
                        # freesurfer环境配置、颅骨去除、未仿射对齐mpz转nii、仿射对齐、仿射对齐mpz转nii.gz格式
                        #recon-all是颅骨去除的命令
                        # mri_convert是进行格式转换，从mgz转到nii.gz，只是为了方便查看
                        # --apply_transform：仿射对齐操作
                        # 转格式
                        filename=filename[:] #根据扩展名的不同，这里需要做更改，只保留文件名即可
                        cur_path=os.path.join(self.path,filename) 
                        print("file name: ",cur_path)
                        cmd = a + b + c \
                              + "recon-all -parallel -i " + image + " -autorecon1 -subjid " + cur_path + "&&" \
                              + "mri_convert " +  cur_path + "/mri/brainmask.mgz " +cur_path + "/mri/"+filename+".nii.gz;"\
                              + "mri_convert " + cur_path + "/mri/brainmask.mgz --apply_transform " + cur_path + "/mri/transforms/talairach.xfm -o " + cur_path + "/mri/brainmask_affine.mgz&&" \
                              + "mri_convert " + cur_path + "/mri/brainmask_affine.mgz " + cur_path + "/mri/"+filename+"_affine.nii.gz;"
                        #print("cmd:\n",cmd)
                        os.system(cmd)
            # os.system('shutdown -h now \n') 关机命令