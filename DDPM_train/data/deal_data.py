import os
from move_file import move_to_dir
from normalize import Normalize
from skullbone import skullboned

# cpath1: 最原始文件所在目录，cpath1 = r"/home/gaofan/data/datas"
# number：待处理数据量
# environment：freesurfer环境变量
# cpath2: 收集处理完成文件所在目录
# dpath2: 移动处理完的文件要保存的目录
# cpath3 = cpath2
# dpath3： 最终保存目录
class deal_datas:
    def __init__(self,cpath1,number,environment,cpath2,dpath2,cpath3,dpath3):
        self.cpath1 = cpath1
        self.number = number
        self.environment = environment
        self.cpath2 = cpath2
        self.dpath2 = dpath2
        self.cpath3 = cpath3
        self.dpath3 = dpath3
    def deal_data(self):
        s = skullboned(self.cpath1,self.number,self.environment)
        s.skullbones()
        
        m = move_to_dir(self.cpath2,self.dpath2)
        m.move_to_dir()

        n = Normalize(self.cpath3,self.dpath3)
        n.normalize()
        
        print('All data has been processed！')

