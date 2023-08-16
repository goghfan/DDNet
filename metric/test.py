import SimpleITK as sitk
import numpy as np
path='E:/data/OAS30001_MR_d0757/mri/aseg.nii.gz'
img_origin=sitk.ReadImage(path)
img_matrix=sitk.GetArrayFromImage(img_origin)
class segment:
#将多结构，包括左右结构等等合到一起，形成一个合理的结构
    def filter_elements(matrix, elements):
        filtered_matrix = np.zeros_like(matrix)  # 创建一个和原始矩阵相同大小的全零矩阵
        for a in elements:
            filtered_matrix[matrix == a] = 255  # 将值为 a 的元素设置为 a
        return filtered_matrix

    def count_non_zero(matrix):
        non_zero_elements = matrix[matrix != 0]  # 使用布尔索引筛选非零元素
        count = non_zero_elements.size  # 获取非零元素的数量
        unique_values = np.unique(non_zero_elements)  # 获取非零元素的唯一值
        return count, unique_values

dict_struct = {'ce_nao_shi':"4-43-5-45-",
               "xiao_nao_bai_zhi":"7-46-",
               "xiao_nao_pi_zhi":"8-47-",
               "di_san_nao_shi":"14-",
               "di_si_nao_shi":"15-",
               "di_wu_nao_shi":"72-",
               "pian_zhi_ti":"251-252-253-254-255-",
               "hai_ma_ti":"17-23-",
               "qiu_nao":"10-49-",
               "xing_ren_he":"18-54-",
               "wei-zhuang-he":"11-50-"
               }
for i,j in dict_struct.items():
    print(i)
    num=[]
    temp=""
    for k in j:
        if k!='-' and k!=len(j)-1:
            temp+=k
        else:
            print(temp)
            num.append(int(temp))
            temp=""
    img=segment.filter_elements(img_matrix,num)
    # img = sitk.GetImageFromArray(img)
    # img.SetDirection(img_origin.GetDirection())
    # img.SetOrigin(img_origin.GetOrigin())
    # img.SetSpacing(img_origin.GetSpacing())
    # sitk.WriteImage(img,"file{}_struct_{}.nii.gz".format(1,{i}))

# counts=count_non_zero(img_matrix)[1]
# for i in counts:
#     img = filter_elements(img_matrix,i)
#     img = sitk.GetImageFromArray(img)
#     img.SetDirection(img_origin.GetDirection())
#     img.SetOrigin(img_origin.GetOrigin())
#     img.SetSpacing(img_origin.GetSpacing())
#     sitk.WriteImage(img,'test{}.nii.gz'.format(i))
#     print("success {}".format(i))
# img = sitk.GetImageFromArray(img)
# sitk.WriteImage(img,'test.nii.gz')