import torch
import numpy as np
from skimage.metrics import structural_similarity
import SimpleITK as sitk
from scipy.spatial.distance import cdist
import torch.nn as nn
import torch.nn.functional as nnf
import scipy

class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('grid', grid)

    def forward(self, src, flow):
        # new locations
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return nnf.grid_sample(src, new_locs, align_corners=True, mode=self.mode)

class ADF:
    def apply_deformation_field(image, deformation_field):
        # 获取形变场的维度信息
        _, depth, height, width = deformation_field.shape

        # 创建目标图像空间的坐标网格
        grid_x, grid_y, grid_z = np.meshgrid(np.arange(width), np.arange(height), np.arange(depth), indexing='ij')

        # 将形变场中的坐标值添加到网格上，得到目标图像空间中的坐标
        warped_x = grid_x + deformation_field[0]
        warped_y = grid_y + deformation_field[1]
        warped_z = grid_z + deformation_field[2]

        # 在目标图像上进行插值，获取相应位置处的图像值
        warped_image = scipy.ndimage.map_coordinates(image, [warped_z, warped_y, warped_x], order=1)

        # 将插值得到的图像值重新调整为与原始图像相同的形状
        warped_image = warped_image.reshape(image.shape)

        return warped_image    



class Dice:
    """
    N-D dice for segmentation
    y_true:真实mask
    y_pred:预测mask
    """
    def dice(y_true, y_source, deformation_field):
        non_zero_elements = np.concatenate([np.unique(a) for a in [y_true,y_source]])
        unique_values = np.unique(non_zero_elements)
        struct_set={}
        deformation_field=torch.tensor(deformation_field)
        STN = SpatialTransformer(y_source.shape)
        for struct in unique_values:
            zeros_matrix = np.zeros_like(y_source)
            zeros_matrix[y_source==struct] = 1
            struct_set[struct] = zeros_matrix
        y_source=torch.tensor(y_source).unsqueeze(0).unsqueeze(0).float()
        dice_list=[]
        result = np.zeros_like(y_true)
        for name,value in struct_set.items():
            value = torch.tensor(value).unsqueeze(0).unsqueeze(0).float()
            y_pred = STN(value,deformation_field).squeeze().detach().numpy()
            y_pred[np.where(y_pred!=0)] = 1
            result[np.where(y_pred!=0)] = name

            zeros_matrix = np.zeros_like(y_true)
            zeros_matrix[np.where(y_true==name)] = 1
            
            intersection = 2*np.sum(np.logical_and(zeros_matrix,y_pred))
            union = np.sum(y_pred) + np.sum(zeros_matrix)
            union = np.maximum(union, np.finfo(float).eps)
            dice_score = (intersection) / (union)
            if dice_score>0.5:
                dice_list.append(dice_score)
        # dices =  2*np.sum(np.logical_and(total_up1,y_true))/(np.sum(total_up1)+np.sum(y_pred))
        # sitk.WriteImage(sitk.GetImageFromArray(result),'mask_result.nii.gz')
        return np.mean(dice_list)

class Jacobians:

    def Get_Jac(displacement):
        '''
        the expected input: displacement of shape(batch, H, W, D, channel),
        obtained in TensorFlow.
        '''
        displacement=np.transpose(displacement,(0,2,3,4,1))
        D_y = (displacement[:,1:,:-1,:-1,:] - displacement[:,:-1,:-1,:-1,:])
        D_x = (displacement[:,:-1,1:,:-1,:] - displacement[:,:-1,:-1,:-1,:])
        D_z = (displacement[:,:-1,:-1,1:,:] - displacement[:,:-1,:-1,:-1,:])
    
        D1 = (D_x[...,0]+1)*((D_y[...,1]+1)*(D_z[...,2]+1) - D_y[...,1]*D_z[...,1])
        D2 = (D_x[...,1])*(D_y[...,0]*(D_z[...,2]+1) - D_y[...,2]*D_z[...,0])
        D3 = (D_x[...,2])*(D_y[...,0]*D_z[...,1] - (D_y[...,1]+1)*D_z[...,0])
        
        D = D1 - D2 + D3

        negative_elements=D[D<0]
        persent = len(negative_elements)/np.size(D)
        std_deviation=np.std(D)
        
        return persent,std_deviation


class JAC:
    @staticmethod
    def calculate_jacobian_metrics(disp):
        """
        Calculate Jacobian related regularity metrics.
        Args:
            disp: (numpy.ndarray, shape (N, ndim, *sizes) Displacement field
        Returns:
            folding_ratio: (scalar) Folding ratio (ratio of Jacobian determinant < 0 points)
            mag_grad_jac_det: (scalar) Mean magnitude of the spatial gradient of Jacobian determinant
        """
        negative_det_J = []
        mag_grad_det_J = []
        std_log_det_J = []
        for n in range(disp.shape[0]):
            disp_n = np.moveaxis(disp[n, ...], 0, -1)  # (*sizes, ndim)
            jac_det_n = JAC.jacobian_det(disp_n)
            negative_det_J += [(jac_det_n < 0).sum() / np.prod(jac_det_n.shape)]
            mag_grad_det_J += [np.abs(np.gradient(jac_det_n)).mean()]
            std_log_det_J += [np.log(jac_det_n.clip(1e-9, 1e9)).std()]
        return {
            'negative_det_J': np.mean(negative_det_J),
            'mag_grad_det_J': np.mean(mag_grad_det_J),
            'std_log_det_J': np.mean(std_log_det_J)
        }

    @staticmethod
    def jacobian_det(disp):
        """
        Calculate Jacobian determinant of displacement field of one image/volume (2D/3D)
        Args:
            disp: (numpy.ndarray, shape (*sizes, ndim)) Displacement field
        Returns:
            jac_det: (numpy.ndarray, shape (*sizes)) Point-wise Jacobian determinant
        """
        disp_img = sitk.GetImageFromArray(disp.astype('float32'), isVector=True)
        jac_det_img = sitk.DisplacementFieldJacobianDeterminant(disp_img)
        jac_det = sitk.GetArrayFromImage(jac_det_img)
        return jac_det

class ASSD:
    def assd(matrix1, matrix2):
        if matrix1.shape != matrix2.shape:
            raise ValueError("两个矩阵的形状不相同")
        diff = np.abs(matrix1 - matrix2)
        diff = np.sum(diff, axis=2)
        diff = np.sum(diff, axis=1)
        return np.mean(diff)
    def calculate_assd(matrix1, matrix2):
        from scipy.spatial import KDTree
        # 将矩阵转换为表面表示
        surface1 = np.argwhere(matrix1 > 0.5)
        surface2 = np.argwhere(matrix2 > 0.5)
        # 构建KD树
        tree = KDTree(surface2)
        # 计算每个表面1上的顶点到表面2的最短距离
        distances, _ = tree.query(surface1)
        # 计算ASSD
        assd = (np.mean(distances) + np.mean(tree.query(surface2)[0])) / 2
        return assd
class SSIM:
    def ssim(y_true,y_pred):
        #https://scikit-image.org/docs/stable/api/skimage.metrics.html
        #mean structural similarity index 
        #参考文档网址
        ssim=structural_similarity(y_pred,y_true)
        return ssim

class PCC:
    def pcc(y_true,y_pred):
        # 将矩阵展平为一维向量
        vector1 = y_pred.flatten()
        vector2 = y_true.flatten()

        # 计算皮尔逊相关系数
        pearson_corr = np.corrcoef(vector1, vector2)[0, 1] 
        return pearson_corr
    def cal_pccs(x, y, n):
        """
        手动计算
        """
        sum_xy = np.sum(np.sum(x*y))
        sum_x = np.sum(np.sum(x))
        sum_y = np.sum(np.sum(y))
        sum_x2 = np.sum(np.sum(x*x))
        sum_y2 = np.sum(np.sum(y*y))
        pcc = (n*sum_xy-sum_x*sum_y)/np.sqrt((n*sum_x2-sum_x*sum_x)*(n*sum_y2-sum_y*sum_y))
        return pcc


class HD:
    def __init__(self) -> None:
        pass
    def hd(self,y_true,y_pred):
        """
        使用simpleitk
        """
        hausdorff_fileter = sitk.HausdorffDistanceImageFilter()
        hausdorff_fileter.Execute(y_true,y_pred)
        hausdorff_distance = hausdorff_fileter.GetHausdorffDistance()
        return hausdorff_distance

    def hd95(self,y_true,y_pred):
        hausdorff_distance = self.hd(y_true,y_pred)
        sorted_distances = np.sort(hausdorff_distance)  # 对距离进行排序
        hd95_index = int(0.95 * len(sorted_distances))  # 获取距离数组中第95%位置的索引
        hd95 = sorted_distances[hd95_index]  # 获取HD95距离值
        return hd95



