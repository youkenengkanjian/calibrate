import numpy as np
import torch
from einops import rearrange
from torchvision import transforms
from tools.tools import resample_rgb
from tools.visualization import tensor2rgb

class MonocularCalibrator(torch.nn.Module):
    def __init__(self, l1_th=0.02):
        """ Calibrate Camera Intrinsic from Incidence Field.

        Args:
            l1_th (float): RANSAC Inlier Count Threshold. Default 0.02.
            RANSAC_num (int): RANSAC Random Sampling Point Number. Default: 20000.
        """

        super().__init__()
        self.RANSAC_num = 20000
        self.l1_th = l1_th

    def initcoords2D(self, b, h, w, device, homogeneous=False):
        """ Init Normalized Pixel Coordinate System
        """

        query_coords = torch.meshgrid(
            (
                torch.linspace(-1 + 1 / h, 1 - 1 / h, h, device=device),
                torch.linspace(-1 + 1 / w, 1 - 1 / w, w, device=device),
            ),
            indexing='ij'
        )
        query_coordsx, query_coordsy = query_coords[1], query_coords[0]

        if homogeneous:
            query_coords = torch.stack((query_coordsx, query_coordsy, torch.ones_like(query_coordsx)), dim=0).view([1, 3, h, w]).expand([b, 3, h, w])
        else:
            query_coords = torch.stack((query_coordsx, query_coordsy), dim=0).view([1, 2, h, w]).expand([b, 2, h, w])
        return query_coords

    @staticmethod
    def norm_intrinsic(intrinsic, b, h, w, device):
        """ Map Intrinsic to Normalized Image Coordinate System [-1, 1]
        """
        scaleM = torch.eye(3).view([1, 3, 3]).expand([b, 3, 3]).to(device)
        scaleM[:, 0, 0] = float(1 / w) * 2
        scaleM[:, 1, 1] = float(1 / h) * 2

        scaleM[:, 0, 2] = -1.0
        scaleM[:, 1, 2] = -1.0
        return scaleM @ intrinsic

    @staticmethod
    def unnorm_intrinsic(intrinsic, b, h, w, device):
        """ Unmap Intrinsic to Image Coordinate System [0.5, h / w - 0.5]
        """
        scaleM = torch.eye(3).view([1, 3, 3]).expand([b, 3, 3]).to(device)
        scaleM[:, 0, 0] = float(1 / w) * 2
        scaleM[:, 1, 1] = float(1 / h) * 2

        scaleM[:, 0, 2] = -1.0
        scaleM[:, 1, 2] = -1.0
        return scaleM.inverse() @ intrinsic

    def intrinsic2incidence(self, intrinsic, b, h, w, device):
        """ Compute Gt Incidence Field from Intrinsic
        """
        coords3d = self.initcoords2D(b, h, w, device, homogeneous=True)
        intrinsic = MonocularCalibrator.norm_intrinsic(intrinsic, b, h, w, device)

        intrinsic = intrinsic.view([b, 1, 1, 3, 3])
        coords3d = rearrange(coords3d, 'b d h w -> b h w d 1')
        coords3d = torch.linalg.inv(intrinsic) @ coords3d
        coords3d = rearrange(coords3d.squeeze(-1), 'b h w d -> b d h w')
        normalray = torch.nn.functional.normalize(coords3d, dim=1)
        return normalray

    # def scoring_function_xy(self, normal_RANSAC, normal_ref):
    #     """ RANSAC Scoring Function
    #     """
    #     xx, yy, _ = torch.split(normal_RANSAC, 1, dim=1)
    #     xxref, yyref, zzref = torch.split(normal_ref, 1, dim=0)
    #     xxref = xxref / zzref
    #     yyref = yyref / zzref
    #
    #     diffx = torch.sum((xx - xxref.unsqueeze(0)).abs() < self.l1_th, dim=[1, 2])
    #     diffy = torch.sum((yy - yyref.unsqueeze(0)).abs() < self.l1_th, dim=[1, 2])
    #
    #     return diffx, diffy

    # TODO 置信度（加权内点评分）
    def scoring_function_xy(self, normal_RANSAC, normal_ref, weights=None):
        """ RANSAC Scoring Function (weighted inlier count if weights is given) """
        xx, yy, _ = torch.split(normal_RANSAC, 1, dim=1)
        xxref, yyref, zzref = torch.split(normal_ref, 1, dim=0)
        xxref = xxref / zzref
        yyref = yyref / zzref

        # (num_hypos, 1, N)
        inlier_x = ((xx - xxref.unsqueeze(0)).abs() < self.l1_th).float()
        inlier_y = ((yy - yyref.unsqueeze(0)).abs() < self.l1_th).float()

        if weights is None:
            # 原逻辑：不加权，直接计数
            diffx = torch.sum(inlier_x, dim=[1, 2])
            diffy = torch.sum(inlier_y, dim=[1, 2])
        else:
            # 加权：把每个像素的 inlier 指示乘以像素权重再求和
            # weights_flat: (N,)  -> (1,1,N) 以便广播
            w = weights.view(1, 1, -1)
            diffx = torch.sum(inlier_x * w, dim=[1, 2])
            diffy = torch.sum(inlier_y * w, dim=[1, 2])

        return diffx, diffy

    # def get_sample_idx(self, h, w, prob=None, seed=None):
    #     if seed is not None:
    #         np.random.seed(seed)
    #
    #     if prob is not None:
    #         prob = prob.view([1, int(h * w)]).squeeze().cpu().numpy()
    #         sampled_index = np.random.choice(
    #             np.arange(int(h * w)),
    #             size=self.RANSAC_num,
    #             replace=False,
    #             p=prob,
    #         )
    #     else:
    #         sampled_index = np.random.choice(
    #             np.arange(int(h * w)),
    #             size=self.RANSAC_num,
    #             replace=False,
    #         )
    #     return sampled_index

    # TODO 置信度
    # def get_sample_idx(self, h, w, prob=None, seed=None):
    #     import numpy as np
    #     import torch
    #
    #     if seed is not None:
    #         np.random.seed(seed)
    #
    #     n = int(h * w)
    #     size = min(self.RANSAC_num, n)  # 安全：避免 size>n 时 replace=False 报错
    #     p = None
    #
    #     if prob is not None:
    #         # 兼容 torch.Tensor / numpy.ndarray / list
    #         if isinstance(prob, torch.Tensor):
    #             p = prob.reshape(-1).detach().cpu().numpy()
    #         else:
    #             p = np.asarray(prob).reshape(-1)
    #
    #         s = float(np.sum(p))
    #         if np.isfinite(s) and s > 0.0:
    #             p = p / s
    #         else:
    #             p = None  # 退化为均匀采样
    #
    #     sampled_index = np.random.choice(np.arange(n), size=size, replace=False, p=p)
    #     return sampled_index

    def get_sample_idx(self, h, w, prob=None, seed=None):
        import numpy as np, torch
        if seed is not None:
            np.random.seed(seed)

        n = int(h * w)
        size = min(self.RANSAC_num, n)
        p = None

        if prob is not None:
            p = np.asarray(prob, dtype=np.float64).reshape(-1)  # 兼容 torch/numpy/list
            p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
            p[p < 0] = 0.0
            s = p.sum()
            if s > 0.0:
                p /= s
                eps = 1e-3  # ← 与均匀分布平滑混合
                p = (1.0 - eps) * p + eps * (1.0 / p.size)
                p /= p.sum()
            else:
                p = None  # 退化为均匀采样（全零/无效时）

        sampled_index = np.random.choice(np.arange(n), size=size, replace=False, p=p)
        return sampled_index

    def sample_wo_neighbour(self, x, sampled_index):
        assert len(x) == 1
        _, ch, h, w = x.shape
        x = x.contiguous().view([ch, int(h * w)])
        return x[:, sampled_index]

    def minimal_solver(self, coords2Ds, normalrays, RANSAC_trial):
        """ RANSAC Minimal Solver
        """
        minimal_sample = 2
        device = coords2Ds.device

        sample_num = int(minimal_sample * RANSAC_trial)
        coords2Dc, normal = coords2Ds[:, 0:sample_num], normalrays[:, 0:sample_num]

        x1, y1, _ = torch.split(coords2Dc, 1, dim=0)
        n1, n2, n3 = torch.split(normal, 1, dim=0)

        n1 = n1 / n3
        n2 = n2 / n3

        x1, y1 = x1.view(minimal_sample, RANSAC_trial), y1.view(minimal_sample, RANSAC_trial)
        n1, n2 = n1.view(minimal_sample, RANSAC_trial), n2.view(minimal_sample, RANSAC_trial)

        fx = (x1[1] - x1[0]) / (n1[1] - n1[0] + 1e-10)
        bx = (x1[0] - n1[0] * fx) * 0.5 + (x1[1] - n1[1] * fx) * 0.5

        fy = (y1[1] - y1[0]) / (n2[1] - n2[0] + 1e-10)
        by = (y1[0] - n2[0] * fy) * 0.5 + (y1[1] - n2[1] * fy) * 0.5

        intrinsic = torch.eye(3).view([1, 3, 3]).repeat([len(fx), 1, 1]).to(device)
        intrinsic[:, 0, 0] = fx
        intrinsic[:, 1, 1] = fy
        intrinsic[:, 0, 2] = bx
        intrinsic[:, 1, 2] = by

        return intrinsic

    # def calibrate_camera_4DoF(self, incidence, RANSAC_trial=2048):
    #     """ 4DoF RANSAC Camera Calibration
    #
    #     Args:
    #         incidence (tensor): Incidence Field
    #         RANSAC_trial (int): RANSAC Iteration Number. Default: 2048.
    #     """
    #     # Calibrate assume a simple pinhole camera model
    #     b, _, h, w = incidence.shape
    #     device = incidence.device
    #     coords2D = self.initcoords2D(b, h, w, device, homogeneous=True)
    #
    #     sampled_index = self.get_sample_idx(h, w)
    #     normalrays = self.sample_wo_neighbour(incidence, sampled_index)
    #     coords2Ds = self.sample_wo_neighbour(coords2D, sampled_index)
    #
    #     # Prepare for RANSAC
    #     intrinsic = self.minimal_solver(coords2Ds, normalrays, RANSAC_trial)
    #
    #     valid = (intrinsic[:, 0, 0] < 1e-2).float() + (intrinsic[:, 1, 1] < 1e-2).float()
    #     valid = valid == 0
    #     intrinsic = intrinsic[valid]
    #
    #     # RANSAC Loop
    #     intrinsic_inv = torch.linalg.inv(intrinsic)
    #     normalray_ransac = intrinsic_inv @ coords2Ds.unsqueeze(0)
    #     diffx, diffy = self.scoring_function_xy(normalray_ransac, normalrays)
    #     intrinsic_x, intrinsic_y = intrinsic, intrinsic
    #
    #     maxid = torch.argmax(diffx)
    #     fx, bx = intrinsic_x[maxid, 0, 0], intrinsic_x[maxid, 0, 2]
    #     maxid = torch.argmax(diffy)
    #     fy, by = intrinsic_y[maxid, 1, 1], intrinsic_y[maxid, 1, 2]
    #
    #     intrinsic_opt = torch.eye(3).to(device)
    #     intrinsic_opt[0, 0] = fx
    #     intrinsic_opt[0, 2] = bx
    #     intrinsic_opt[1, 1] = fy
    #     intrinsic_opt[1, 2] = by
    #
    #     intrinsic_opt = MonocularCalibrator.unnorm_intrinsic(intrinsic_opt.unsqueeze(0), b, h, w, device)
    #     return intrinsic_opt.squeeze(0)

    # TODO 置信度
    def calibrate_camera_4DoF(
        self,
        incidence,
        RANSAC_trial=2048,
        weights=None,
        sampling_weights=None,
        scoring_weights=None,
        seed=None,
    ):
        """
        incidence: (1,3,H,W) 单位方向（n_x,n_y,n_z）
        weights: legacy shorthand that enables both confidence-guided sampling and
            confidence-weighted scoring.
        sampling_weights: optional (1,1,H,W) weights used only for sampling.
        scoring_weights: optional (1,1,H,W) weights used only for scoring.
        seed: optional NumPy sampling seed. Reusing a seed makes ablations that
            share a sampling strategy use exactly the same sampled pixels.
        """
        b, _, h, w = incidence.shape
        device = incidence.device
        coords2D = self.initcoords2D(b, h, w, device, homogeneous=True)

        if weights is not None:
            if sampling_weights is None:
                sampling_weights = weights
            if scoring_weights is None:
                scoring_weights = weights

        # === 采样概率 ===
        prob = None
        if sampling_weights is not None:
            # 展平并正规化成概率分布
            prob_flat = sampling_weights.view(1, -1).squeeze().detach()
            prob = (prob_flat / (prob_flat.sum() + 1e-12)).cpu().numpy()

        # === 抽样 ===
        sampled_index = self.get_sample_idx(h, w, prob=prob, seed=seed)
        normalrays = self.sample_wo_neighbour(incidence, sampled_index)  # (3, N)
        coords2Ds = self.sample_wo_neighbour(coords2D, sampled_index)  # (3, N)

        # 若有权重，取抽样位置对应的权重子集（形状变成 (N,)）
        weights_samp = None
        if scoring_weights is not None:
            weights_samp = self.sample_wo_neighbour(scoring_weights, sampled_index)[0]

        # === 最小二乘求解所有假设（与你原来的 minimal_solver 完全一致） ===
        intrinsic = self.minimal_solver(coords2Ds, normalrays, RANSAC_trial)

        # 筛掉明显非法的假设
        valid = (intrinsic[:, 0, 0] < 1e-2).float() + (intrinsic[:, 1, 1] < 1e-2).float()
        valid = valid == 0
        intrinsic = intrinsic[valid]

        # === RANSAC 打分（这里用“加权内点数”替代“内点个数”） ===
        intrinsic_inv = torch.linalg.inv(intrinsic)
        normalray_ransac = intrinsic_inv @ coords2Ds.unsqueeze(0)  # (num_hypos,3,N)
        diffx, diffy = self.scoring_function_xy(normalray_ransac, normalrays, weights=weights_samp)

        intrinsic_x, intrinsic_y = intrinsic, intrinsic
        maxid = torch.argmax(diffx)
        fx, bx = intrinsic_x[maxid, 0, 0], intrinsic_x[maxid, 0, 2]
        maxid = torch.argmax(diffy)
        fy, by = intrinsic_y[maxid, 1, 1], intrinsic_y[maxid, 1, 2]

        intrinsic_opt = torch.eye(3).to(device)
        intrinsic_opt[0, 0] = fx
        intrinsic_opt[0, 2] = bx
        intrinsic_opt[1, 1] = fy
        intrinsic_opt[1, 2] = by

        intrinsic_opt = MonocularCalibrator.unnorm_intrinsic(intrinsic_opt.unsqueeze(0), b, h, w, device)
        return intrinsic_opt.squeeze(0)

    # def calibrate_camera_1DoF(self, incidence, r, RANSAC_trial=2048):
    #     """ 1DoF RANSAC Camera Calibration
    #
    #     Args:
    #         incidence (tensor): Incidence Field.
    #         r: Aspect Ratio Restoration from Network Inference Resolution (480 x 640) to the Original Resolution
    #         RANSAC_trial (int): RANSAC Iteration Number. Default: 2048.
    #     """
    #
    #     # Calibrate assume a simple pinhole camera model
    #     b, _, h, w = incidence.shape
    #     assert b == 1
    #     # r = (scaleM[0, 1, 1] / scaleM[0, 0, 0]).item()
    #     device = incidence.device
    #     coords2D = self.initcoords2D(b, h, w, device, homogeneous=True)
    #
    #     sampled_index = self.get_sample_idx(h, w)
    #     normalrays = self.sample_wo_neighbour(incidence, sampled_index)
    #     coords2Ds = self.sample_wo_neighbour(coords2D, sampled_index)
    #
    #     # Prepare for RANSAC
    #     fs = torch.linspace(100, 4096, steps=RANSAC_trial)
    #     intrinsic = torch.eye(3).view([1, 3, 3]).expand([2048, 3, 3]).contiguous().to(device)
    #     intrinsic[:, 0, 2] = float(w / 2)
    #     intrinsic[:, 1, 2] = float(h / 2)
    #     intrinsic[:, 0, 0] = fs
    #     intrinsic[:, 1, 1] = fs * r
    #     intrinsic = self.norm_intrinsic(intrinsic, b, h, w, device)
    #
    #     # RANSAC Loop
    #     intrinsic_inv = torch.linalg.inv(intrinsic)
    #     normalray_ransac = intrinsic_inv @ coords2Ds.unsqueeze(0)
    #     diffx, diffy = self.scoring_function_xy(normalray_ransac, normalrays)
    #
    #     maxid = torch.argmax(diffx + diffy)
    #     fx, bx = intrinsic[maxid, 0, 0], intrinsic[maxid, 0, 2]
    #     fy, by = intrinsic[maxid, 1, 1], intrinsic[maxid, 1, 2]
    #
    #     intrinsic_opt = torch.eye(3).to(device)
    #     intrinsic_opt[0, 0] = fx
    #     intrinsic_opt[0, 2] = bx
    #     intrinsic_opt[1, 1] = fy
    #     intrinsic_opt[1, 2] = by
    #
    #     intrinsic_opt = MonocularCalibrator.unnorm_intrinsic(intrinsic_opt.unsqueeze(0), b, h, w, device)
    #     return intrinsic_opt.squeeze(0)



    # TODO 置信度
    def calibrate_camera_1DoF(
        self,
        incidence,
        r,
        RANSAC_trial=2048,
        weights=None,
        sampling_weights=None,
        scoring_weights=None,
        seed=None,
    ):
        b, _, h, w = incidence.shape
        device = incidence.device
        coords2D = self.initcoords2D(b, h, w, device, homogeneous=True)

        if weights is not None:
            if sampling_weights is None:
                sampling_weights = weights
            if scoring_weights is None:
                scoring_weights = weights

        prob = None
        if sampling_weights is not None:
            prob_flat = sampling_weights.view(1, -1).squeeze().detach()
            prob = (prob_flat / (prob_flat.sum() + 1e-12)).cpu().numpy()

        sampled_index = self.get_sample_idx(h, w, prob=prob, seed=seed)
        normalrays = self.sample_wo_neighbour(incidence, sampled_index)
        coords2Ds = self.sample_wo_neighbour(coords2D, sampled_index)

        if scoring_weights is not None:
            weights_samp = self.sample_wo_neighbour(scoring_weights, sampled_index)[0]
        else:
            weights_samp = None

        # 与原实现相同：枚举 fs，构造假设
        fs = torch.linspace(100, 4096, steps=RANSAC_trial)
        intrinsic = torch.eye(3).view([1, 3, 3]).expand([RANSAC_trial, 3, 3]).contiguous().to(device)
        intrinsic[:, 0, 2] = float(w / 2)
        intrinsic[:, 1, 2] = float(h / 2)
        intrinsic[:, 0, 0] = fs
        intrinsic[:, 1, 1] = fs * r
        intrinsic = self.norm_intrinsic(intrinsic, b, h, w, device)

        intrinsic_inv = torch.linalg.inv(intrinsic)
        normalray_ransac = intrinsic_inv @ coords2Ds.unsqueeze(0)
        diffx, diffy = self.scoring_function_xy(normalray_ransac, normalrays, weights=weights_samp)
        maxid = torch.argmax(diffx + diffy)
        fx, bx = intrinsic[maxid, 0, 0], intrinsic[maxid, 0, 2]
        fy, by = intrinsic[maxid, 1, 1], intrinsic[maxid, 1, 2]

        intrinsic_opt = torch.eye(3).to(device)
        intrinsic_opt[0, 0] = fx
        intrinsic_opt[0, 2] = bx
        intrinsic_opt[1, 1] = fy
        intrinsic_opt[1, 2] = by

        intrinsic_opt = MonocularCalibrator.unnorm_intrinsic(intrinsic_opt.unsqueeze(0), b, h, w, device)
        return intrinsic_opt.squeeze(0)

    def restore_image(self, image, intrinsic, fixcrop=True):
        # Adjust Intrinsic with Crop and Resize
        w, h = image.size
        wt, ht = image.size

        # Fix Aspect Ratio, Avoid Image Reduced
        resizeM = np.eye(3)
        if intrinsic[0, 0] > intrinsic[1, 1]:
            r = intrinsic[0, 0] / intrinsic[1, 1]
            resizeM[1, 1] = r
            wt, ht = wt, ht * r
        else:
            r = intrinsic[1, 1] / intrinsic[0, 0]
            resizeM[0, 0] = r
            wt, ht = wt * r, ht
        wt, ht = int(np.ceil(wt).item()), int(np.ceil(ht).item())

        # Fix Crop
        cropM = np.eye(3)
        if fixcrop:
            intrinsic_ = resizeM @ intrinsic
            padding_lr, padding_ud = intrinsic_[0, 2] - wt / 2, intrinsic_[1, 2] - ht / 2
            if padding_lr < 0:
                cropM[0, 2] = -padding_lr
            if padding_ud < 0:
                cropM[1, 2] = -padding_ud

            wt, ht = int(np.ceil(wt + np.abs(padding_lr)).item()), int(np.ceil(ht + np.abs(padding_ud)).item())

        resample_matrix = np.linalg.inv(cropM @ resizeM)
        totensor = transforms.ToTensor()
        image_restore = resample_rgb(
            totensor(image).unsqueeze(0),
            torch.from_numpy(resample_matrix).float().view([1, 3, 3]),
            batch=1, ht=ht, wd=wt, device=torch.device("cpu")
        )

        return tensor2rgb(image_restore, viewind=0)
