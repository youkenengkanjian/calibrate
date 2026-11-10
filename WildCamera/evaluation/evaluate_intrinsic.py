import sys, os, time, inspect, tabulate
import tqdm
import numpy as np

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from loguru import logger
import torch.nn.functional as F
from tools.tools import to_cuda
from tools.calibrator import MonocularCalibrator
from tools.evaluation import compute_intrinsic_measure
from tools.tools import DistributedSamplerNoEvenlyDivisible
from WildCamera.datasets.IncdDataset import IncdDataset

class EvaluateIntrinsic:
    def __init__(self):
        self.min_focal = 1e10

    def evaluate(self, model, args, steps, writer=None, group=None, wtassumption=False):

        if len(args.datasets_eval) > 1:
            autosave = False
        else:
            autosave = True

        measurements_all = dict()

        for dataaset in args.datasets_eval:

            if dataaset in args.datasets_train:
                augmentation = True
            else:
                augmentation = False

            val_dataset = IncdDataset(
                data_root=args.data_path,
                datasets_included=[dataaset],
                ht=args.input_height,
                wt=args.input_width,
                augmentation=augmentation,
                shuffleseed=None,
                split='test',
                augscale=args.augscale,
                no_change_prob=0.0,
                transformcategory='transform_calibration'
            )
            measurements = evaluate_intrinsic(model, val_dataset, args, group, wtassumption=wtassumption)
            measurements_all[dataaset] = measurements

            if args.gpu == 0:
                if writer is not None and (steps > 0):
                    for k in measurements.keys():
                        writer.add_scalar('Eval_{}/{}'.format(dataaset, k), measurements[k], steps)

            if args.gpu == 0 and autosave and (steps > 0):
                if measurements['error_f'] < self.min_focal:
                    self.min_focal = measurements['error_f']
                    svpath = os.path.join(args.saving_location, 'model_zoo', args.experiment_name, 'min_focal.ckpt')
                    torch.save(model.state_dict(), svpath)
                    logger.info("Save to %s" % svpath)

        if args.gpu == 0 and (not autosave) and (steps > 0):
            torch.save(model.state_dict(), os.path.join(args.saving_location, 'model_zoo', args.experiment_name, 'step_{}.ckpt'.format(str(steps))))

        if args.gpu == 0:
            errors_all_mean = [['Datasets', 'error_fx', 'error_fy', 'error_f', 'error_bx', 'error_by', 'error_b']]
            for key in measurements_all.keys():
                measurements = measurements_all[key]
                errors_all_mean.append(
                    [
                        key,
                        measurements['error_fx'],
                        measurements['error_fy'],
                        measurements['error_f'],
                        measurements['error_bx'],
                        measurements['error_by'],
                        measurements['error_b']
                    ]
                )
            logger.info(
                "\n====================  Performance at Step %d with Assumption %d ====================\n" % (steps, wtassumption) +
                tabulate.tabulate(errors_all_mean, headers='firstrow', tablefmt='fancy_grid', numalign="center", floatfmt=".3f"))

        torch.cuda.synchronize()
        return

@torch.no_grad()
def evaluate_intrinsic(model, val_dataset, args, group=None, wtassumption=False):
    """ In Validation, random sample N Images to mimic a real test set situation  """
    monocalibrator = MonocularCalibrator(l1_th=args.l1_th)
    model.eval()
    if args.world_size > 1:
        sampler = DistributedSamplerNoEvenlyDivisible(val_dataset, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=1, sampler=sampler, pin_memory=False, shuffle=False, num_workers=args.eval_workers, drop_last=False)
    else:
        val_loader = DataLoader(val_dataset, batch_size=1, pin_memory=False, shuffle=False, num_workers=args.eval_workers, drop_last=False)

    measurements = torch.zeros(len(val_dataset), 6).cuda(device=args.gpu)
    world_size = args.world_size

    for val_id, data_blob in enumerate(tqdm.tqdm(val_loader, disable=False)):
        sample_batched = to_cuda(data_blob)
        rgb, K, r = sample_batched['rgb'], sample_batched['K'], sample_batched['aspect_ratio_restoration']
        # TODO 置信度
        # incidence = model(rgb)
        out = model(rgb)  # 现在模型可能返回 dict
        # incidence = out["incidence"] if isinstance(out, dict) else out
        # 1) 取 incidence 和 conf（若模型未返回字典就按 None 处理）
        if isinstance(out, dict):
            incidence = out["incidence"]
            weights = out.get("conf", None)  # (B,1,H,W)
        else:
            incidence = out
            weights = None
        # 2) 尺寸不一致就插值到 incidence 的大小
        # TODO 第一次改动
        # if weights is not None and (weights.shape[-2:] != incidence.shape[-2:]):
        #     weights = torch.nn.functional.interpolate(weights, size=incidence.shape[-2:], mode='bilinear',
        #                                               align_corners=False)
        # TODO 第二次改动
        if weights is not None:
            # 1) 与 incidence 对齐尺寸
            if weights.shape[-2:] != incidence.shape[-2:]:
                weights = F.interpolate(weights, size=incidence.shape[-2:], mode="bilinear", align_corners=False)

            # 2) 映射到 [0,1]
            wmin = weights.amin(dim=(2, 3), keepdim=True)
            wmax = weights.amax(dim=(2, 3), keepdim=True)
            weights = (weights - wmin) / (wmax - wmin + 1e-6)
            weights = weights.clamp(0.0, 1.0)

            # 3) 每张图取置信度最高的前 5%
            TOP_P = 0.05
            B, _, H, W = weights.shape
            flat = weights.view(B, -1)           # (B, H*W)
            N = flat.size(1)
            k = max(int((1.0 - TOP_P) * N), 1)   # 95% 分位
            thresh, _ = torch.kthvalue(flat, k, dim=1, keepdim=True)  # (B,1)
            thresh = thresh.view(B, 1, 1, 1)

            top_mask = (weights >= thresh)       # bool
            weights = top_mask.float()           # 只用前 5% 区域


        # if not wtassumption:
        #     Kgt, Kest = torch.clone(K).squeeze(0), monocalibrator.calibrate_camera_4DoF(incidence, RANSAC_trial=2048)
        # else:
        #     Kgt, Kest = torch.clone(K).squeeze(0), monocalibrator.calibrate_camera_1DoF(incidence, r=r.item())
        # 3) 传入加权 RANSAC
        if not wtassumption:
            Kgt, Kest = torch.clone(K).squeeze(0), monocalibrator.calibrate_camera_4DoF(
                incidence, RANSAC_trial=2048, weights=weights)
        else:
            Kgt, Kest = torch.clone(K).squeeze(0), monocalibrator.calibrate_camera_1DoF(
                incidence, r=r.item(), RANSAC_trial=2048, weights=weights)
        b, _, h, w = rgb.shape
        device = rgb.device

        error_fx, error_fy, error_f, error_bx, error_by, error_b = compute_intrinsic_measure(
            Kest=torch.clone(Kest),
            Kgt=torch.clone(Kgt),
            h=h,
            w=w
        )

        error_all = np.array([error_fx, error_fy, error_f, error_bx, error_by, error_b])
        error_all = torch.from_numpy(error_all).float().to(device)

        measurements[val_id * world_size + args.gpu] += error_all

    if args.world_size > 1:
        dist.all_reduce(tensor=measurements, op=dist.ReduceOp.SUM, group=group)

    zero_entry = torch.sum(measurements.abs(), dim=1) == 0
    assert torch.sum(zero_entry) == 0

    measurements = torch.mean(measurements, dim=0)
    measurements = {
        'error_fx': measurements[0].item(),
        'error_fy': measurements[1].item(),
        'error_f': measurements[2].item(),
        'error_bx': measurements[3].item(),
        'error_by': measurements[4].item(),
        'error_b': measurements[5].item()
    }

    return measurements