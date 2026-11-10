from __future__ import print_function, division
import os, sys, inspect
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))))
sys.path.insert(0, project_root)

import warnings
warnings.filterwarnings('ignore')
import torch
import argparse
from loguru import logger
from pprint import pprint

from WildCamera.newcrfs.newcrf_incidencefield import NEWCRFIF


parser = argparse.ArgumentParser(description='NeWCRFs PyTorch implementation.', fromfile_prefix_chars='@')
parser.add_argument('--load_ckpt',                 type=str,   help='path of ckpt' ,default="E:/hsh/second/WildCamera/model_zoo/Release")
parser.add_argument('--data_path',                 type=str,   help='path to the data', default='data/MonoCalib')
# parser.add_argument('--experiment_name',           type=str,   help='name of the experiment', required=True, choices=['in_the_wild', 'gsv'])
parser.add_argument('--experiment_name',           type=str,   help='name of the experiment', default='in_the_wild')
def main_worker(args, wtassumption=False):
    args.gpu = 0
    pprint(vars(args))

    model = NEWCRFIF(version='large07', pretrained=None)
    # model.load_state_dict(torch.load(args.load_ckpt, map_location="cpu"), strict=True)
    state = torch.load(args.load_ckpt, map_location="cpu")

    # 兼容多种保存格式：{model: state_dict}, {state_dict: ...}, 直接 state_dict
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    elif isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    # 如来自 DataParallel, 去掉 'module.' 前缀
    from collections import OrderedDict

    def strip_module_prefix(sd):
        new_sd = OrderedDict()
        for k, v in sd.items():
            if k.startswith("module."):
                new_sd[k[7:]] = v
            else:
                new_sd[k] = v
        return new_sd

    state = torch.load(args.load_ckpt, map_location="cpu")
    # 兼容不同保存格式
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    elif isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    state = strip_module_prefix(state)

    # 尽量用 strict=True，避免默默漏载造成劣化
    missing, unexpected = model.load_state_dict(state, strict=False)  # 先宽松看情况
    print("=> missing:", missing)
    print("=> unexpected:", unexpected)

    # 建议：如果是同架构评估，改 strict=True，并在有缺失/意外时直接报错
    # model.load_state_dict(state, strict=True)

    model.eval()
    model.cuda()
    logger.info("Load Model from %s" % args.load_ckpt)

    if args.experiment_name == 'in_the_wild':
        from WildCamera.evaluation.evaluate_intrinsic import EvaluateIntrinsic
        evaluate_intrinsic = EvaluateIntrinsic()
        evaluate_intrinsic.evaluate(model, args, steps=0, writer=None, group=None, wtassumption=wtassumption)
    elif args.experiment_name == 'gsv':
        from WildCamera.evaluation.evaluate_fov import EvaluateFov
        evaluate_fov = EvaluateFov()
        evaluate_fov.evaluate(model, args, steps=0, writer=None, group=None, wtassumption=wtassumption)


if __name__ == '__main__':
    args = parser.parse_args()

    args.world_size = torch.cuda.device_count()
    args.augscale = 2.0
    args.input_height = 480
    args.input_width = 640

    args.eval_workers = 4
    args.l1_th = 0.02

    if args.experiment_name == 'in_the_wild':
        args.load_ckpt = os.path.join(project_root, 'model_zoo/Release', 'wild_camera_all.pth')
        args.datasets_train = [
            'Nuscenes',
            'KITTI',
            'Cityscapes',
            'NYUv2',
            'ARKitScenes',
            'MegaDepth',
            'SUN3D',
            'MVImgNet',
            'Objectron',
        ]
        args.datasets_eval = [
            'Nuscenes',
            'KITTI',
            'Cityscapes',
            'NYUv2',
            'ARKitScenes',
            'MegaDepth',
            'SUN3D',
            'MVImgNet',
            'Objectron',
            'Waymo',
            'BIWIRGBDID',
            'RGBD',
            'ScanNet',
            'CAD120',
            'MVS',
            'Scenes11',
        ]
        main_worker(args, wtassumption=False)
        main_worker(args, wtassumption=True)
    elif args.experiment_name == 'gsv':
        args.load_ckpt = os.path.join(project_root, 'model_zoo/Release', 'wild_camera_gsv.pth')
        args.datasets_train = [
            'GSV'
        ]
        args.datasets_eval = [
            'GSV'
        ]
        main_worker(args, wtassumption=False)
        main_worker(args, wtassumption=True)
    else:
        raise NotImplementedError()


