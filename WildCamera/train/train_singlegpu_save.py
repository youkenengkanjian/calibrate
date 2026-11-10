from __future__ import print_function, division
import os, sys, inspect, time, copy, warnings
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))))
sys.path.insert(0, project_root)
warnings.filterwarnings('ignore')

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import ConcatDataset
from torchvision import transforms

import argparse
import numpy as np
from loguru import logger
from einops import rearrange
from pprint import pprint

# === 进度条：优先用 tqdm，没有则静默降级 ===
try:
    from tqdm.auto import tqdm
except Exception:
    class tqdm:  # type: ignore
        def __init__(self, total=None, desc=None, unit=None, dynamic_ncols=None, leave=None): self.total = total
        def __enter__(self): return self
        def __exit__(self, exc_type, exc, tb): pass
        def update(self, n=1): pass
        def set_postfix(self, **kw): pass

from torch.utils.data.dataloader import default_collate

from WildCamera.newcrfs.newcrf_incidencefield import NEWCRFIF
from WildCamera.datasets.IncdDataset import IncdDataset
from tools.tools import to_cuda, intrinsic2incidence, resample_rgb, IncidenceLoss
from tools.visualization import tensor2disp, tensor2rgb

"""
Single-GPU training script for NeWCRFs incidence field.
Changes vs. original:
1) Removed distributed / multiprocess logic. Always uses a single visible GPU.
2) Replaced DistributedSamplerNoEvenlyDivisible with standard shuffling DataLoader.
3) Optimizer now uses model.parameters() (no .module indirection).
4) Always creates SummaryWriter on single process.
5) （已按你的要求）不再保存任何 checkpoint。
6) （已按你的要求）不再对坏的/损坏的 h5py 样本做“跳过”处理，遇到异常将直接抛出。
"""

parser = argparse.ArgumentParser(description='NeWCRFs PyTorch (single-GPU).', fromfile_prefix_chars='@')

parser.add_argument('--experiment_name',           type=str,   help='experiment name', default='multi-modality newcrfs')
parser.add_argument('--experiment_set',            type=str,   choices=['gsv', 'in_the_wild'], required=False,default='in_the_wild')
parser.add_argument('--saving_location',           type=str,   help='saving location', default=None)
parser.add_argument('--encoder',                   type=str,   help='type of encoder, base07, large07', default='large07')
parser.add_argument('--pretrain',                  type=str,   help='path of pretrained encoder', default='E:/hsh/second/WildCamera/model_zoo/swin_transformer/swin_large_patch4_window7_224_22k.pth')
# parser.add_argument('--load_ckpt',                 type=str,   help='path of ckpt', default="E:/hsh/second/WildCamera/model_zoo/Release/wild_camera_all.pth")
parser.add_argument('--load_ckpt',                 type=str,   help='path of ckpt', default=None)
parser.add_argument('--evaluation_only',           action="store_true")
parser.add_argument('--l1_th',                     type=int, help='RANSAC threshold', default=0.02)

# Dataset
parser.add_argument('--data_path',                 type=str,   help='path to the data', default='data/MonoCalib')
parser.add_argument('--input_height',              type=int,   help='input height', default=480)
parser.add_argument('--input_width',               type=int,   help='input width',  default=640)

# Training
parser.add_argument('--batch_size',                type=int,   help='batch size', default=12)
parser.add_argument('--loss',                      type=str,   default='cosine', help='You can also choose l1 loss')
parser.add_argument('--steps_per_epoch',           type=int,   help='steps (batches) per epoch', default=1000)
parser.add_argument('--termination_epoch',         type=int,   help='epoch to stop training', default=19)

# Training Misc
parser.add_argument('--weight_decay',              type=float, help='weight decay factor for optimization', default=1e-2)
parser.add_argument('--adam_eps',                  type=float, help='epsilon in Adam optimizer', default=1e-6)
parser.add_argument('--train_workers',             type=int,   help='dataloader workers', default=8)
parser.add_argument('--eval_workers',              type=int,   help='dataloader workers', default=2)
parser.add_argument('--learning_rate',             type=float, help='initial learning rate', default=1e-5)
parser.add_argument('--end_learning_rate',         type=float, help='end learning rate (if -1 uses 0.1 * lr)', default=-1)

# Augmentation
parser.add_argument('--dataset_favour_long',       type=float, default=0.1, help='whether in training will see more samples from large dataset')
parser.add_argument('--augscale',                  type=float, default=2.0, help='The scale of Augmentation')
parser.add_argument('--no_change_prob',            type=float, default=0.1, help='The probability of seeing original image')
parser.add_argument('--coloraugmentation',         action="store_true")
parser.add_argument('--coloraugmentation_scale',   type=float, default=0.0)

# Single-GPU selection
parser.add_argument('--gpu',                       type=int,   help='GPU id to use', default=0)
# （已移除）--debug_bad_h5 参数及其相关逻辑

def main_worker(args):
    # （已移除）h5py.File 守护安装

    # Device
    assert torch.cuda.is_available(), "CUDA is required for single-GPU training."
    torch.cuda.set_device(args.gpu)
    device = torch.device(f'cuda:{args.gpu}')

    # Model
    model = NEWCRFIF(version=args.encoder, pretrained=args.pretrain)
    if args.load_ckpt is not None:
        state = torch.load(args.load_ckpt, map_location='cpu')
        model.load_state_dict(state, strict=True)
        logger.info(f"Loaded checkpoint from {args.load_ckpt}")
    model.to(device)

    # Optimizer
    optimizer = torch.optim.Adam([{'params': model.parameters()}], lr=args.learning_rate, eps=args.adam_eps, weight_decay=args.weight_decay)

    cudnn.benchmark = True

    # Criterion
    incidence_criterion = IncidenceLoss(loss=args.loss)

    end_learning_rate = args.end_learning_rate if args.end_learning_rate != -1 else 0.1 * args.learning_rate

    steps_per_epoch = args.steps_per_epoch
    num_total_steps = 20000
    total_epochs = int(num_total_steps / steps_per_epoch)

    inv_normalize = transforms.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
    )

    # Logging（仍保留 TensorBoard 与日志文件，方便观察训练）
    if args.saving_location is None:
        args.saving_location = project_root
    log_dir = os.path.join(args.saving_location, 'model_zoo', args.experiment_name)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir, flush_secs=30)
    logger.add(os.path.join(log_dir, f"{args.experiment_name}.log"))

    # Evaluator
    if args.experiment_set == 'gsv':
        from WildCamera.evaluation.evaluate_fov import EvaluateFov
        evaluator = EvaluateFov()
    elif args.experiment_set == 'in_the_wild':
        from WildCamera.evaluation.evaluate_intrinsic import EvaluateIntrinsic
        evaluator = EvaluateIntrinsic()
    else:
        raise NotImplementedError()

    # Evaluation-only path
    if args.evaluation_only:
        model.eval()
        evaluator.evaluate(model, args, steps=0, writer=writer, group=None, wtassumption=False)
        evaluator.evaluate(model, args, steps=0, writer=writer, group=None, wtassumption=True)
        writer.close()
        return

    # Build dataset(s)
    if args.experiment_set == 'gsv':
        train_sets = ['GSV']
        eval_sets  = ['GSV']
    elif args.experiment_set == 'in_the_wild':
        train_sets = [
            'Nuscenes',
            'KITTI',
            'Cityscapes',
            'NYUv2',
            'ARKitScenes',
            'MegaDepth',
            'SUN3D',
            'MVImgNet',
            'Objectron'
        ]
        eval_sets  = [
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
            'MVS',
            'Scenes11'
        ]
    else:
        raise NotImplementedError()

    args.datasets_train = train_sets
    args.datasets_eval = eval_sets

    # Training loop
    global_step = 0
    for epoch in range(total_epochs):
        if epoch > args.termination_epoch:
            break

        base_dataset = IncdDataset(
            data_root=args.data_path,
            datasets_included=args.datasets_train,
            ht=args.input_height,
            wt=args.input_width,
            augmentation=True,
            split='train',
            shuffleseed=int(epoch + 1),
            dataset_favour_long=args.dataset_favour_long,
            augscale=args.augscale,
            no_change_prob=args.no_change_prob,
            coloraugmentation=args.coloraugmentation,
            coloraugmentation_scale=args.coloraugmentation_scale
        )

        # 直接使用原始 dataset（不再包装“安全跳过”逻辑）
        dataloader = torch.utils.data.DataLoader(
            base_dataset,
            shuffle=True,
            batch_size=int(args.batch_size),
            num_workers=int(args.train_workers),
            pin_memory=True,
            drop_last=True,
        )

        data_iter = iter(dataloader)

        # 如果一个 epoch 需要的 step 多于 dataset 提供的批次数，则循环使用 dataloader
        def next_batch():
            nonlocal data_iter
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)
            return batch

        epoch_loss_sum = 0.0
        epoch_loss_count = 0

        model.train()
        with tqdm(total=steps_per_epoch, desc=f"Epoch {epoch}", unit="step", dynamic_ncols=True, leave=False) as pbar:
            for local_step in range(steps_per_epoch):
                sample_batched = next_batch()
                sample_batched = to_cuda(sample_batched)

                rgb, K = sample_batched['rgb'], sample_batched['K']

                incidence = model(rgb)
                optimizer.zero_grad(set_to_none=True)

                loss = incidence_criterion(incidence, K)
                loss.backward()

                # Poly decay (same formula as original)
                current_lr = (args.learning_rate - end_learning_rate) * (1 - global_step / num_total_steps) ** 0.9 + end_learning_rate
                for param_group in optimizer.param_groups:
                    param_group['lr'] = current_lr

                optimizer.step()

                # accumulate epoch loss for reporting
                epoch_loss_sum += float(loss.item())
                epoch_loss_count += 1

                # Logging
                if global_step % 1000 == 0:
                    b = 1
                    _, _, h, w = rgb.shape
                    with torch.no_grad():
                        rgb_vis = inv_normalize(rgb)
                        vls1 = tensor2rgb(rgb_vis, viewind=0)
                        device_rgb = rgb.device
                        incidence_gt = intrinsic2incidence(K[0:1], b, h, w, device_rgb)
                        incidence_gt = rearrange(incidence_gt.squeeze(dim=4), 'b h w d -> b d h w')
                        vls3 = tensor2rgb((incidence + 1) / 2, viewind=0)
                        vls4 = tensor2rgb((incidence_gt + 1) / 2, viewind=0)
                        vls = np.concatenate([np.array(vls1), np.array(vls3), np.array(vls4)], axis=0)
                        writer.add_image('visualization', (torch.from_numpy(vls).float() / 255).permute([2, 0, 1]), global_step)

                writer.add_scalar('loss/loss_incidence', loss.item(), global_step)
                if global_step % 500 == 0:
                    logger.info(f'Step {global_step}, Epoch {epoch}, loss {loss.item():.3f}, lr {current_lr:.2e}')

                pbar.set_postfix(loss=f"{loss.item():.3f}", lr=f"{current_lr:.2e}")
                pbar.update(1)

                global_step += 1

        # End of epoch evaluation（仍保留）
        model.eval()
        evaluator.evaluate(model, args, steps=global_step, writer=writer, group=None, wtassumption=False)
        evaluator.evaluate(model, args, steps=global_step, writer=writer, group=None, wtassumption=True)

        # （已移除）epoch 平均损失用于 best/ckpt 的逻辑
        epoch_avg_loss = (epoch_loss_sum / max(1, epoch_loss_count))
        logger.info(f"Epoch {epoch} done, avg_loss {epoch_avg_loss:.4f}")

        model.train()

    writer.close()


def main():
    args = parser.parse_args()
    torch.cuda.empty_cache()

    # Force single-GPU mode
    args.world_size = 1
    args.distributed = False

    # （已移除）debug_bad_h5 相关逻辑

    if args.experiment_set == 'gsv':
        args.datasets_train = ['GSV']
        args.datasets_eval = ['GSV']
    elif args.experiment_set == 'in_the_wild':
        args.datasets_train = [
            'Nuscenes','KITTI','Cityscapes','NYUv2',
            'ARKitScenes',
            'MegaDepth','SUN3D','MVImgNet','Objectron'
        ]
        args.datasets_eval = [
            'Nuscenes','KITTI','Cityscapes','NYUv2',
            'ARKitScenes',
            # 'MegaDepth','SUN3D','MVImgNet','Objectron',
            # 'Waymo','BIWIRGBDID','RGBD','ScanNet','MVS','Scenes11'
        ]
    else:
        raise NotImplementedError()

    pprint(vars(args))
    main_worker(args)


if __name__ == '__main__':
    main()
