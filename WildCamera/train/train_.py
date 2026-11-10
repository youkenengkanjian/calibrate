from __future__ import print_function, division
import os, sys, inspect, time, copy, warnings
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))))
sys.path.insert(0, project_root)
warnings.filterwarnings('ignore')

import torch
import torch.backends.cudnn as cudnn
# 1. 移除分布式相关导入（dist/mp）
# import torch.distributed as dist
# import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import ConcatDataset
from torchvision import transforms

import argparse
import numpy as np
from loguru import logger
from einops import rearrange
from pprint import pprint

from WildCamera.newcrfs.newcrf_incidencefield import NEWCRFIF
from WildCamera.datasets.IncdDataset import IncdDataset
# 2. 移除分布式采样器（仅保留单卡数据加载逻辑）
from tools.tools import to_cuda, intrinsic2incidence, resample_rgb, IncidenceLoss  # 删掉 DistributedSamplerNoEvenlyDivisible
from tools.visualization import tensor2disp, tensor2rgb

parser = argparse.ArgumentParser(description='NeWCRFs PyTorch implementation.', fromfile_prefix_chars='@')

parser.add_argument('--experiment_name',           type=str,   help='experiment name', default='multi-modality newcrfs')
parser.add_argument('--experiment_set',            type=str,   choices=['gsv', 'in_the_wild'], required=False ,default="in_the_wild")
parser.add_argument('--saving_location',           type=str,   help='saving location', default=None)
parser.add_argument('--encoder',                   type=str,   help='type of encoder, base07, large07', default='large07')
parser.add_argument('--pretrain',                  type=str,   help='path of pretrained encoder', default='E:/hsh/second/WildCamera/model_zoo/swin_transformer/swin_large_patch4_window7_224_22k.pth')
# parser.add_argument('--load_ckpt',                 type=str,   help='path of ckpt', default="D:/second/WildCamera/model_zoo/Release/wild_camera_all.pth")
parser.add_argument('--load_ckpt',                 type=str,   help='path of ckpt', default=None)
parser.add_argument('--evaluation_only',           action="store_true")
parser.add_argument('--l1_th',                     type=int,  help='RANSAC threshold', default=0.02)

# Dataset
parser.add_argument('--data_path',                 type=str,   help='path to the data', default='data/MonoCalib')
parser.add_argument('--input_height',              type=int,   help='input height', default=480)
parser.add_argument('--input_width',               type=int,   help='input width',  default=640)

# Training
parser.add_argument('--batch_size',                type=int,   help='batch size (单卡直接使用此值，无需拆分)', default=16)
parser.add_argument('--loss',                      type=str,   default='cosine', help='You can also choose l1 loss')
parser.add_argument('--steps_per_epoch',           type=int,   help='frequency for evaluation', default=1000)
parser.add_argument('--termination_epoch',         type=int,   help='epoch to stop training', default=25)

# Training Misc
parser.add_argument('--weight_decay',              type=float, help='weight decay factor for optimization', default=1e-2)
parser.add_argument('--adam_eps',                  type=float, help='epsilon in Adam optimizer', default=1e-6)
parser.add_argument('--train_workers',             type=int,   help='dataloader workers (单卡直接使用此值)', default=8)  # 建议降低（单卡无需32线程）
parser.add_argument('--eval_workers',              type=int,   help='dataloader workers', default=2)
parser.add_argument('--learning_rate',             type=float, help='initial learning rate (单卡无需缩放)', default=5e-6)
parser.add_argument('--end_learning_rate',         type=float, help='end learning rate', default=-1)

# Augmentation
parser.add_argument('--dataset_favour_long',       type=float, default=0.1, help='whether in training will see more samples from large dataset')
parser.add_argument('--augscale',                  type=float, default=2.0, help='The scale of Augmentation')
parser.add_argument('--no_change_prob',            type=float, default=0.1, help='The probability of seeing original image')
parser.add_argument('--coloraugmentation',         action="store_true")
parser.add_argument('--coloraugmentation_scale',   type=float, default=0.0)

# 3. 移除分布式相关参数（gpu/dist_url/dist_backend），单卡默认使用第0块GPU
parser.add_argument('--gpu_id',                    type=int,   help='GPU id to use (单卡固定为0即可)', default=0)

# ------------------------------
# 4. 移除分布式main_worker，直接写单卡训练逻辑
# ------------------------------
def main():
    args = parser.parse_args()
    torch.cuda.empty_cache()
    # 5. 单卡配置：固定使用指定GPU，关闭分布式标记
    args.gpu = args.gpu_id
    torch.cuda.set_device(args.gpu)  # 绑定GPU
    args.distributed = False         # 强制关闭分布式
    args.world_size = 1              # 单卡世界大小为1

    # 处理保存路径（与原逻辑一致）
    if args.saving_location is None:
        args.saving_location = project_root

    # 数据集配置（与原逻辑一致）
    if args.experiment_set == 'gsv':
        args.datasets_train = ['GSV']
        args.datasets_eval = ['GSV']
        from WildCamera.evaluation.evaluate_fov import EvaluateFov
        evaluator = EvaluateFov()
    elif args.experiment_set == 'in_the_wild':
        args.datasets_train = [
            'Nuscenes', 'KITTI', 'Cityscapes', 'NYUv2', 'ARKitScenes',
            'MegaDepth', 'SUN3D', 'MVImgNet', 'Objectron'
        ]
        args.datasets_eval = [
            'Nuscenes', 'KITTI', 'Cityscapes', 'NYUv2',
            'ARKitScenes',
            'MegaDepth',
            'SUN3D', 'MVImgNet', 'Objectron', 'Waymo',
            'BIWIRGBDID', 'RGBD', 'ScanNet', 'MVS', 'Scenes11'
        ]
        from WildCamera.evaluation.evaluate_intrinsic import EvaluateIntrinsic
        evaluator = EvaluateIntrinsic()
    else:
        raise NotImplementedError()

    pprint(vars(args))
    logger.add(sys.stderr, level="INFO")  # 单卡直接输出日志到控制台

    # ------------------------------
    # 6. 模型初始化（单卡版本）
    # ------------------------------
    model = NEWCRFIF(version=args.encoder, pretrained=args.pretrain)
    model = model.cuda(args.gpu)  # 单卡直接放GPU
    cudnn.benchmark = True        # 加速GPU计算

    # 加载预训练权重（与原逻辑一致，但无需分布式映射）
    if args.load_ckpt is not None:
        checkpoint = torch.load(args.load_ckpt, map_location=f'cuda:{args.gpu}')  # 直接映射到目标GPU
        model.load_state_dict(checkpoint, strict=True)
        model.eval()
        logger.info(f"Load Model from {args.load_ckpt}")

    # ------------------------------
    # 7. 优化器与学习率配置（单卡无需拆分参数）
    # ------------------------------
    optimizer = torch.optim.Adam(
        [{'params': model.parameters()}],  # 单卡直接用model.parameters()（原分布式用model.module）
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        eps=args.adam_eps
    )

    # 学习率终点配置（与原逻辑一致）
    end_learning_rate = args.end_learning_rate if args.end_learning_rate != -1 else 0.1 * args.learning_rate
    steps_per_epoch = args.steps_per_epoch
    num_total_steps = 250000
    epoch = int(num_total_steps / steps_per_epoch)

    # 损失函数（与原逻辑一致）
    incidence_criterion = IncidenceLoss(loss=args.loss)

    # 图像反归一化（与原逻辑一致）
    inv_normalize = transforms.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
    )

    # ------------------------------
    # 8. TensorBoard与日志配置（单卡直接创建）
    # ------------------------------
    checkpoint_dir = os.path.join(args.saving_location, 'model_zoo', args.experiment_name)
    os.makedirs(checkpoint_dir, exist_ok=True)  # 确保目录存在
    writer = SummaryWriter(checkpoint_dir, flush_secs=30)
    logger.add(os.path.join(checkpoint_dir, f"{args.experiment_name}.log"), level="INFO")

    # ------------------------------
    # 9. 仅评估模式（与原逻辑一致，移除分布式group参数）
    # ------------------------------
    if args.evaluation_only:
        evaluator.evaluate(
            model,
            args,
            steps=0,
            writer=writer,
            # group=group,  # 单卡无分布式group，删除此参数（需确保evaluator兼容，若报错则在evaluator中注释group相关逻辑）
            wtassumption=False,
        )
        evaluator.evaluate(
            model,
            args,
            steps=0,
            writer=writer,
            # group=group,
            wtassumption=True,
        )
        return

    # ------------------------------
    # 10. 单卡训练循环（核心修改：移除分布式采样器，用普通DataLoader）
    # ------------------------------
    for n_0 in range(epoch):
        # 超过终止epoch则停止（与原逻辑一致）
        if n_0 > args.termination_epoch:
            break

        # 加载训练集（单卡用普通DataLoader，无需分布式采样器）
        incdataset = IncdDataset(
            data_root=args.data_path,
            datasets_included=args.datasets_train,
            ht=args.input_height,
            wt=args.input_width,
            augmentation=True,
            split='train',
            shuffleseed=int(n_0 + 1),  # 每个epoch不同种子打乱
            dataset_favour_long=args.dataset_favour_long,
            augscale=args.augscale,
            no_change_prob=args.no_change_prob,
            coloraugmentation=args.coloraugmentation,
            coloraugmentation_scale=args.coloraugmentation_scale
        )

        # 单卡DataLoader（shuffle=True替代分布式采样器的打乱）
        dataloader = torch.utils.data.DataLoader(
            incdataset,
            batch_size=args.batch_size,  # 单卡直接用设置的batch_size
            shuffle=True,               # 训练集打乱
            num_workers=args.train_workers,
            pin_memory=True,            # 加速GPU数据传输
            drop_last=True              # 避免最后一个不完整batch
        )

        # 确保dataloader步数足够（与原逻辑一致）
        assert len(dataloader) > steps_per_epoch
        dataloader = iter(dataloader)

        # 单卡训练步循环
        for global_step in range(int(n_0 * steps_per_epoch), int((n_0 + 1) * steps_per_epoch)):
            sample_batched = next(dataloader)
            sample_batched = to_cuda(sample_batched)  # 自动使用已绑定的GPU（args.gpu）

            # 取出数据（与原逻辑一致）
            rgb, K = sample_batched['rgb'], sample_batched['K']

            # 模型前向传播（单卡直接调用model，无需model.module）
            model.train()
            incidence = model(rgb)
            optimizer.zero_grad()  # 清空梯度

            # 计算损失（与原逻辑一致）
            loss = incidence_criterion(incidence, K)
            loss.backward()        # 反向传播

            # 动态调整学习率（与原逻辑一致）
            for param_group in optimizer.param_groups:
                current_lr = (args.learning_rate - end_learning_rate) * \
                             (1 - global_step / num_total_steps) ** 0.9 + end_learning_rate
                param_group['lr'] = current_lr

            optimizer.step()  # 更新参数

            # ------------------------------
            # 11. 可视化与日志（单卡直接执行，无需判断gpu==0）
            # ------------------------------
            if np.mod(global_step, 1000) == 0:
                b = 1
                _, _, h, w = rgb.shape

                # 图像反归一化并可视化
                rgb = inv_normalize(rgb)
                vls1 = tensor2rgb(rgb, viewind=0)

                # 计算GT incidence并可视化
                incidence_gt = intrinsic2incidence(K[0:1], b, h, w, device=rgb.device)
                incidence_gt = rearrange(incidence_gt.squeeze(dim=4), 'b h w d -> b d h w')
                vls3 = tensor2rgb((incidence + 1) / 2, viewind=0)
                vls4 = tensor2rgb((incidence_gt + 1) / 2, viewind=0)

                # 拼接可视化结果并写入TensorBoard
                vls = np.concatenate([np.array(vls1), np.array(vls3), np.array(vls4)], axis=0)
                writer.add_image('visualization', (torch.from_numpy(vls).float() / 255).permute([2, 0, 1]), global_step)

            # 写入损失日志（与原逻辑一致）
            writer.add_scalar('loss/loss_incidence', loss.item(), global_step)
            if np.mod(global_step, 500) == 0:
                logger.info(f'Step {global_step}, Epoch {n_0}, loss {loss.item():.3f}')

        # ------------------------------
        # 12. 每个epoch结束后评估（单卡无需分布式group）
        # ------------------------------
        evaluator.evaluate(
            model,
            args,
            steps=global_step,
            writer=writer,
            # group=group,  # 同评估模式，删除group参数
            wtassumption=False,
        )
        evaluator.evaluate(
            model,
            args,
            steps=global_step,
            writer=writer,
            # group=group,
            wtassumption=True,
        )

    # 训练结束关闭资源
    writer.close()
    logger.info("Single-GPU Training Finished!")

if __name__ == '__main__':
    main()