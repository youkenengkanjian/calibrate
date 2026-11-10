# Calibrate

面向单张图像的相机内参标定研究代码，结合入射场预测与像素级置信度，实现置信度加权的几何估计。

## 主要功能

- 入射场与像素置信度联合预测。
- 置信度加权 RANSAC 标定。
- 模型训练、标定评估、置信度消融与结果可视化。

## 使用

使用 Python 3.9 或更新版本，先安装相互匹配的 PyTorch、torchvision 和 MMCV，再安装其余依赖：

```bash
pip install -r requirements.txt
```

训练入口见 [train](WildCamera/train/)，评估与消融入口见 [benchmark](WildCamera/benchmark/)。数据集与模型权重需另行准备，加载时请使用与模型输出头匹配的权重。
