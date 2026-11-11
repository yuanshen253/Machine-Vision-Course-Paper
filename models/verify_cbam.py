# -*- coding: utf-8 -*-
"""
用途：训练前空跑验证三个 CBAM 模型的通道对齐是否正确，避免浪费 GPU 时间。
用法：在 conda 环境（yolov8）下执行
    /workspace/new/conda_envs/yolov8/bin/python /workspace/new/conda_envs/yolov8/models/verify_cbam.py
只要不报维度错误，就说明 yaml 里 CBAM 的通道数写对了，可以拿去训练。
"""
import torch
from pathlib import Path

from ultralytics.nn.tasks import DetectionModel

# 脚本所在目录（models/），用绝对路径避免运行时工作目录不同导致找不到文件
MODEL_DIR = Path(__file__).resolve().parent


def verify_model(model_path: str) -> None:
    """构造指定 yaml 的模型，跑一次随机前向，检查通道是否对齐。

    参数:
        model_path: 模型 yaml 文件的路径。
    """
    # 用 nc=1 显式指定仪表检测为单类别，构造网络结构但不加载权重
    model = DetectionModel(model_path, nc=1)
    model.eval()
    # 模拟一张 640x640 的 RGB 图做前向，验证各层通道能顺畅对齐
    with torch.no_grad():
        outputs = model(torch.zeros(1, 3, 640, 640))
    # 统计可训练参数量，用于后续论文里的 Params 对比
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[OK] {model_path} 前向成功 | 输出张量数={len(outputs)} | 参数量={total_params / 1e6:.3f}M")


if __name__ == "__main__":
    # 依次验证 E2 / E3 / E4 三个模型定义文件
    for name in [
        "yolov8n-cbam-backbone.yaml",
        "yolov8n-cbam-neck.yaml",
        "yolov8n-cbam-full.yaml",
    ]:
        verify_model(str(MODEL_DIR / name))