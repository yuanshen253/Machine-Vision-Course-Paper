#!/bin/bash
# 本脚本：等待 GPU 恢复后，自动后台启动 E2/E3 训练
# 用法：nohup bash run_train_auto.sh > auto_train.log 2>&1 &
# 说明：即使 SSH 断开，脚本和训练都会继续运行

cd /workspace/new/conda_envs/yolov8
PY=./bin/python

echo "[$(date '+%F %T')] 开始等待 GPU 恢复..."

# 循环检测 GPU，直到 torch 至少能看到一张卡
while true; do
    # 关键：只在真正能拿到 GPU 时 device_count 才是正数
    cnt=$($PY -c "import torch; print(torch.cuda.device_count())" 2>/dev/null)
    if [ "${cnt:-0}" -gt 0 ] 2>/dev/null; then
        echo "[$(date '+%F %T')] GPU 已恢复，可用卡数 = $cnt"
        break
    fi
    echo "[$(date '+%F %T')] GPU 仍不可用，60 秒后重试..."
    sleep 60
done

# 启动 E3（neck 消融）→ GPU 1
echo "[$(date '+%F %T')] 启动 E3 (cbam_neck) on GPU 1"
nohup ./bin/yolo train model=models/yolov8n-cbam-neck.yaml data=dataset/data.yaml \
    epochs=100 batch=64 imgsz=640 seed=0 name=cbam_neck device=1 \
    > cbam_neck_train.log 2>&1 &

# 启动 E2（backbone 消融）→ GPU 2
echo "[$(date '+%F %T')] 启动 E2 (cbam_backbone) on GPU 2"
nohup ./bin/yolo train model=models/yolov8n-cbam-backbone.yaml data=dataset/data.yaml \
    epochs=100 batch=64 imgsz=640 seed=0 name=cbam_backbone device=2 \
    > cbam_backbone_train.log 2>&1 &

echo "[$(date '+%F %T')] 两个训练均已后台启动，脚本任务结束"