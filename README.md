# Machine Vision Course Paper

YOLOv8n meter detection experiments with CBAM attention modules.

- `models/`: backbone, neck, and combined CBAM configurations and a model verification script.
- `run_train_auto.sh`: original training launcher from the server; its paths and GPU indices are specific to that machine.
- `runs/`: experiment metrics, plots, configurations, and model checkpoints. The baseline checkpoint is in `runs/detect/train/weights/best.pt` (with `last.pt` as the final-epoch checkpoint); attention-model checkpoints are stored in their respective experiment directories. Training and validation batch images are excluded.
- `lib/python3.10/site-packages/ultralytics/`: the Ultralytics 8.4.170 source used by the server environment.
- `yolov8n.pt`: YOLOv8n initialization weights used for the baseline training run.

The `dataset/` directory is excluded, including training and validation data. The rest of the original Conda environment is also excluded; dependencies must be installed separately before running the scripts. The source snapshot is included to document the actual implementation used in the experiments, and is not a complete Conda environment.
