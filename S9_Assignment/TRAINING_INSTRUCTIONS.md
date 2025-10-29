# ResNet50 Training Instructions

## Overview
This training script (`train_resnet50.py`) is a command-line version of the Jupyter notebook that trains ResNet50 on ImageNet. It includes:
- Dual logging to both file and terminal
- Checkpoint saving and resuming
- Learning rate finder
- TensorBoard integration
- Graceful interrupt handling
- Background execution capability

## Quick Start

### 1. Basic Training Command
```bash
python train_resnet50.py
```

### 2. Run in Background (Recommended for AWS)
```bash
./run_training.sh
```

This will:
- Start training in the background using `nohup`
- Save the process ID to `training.pid`
- Continue running even after you close the terminal
- Log all output to `logs/training_output.log`

### 3. Monitor Training Progress
```bash
# Check training status
./monitor_training.sh

# Watch real-time logs
tail -f logs/training_output.log

# View full log
less logs/training_output.log

# Monitor with TensorBoard
tensorboard --logdir=runs
```

## Command Line Options

```bash
python train_resnet50.py [OPTIONS]

Options:
  --batch-size        Batch size for training (default: 128)
  --epochs           Number of epochs to train (default: 100)
  --lr               Learning rate (default: None, uses LR finder)
  --workers          Number of data loading workers (default: 4)
  --name             Experiment name (default: resnet_50_sgd1)
  --resume           Resume from checkpoint
  --checkpoint-dir   Checkpoint directory (default: checkpoints)
  --log-dir          Log directory (default: logs)
  --data-path        Path to training data
  --val-path         Path to validation data
  --skip-lr-finder   Skip LR finder and use provided LR
```

## Examples

### Custom Configuration
```bash
python train_resnet50.py --batch-size 256 --epochs 50 --lr 0.001 --workers 8
```

### Resume Training
```bash
python train_resnet50.py --resume
```

### Run on AWS with Custom Paths
```bash
nohup python train_resnet50.py \
    --data-path /path/to/imagenet/train \
    --val-path /path/to/imagenet/val \
    --batch-size 256 \
    --workers 16 \
    --resume \
    > logs/training.log 2>&1 &
```

## Using Screen (Alternative to nohup)

```bash
# Create a new screen session
screen -S resnet50_training

# Run training
python train_resnet50.py --resume

# Detach from screen (Ctrl+A, then D)
# Training continues in background

# Reattach to screen
screen -r resnet50_training

# List all screens
screen -ls
```

## Using tmux (Another Alternative)

```bash
# Create new tmux session
tmux new -s resnet50_training

# Run training
python train_resnet50.py --resume

# Detach from tmux (Ctrl+B, then D)
# Training continues in background

# Reattach to tmux
tmux attach -t resnet50_training

# List all tmux sessions
tmux ls
```

## Output Files

The training script creates the following files and directories:

```
.
├── checkpoints/
│   └── resnet_50_sgd1/
│       ├── checkpoint.pth      # Latest checkpoint
│       ├── best_model.pth      # Best validation accuracy model
│       └── model_*.pth         # Epoch-specific checkpoints
├── logs/
│   ├── training_output.log     # Full training log
│   ├── resnet_50_sgd1_*.log   # Timestamped training logs
│   ├── resnet_50_sgd1_history.json  # Training metrics history
│   └── resnet_50_sgd1_config.json   # Training configuration
├── plots/
│   ├── lr_finder_curve.png     # Learning rate finder results
│   └── resnet50_training_curves.png  # Training progress plots
├── runs/
│   └── resnet_50_sgd1/         # TensorBoard logs
└── training.pid                # Process ID file (when using run_training.sh)
```

## Stopping Training

### Graceful Shutdown (Recommended)
```bash
# If you know the PID
kill -SIGINT <PID>

# If using run_training.sh
kill -SIGINT $(cat training.pid)
```

The script will:
- Save the current checkpoint
- Save training history
- Generate final plots
- Close all files properly

### Force Stop (Not Recommended)
```bash
kill -9 <PID>
```

## Troubleshooting

### Out of Memory
- Reduce batch size: `--batch-size 64`
- Reduce number of workers: `--workers 2`

### Training Not Starting
- Check CUDA availability: `python -c "import torch; print(torch.cuda.is_available())"`
- Verify data paths exist
- Check available disk space for checkpoints

### Can't Find Process After Closing Terminal
- Check if still running: `ps aux | grep train_resnet50.py`
- Check the PID file: `cat training.pid`
- Check the log file: `tail logs/training_output.log`

## AWS EC2 Specific Tips

1. **Use Spot Instances with Checkpointing**
   - The script automatically saves checkpoints every epoch
   - Use `--resume` to continue from the last checkpoint

2. **EBS Volume for Data**
   - Mount ImageNet data on a separate EBS volume
   - Use `--data-path` and `--val-path` to point to the mounted location

3. **Instance Types**
   - Recommended: p3.2xlarge (1x V100 GPU)
   - For faster training: p3.8xlarge (4x V100 GPUs)

4. **Auto-shutdown After Training**
   ```bash
   nohup bash -c "python train_resnet50.py && sudo shutdown -h now" &
   ```

## Performance Tips

1. **Data Loading**
   - Increase workers for faster data loading: `--workers 8`
   - Ensure data is on fast SSD storage

2. **Mixed Precision Training**
   - The script uses FP32 by default
   - For mixed precision, modify the script to use torch.cuda.amp

3. **Multi-GPU Training**
   - Current script uses single GPU
   - For multi-GPU, wrap model with `nn.DataParallel`

## Monitoring Training Remotely

### SSH Port Forwarding for TensorBoard
```bash
# On your local machine
ssh -L 6006:localhost:6006 user@aws-instance

# On AWS instance
tensorboard --logdir=runs --port=6006

# Open browser to http://localhost:6006
```

### Download Logs and Plots
```bash
# From local machine
scp user@aws-instance:~/project/logs/*.json ./
scp user@aws-instance:~/project/plots/*.png ./