#!/bin/bash

# ResNet50 Training Script Runner
# This script runs the training in the background and keeps it running even after terminal closure

echo "Starting ResNet50 training..."

# Create necessary directories
mkdir -p logs checkpoints plots runs

# Run training with nohup to keep it running after terminal closure
# The output will be saved to logs/training_output.log
nohup python train_resnet50.py \
    --batch-size 128 \
    --epochs 100 \
    --workers 4 \
    --name resnet_50_sgd1 \
    --resume \
    > logs/training_output.log 2>&1 &

# Get the process ID
PID=$!
echo "Training started with PID: $PID"
echo "PID saved to: training.pid"
echo $PID > training.pid

echo ""
echo "Training is running in the background."
echo "To monitor progress:"
echo "  - Real-time log: tail -f logs/training_output.log"
echo "  - Full log: less logs/training_output.log"
echo "  - TensorBoard: tensorboard --logdir=runs"
echo ""
echo "To stop training:"
echo "  - kill $PID"
echo "  - or: kill \$(cat training.pid)"
echo ""
echo "The training will continue even if you close this terminal."