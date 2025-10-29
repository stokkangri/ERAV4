#!/bin/bash

# Training Monitor Script

echo "ResNet50 Training Monitor"
echo "========================"

# Check if training is running
if [ -f training.pid ]; then
    PID=$(cat training.pid)
    if ps -p $PID > /dev/null; then
        echo "Training is RUNNING (PID: $PID)"
    else
        echo "Training is NOT RUNNING (PID file exists but process not found)"
    fi
else
    echo "No training.pid file found"
fi

echo ""

# Show latest log entries
if [ -f logs/training_output.log ]; then
    echo "Latest training output:"
    echo "----------------------"
    tail -n 20 logs/training_output.log
else
    echo "No training log found"
fi

echo ""

# Check for latest checkpoint
if [ -d checkpoints/resnet_50_sgd1 ]; then
    echo "Checkpoints found:"
    echo "-----------------"
    ls -lht checkpoints/resnet_50_sgd1/*.pth | head -5
fi

echo ""
echo "To see real-time logs: tail -f logs/training_output.log"
echo "To stop training: kill \$(cat training.pid)"