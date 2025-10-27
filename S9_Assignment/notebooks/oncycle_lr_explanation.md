# OneCycleLR Configuration Explanation

## The Issue with Current Setup

In your notebook, after LR finder:
```python
config['learning_rate'] = suggested_lr * 0.5
config['max_lr'] = suggested_lr * 0.5
```

This is **redundant and confusing**! Here's why:

## How OneCycleLR Works

1. **OneCycleLR ignores the optimizer's initial LR**
2. It only uses `max_lr` parameter
3. The learning rate schedule:
   - Starts at `max_lr / div_factor` (default div_factor=25)
   - Increases to `max_lr` over first 30% of training
   - Decreases back down to `max_lr / final_div_factor`

## Current Problem

Your code sets both to the same value, but:
- `config['learning_rate']` is used for the optimizer (line 439)
- `config['max_lr']` is used for OneCycleLR (line 459)
- OneCycleLR will override whatever LR the optimizer has!

## Correct Approaches

### Option 1: Use OneCycleLR (Recommended)
```python
# After LR finder
config['max_lr'] = suggested_lr * 0.5  # Only set max_lr
# Don't worry about config['learning_rate'] - it will be overridden

# OneCycleLR will handle the schedule
scheduler = OneCycleLR(
    optimizer,
    max_lr=config['max_lr'],  # This is what matters
    epochs=config['epochs'],
    steps_per_epoch=len(train_loader),
    ...
)
```

### Option 2: Use Fixed LR (For Debugging)
```python
# After LR finder
config['learning_rate'] = suggested_lr * 0.1  # Use lower value
config['scheduler'] = None  # Disable OneCycleLR!

# Now the optimizer's LR will be used
optimizer = optim.SGD(
    model.parameters(),
    lr=config['learning_rate'],  # This will be used
    ...
)
```

### Option 3: Use Different Scheduler
```python
config['learning_rate'] = suggested_lr * 0.5
config['scheduler'] = 'cosine'  # Not 'onecycle'

# CosineAnnealingLR uses the optimizer's LR
scheduler = CosineAnnealingLR(
    optimizer,
    T_max=config['epochs'],
    eta_min=1e-6
)
```

## Why Your Training Might Be Failing

With OneCycleLR and your current setup:
- Initial LR = max_lr / 25 = (suggested_lr * 0.5) / 25
- This might be too low to start!
- Peak LR = suggested_lr * 0.5 (might still be too conservative)

## Recommendations

1. **For OneCycleLR**: Use full suggested LR or even higher
   ```python
   config['max_lr'] = suggested_lr  # Don't reduce by 0.5
   ```

2. **For debugging**: Disable OneCycleLR entirely
   ```python
   config['scheduler'] = None
   config['learning_rate'] = 0.01  # Fixed, reasonable value
   ```

3. **Check what LR is actually being used**:
   ```python
   # In your training loop, print actual LR
   print(f"Current LR: {optimizer.param_groups[0]['lr']:.6f}")
   ```

## Quick Fix for Your Notebook

Replace lines after LR finder with:
```python
if config['scheduler'] == 'onecycle':
    # For OneCycleLR, only max_lr matters
    config['max_lr'] = suggested_lr  # Use full suggested LR
    print(f"OneCycleLR will use max_lr={config['max_lr']:.2e}")
else:
    # For other schedulers or no scheduler
    config['learning_rate'] = suggested_lr * 0.5
    print(f"Using learning_rate={config['learning_rate']:.2e}")