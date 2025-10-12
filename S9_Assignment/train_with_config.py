"""
Train ResNet-50 on ImageNet using configuration file
"""

import argparse
import yaml
import sys
from pathlib import Path

# Import the main trainer
from train_imagenet import ImageNetTrainer, get_default_config


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description='Train ResNet-50 using config file')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration YAML file')
    parser.add_argument('--override', nargs='*', default=[],
                       help='Override config parameters (e.g., --override epochs=10 lr=0.01)')
    
    args = parser.parse_args()
    
    # Load base configuration
    config = get_default_config()
    
    # Load configuration from file
    file_config = load_config(args.config)
    config.update(file_config)
    
    # Apply command line overrides
    for override in args.override:
        key, value = override.split('=')
        # Try to parse value as number
        try:
            value = float(value)
            if value.is_integer():
                value = int(value)
        except ValueError:
            # Keep as string
            if value.lower() == 'true':
                value = True
            elif value.lower() == 'false':
                value = False
            elif value.lower() == 'null' or value.lower() == 'none':
                value = None
        
        config[key] = value
    
    # Print configuration
    print("\nConfiguration:")
    print("-" * 50)
    for key, value in sorted(config.items()):
        print(f"{key:25s}: {value}")
    print("-" * 50)
    
    # Create trainer
    trainer = ImageNetTrainer(config)
    
    # Find optimal LR if requested
    if config.get('find_lr', False):
        suggested_lr = trainer.find_lr(num_iter=config.get('lr_finder_iterations', 200))
        config['learning_rate'] = suggested_lr
        config['max_lr'] = suggested_lr
        trainer.setup_training()  # Reinitialize with new LR
    
    # Train model
    history = trainer.train()
    
    print("\nTraining completed successfully!")
    print(f"Configuration file: {args.config}")


if __name__ == "__main__":
    # Check if PyYAML is installed
    try:
        import yaml
    except ImportError:
        print("PyYAML is required. Install it with: pip install pyyaml")
        sys.exit(1)
    
    main()