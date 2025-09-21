import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import traceback

class ModelShapeDebugger:
    """
    A debugging tool to trace tensor shapes through a PyTorch model's forward pass
    and identify size mismatches.
    """
    
    def __init__(self, model, input_shape=(1, 1, 28, 28), device='cpu'):
        """
        Initialize the debugger.
        
        Args:
            model: PyTorch model to debug
            input_shape: Input tensor shape (batch_size, channels, height, width)
            device: Device to run the model on
        """
        self.model = model.to(device)
        self.input_shape = input_shape
        self.device = device
        self.layer_outputs = OrderedDict()
        self.hooks = []
        
    def hook_fn(self, name):
        """Create a hook function to capture layer outputs."""
        def hook(module, input, output):
            # Store input and output shapes
            input_shapes = []
            if isinstance(input, tuple):
                for inp in input:
                    if isinstance(inp, torch.Tensor):
                        input_shapes.append(list(inp.shape))
            else:
                input_shapes.append(list(input.shape))
            
            output_shape = list(output.shape) if isinstance(output, torch.Tensor) else str(output)
            
            self.layer_outputs[name] = {
                'module': module.__class__.__name__,
                'input_shapes': input_shapes,
                'output_shape': output_shape,
                'module_repr': str(module)
            }
        return hook
    
    def register_hooks(self):
        """Register forward hooks on all layers."""
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:  # Only hook leaf modules
                hook = module.register_forward_hook(self.hook_fn(name))
                self.hooks.append(hook)
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def trace_forward_pass(self):
        """
        Trace the forward pass and capture shapes at each layer.
        Returns True if successful, False if error encountered.
        """
        self.layer_outputs.clear()
        self.register_hooks()
        
        # Create dummy input
        x = torch.randn(*self.input_shape).to(self.device)
        
        try:
            # Run forward pass
            with torch.no_grad():
                output = self.model(x)
            
            self.remove_hooks()
            return True, None
            
        except Exception as e:
            self.remove_hooks()
            return False, e
    
    def analyze_model(self):
        """
        Analyze the model and print layer-by-layer shape information.
        Identifies the first size mismatch if any.
        """
        print("="*80)
        print("MODEL SHAPE ANALYSIS")
        print("="*80)
        print(f"Input Shape: {self.input_shape}")
        print("-"*80)
        
        success, error = self.trace_forward_pass()
        
        if success:
            self._print_successful_trace()
        else:
            self._print_error_trace(error)
    
    def _print_successful_trace(self):
        """Print the successful trace of all layers."""
        print("\n✓ Forward pass completed successfully!\n")
        print("Layer-by-Layer Shape Trace:")
        print("-"*80)
        
        for i, (name, info) in enumerate(self.layer_outputs.items(), 1):
            if name:  # Skip empty names
                print(f"\n[{i}] Layer: {name}")
                print(f"    Type: {info['module']}")
                print(f"    Input Shape(s): {info['input_shapes']}")
                print(f"    Output Shape: {info['output_shape']}")
                if 'Conv2d' in info['module'] or 'Linear' in info['module']:
                    print(f"    Layer Config: {info['module_repr']}")
    
    def _print_error_trace(self, error):
        """Print the trace up to the error point."""
        print("\n✗ ERROR DETECTED!\n")
        print("Successful layers before error:")
        print("-"*80)
        
        for i, (name, info) in enumerate(self.layer_outputs.items(), 1):
            if name:
                print(f"\n[{i}] Layer: {name}")
                print(f"    Type: {info['module']}")
                print(f"    Input Shape(s): {info['input_shapes']}")
                print(f"    Output Shape: {info['output_shape']}")
                if 'Conv2d' in info['module'] or 'Linear' in info['module']:
                    print(f"    Layer Config: {info['module_repr']}")
        
        print("\n" + "="*80)
        print("ERROR DETAILS:")
        print("="*80)
        print(f"Error Type: {type(error).__name__}")
        print(f"Error Message: {str(error)}")
        
        # Try to identify the problematic layer
        self._identify_problematic_layer(error)
    
    def _identify_problematic_layer(self, error):
        """Try to identify which layer caused the error."""
        error_msg = str(error)
        
        print("\n" + "-"*80)
        print("DIAGNOSIS:")
        print("-"*80)
        
        if "size mismatch" in error_msg.lower() or "shape" in error_msg.lower():
            print("✗ Tensor size mismatch detected!")
            
            # Parse error message for size information
            import re
            numbers = re.findall(r'\d+', error_msg)
            if numbers:
                print(f"Conflicting dimensions found in error: {numbers}")
            
            # Check the last successful layer
            if self.layer_outputs:
                last_layer = list(self.layer_outputs.items())[-1]
                print(f"\nLast successful layer: {last_layer[0]}")
                print(f"Last output shape: {last_layer[1]['output_shape']}")
                print("\nThe error likely occurs in the NEXT layer after this one.")
        
        elif "expected" in error_msg.lower() and "got" in error_msg.lower():
            print("✗ Input channel mismatch detected!")
            print("The layer expects a different number of input channels than provided.")
        
        print("\nRecommendation: Check the layer definition that comes after the last")
        print("successful layer in your forward() method.")


def debug_model_shapes(model_class, input_shape=(1, 1, 28, 28), device='cpu'):
    """
    Convenience function to debug a model class.
    
    Args:
        model_class: The model class (not instance) to debug
        input_shape: Input tensor shape
        device: Device to run on
    
    Returns:
        ModelShapeDebugger instance
    """
    try:
        model = model_class()
    except Exception as e:
        print(f"Error instantiating model: {e}")
        return None
    
    debugger = ModelShapeDebugger(model, input_shape, device)
    debugger.analyze_model()
    return debugger


def manual_trace_forward(model, input_shape=(1, 1, 28, 28), device='cpu'):
    """
    Manually trace through the forward pass, printing shapes at each step.
    This mimics the actual forward() method execution.
    """
    print("="*80)
    print("MANUAL FORWARD PASS TRACE")
    print("="*80)
    
    model = model.to(device)
    x = torch.randn(*input_shape).to(device)
    
    print(f"Input: {list(x.shape)}")
    print("-"*80)
    
    try:
        # Manually execute forward pass step by step
        # This should match your forward() method
        
        print("\n[Step 1] conv0 + bn0 + relu")
        x = model.conv0(x)
        print(f"  After conv0: {list(x.shape)}")
        x = model.bn0(x)
        x = model.relu(x)
        x = model.dropout0(x)
        print(f"  After full block: {list(x.shape)}")
        
        print("\n[Step 2] conv01 + bn01 + relu")
        x = model.conv01(x)
        print(f"  After conv01: {list(x.shape)}")
        x = model.bn01(x)
        x = model.relu(x)
        print(f"  After full block: {list(x.shape)}")
        
        print("\n[Step 3] conv1 + bn1 + relu")
        print(f"  Input to conv1: {list(x.shape)}")
        print(f"  conv1 expects: in_channels={model.conv1.in_channels}, out_channels={model.conv1.out_channels}")
        
        # This is where the error will occur
        x = model.conv1(x)
        print(f"  After conv1: {list(x.shape)}")
        x = model.bn1(x)
        x = model.relu(x)
        x = model.dropout1(x)
        print(f"  After full block: {list(x.shape)}")
        
        # Continue with rest of forward pass...
        print("\n✓ Forward pass completed successfully!")
        
    except Exception as e:
        print("\n" + "="*80)
        print("✗ ERROR ENCOUNTERED!")
        print("="*80)
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Message: {str(e)}")
        print("\nThe error occurred at the step shown above.")
        print("Check the layer configuration for input/output channel mismatch.")
        
        # Additional diagnostic info
        if hasattr(model, 'conv1'):
            print(f"\nDiagnostic Info:")
            print(f"  conv1 configuration: Conv2d({model.conv1.in_channels}, {model.conv1.out_channels}, kernel_size={model.conv1.kernel_size})")
            print(f"  Actual input channels received: {x.shape[1]}")
            print(f"  ✗ Mismatch: conv1 expects {model.conv1.in_channels} channels but got {x.shape[1]} channels")


# Example usage function
def test_with_your_model():
    """
    Example of how to use the debugger with your Net class.
    """
    # Assuming your Net class is defined
    from your_model_file import Net  # Replace with actual import
    
    # Method 1: Automatic debugging
    print("METHOD 1: Automatic Shape Debugging")
    print("="*80)
    debugger = debug_model_shapes(Net, input_shape=(1, 1, 28, 28))
    
    print("\n\n")
    
    # Method 2: Manual step-by-step trace
    print("METHOD 2: Manual Step-by-Step Trace")
    print("="*80)
    model = Net()
    manual_trace_forward(model, input_shape=(1, 1, 28, 28))


if __name__ == "__main__":
    print("Model Shape Debugger")
    print("====================")
    print("\nThis tool helps identify tensor shape mismatches in PyTorch models.")
    print("\nUsage:")
    print("  from model_shape_debugger import debug_model_shapes, manual_trace_forward")
    print("  from your_model import Net")
    print("  ")
    print("  # Automatic debugging")
    print("  debugger = debug_model_shapes(Net, input_shape=(1, 1, 28, 28))")
    print("  ")
    print("  # Manual trace")
    print("  model = Net()")
    print("  manual_trace_forward(model, input_shape=(1, 1, 28, 28))")