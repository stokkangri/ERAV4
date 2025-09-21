import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import numpy as np

def summary_with_rf(model, input_size, batch_size=-1, device="cuda", dtypes=None, debug=False):
    """
    Enhanced torchsummary that calculates and displays receptive fields
    
    Args:
        model: PyTorch model
        input_size: Input tensor size (C, H, W) or tuple of sizes for multiple inputs
        batch_size: Batch size for summary (default: -1)
        device: Device to run model on (default: "cuda")
        dtypes: Data types for inputs (default: None, uses torch.FloatTensor)
        debug: If True, prints detailed RF calculations (Nin, Nout, k, s, p, Jin, Jout, Rin, Rout)
        
    Returns:
        None (prints summary table)
    """
    
    def register_hook(module):
        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)
            
            m_key = "%s-%i" % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()
            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["input_shape"][0] = batch_size
            
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [
                    [-1] + list(o.size())[1:] for o in output
                ]
            else:
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = batch_size
                
            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params
            
            # Calculate receptive field
            rf_info = calculate_rf_for_layer(module, m_key)
            summary[m_key]["receptive_field"] = rf_info

        if (
            not isinstance(module, nn.Sequential)
            and not isinstance(module, nn.ModuleList)
            and not (module == model)
        ):
            hooks.append(module.register_forward_hook(hook))

    # Initialize RF tracking
    current_rf = 1
    current_jump = 1
    current_size = input_size[1] if isinstance(input_size, (tuple, list)) and len(input_size) > 1 else input_size[0] if isinstance(input_size, (tuple, list)) else 28
    original_input_size = current_size  # Keep track of original input size

    def calculate_rf_for_layer(module, layer_name):
        nonlocal current_rf, current_jump, current_size
        
        if isinstance(module, nn.Conv2d):
            kernel_size = module.kernel_size[0] if isinstance(module.kernel_size, tuple) else module.kernel_size
            stride = module.stride[0] if isinstance(module.stride, tuple) else module.stride
            padding = module.padding[0] if isinstance(module.padding, tuple) else module.padding
            
            # Store old values for debug
            old_size = current_size
            old_rf = current_rf
            old_jump = current_jump
            
            # Calculate output size: Nout = 1 + (Nin + 2*p - k)/s
            new_size = 1 + (current_size + 2 * padding - kernel_size) // stride
            
            # Calculate new RF: RFout = RFin + (k-1) * Jin
            new_rf = current_rf + (kernel_size - 1) * current_jump
            
            # Calculate new Jump: Jout = s * Jin
            new_jump = stride * current_jump
            
            if debug:
                print(f"\n[DEBUG] {layer_name} (Conv2d):")
                print(f"  Nin={old_size}, Nout={new_size}, k={kernel_size}, s={stride}, p={padding}")
                print(f"  Jin={old_jump}, Jout={new_jump}")
                print(f"  Rin={old_rf}, Rout={new_rf}")
                print(f"  Formula: Nout = 1 + ({old_size} + 2*{padding} - {kernel_size})/{stride} = {new_size}")
                print(f"  Formula: Rout = {old_rf} + ({kernel_size}-1)*{old_jump} = {new_rf}")
                print(f"  Formula: Jout = {stride}*{old_jump} = {new_jump}")
            
            current_size = new_size
            current_rf = new_rf
            current_jump = new_jump
            
            return new_rf
            
        elif isinstance(module, nn.MaxPool2d):
            kernel_size = module.kernel_size
            stride = module.stride if module.stride is not None else kernel_size
            padding = module.padding
            
            # Store old values for debug
            old_size = current_size
            old_rf = current_rf
            old_jump = current_jump
            
            # Calculate output size: Nout = 1 + (Nin + 2*p - k)/s
            new_size = 1 + (current_size + 2 * padding - kernel_size) // stride
            
            # Calculate new RF: RFout = RFin + (k-1) * Jin
            new_rf = current_rf + (kernel_size - 1) * current_jump
            
            # Calculate new Jump: Jout = s * Jin
            new_jump = stride * current_jump
            
            if debug:
                print(f"\n[DEBUG] {layer_name} (MaxPool2d):")
                print(f"  Nin={old_size}, Nout={new_size}, k={kernel_size}, s={stride}, p={padding}")
                print(f"  Jin={old_jump}, Jout={new_jump}")
                print(f"  Rin={old_rf}, Rout={new_rf}")
                print(f"  Formula: Nout = 1 + ({old_size} + 2*{padding} - {kernel_size})/{stride} = {new_size}")
                print(f"  Formula: Rout = {old_rf} + ({kernel_size}-1)*{old_jump} = {new_rf}")
                print(f"  Formula: Jout = {stride}*{old_jump} = {new_jump}")
            
            current_size = new_size
            current_rf = new_rf
            current_jump = new_jump
            
            return new_rf
            
        elif isinstance(module, nn.AdaptiveAvgPool2d):
            # FIXED: Properly handle AdaptiveAvgPool2d
            output_size = module.output_size
            
            # Store old values for debug
            old_size = current_size
            old_rf = current_rf
            old_jump = current_jump
            
            # Check if it's global pooling (output size is 1x1)
            if (isinstance(output_size, int) and output_size == 1) or \
               (isinstance(output_size, (tuple, list)) and output_size[0] == 1 and output_size[1] == 1):
                # Global Average Pooling - RF becomes the entire feature map
                # The RF now covers the entire spatial dimension
                global_rf = f"GLOBAL({current_size}x{current_size})"
                
                if debug:
                    print(f"\n[DEBUG] {layer_name} (AdaptiveAvgPool2d - Global):")
                    print(f"  Nin={old_size}, Nout=1, k={old_size} (adaptive), s={old_size}, p=0")
                    print(f"  Jin={old_jump}, Jout=1 (global pooling)")
                    print(f"  Rin={old_rf}, Rout=GLOBAL({old_size}x{old_size})")
                    print(f"  Note: Global Average Pooling - each output sees entire {old_size}x{old_size} feature map")
                    print(f"  Effective RF in original input space: {original_input_size}x{original_input_size}")
                
                current_size = 1
                # For subsequent layers, we'll track that RF is global
                current_rf = original_input_size  # Use original input size as effective RF
                current_jump = 1  # Jump doesn't matter after global pooling
                return global_rf
            else:
                # For other adaptive pool sizes, calculate approximate RF
                # This is a simplification - actual RF depends on input/output ratio
                if isinstance(output_size, int):
                    out_h = out_w = output_size
                else:
                    out_h, out_w = output_size
                
                # Approximate kernel size based on input/output ratio
                approx_kernel_h = current_size // out_h
                approx_kernel_w = current_size // out_w
                approx_kernel = max(approx_kernel_h, approx_kernel_w)
                
                # Update RF based on approximate kernel
                new_rf = current_rf + (approx_kernel - 1) * current_jump
                
                if debug:
                    print(f"\n[DEBUG] {layer_name} (AdaptiveAvgPool2d):")
                    print(f"  Nin={old_size}, Nout={out_h}, k≈{approx_kernel} (adaptive), s≈{approx_kernel}, p=0")
                    print(f"  Jin={old_jump}, Jout={old_jump}")
                    print(f"  Rin={old_rf}, Rout={new_rf}")
                    print(f"  Note: Non-global adaptive pooling to {out_h}x{out_w}")
                
                current_size = out_h  # Assuming square output
                current_rf = new_rf
                
                return new_rf
                
        elif isinstance(module, (nn.BatchNorm2d, nn.Dropout, nn.ReLU)):
            # These layers don't change receptive field
            if debug:
                print(f"\n[DEBUG] {layer_name} ({module.__class__.__name__}):")
                print(f"  No change in RF: Nin={current_size}, Nout={current_size}")
                print(f"  Jin={current_jump}, Jout={current_jump}")
                print(f"  Rin={current_rf}, Rout={current_rf}")
            return current_rf
            
        elif isinstance(module, nn.Linear):
            # Linear layers typically come after flattening, RF stays the same
            if debug:
                print(f"\n[DEBUG] {layer_name} (Linear):")
                print(f"  After flattening - RF stays at {current_rf}")
            return current_rf
            
        else:
            # For unknown layers, assume RF doesn't change
            if debug:
                print(f"\n[DEBUG] {layer_name} ({module.__class__.__name__}):")
                print(f"  Unknown layer type - RF unchanged at {current_rf}")
            return current_rf

    # Check device
    device = device.lower()
    assert device in [
        "cuda",
        "cpu",
    ], "Input device is not valid, please specify 'cuda' or 'cpu'"

    if device == "cuda" and torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    # Multiple inputs to the network
    if isinstance(input_size, tuple):
        input_size = [input_size]
    if isinstance(input_size[0], (list, tuple)):
        input_size = input_size
    else:
        input_size = [input_size]

    # Batch_size of 2 for batchnorm
    x = [torch.rand(2, *in_size).type(dtype) for in_size in input_size]

    # Create properties
    summary = OrderedDict()
    hooks = []

    # Register hook
    model.apply(register_hook)

    # Make a forward pass
    model(*x)

    # Remove these hooks
    for h in hooks:
        h.remove()

    # Print summary
    if debug:
        print("\n" + "=" * 80)
        print("DEBUG MODE: Detailed RF Calculations Complete")
        print("=" * 80 + "\n")
    
    print("=" * 80)
    line_new = "{:>25}  {:>25} {:>15} {:>15}".format(
        "Layer (type)", "Output Shape", "Param #", "RF"
    )
    print(line_new)
    print("=" * 80)
    
    total_params = 0
    total_output = 0
    trainable_params = 0
    
    for layer in summary:
        # Input_shape, output_shape, trainable, nb_params
        rf_display = str(summary[layer]["receptive_field"])
        line_new = "{:>25}  {:>25} {:>15} {:>15}".format(
            layer,
            str(summary[layer]["output_shape"]),
            "{0:,}".format(summary[layer]["nb_params"]),
            rf_display,
        )
        total_params += summary[layer]["nb_params"]

        total_output_size = abs(np.prod(summary[layer]["output_shape"]))
        total_output += total_output_size

        if "trainable" in summary[layer]:
            if summary[layer]["trainable"] == True:
                trainable_params += summary[layer]["nb_params"]
        print(line_new)

    # Assume 4 bytes/number (float on cuda).
    total_input_size = abs(np.prod(sum(input_size, ())))
    total_output_size = abs(2.0 * total_output * 4.0 / (1024 ** 2.0))  # x2 for gradients
    total_params_size = abs(total_params * 4.0 / (1024 ** 2.0))
    total_input_size = abs(total_input_size * 4.0 / (1024 ** 2.0))

    print("=" * 80)
    print("Total params: {0:,}".format(total_params))
    print("Trainable params: {0:,}".format(trainable_params))
    print("Non-trainable params: {0:,}".format(total_params - trainable_params))
    print("-" * 80)
    print("Input size (MB): %0.2f" % total_input_size)
    print("Forward/backward pass size (MB): %0.2f" % total_output_size)
    print("Params size (MB): %0.2f" % total_params_size)
    print("Estimated Total Size (MB): %0.2f" % (total_params_size + total_output_size + total_input_size))
    print("=" * 80)
    # Display final RF
    final_rf_display = f"GLOBAL({original_input_size}x{original_input_size})" if isinstance(current_rf, int) and current_rf >= original_input_size else str(current_rf)
    print(f"Final Receptive Field: {final_rf_display}")
    print("=" * 80)


# Helper function to calculate RF for functional operations
def calculate_functional_rf(operation_name, current_rf, current_jump, current_size, **kwargs):
    """
    Calculate RF for functional operations like F.max_pool2d, F.conv2d, etc.
    
    Args:
        operation_name: Name of the operation ('max_pool2d', 'conv2d', etc.)
        current_rf: Current receptive field
        current_jump: Current jump
        current_size: Current spatial size
        **kwargs: Operation-specific parameters
        
    Returns:
        tuple: (new_rf, new_jump, new_size)
    """
    
    if operation_name == 'max_pool2d':
        kernel_size = kwargs.get('kernel_size', 2)
        stride = kwargs.get('stride', kernel_size)
        padding = kwargs.get('padding', 0)
        
        # Calculate output size: Nout = 1 + (Nin + 2*p - k)/s
        new_size = 1 + (current_size + 2 * padding - kernel_size) // stride
        
        # Calculate new RF: RFout = RFin + (k-1) * Jin
        new_rf = current_rf + (kernel_size - 1) * current_jump
        
        # Calculate new Jump: Jout = s * Jin
        new_jump = stride * current_jump
        
        return new_rf, new_jump, new_size
        
    elif operation_name == 'avg_pool2d':
        # Same calculation as max_pool2d
        return calculate_functional_rf('max_pool2d', current_rf, current_jump, current_size, **kwargs)
        
    elif operation_name == 'adaptive_avg_pool2d':
        # FIXED: Handle adaptive pooling
        output_size = kwargs.get('output_size', 1)
        
        if output_size == 1 or output_size == (1, 1):
            # Global pooling - RF becomes the entire feature map
            return 'GLOBAL', 1, 1
        else:
            # For other sizes, approximate
            if isinstance(output_size, int):
                out_size = output_size
            else:
                out_size = output_size[0]  # Assuming square
            
            approx_kernel = current_size // out_size
            new_rf = current_rf + (approx_kernel - 1) * current_jump
            return new_rf, current_jump, out_size
        
    elif operation_name in ['relu', 'dropout']:
        # These don't change RF
        return current_rf, current_jump, current_size
        
    else:
        # Unknown operation, assume no change
        return current_rf, current_jump, current_size


# Example usage function
def demo_rf_calculator():
    """
    Demonstration function showing how to use the RF calculator
    """
    
    # Example CNN similar to the one in the notebook
    class ExampleNet(nn.Module):
        def __init__(self):
            super(ExampleNet, self).__init__()
            self.conv1 = nn.Conv2d(1, 16, kernel_size=3)  # RF: 3
            self.conv2 = nn.Conv2d(16, 32, kernel_size=3)  # RF: 5
            self.conv3 = nn.Conv2d(32, 32, kernel_size=3)  # RF: 10
            self.conv1x1 = nn.Conv2d(32, 16, kernel_size=1)  # RF: 12
            self.fc = nn.Linear(16*5*5, 10)

        def forward(self, x):
            x = F.relu(self.conv1(x))  # 26x26
            x = F.relu(F.max_pool2d(self.conv2(x), 2))  # 12x12
            x = F.relu(F.max_pool2d(self.conv3(x), 2))  # 5x5
            x = self.conv1x1(x)  # 5x5
            x = x.view(-1, 16*5*5)
            x = self.fc(x)
            return F.log_softmax(x, dim=1)
    
    model = ExampleNet()
    print("Example RF Calculation:")
    summary_with_rf(model, input_size=(1, 28, 28), device="cpu")
    
    print("\n\n" + "=" * 80)
    print("Example with DEBUG mode enabled:")
    print("=" * 80)
    summary_with_rf(model, input_size=(1, 28, 28), device="cpu", debug=True)


if __name__ == "__main__":
    demo_rf_calculator()