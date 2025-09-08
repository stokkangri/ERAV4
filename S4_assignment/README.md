Colab Link: https://colab.research.google.com/drive/1doENEEjbPOFUgwL1lbUkaJHTa5JGvl8K?usp=sharing

MNIST Classifier with a Convolutional Neural Network
This notebook demonstrates a simple Convolutional Neural Network (CNN) for classifying handwritten digits from the MNIST dataset using PyTorch.

Network Architecture (Class Net)
The neural network is defined in the Net class. It consists of the following layers:

Convolutional Layer 1 (conv1): Takes a single input channel (grayscale MNIST images) and applies 16 filters with a kernel size of 3x3.
Batch Normalization 1 (bn1): Normalizes the output of conv1.
Dropout 1 (dropout1): Applies dropout with a probability of 0.01 to the output of the first batch normalization layer.
Convolutional Layer 2 (conv2): Takes 16 input channels (from conv1) and applies 32 filters with a kernel size of 3x3.
Batch Normalization 2 (bn2): Normalizes the output of conv2.
Dropout 2 (dropout2): Applies dropout with a probability of 0.01 to the output of the second batch normalization layer.
Max Pooling 1: Applies a 2x2 max pooling operation after the second convolutional block to reduce spatial dimensions.
Convolutional Layer 3 (conv3): Takes 32 input channels (from the previous block) and applies 64 filters with a kernel size of 3x3.
Batch Normalization 3 (bn3): Normalizes the output of conv3.
Dropout 3 (dropout3): Applies dropout with a probability of 0.01 to the output of the third batch normalization layer.
Convolutional Layer 4 (conv4): Takes 64 input channels (from the previous block) and applies 128 filters with a kernel size of 3x3.
Batch Normalization 4 (bn4): Normalizes the output of conv4.
Dropout 4 (dropout4): Applies dropout with a probability of 0.01 to the output of the fourth batch normalization layer.
Max Pooling 2: Applies a 2x2 max pooling operation after the fourth convolutional block to further reduce spatial dimensions.
1x1 Convolution (conv1x1): Applies a 1x1 convolution with 10 output channels. This is often used to reduce the number of channels before global average pooling.
Global Average Pooling (global_avg_pool): Applies global average pooling to reduce the spatial dimensions to 1x1, effectively averaging the feature maps.
Flatten: The output is flattened into a 1D tensor with 10 elements (corresponding to the 10 classes).
Log Softmax: Applies the log-softmax function to the output, providing log-probabilities for each class.
The forward method defines the flow of data through these layers, applying ReLU activation functions after each convolutional and batch normalization step. Dropout is applied after the batch normalization layers. Max pooling is applied after the second and fourth convolutional blocks. A 1x1 convolution and global average pooling are used before the final output.

Training Logs
The notebook includes a training loop that trains the model for a specified number of epochs. The train function iterates over the training data, calculates the loss, and updates the model's weights using an optimizer. The test function evaluates the model's performance on the test data after each epoch.

Here are some of the logs observed during the training:

CUDA Availability: The notebook checks for CUDA availability, indicating whether the training will run on a GPU (if available) or CPU.
Model Summary: The torchsummary library is used to print a summary of the model, showing the output shape and number of parameters for each layer. This helps in understanding the model's structure and parameter count.
Training Progress (tqdm): The tqdm library is used to display a progress bar during training, showing the current batch loss, batch ID, and training accuracy.
Test Set Evaluation: After each epoch, the model is evaluated on the test set, and the average loss and accuracy are printed.
The notebook also includes code to plot the training and testing loss and accuracy over epochs, providing a visual representation of the model's learning progress.



1. torchsummary Output :

This output provides a layer-by-layer breakdown of your Net model. It's incredibly useful for understanding the architecture and parameter count.

Layer (type): Shows the type of layer (e.g., Conv2d, BatchNorm2d, Dropout, AdaptiveAvgPool2d, Linear).
Output Shape: This is the shape of the tensor after passing through that specific layer. The -1 at the beginning represents the batch size, which can vary. The other numbers represent the dimensions of the output (e.g., channels, height, width). For example, [-1, 16, 26, 26] means the output is a batch of tensors with 16 channels, 26 pixels in height, and 26 pixels in width.
Param #: This is the number of trainable parameters in that specific layer.
For Conv2d layers, the parameters are the weights and biases. The number is calculated as (input_channels * kernel_height * kernel_width + bias) * output_channels. (Bias is usually 1 per output channel).
For BatchNorm2d layers, the parameters are the learned mean and variance (gamma and beta).
Dropout and AdaptiveAvgPool2d layers have no trainable parameters.
For Linear layers, the parameters are the weights and biases, calculated as (input_features + bias) * output_features.
Total params: The sum of parameters across all layers.
Trainable params: The number of parameters that the optimizer will update during training. In your case, all parameters are trainable.
Non-trainable params: Parameters that are fixed during training (e.g., if you freeze some layers).
Input size (MB): The estimated memory size of the input data.
Forward/backward pass size (MB): The estimated memory required for a single forward and backward pass through the network.
Params size (MB): The estimated memory size of the model's parameters.
Estimated Total Size (MB): The estimated total memory usage of the model during training.
This summary helps you see how the data's shape changes through the network and which layers contribute the most to the model's size in terms of parameters.

2. Training and Test Output (from cell oYWaxfPOG1__):

The output from the training loop provides insights into the model's learning progress during each epoch.

Epoch X: Indicates the start of a new training epoch.
Train: Loss=X.XXXX Batch_id=Y Accuracy=Z.ZZ: This is the output from the tqdm progress bar within the train function.
Loss=X.XXXX: The loss calculated for the current batch. This tells you how well the model's predictions match the actual labels for that batch. A lower loss indicates better performance.
Batch_id=Y: The index of the current batch being processed in the epoch.
Accuracy=Z.ZZ: The cumulative accuracy on the training data processed so far in the current epoch. This is the percentage of correctly classified samples.
Test set: Average loss: X.XXXX, Accuracy: A/B (C.CC%): This is the output from the test function, which evaluates the model on the test dataset after each epoch.
Average loss: X.XXXX: The average loss over the entire test dataset.
Accuracy: A/B: The number of correctly classified samples (A) out of the total number of samples in the test dataset (B).
(C.CC%): The percentage of correctly classified samples on the test dataset.
These logs are crucial for monitoring training. You want to see the training loss decrease and training accuracy increase over time. The test set evaluation tells you how well your model generalizes to unseen data. It's important to track both to detect issues like overfitting (where the model performs very well on training data but poorly on test data).

3. Other Logs/Run Logs:

Beyond the specific outputs you asked about, a complete run log would typically include:

Setup information: Details about the environment (e.g., Python version, library versions), hardware (CPU/GPU), and any specific configurations used.
Data loading details: Information about the datasets loaded, their sizes, and any transformations applied.
Model initialization: Confirmation that the model was created and moved to the correct device.
Optimizer and scheduler details: Information about the optimizer and learning rate scheduler used.
Error messages and warnings: Any errors or warnings that occurred during the run, which are critical for debugging.
Execution times: How long different parts of the code took to run.
In your notebook, the outputs from cells like checking CUDA availability (CUDA Available? True) and the progress bar from tqdm are part of the run log, providing information about the execution environment and the training process.
