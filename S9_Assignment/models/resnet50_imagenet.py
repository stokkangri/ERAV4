"""
ResNet-50 model implementation for ImageNet
"""

import torch
import torch.nn as nn
from typing import Type, List, Optional, Callable


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        groups=groups,
        bias=False,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    """
    Bottleneck block for ResNet-50/101/152
    """
    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        base_width: int = 64,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0))
        
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """
    ResNet model
    """
    def __init__(
        self,
        block: Type[Bottleneck],
        layers: List[int],
        num_classes: int = 1000,  # ImageNet has 1000 classes
        width_per_group: int = 64,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        replace_maxpool_with_conv: bool = False,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.base_width = width_per_group
        self.replace_maxpool_with_conv = replace_maxpool_with_conv
        
        # Standard ResNet for ImageNet (224x224 images)
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        
        # Option to replace maxpool with strided conv
        if replace_maxpool_with_conv:
            # Use 3x3 conv with stride 2 instead of maxpool
            self.maxpool = nn.Sequential(
                nn.Conv2d(self.inplanes, self.inplanes, kernel_size=3, stride=2, padding=1, bias=False),
                norm_layer(self.inplanes),
                nn.ReLU(inplace=True)
            )
        else:
            # Standard maxpool
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(
        self,
        block: Type[Bottleneck],
        planes: int,
        blocks: int,
        stride: int = 1,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.base_width, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    base_width=self.base_width,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def resnet50(num_classes: int = 1000, pretrained: bool = False, replace_maxpool_with_conv: bool = True, **kwargs) -> ResNet:
    """
    ResNet-50 model
    Args:
        num_classes: Number of output classes (default: 1000 for ImageNet)
        pretrained: If True, loads pretrained weights
        replace_maxpool_with_conv: If True, replaces MaxPool with strided Conv (default: True)
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes,
                   replace_maxpool_with_conv=replace_maxpool_with_conv, **kwargs)
    
    if pretrained:
        try:
            import torchvision.models as models
            # Load pretrained weights
            pretrained_dict = models.resnet50(pretrained=True).state_dict()
            model_dict = model.state_dict()
            
            # Filter out fc layer if num_classes is different
            if num_classes != 1000:
                pretrained_dict = {k: v for k, v in pretrained_dict.items()
                                 if k not in ['fc.weight', 'fc.bias']}
            
            # If using conv instead of maxpool, skip maxpool weights
            if replace_maxpool_with_conv:
                pretrained_dict = {k: v for k, v in pretrained_dict.items()
                                 if not k.startswith('maxpool')}
            
            # Update model dict
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict, strict=False)
            print(f"Loaded pretrained weights (except fc layer) for ResNet-50")
        except Exception as e:
            print(f"Could not load pretrained weights: {e}")
    
    return model


def resnet101(num_classes: int = 1000, **kwargs) -> ResNet:
    """
    ResNet-101 model
    Args:
        num_classes: Number of output classes (default: 1000 for ImageNet)
    """
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, **kwargs)


def resnet152(num_classes: int = 1000, **kwargs) -> ResNet:
    """
    ResNet-152 model
    Args:
        num_classes: Number of output classes (default: 1000 for ImageNet)
    """
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes=num_classes, **kwargs)


if __name__ == "__main__":
    # Test the model
    print("Testing ResNet-50 for ImageNet...")
    
    # Test with ImageNet dimensions
    model = resnet50(num_classes=1000)
    x = torch.randn(2, 3, 224, 224)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test with pretrained weights
    print("\nTesting with pretrained weights...")
    model_pretrained = resnet50(num_classes=1000, pretrained=True)
    output_pretrained = model_pretrained(x)
    print(f"Pretrained model output shape: {output_pretrained.shape}")
    
    # Test with conv instead of maxpool
    print("\nTesting with Conv instead of MaxPool...")
    model_conv = resnet50(num_classes=1000, replace_maxpool_with_conv=True)
    output_conv = model_conv(x)
    print(f"Model with Conv output shape: {output_conv.shape}")
    print(f"Conv model parameters: {sum(p.numel() for p in model_conv.parameters()):,}")