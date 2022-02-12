import torch
import torchvision
from ..utils import modules


class ResNet18(torch.nn.Module):
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    def __init__(self, latent_dims, pretrained):
        super().__init__()
        assert isinstance(latent_dims, int) and latent_dims > 0
        assert isinstance(pretrained, bool)

        self.pretrained = pretrained
        if pretrained:
            # Reference: https://pytorch.org/vision/stable/models.html#codecell2
            self.normalize = torchvision.transforms.Normalize(
                mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD
            )
        self.core = torchvision.models.resnet18(pretrained=pretrained)
        # replace classifier fully connected layer
        self.core.fc = modules.build_linear_relu(
            self.core.fc.in_features, latent_dims
        )

    def forward(self, image):
        """
        Args:
            pcl_nml (torch.Tensor): Batch of images with shape (N, C, H, W)
        """
        if self.pretrained:
            image = self.normalize(image)
        return self.core(image)