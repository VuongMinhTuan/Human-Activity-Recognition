import torch
from torch import nn
from torchvision.models import vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn
from torchvision.models import VGG11_BN_Weights, VGG13_BN_Weights, VGG16_BN_Weights, VGG19_BN_Weights


MODEL_VERSIONS = {
    "11": ( vgg11_bn, VGG11_BN_Weights.DEFAULT ),
    "13": ( vgg13_bn, VGG13_BN_Weights.DEFAULT ),
    "16": ( vgg16_bn, VGG16_BN_Weights.DEFAULT ),
    "19": ( vgg19_bn, VGG19_BN_Weights.DEFAULT )
}


class VGG(nn.Module):
    def __init__(
        self,
        version: str,
        num_classes: int,
        dropout: float = 0.0,
        hidden_features: int = 4096,
        pretrained: bool = False,
        freeze: bool = False
    ):
        
        super().__init__()

        self.version = version
        self.num_classes = num_classes
        self.dropout = dropout
        self.hidden_features = hidden_features
        self.pretrained = pretrained
        self.freeze = freeze

        # Check version of model
        if self.version not in MODEL_VERSIONS:
            raise ValueError(f"Version {self.version} is not available!!!\nOnly these versions are supported: {list(MODEL_VERSIONS.keys())}")
        

        # Get model and its weight
        model, weight = MODEL_VERSIONS.get(self.version)


        # Get feature layers
        self.features: nn.Module = model(
            weights = weight if pretrained else None,
            dropout = dropout
        )

        
        # Freeze feature layers
        if freeze:
            for param in self.features.parameters():
                param.requires_grad = False


        # Replace the classifier layer
        self.features.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, hidden_features),
            nn.ReLU(True),
            nn.Dropout(p = dropout),
            nn.Linear(hidden_features, hidden_features),
            nn.ReLU(True),
            nn.Dropout(p = dropout),
            nn.Linear(hidden_features, num_classes),
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)