from torch import nn
from torchvision.models import vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn, VGG11_BN_Weights, VGG13_BN_Weights, VGG16_BN_Weights, VGG19_BN_Weights



# Available versions of the ViT model with associated architectures and weights
VERSIONS = {
    "11": (vgg11_bn, VGG11_BN_Weights.DEFAULT),
    "13": (vgg13_bn, VGG13_BN_Weights.DEFAULT),
    "16": (vgg16_bn, VGG16_BN_Weights.DEFAULT),
    "19": (vgg19_bn, VGG19_BN_Weights.DEFAULT)
}



class VGG(nn.Module):
    def __init__(
        self,
        version: str,
        num_classes: int,
        hidden_features: int = 4096,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        pretrained: bool = False,
        freeze: bool = False
    ):
        
        super().__init__()

        self.version = version
        self.num_classes = num_classes
        self.hidden_features = hidden_features
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.pretrained = pretrained
        self.freeze = freeze


        # Check if the version of model is available
        if self.version not in VERSIONS:
            raise ValueError(f"{self.version} is not supported! Version available: {list(VERSIONS.keys())}")
        
        # Get the features layer
        model, weight = VERSIONS.get(self.version)

        self.features: nn.Module = model (
            weights = weight if self.pretrained else None,
            dropout = self.dropout,
            attention_dropout = attention_dropout
        )

        # Freeze the features layer
        if freeze:
            for param in self.features.parameters():
                param.requires_grad = False


        # Replace the fully-connected layer
        self.features.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, self.hidden_features),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(self.hidden_features, self.hidden_features),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(self.hidden_features, num_classes),
        )


    def forward(self, x):
        return self.features(x)