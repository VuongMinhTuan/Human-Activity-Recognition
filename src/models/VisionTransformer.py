import torch
from torch import nn
from torchvision.models import vit_b_16, vit_b_32, vit_l_16, vit_l_32, vit_h_14
from torchvision.models import ViT_B_16_Weights, ViT_B_32_Weights, ViT_L_16_Weights, ViT_L_32_Weights, ViT_H_14_Weights


MODEL_VERSIONS = {
    "B_16": ( vit_b_16, ViT_B_16_Weights.DEFAULT ),
    "B_32": ( vit_b_32, ViT_B_32_Weights.DEFAULT ),
    "L_16": ( vit_l_16, ViT_L_16_Weights.DEFAULT ),
    "L_32": ( vit_l_32, ViT_L_32_Weights.DEFAULT ),
    "H_14": ( vit_h_14, ViT_H_14_Weights.DEFAULT ),
}


class ViT(nn.Module):
    def __init__(
        self,
        version: str,
        num_classes: int,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        pretrained: bool = False,
        freeze: bool = False
    ):
        
        super().__init__()

        self.version = version
        self.num_classes = num_classes
        self.dropout = dropout
        self.attention_dropout = attention_dropout
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
            dropout = dropout,
            attention_dropout = attention_dropout
        )

        
        # Freeze feature layers
        if freeze:
            for param in self.features.parameters():
                param.requires_grad = False

        
        # Replace the fully-connected layer
        self.features.heads = nn.Linear(768, num_classes)



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)

