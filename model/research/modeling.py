import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

from .device import get_device

try:
    import timm
except ImportError:
    timm = None


class ResearchModel(nn.Module):
    def __init__(
        self,
        num_binary_classes=2,
        num_method_classes=7,
        model_name="EfficientNet-b0",
        lstm_layers=1,
        hidden_dim=2048,
        bidirectional=False,
        dropout=0.5,
    ):
        super().__init__()
        self.model_name = model_name

        if self.model_name == "resnext50_32x4d":
            model = models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.DEFAULT)
            self.model = nn.Sequential(*list(model.children())[:-2])
            self.latent_dim = 2048
        elif self.model_name == "xception":
            if timm is None:
                raise ImportError("Install timm to use the Xception backbone.")
            self.latent_dim = 2048
            model = timm.create_model("xception", pretrained=True, features_only=False)
            self.model = nn.Sequential(*list(model.children())[:-2])
        elif self.model_name == "EfficientNet-b0":
            self.latent_dim = 1280
            weights = EfficientNet_B0_Weights.DEFAULT
            model = efficientnet_b0(weights=weights)
            self.model = nn.Sequential(*list(model.features))
        else:
            raise ValueError(f"Unsupported model_name: {self.model_name}")

        self.lstm = nn.LSTM(self.latent_dim, hidden_dim, lstm_layers, bidirectional)
        self.relu = nn.LeakyReLU()
        self.dp = nn.Dropout(dropout)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        # 두 개의 출력: 이진 분류와 method 분류
        self.binary_classifier = nn.Linear(hidden_dim, num_binary_classes)
        self.method_classifier = nn.Linear(hidden_dim, num_method_classes)

    def forward(self, x):
        batch_size, seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)
        fmap = self.model(x)
        x = self.avgpool(fmap)
        x = x.view(batch_size, seq_length, self.latent_dim)
        x_lstm, _ = self.lstm(x, None)
        pooled = torch.mean(x_lstm, dim=1)
        return fmap, self.binary_classifier(self.dp(pooled)), self.method_classifier(self.dp(pooled))


def load_model(checkpoint_path, model_name="EfficientNet-b0", device=None, dropout=0.5):
    if device is None:
        device = get_device()
    model = ResearchModel(
        num_binary_classes=2,
        num_method_classes=7,
        model_name=model_name,
        dropout=dropout,
    ).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model
