import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

try:
    import timm
except ImportError:
    timm = None


class AblationModel(nn.Module):
    """CNN/LSTM/multitask variants for reviewer-facing ablation study.

    Variants:
    - cnn_only: CNN frame features are mean-pooled across time.
    - cnn_lstm: CNN features are passed through LSTM, binary head only.
    - cnn_lstm_multitask: CNN + LSTM with binary and method heads.
    """

    def __init__(
        self,
        variant="cnn_lstm_multitask",
        num_binary_classes=2,
        num_method_classes=7,
        model_name="EfficientNet-b0",
        lstm_layers=1,
        hidden_dim=2048,
        bidirectional=False,
        dropout=0.5,
        feature_chunk_size=32,
        gradient_checkpointing=False,
    ):
        super().__init__()

        valid_variants = {"cnn_only", "cnn_lstm", "cnn_lstm_multitask"}
        if variant not in valid_variants:
            raise ValueError(f"variant must be one of {sorted(valid_variants)}")

        self.variant = variant
        self.model_name = model_name
        self.use_lstm = variant in {"cnn_lstm", "cnn_lstm_multitask"}
        self.use_multitask = variant == "cnn_lstm_multitask"
        self.feature_chunk_size = feature_chunk_size
        self.gradient_checkpointing = gradient_checkpointing

        self.feature_extractor, self.latent_dim = self._build_backbone(model_name)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dp = nn.Dropout(dropout)

        if self.use_lstm:
            self.lstm = nn.LSTM(
                self.latent_dim,
                hidden_dim,
                lstm_layers,
                bidirectional=bidirectional,
                batch_first=True,
            )
            feature_dim = hidden_dim * (2 if bidirectional else 1)
        else:
            self.lstm = None
            feature_dim = self.latent_dim

        self.binary_classifier = nn.Linear(feature_dim, num_binary_classes)
        self.method_classifier = (
            nn.Linear(feature_dim, num_method_classes)
            if self.use_multitask
            else None
        )

    def _build_backbone(self, model_name):
        if model_name == "EfficientNet-b0":
            weights = EfficientNet_B0_Weights.DEFAULT
            model = efficientnet_b0(weights=weights)
            return nn.Sequential(*list(model.features)), 1280

        if model_name == "resnext50_32x4d":
            weights = models.ResNeXt50_32X4D_Weights.DEFAULT
            model = models.resnext50_32x4d(weights=weights)
            return nn.Sequential(*list(model.children())[:-2]), 2048

        if model_name == "xception":
            if timm is None:
                raise ImportError("xception requires timm. Install with: pip install timm")
            model = timm.create_model("xception", pretrained=True, features_only=False)
            return nn.Sequential(*list(model.children())[:-2]), 2048

        raise ValueError(f"Unsupported model_name: {model_name}")

    def forward(self, x):
        batch_size, seq_length, channels, height, width = x.shape
        x = x.reshape(batch_size * seq_length, channels, height, width)

        feature_chunks = []
        last_fmap = None
        for chunk in x.split(self.feature_chunk_size):
            if self.training and self.gradient_checkpointing:
                last_fmap = checkpoint(self.feature_extractor, chunk, use_reentrant=False)
            else:
                last_fmap = self.feature_extractor(chunk)
            feature_chunks.append(self.avgpool(last_fmap).flatten(1))

        features = torch.cat(feature_chunks, dim=0).reshape(batch_size, seq_length, self.latent_dim)

        if self.use_lstm:
            features, _ = self.lstm(features)
            pooled = torch.mean(features, dim=1)
        else:
            pooled = torch.mean(features, dim=1)

        binary_logits = self.binary_classifier(self.dp(pooled))
        method_logits = (
            self.method_classifier(self.dp(pooled))
            if self.method_classifier is not None
            else None
        )

        return last_fmap, binary_logits, method_logits
