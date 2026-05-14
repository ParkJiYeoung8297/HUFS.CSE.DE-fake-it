import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


class Model(nn.Module):
    def __init__(self, num_binary_classes=2, num_method_classes=7,model_name="EfficientNet-b0", lstm_layers=1 , hidden_dim = 2048, bidirectional = False):
        super(Model, self).__init__()
        self.model_name = model_name
        self.latent_dim = 1280 # efficient
        weights = EfficientNet_B0_Weights.DEFAULT
        model = efficientnet_b0(weights=weights)
        self.model = nn.Sequential(*list(model.features))
        self.lstm = nn.LSTM(self.latent_dim,hidden_dim, lstm_layers,  bidirectional)
        self.relu = nn.LeakyReLU() 
        self.dp = nn.Dropout(0.5)
        self.avgpool = nn.AdaptiveAvgPool2d(1)


        # 두 개의 출력: 이진 분류와 method 분류
        self.binary_classifier = nn.Linear(hidden_dim, num_binary_classes)
        self.method_classifier = nn.Linear(hidden_dim, num_method_classes)

    def forward(self, x):
        batch_size,seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)
        fmap = self.model(x)
        x = self.avgpool(fmap)
        x = x.view(batch_size,seq_length,self.latent_dim) # resnext50_32x4d, xception : 2048, efficientnet-b0 : 1280
        x_lstm,_ = self.lstm(x,None)
        pooled = torch.mean(x_lstm, dim=1)
        return fmap, self.binary_classifier(self.dp(pooled)), self.method_classifier(self.dp(pooled))