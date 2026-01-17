import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.video import r3d_18

# --- MODEL A: CNN + LSTM ---
class CNNLSTM(nn.Module):
    def __init__(self, num_classes=2, hidden_size=128):
        super(CNNLSTM, self).__init__()
        resnet = models.resnet18(pretrained=True)
        self.cnn = nn.Sequential(*list(resnet.children())[:-1])
        self.lstm = nn.LSTM(input_size=512, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        batch_size, frames, c, h, w = x.size()
        c_in = x.view(batch_size * frames, c, h, w)
        features = self.cnn(c_in)
        features = features.view(batch_size, frames, -1)
        lstm_out, (hn, cn) = self.lstm(features)
        out = self.fc(hn[-1])
        return out

# --- MODEL B: 3D-CNN ---
class C3D_ResNet(nn.Module):
    def __init__(self, num_classes=2):
        super(C3D_ResNet, self).__init__()
        self.model = r3d_18(pretrained=True)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4)
        return self.model(x)
