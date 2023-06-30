import torch
import torch.nn as nn
from torchvision.models import inception_v3, resnet50
from typing import Optional
from utils import get_device

def get_cub_model(args):
    device = get_device(args.cuda, args.device)
    if  not hasattr(args, 'active_layers') or  args.active_layers is None:
        active_layers = None
    else:
        active_layers = args.active_layers
    if args.model == "inception":
        return CUBClassifier(active_layers=active_layers).to(device)
    elif args.model == "resnet":
        return CUBClassifier(active_layers=active_layers).to(device)
    else:
        raise NotImplementedError

class LinearLayerConcept(nn.Module):
    def __init__(self, input_dim, output_dim, bias = True):
        super(LinearLayerConcept, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)  
     

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x

class SmallLayerConcept(nn.Module):
    def __init__(self, input_dim, output_dim, bias = True):
        super(SmallLayerConcept, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(input_dim, 10)  
        self.fc2 = nn.Linear(10, 10)  
        self.fc3 = nn.Linear(10, output_dim)  
     

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)        
        x = self.relu(x)
        x = self.fc3(x)   
        return x
    
class CUBClassifier(nn.Module):
    def __init__(self, active_layers = None):
        super().__init__()
        self.inception = inception_v3(pretrained=True)
        self.fc = nn.Linear(2048, 200)

    def forward(self, x):
        x = self.input_to_representation(x)
        x = self.fc(x)
        # N x 200
        return x

    def input_to_representation(self, x):
        # N x 3 x 299 x 299
        x = self.inception.Conv2d_1a_3x3(x)
        # N x 32 x 149 x 149
        x = self.inception.Conv2d_2a_3x3(x)
        # N x 32 x 147 x 147
        x = self.inception.Conv2d_2b_3x3(x)
        # N x 64 x 147 x 147
        x = self.inception.maxpool1(x)
        # N x 64 x 73 x 73
        x = self.inception.Conv2d_3b_1x1(x)
        # N x 80 x 73 x 73
        x = self.inception.Conv2d_4a_3x3(x)
        # N x 192 x 71 x 71
        x = self.inception.maxpool2(x)
        # N x 192 x 35 x 35
        x = self.inception.Mixed_5b(x)
        # N x 256 x 35 x 35
        x = self.inception.Mixed_5c(x)
        # N x 288 x 35 x 35
        x = self.inception.Mixed_5d(x)
        # N x 288 x 35 x 35
        x = self.inception.Mixed_6a(x)
        # N x 768 x 17 x 17
        x = self.inception.Mixed_6b(x)
        # N x 768 x 17 x 17
        x = self.inception.Mixed_6c(x)
        # N x 768 x 17 x 17
        x = self.inception.Mixed_6d(x)
        # N x 768 x 17 x 17
        x = self.inception.Mixed_6e(x)
        # N x 768 x 17 x 17
        aux: Optional[torch.Tensor] = None
        if self.inception.AuxLogits is not None:
            if self.inception.training:
                aux = self.inception.AuxLogits(x)
        # N x 768 x 17 x 17
        x = self.inception.Mixed_7a(x)
        # N x 1280 x 8 x 8
        x = self.inception.Mixed_7b(x)
        # N x 2048 x 8 x 8
        x = self.inception.Mixed_7c(x)
        # N x 2048 x 8 x 8
        # Adaptive average pooling
        x = self.inception.avgpool(x)
        # N x 2048 x 1 x 1
        x = self.inception.dropout(x)
        # N x 2048 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 2048
        return x

    def mixed_7b_rep(self, x):
        # N x 3 x 299 x 299
        x = self.inception.Conv2d_1a_3x3(x)
        # N x 32 x 149 x 149
        x = self.inception.Conv2d_2a_3x3(x)
        # N x 32 x 147 x 147
        x = self.inception.Conv2d_2b_3x3(x)
        # N x 64 x 147 x 147
        x = self.inception.maxpool1(x)
        # N x 64 x 73 x 73
        x = self.inception.Conv2d_3b_1x1(x)
        # N x 80 x 73 x 73
        x = self.inception.Conv2d_4a_3x3(x)
        # N x 192 x 71 x 71
        x = self.inception.maxpool2(x)
        # N x 192 x 35 x 35
        x = self.inception.Mixed_5b(x)
        # N x 256 x 35 x 35
        x = self.inception.Mixed_5c(x)
        # N x 288 x 35 x 35
        x = self.inception.Mixed_5d(x)
        # N x 288 x 35 x 35
        x = self.inception.Mixed_6a(x)
        # N x 768 x 17 x 17
        x = self.inception.Mixed_6b(x)
        # N x 768 x 17 x 17
        x = self.inception.Mixed_6c(x)
        # N x 768 x 17 x 17
        x = self.inception.Mixed_6d(x)
        # N x 768 x 17 x 17
        x = self.inception.Mixed_6e(x)
        # N x 768 x 17 x 17
        aux: Optional[torch.Tensor] = None
        if self.inception.AuxLogits is not None:
            if self.inception.training:
                aux = self.inception.AuxLogits(x)
        # N x 768 x 17 x 17
        x = self.inception.Mixed_7a(x)
        # N x 1280 x 8 x 8
        x = self.inception.Mixed_7b(x)
        return x

    def representation_to_output(self, h):
        return self.fc(h)

    def mixed_7b_rep_to_output(self, h):
        # N x 2048 x 8 x 8
        h = self.inception.Mixed_7c(h)
        # N x 2048 x 8 x 8
        # Adaptive average pooling
        h = self.inception.avgpool(h)
        # N x 2048 x 1 x 1
        h = self.inception.dropout(h)
        # N x 2048 x 1 x 1
        h = torch.flatten(h, 1)
        # N x 2048
        return self.fc(h)
    
class CUBResNet(nn.Module):
    def __init__(self, active_layers = None):
        super().__init__()
        self.n_class = 200
        self.base_model = resnet50(pretrained=True)
        self.base_model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.base_model.fc = nn.Linear(512 * 4, self.n_class)

    def forward(self, x):
        return self.base_model(x)

    def input_to_representation(self, x):
        return x

    def representation_to_output(self, h):
        return self.base_model.fc(h)
    