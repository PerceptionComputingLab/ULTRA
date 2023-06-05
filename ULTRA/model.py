from torchvision.models import resnet18, resnet34, ResNet, GoogLeNet
import torch
import torchvision
import torch.nn as nn
import torch.functional as F
import inspect
import pretrainedmodels


class MLP(nn.Module):
    # A simple multi-layer perception implementation

    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.activation = nn.ReLU()
        self.layer1 = nn.Linear(input_dim, 256)
        self.layer2 = nn.Linear(256, 128)
        self.layer3 = nn.Linear(128, output_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.layer4 = nn.Linear(128,1)
        self.sigmoid = nn.Sigmoid()

        # Test add Dropout layer
        # self.dropout1 = nn.Dropout(0.4)
        # self.dropout2 = nn.Dropout(0.5)


    def forward(self, x):
        x = self.activation(self.layer1(x))
        # Test Dropout
        # x = self.dropout1(x)
        x = self.activation(self.layer2(x))
        # Test Dropout
        # x = self.dropout2(x)

        output1 = self.softmax(self.layer3(x))
        output2 = self.sigmoid(self.layer4(x))
        return output1, output2


def get_backbone_model(model,  pretrained=True, num_features=512, as_backbone=True):
    '''
    Get the models implemented in the torchvision
    model: model name (string)
    num_features: output feature channels
    pretrained: Whether pretrained or not, default True.
    as_backbone: whether used as backbone or an independent model
    '''
    model_dict = {}
    for module_name in dir(torchvision.models):
        _models = getattr(torchvision.models, module_name)
        if inspect.isfunction(_models):
            model_dict[_models.__name__] = _models

    # if model == "inception_v3":
    #     return model_dict[model](aux_logits=False,pretrained=pretrained, num_classes=num_features)
    # else:
    #     return model_dict[model](pretrained=pretrained, num_classes =num_features)

    if model == "inception_v3":
        model_s= model_dict[model](aux_logits=False,pretrained=pretrained)
    elif model == "xception":
        model_s = pretrainedmodels.__dict__[model](num_classes=1000)
    else:
        model_s= model_dict[model](pretrained=pretrained)

    num_f = list(model_s.modules())[-1].in_features
    if as_backbone:
        if model == "densenet121":
            model_s.classifier = nn.Sequential(nn.Linear(in_features=num_f, out_features=num_features, bias=True))
        elif model == "xception":
            model_s.last_linear = nn.Sequential(nn.Linear(in_features=num_f, out_features=num_features, bias=True))
        else:
            model_s.fc = nn.Sequential(nn.Linear(in_features=num_f, out_features=num_features, bias=True))
    else:
        if model == "densenet121":
            model_s.classifier = nn.Sequential(nn.Linear(in_features=num_f, out_features=num_features, bias=True),
                                       nn.Linear(in_features=num_features, out_features=1, bias=True))
        elif model == "xception":
            model_s.last_linear = nn.Sequential(nn.Linear(in_features=num_f, out_features=num_features, bias=True),
                                        nn.Linear(in_features=num_features, out_features=1, bias=True))
        else:
            model_s.fc = nn.Sequential(nn.Linear(in_features=num_f, out_features=num_features, bias=True),
                                        nn.Linear(in_features=num_features, out_features=1, bias=True))
    
    return model_s