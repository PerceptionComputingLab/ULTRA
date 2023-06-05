import torch
from torch import nn
import torchvision.models as models
import numpy as np
import torch.nn.functional as F
from torch.distributions import Normal, Independent, kl
import timm

class Classifier(nn.Module):
    def __init__(self, in_features, num_classes):
        super(Classifier, self).__init__()

        self.fc1 = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(True),
            nn.Linear(256, 128),
            nn.ReLU(True))
        self.fc2 = nn.Linear(128, num_classes)
        self.fc3 = nn.Linear(128, 1)
        self.softmax = nn.Softmax(dim=-1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        score = self.fc1(x)
        dist = self.softmax(self.fc2(score))
        score = self.sig(self.fc3(score))
        return dist, score


class Baseline(nn.Module):

    def __init__(self, num_classes=512, backbone="resnet34"):
        super(Baseline, self).__init__()
        if backbone == "resnet34":
            model = models.resnet34(pretrained=False)
            model.fc = torch.nn.Sequential()      
            self.model = model
            self.fc = nn.Sequential(nn.Linear(512, 256), nn.ReLU(True), nn.Linear(256, num_classes))
        elif backbone == "vit_base_patch16_384":
            model = timm.create_model('vit_base_patch16_384', num_classes=1000)
            self.model = model
            self.fc = nn.Sequential(nn.Linear(1000, 256), nn.ReLU(True), nn.Linear(256, num_classes))
        elif backbone == "swin_base_patch4_window12_384":
            model = timm.create_model('swin_base_patch4_window12_384', num_classes=1000)
            self.model = model
            self.fc = nn.Sequential(nn.Linear(1000, 256), nn.ReLU(True), nn.Linear(256, num_classes))       


    def forward(self, image):
        features = self.model(image)
        features = self.fc(features)
        return features


class Encoder(nn.Module):

    def __init__(self, in_features, backbone = "resnet34"):
        super(Encoder, self).__init__()
        if backbone == "resnet34":
            model = models.resnet34(pretrained=False)
            model.fc = torch.nn.Sequential()      
            self.model = model
            self.fc = nn.Sequential(nn.Linear(512, 256), nn.ReLU(True), nn.Linear(256, in_features))
        elif backbone == "vit_base_patch16_384":
            model = timm.create_model('vit_base_patch16_384', num_classes=1000)
            self.model = model
            self.fc = nn.Sequential(nn.Linear(1000, 256), nn.ReLU(True), nn.Linear(256, in_features))
        elif backbone == "swin_base_patch4_window12_384":
            model = timm.create_model('swin_base_patch4_window12_384', num_classes=1000)
            self.model = model
            self.fc = nn.Sequential(nn.Linear(1000, 256), nn.ReLU(True), nn.Linear(256, in_features))       


    def forward(self, image):
        features = self.model(image)
        features = self.fc(features)
        return features


class MLP(nn.Module):
    def __init__(self, in_features, num_filters, num_classes, training=True):
        super(MLP, self).__init__()

        if training:
            # 输入特征维度(256)加上标签分布特征（13或100）
            in_features += num_classes
        self.mlp = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(True),
            nn.Linear(128, num_filters))

    def forward(self, input, target=None):
        if target is not None:
            input = torch.cat((input, target), dim=1)
        latent = self.mlp(input)
        return latent


class Expertness(nn.Module):
    def __init__(self, in_features, out_features, ):
        super(Expertness, self).__init__()
        self.mlp = nn.Linear(in_features, out_features)
        self.sig = nn.Sigmoid()

    def forward(self, feature_map):
        feature_map = self.mlp(feature_map)
        weight = self.sig(feature_map)
        return weight


class ESFG(nn.Module):

    def __init__(self, in_features, out_features, latent_dim, num_classes, n_raters, device, training, backbone):
        super(ESFG, self).__init__()
        self.encoder = Encoder(in_features, backbone)
        self.expert = [Expertness(in_features, out_features).to(device) for i in range(n_raters)]
        self.mlp = MLP(in_features, latent_dim * 2, num_classes, training)
        self.latent_dim = latent_dim

    def forward(self, image, target=None):
        encoding = self.encoder(image)
        mu_log_sigma = self.mlp(encoding, target)

        mu = mu_log_sigma[:, :self.latent_dim]
        log_sigma = mu_log_sigma[:, self.latent_dim:]

        dist = Independent(Normal(loc=mu, scale=torch.exp(log_sigma)), 1)
        expertness = [expert(encoding) for expert in self.expert]
        return dist, expertness


class EIFG(nn.Module):

    def __init__(self, in_features, num_filters, num_classes):
        super(EIFG, self).__init__()

        self.encoder = Encoder(in_features)
        self.mlp = MLP(in_features, num_filters, num_classes, training=False)

    def forward(self, image):
        encoding = self.encoder(image)
        features = self.mlp(encoding)
        return features


class Fcomb(nn.Module):
    def __init__(self, num_filters, latent_dim, out_features):
        super(Fcomb, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(num_filters + latent_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, out_features))
        self.mlp2 = nn.Sequential(
            nn.Linear(num_filters, 128),
            nn.ReLU(True),
            nn.Linear(128, out_features))

    def forward(self, feature_map, z):
        Z_feature = torch.cat((feature_map, z), dim=1)
        output = self.mlp(Z_feature) + self.mlp2(feature_map)
        # output = self.mlp(Z_feature)
        return output


class Fusion(nn.Module):
    def __init__(self):
        super(Fusion, self).__init__()
        self.sig = nn.Sigmoid()

    def forward(self, feature_map):
        weight = self.sig(feature_map)
        output = feature_map + weight * feature_map
        # output = weight * feature_map
        return output


class ULTRA(nn.Module):

    def __init__(self, in_features=256, out_features=128, num_classes=13, num_filters=20,
                 latent_dim=6, n_raters=3, training=True, device=None, backbone="resnet34"):
        super(ULTRA, self).__init__()
        if device is None:
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # esfg提取专家信息，eifg提取图像特征，前者作为后者的权重
        self.esfg_prior = ESFG(in_features, out_features, latent_dim, num_classes, n_raters, device,training=False, backbone = backbone).to(
            device)  # 验证或者测试时用到，不包含标签信息
        self.esfg_posterior = ESFG(in_features, out_features, latent_dim, num_classes, n_raters, device, training, backbone = backbone).to(
            device)  # 验证或者测试时用到，包含标签信息
        self.eifg = EIFG(in_features, num_filters, num_classes).to(device)
        self.fcomb = Fcomb(num_filters, latent_dim, out_features).to(device)

    def forward(self, image, target=None):
        self.prior_latent_space, self.prior_expertness = self.esfg_prior(image)
        self.posterior_latent_space, self.posterior_expertness = self.esfg_posterior(image, target)
        self.eifg_features = self.eifg(image)

    def reconstruct(self, t):
        z_posterior = self.prior_latent_space.rsample()
        reconstruction = self.fcomb.forward(self.eifg_features, z_posterior)
        reconstruction = reconstruction + reconstruction * self.prior_expertness[t]
        return reconstruction

    def fdmrm(self, t):
        vae_loss = torch.mean(kl.kl_divergence(self.posterior_latent_space, self.prior_latent_space))
        z_posterior = self.posterior_latent_space.rsample()
        reconstruction = self.fcomb.forward(self.eifg_features, z_posterior)
        reconstruction = reconstruction + reconstruction * self.posterior_expertness[t]
        return vae_loss, reconstruction
