import torch
import torch.nn as nn
import torchvision.models as models

class DiseaseDetectionModel(nn.Module):
    def __init__(self, num_classes):
        super(DiseaseDetectionModel, self).__init__()
        # Use ResNet50 as backbone
        resnet = models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        
        # Disease classification head
        self.classification_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
        # Severity estimation head (regression)
        self.severity_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()  # Normalize severity between 0-1
        )
        
    def forward(self, x):
        features = self.backbone(x)
        disease_pred = self.classification_head(features)
        severity_pred = self.severity_head(features)
        return disease_pred, severity_pred
