import torchvision.models as models
import torch.nn as nn

vgg19_model = models.vgg19(pretrained=True)
feature_extractor = nn.Sequential(*list(vgg19_model.features.children())[:18])
print(feature_extractor)


vgg = models.vgg19(pretrained = True)
feature_vgg = nn.Sequential(
*list(vgg.features.children())[:-4])
print(feature_vgg)