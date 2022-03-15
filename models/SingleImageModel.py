import torch.nn as nn
import torchvision.models as models

class SingleImageModel(nn.Module):

    def __init__(self):
        '''
            param device: set device to 'cpu' or 'cuda', default 'cpu'
        '''
        super(SingleImageModel, self).__init__()

        self.image_encoder = self.get_image_encoder()
        self.prediction_head = self.get_prediction_head()

        # freeze resnet
        for param in self.image_encoder.parameters():
            param.requires_grad = False

        self.image_encoder.fc = self.prediction_head

    def get_image_encoder(self):
        '''
            Use ResNet's feature extractors as the image encoder
        '''
        resnet = models.resnet18(pretrained=True)
        return resnet

    def get_prediction_head(self):
        prediction_head = nn.Linear(in_features=512, out_features=1, bias=True)
        return prediction_head

    def forward(self, x):
        x = self.image_encoder(x)
        return x

    

    
