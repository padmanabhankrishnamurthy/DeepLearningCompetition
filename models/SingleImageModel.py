import torch.nn as nn
import torchvision.models as models

class SingleImageModel(nn.Module):

    def __init__(self):
        '''
            param device: set device to 'cpu' or 'cuda', default 'cpu'
        '''
        super(SingleImageModel, self).__init__()
        self.model = self.build_model()

    def build_model(self):
        '''
            Use ResNet's feature extractors as the image encoder
        '''
        resnet = models.resnet18(pretrained=True)
        prediction_head = self.get_prediction_head()
        resnet.fc = prediction_head

        # freeze resnet
        for param in resnet.parameters():
            param.requires_grad = False

        # unfreeze prediction head
        for param in resnet.fc.parameters():
            param.requires_grad = True
        
        return resnet

    def get_prediction_head(self):
        prediction_head = nn.Sequential(
            nn.Linear(in_features=512, out_features=256, bias=True),
            nn.Linear(in_features=256, out_features=128, bias=True),
            nn.Linear(in_features=128, out_features=1, bias=True)
        )

        return prediction_head

    def forward(self, x):
        x = self.modelr(x)
        return x

    
if __name__ == '__main__':
    model = SingleImageModel()
    print(model.model)

    for param in model.model.parameters():
        if param.requires_grad:
            print(param.shape)
    
