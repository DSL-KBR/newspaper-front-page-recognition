import torch
import torch.nn as nn

from models.base_model import BaseModel
from networks.ResNeSt import FullyConvolutionalResNeSt


class FPRNet(BaseModel):

    def __init__(self, checkpoints_dir, model_name, num_class=2, isTrain=True, gpu_ids=[]):

        super().__init__(checkpoints_dir=checkpoints_dir, model_name=model_name, isTrain=isTrain, gpu_ids=gpu_ids)
        self.model_names = ['ResNeSt', 'Classifier']
        self.num_class = num_class
        self.loss_names = ['pred']

        # define networks
        self.netResNeSt = FullyConvolutionalResNeSt().eval().to(self.device)
        self.netClassifier = FullyConnectedClassifier(num_class=self.num_class).to(self.device)

        if self.isTrain:
            # define loss functions
            self.criterionPred = nn.CrossEntropyLoss()

            # optimizers and schedulers
            self.optimizer_ResNeSt = torch.optim.Adam(self.netResNeSt.parameters(), lr=1e-5)
            self.optimizer_Classifier = torch.optim.Adam(self.netClassifier.parameters(), lr=1e-5)

            self.optimizers.append(self.optimizer_ResNeSt)
            self.optimizers.append(self.optimizer_Classifier)

        # End of Initialization

    def set_input(self, inputs):

        self.sample = inputs['Sample'].to(self.device)
        self.label = inputs['Label'].to(self.device)
        self.metadata = inputs['Metadata']

    def forward(self):

        # Upper path through only ResNeSt
        self.featResNeSt = self.netResNeSt(self.sample)
        # quality/pregnancy prediction
        self.pred = self.netClassifier(self.featResNeSt)

    def backward_R(self):
        self.loss_pred = self.criterionPred(self.pred, self.label.long())
        self.loss_pred.backward()

    def optimize_parameters(self):
        self.forward()

        self.optimizer_ResNeSt.zero_grad()
        self.optimizer_Classifier.zero_grad()

        self.backward_R()

        self.optimizer_ResNeSt.step()
        self.optimizer_Classifier.step()


class FullyConnectedClassifier(nn.Module):
    def __init__(self, num_class):
        super().__init__()
        self.pathResNeSt = nn.Sequential(*[nn.Conv2d(in_channels=1000, out_channels=num_class, kernel_size=1),
                                           nn.BatchNorm2d(num_features=num_class),
                                           nn.ReLU(inplace=True)])
        self.pathMain = nn.Linear(in_features=32, out_features=num_class)

    def forward(self, x):
        self.featLinear = self.pathResNeSt(x)
        x = self.featLinear.view(x.shape[0], -1)
        return self.pathMain(x)

        # End of Initialization
