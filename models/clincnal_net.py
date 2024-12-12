from torch import nn


class clincnal_net(nn.Module):
    def __init__(self):
        super(clincnal_net, self).__init__()
        self.clinical = nn.Linear(12, 60)
        self.clinical1 = nn.Linear(60, 60)
        self.classifier = nn.Linear(60, 2)

    def forward(self, clinical_features):
        x = self.clinical(clinical_features)
        x= self.clinical1(x)
        # x = self.sigmod(x)
        x = self.classifier(x)
        return x