import torch
import numpy as np
import torch.nn as nn

class XRD_convnet(nn.Module):
    def __init__(self, in_channels, output_dim):
        super(XRD_convnet, self).__init__()
        self.flatten = nn.Flatten()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels, 80, kernel_size = 100, stride=5),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.BatchNorm1d(80),
            nn.Conv1d(80, 80, 50, stride=5),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.BatchNorm1d(80),
            nn.Conv1d(80, 80, 25, stride=2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.BatchNorm1d(80),
        )

         # Calculate flattened_size dynamically
        self.flattened_size = self._get_flattened_size(input_shape=(1, in_channels, 8500))

        self.MLP = nn.Sequential(
            nn.Linear(self.flattened_size, 2300),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2300, 1150),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1150, output_dim)
        )

    def _get_flattened_size(self, input_shape):
        dummy_input = torch.zeros(input_shape)
        with torch.no_grad():
            dummy_output = self.conv_layers(dummy_input)
        return int(np.prod(dummy_output.shape))
    
    def forward(self, x,): 

        x = self.conv_layers(x)
        x = self.flatten(x)
        x = self.MLP(x)

        return x


class composition_MLP(nn.Module):
    def __init__(self):
        super(composition_MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.composition_net = nn.Sequential(
            nn.Linear(100, 230),
            nn.ReLU(),
            nn.BatchNorm1d(230),  
            nn.Linear(230, 230),
            nn.ReLU(),
            nn.BatchNorm1d(230),  
            nn.Linear(230, 230),
            nn.ReLU(),
            nn.BatchNorm1d(230),  
            nn.Linear(230, 230),
            nn.ReLU(),
        )

    def forward(self, c): 
        c = c.squeeze(1)
        c = self.composition_net(c)
        return c

class composition_CNN(nn.Module):
    def __init__(self, in_channels, output_dim):
        super(composition_CNN, self).__init__()
        self.flatten = nn.Flatten()

        self.composition_convnet = nn.Sequential(
            nn.Conv2d(1, 20, 5, 1, 0),
            nn.ReLU(),
            nn.BatchNorm2d(20),
            nn.Conv2d(20, 10, 3, 1, 0),
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Conv2d(10, 5, 3, 1, 0),
            nn.ReLU(),
            nn.BatchNorm2d(5),
        )

        self.flattened_size2 = self._get_flattened_comp_size(input_shape=(1, 1, 21, 100))

        self.post_composition_convnet = nn.Sequential(
                nn.Linear(self.flattened_size2, 230),
                nn.ReLU(),
                nn.Linear(230, 230),
                nn.ReLU(),
                nn.Linear(230, 230),
        )

    def _get_flattened_comp_size(self, input_shape):
        dummy_input = torch.zeros(input_shape)

        with torch.no_grad():
            dummy_output = self.composition_convnet(dummy_input)
        
        return int(np.prod(dummy_output.shape))
    
    def forward(self, c): 
        c = self.composition_convnet(c)
        c = self.flatten(c)
        c = self.post_composition_convnet(c)
        return c


class XRD_C_SymNet(nn.Module):
    def __init__(self, in_channels, output_dim, composition_model,  xrd_model = "cnn"):
        super(XRD_C_SymNet, self).__init__()

        if xrd_model == "cnn":
            self.xrd_module = XRD_convnet(in_channels, output_dim)
        else:
            self.xrd_module = None
        
        if composition_model == "mlp":
            self.composition_module = composition_MLP()
        elif composition_model == "cnn":
            self.composition_module = composition_CNN()
        else:
            self.composition_module = None
    
        if (xrd_model is not None) and (composition_model is not None):
            self.merge_net = nn.Sequential(
                nn.Linear(460, 230),
                nn.ReLU(),
                nn.Linear(230, 230)
            )
        else: 
            self.merge_net = None

    def forward(self, x, c):
        if self.xrd_module: 
            x = self.xrd_module(x)
    
        if self.composition_module:
            c = self.composition_module(c)

        if self.merge_net:
            m = torch.cat((x, c), dim = 1)
            m = self.merge_net(m)
            return m
        
        elif self.xrd_module:
            return x
        elif self.composition_module:
            return c    