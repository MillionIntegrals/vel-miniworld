import numpy as np

import torch
import torch.nn as nn
import torch.nn.init as init

import vel.util.network as net_util

from vel.modules.layers import OneHotEncode
from vel.api.base import LinearBackboneModel, ModelFactory


class MinigridObsFc(LinearBackboneModel):
    """ Feedforward fully-connected network properly decoding Minigrid 7x7x3 observations """
    def __init__(self, input_width=7, input_height=3, hidden_layers=None, activation='relu'):
        super().__init__()

        if hidden_layers is None:
            hidden_layers = [512, 256, 128]

        assert hidden_layers, "Hidden layers cannot be empty"
        self.hidden_layers = hidden_layers

        self._output_dim = self.hidden_layers[-1]

        self.layer1encode = OneHotEncode(10)
        self.layer2encode = OneHotEncode(6)
        self.layer3encode = OneHotEncode(2)

        self.input_width = input_width
        self.input_height = input_height

        self.concatenated_size = (
                self.input_width * self.input_height *
                (self.layer1encode.num_classes + self.layer2encode.num_classes + self.layer3encode.num_classes)
        )

        current_size = self.concatenated_size

        hidden_layer_objects = []

        for layer_size in self.hidden_layers:
            linear_layer = nn.Linear(current_size, layer_size)
            activation_layer = net_util.activation(activation)()

            hidden_layer_objects.append(linear_layer)
            hidden_layer_objects.append(activation_layer)

            current_size = layer_size

        self.model = nn.Sequential(*hidden_layer_objects)

    @property
    def output_dim(self) -> int:
        """ Final dimension of model output """
        return self._output_dim

    def reset_weights(self):
        """ Call proper initializers for the weights """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                init.orthogonal_(m.weight, gain=np.sqrt(2))
                init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                # init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                init.orthogonal_(m.weight, gain=np.sqrt(2))
                init.constant_(m.bias, 0.0)

    def forward(self, image):
        layer1 = self.layer1encode(image[:, :, :, 0].view(image.size(0), -1))
        layer2 = self.layer2encode(image[:, :, :, 1].view(image.size(0), -1))
        layer3 = self.layer3encode(image[:, :, :, 2].view(image.size(0), -1))

        flat_observation = torch.cat([layer1, layer2, layer3], dim=2).view(image.size(0), -1)

        model_output = self.model(flat_observation)

        return model_output


def create(input_width=7, input_height=7, hidden_layers=None, activation='relu'):
    def instantiate(**_):
        return MinigridObsFc(
            input_width=input_width, input_height=input_height, hidden_layers=hidden_layers,
            activation=activation
        )

    return ModelFactory.generic(instantiate)


# Add this to make nicer scripting interface
MinigridObsFcFactory = create

