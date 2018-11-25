import numpy as np
import typing

import torch
import torch.nn as nn
import torch.nn.init as init

import vel.util.network as net_util

from vel.modules.layers import OneHotEncode
from vel.api.base import RnnLinearBackboneModel, ModelFactory


class MinigridObsLstm(RnnLinearBackboneModel):
    """ LSTM network properly decoding Minigrid 7x7x3 observations """
    def __init__(self, input_width=7, input_height=3, hidden_layers=None, lstm_dim=128, activation='relu',
                 normalization: typing.Optional[str]=None):
        super().__init__()

        if hidden_layers is None:
            hidden_layers = [512, 256]

        assert hidden_layers, "Hidden layers cannot be empty"
        self.hidden_layers = hidden_layers

        self.fc_output_dim = self.hidden_layers[-1]
        self.lstm_dim = lstm_dim
        self.normalization = normalization

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

            if self.normalization:
                hidden_layer_objects.append(net_util.normalization(normalization)(layer_size))

            hidden_layer_objects.append(activation_layer)

            current_size = layer_size

        self.model = nn.Sequential(*hidden_layer_objects)

        self.lstm = nn.LSTMCell(self.fc_output_dim, self.lstm_dim)

    @property
    def output_dim(self) -> int:
        """ Final dimension of model output """
        return self.lstm_dim

    @property
    def state_dim(self) -> int:
        """ Initial state of the network """
        return 2 * self.lstm_dim

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

        init.orthogonal_(self.lstm.weight_ih, gain=1.0)
        init.orthogonal_(self.lstm.weight_hh, gain=1.0)
        init.zeros_(self.lstm.bias_ih)
        init.zeros_(self.lstm.bias_hh)

    def forward(self, image, state):
        layer1 = self.layer1encode(image[:, :, :, 0].view(image.size(0), -1))
        layer2 = self.layer2encode(image[:, :, :, 1].view(image.size(0), -1))
        layer3 = self.layer3encode(image[:, :, :, 2].view(image.size(0), -1))

        flat_observation = torch.cat([layer1, layer2, layer3], dim=2).view(image.size(0), -1)

        fc_output = self.model(flat_observation)

        hidden_state, cell_state = torch.split(state, self.lstm_dim, 1)
        hidden_state, cell_state = self.lstm(fc_output, (hidden_state, cell_state))

        new_state = torch.cat([hidden_state, cell_state], dim=1)

        return hidden_state, new_state


def create(input_width=7, input_height=7, hidden_layers=None, lstm_dim=None, activation='relu', normalization=None):
    def instantiate(**_):
        return MinigridObsLstm(
            input_width=input_width, input_height=input_height, hidden_layers=hidden_layers, lstm_dim=lstm_dim,
            normalization=normalization, activation=activation
        )

    return ModelFactory.generic(instantiate)


# Add this to make nicer scripting interface
MinigridObsLstmFactory = create
