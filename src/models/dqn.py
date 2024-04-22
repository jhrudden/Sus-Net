from typing import List
import torch
from torch import nn
from enum import StrEnum, auto

from src.utils import calculate_cnn_output_dim


class ModelType(StrEnum):
    RANDOM = auto()
    SPATIAL_DQN = auto()
    MLP = auto()

    @staticmethod
    def build(model_type: str, **kwargs):
        assert model_type in [
            m.value for m in ModelType
        ], f"Invalid model type: {model_type}"
        if model_type == ModelType.RANDOM:
            assert (
                kwargs.get("pretrained_model_path", None) is None
            ), "Random model does not support pretrained model"
            return RandomEquiprobable(kwargs["n_actions"])
        elif model_type == ModelType.SPATIAL_DQN:
            if kwargs.get("pretrained_model_path", None) is not None:
                return SpatialDQN.load_from_checkpoint(kwargs["pretrained_model_path"])
            kwargs.pop("pretrained_model_path", None)
            return SpatialDQN(**kwargs)
        elif model_type == ModelType.MLP:
            if kwargs.get("pretrained_model_path", None) is not None:
                return MLP.load_from_checkpoint(kwargs["pretrained_model_path"])
            kwargs.pop("pretrained_model_path", None)
            return MLP(**kwargs)


class ActivationType(StrEnum):
    RELU = "relu"
    SIGMOID = "sigmoid"
    PRELU = "prelu"

    def build(self):
        if self == ActivationType.RELU:
            return nn.ReLU()
        if self == ActivationType.PRELU:
            return nn.PReLU()
        elif self == ActivationType.SIGMOID:
            return nn.Sigmoid()
        else:
            raise ValueError(f"Activation function {self} not supported")


class Q_Estimator(nn.Module):
    def __init__(self):
        super(Q_Estimator, self).__init__()

    @property
    def model_type(self):
        raise NotImplementedError("model_type property not implemented")

    def dump_to_checkpoint(self, filepath):
        raise NotImplementedError("dump_to_checkpoint method not implemented")

    def load_from_checkpoint(self, filepath):
        raise NotImplementedError("load_from_checkpoint method not implemented")


class MLP(nn.Module):
    def __init__(
        self,
        layer_dims,
    ):
        super(MLP, self).__init__()
        self.layer_dims = layer_dims
        self.model = make_mlp(layer_dims, activation_fn=ActivationType.PRELU)
        self.config = {"layer_dims": layer_dims}

    @property
    def model_type(self):
        return ModelType.MLP

    def forward(self, spatial_x, non_spatial_x):
        batch_size, timesteps, C, H, W = spatial_x.size()
        x = torch.cat(
            (spatial_x.view(batch_size, -1), non_spatial_x.view(batch_size, -1)), dim=1
        )
        out = self.model(x)
        return out

    def dump_to_checkpoint(model, filepath):
        checkpoint = {"state_dict": model.state_dict(), "config": model.config}
        torch.save(checkpoint, filepath)
        print(f"Model checkpoint saved to {filepath}")


class RandomEquiprobable(Q_Estimator):
    def __init__(self, n_outputs: int):
        super(RandomEquiprobable, self).__init__()
        self.n_outputs = n_outputs

    def forward(self, *inputs):
        batch_size = 1  # default batch size if no inputs are provided
        if inputs:
            batch_size = inputs[0].shape[0]

        random_indices = torch.randint(0, self.n_outputs, (batch_size,))
        outputs = torch.zeros(batch_size, self.n_outputs)
        outputs[torch.arange(batch_size), random_indices] = 1

        return outputs

    @property
    def model_type(self):
        return ModelType.RANDOM

    def dump_to_checkpoint(model, filepath):
        pass

    def load_from_checkpoint(filepath):
        raise NotImplementedError("load_from_checkpoint method not implemented")


class CNNModel(nn.Module):
    def __init__(
        self,
        n_channels: List[int],
        strides: List[int],
        paddings: List[int],
        kernel_sizes: List[int],
    ):
        super(CNNModel, self).__init__()

        layers = []
        for idx, (channels, stride, kernel_size, padding) in enumerate(
            zip(n_channels[:-1], strides, kernel_sizes, paddings)
        ):
            layers.append(
                nn.Conv2d(
                    in_channels=channels,
                    out_channels=n_channels[idx + 1],
                    stride=stride,
                    kernel_size=kernel_size,
                    padding=padding,
                )
            )
            layers.append(nn.ReLU())

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class RNNModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        n_layers: int,
        hidden_dim: int,
        dropout: float,
    ):
        super(RNNModel, self).__init__()
        self.model = nn.RNN(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self, x):
        return self.model(x)  # (output, hidden state)


class SpatialDQN(Q_Estimator):

    def __init__(
        self,
        # Feature size args
        input_image_size: int,
        non_spatial_input_size: int,
        # CNN arguments
        n_channels: List[int],
        strides: List[int],
        paddings: List[int],
        kernel_sizes: List[int],
        # RNN arguments
        rnn_layers: int,
        rnn_hidden_dim: int,
        rnn_dropout: float,
        # MLP arguments
        mlp_hidden_layer_dims: List[int],
        n_actions: int,
    ):
        super(SpatialDQN, self).__init__()

        self.config = {
            "input_image_size": input_image_size,
            "non_spatial_input_size": non_spatial_input_size,
            "n_channels": n_channels,
            "strides": strides,
            "paddings": paddings,
            "kernel_sizes": kernel_sizes,
            "rnn_layers": rnn_layers,
            "rnn_hidden_dim": rnn_hidden_dim,
            "rnn_dropout": rnn_dropout,
            "mlp_hidden_layer_dims": mlp_hidden_layer_dims,
            "n_actions": n_actions,
        }

        self.cnn = CNNModel(
            n_channels=n_channels,
            strides=strides,
            paddings=paddings,
            kernel_sizes=kernel_sizes,
        )

        # calculating the number of features out of CNN
        self.cnn_ouput_dim = calculate_cnn_output_dim(
            input_size=input_image_size,
            kernel_sizes=kernel_sizes,
            strides=strides,
            paddings=paddings,
        )
        self.rnn_in_dim = (
            self.cnn_ouput_dim**2 * n_channels[-1] + non_spatial_input_size
        )

        # Making RNN
        self.rnn = RNNModel(
            input_dim=self.rnn_in_dim,
            n_layers=rnn_layers,
            hidden_dim=rnn_hidden_dim,
            dropout=rnn_dropout,
        )

        # MLP Prediction head
        self.n_actions = n_actions
        self.mlp_dims = [rnn_hidden_dim] + mlp_hidden_layer_dims + [n_actions]
        self.prediction_head = make_mlp(
            layer_dims=self.mlp_dims, activation_fn=ActivationType.PRELU
        )

    @property
    def model_type(self):
        return ModelType.SPATIAL_DQN

    def forward(self, spatial_x, non_spatial_x):

        # running through CNN
        batch_size, timesteps, C, H, W = spatial_x.size()
        cnn_in = spatial_x.view(batch_size * timesteps, C, H, W)
        cnn_out = self.cnn(cnn_in)
        # Reshape the output for the RNN

        print(f"CNN OUT SUM: {cnn_out.sum()}")

        cnn_out = cnn_out.view(batch_size, timesteps, -1)
        # appending non-spatial features
        rnn_in = torch.cat((cnn_out, non_spatial_x), dim=2)

        rnn_out, _ = self.rnn(rnn_in)
        # Use the last hidden state to predict with MLP
        mlp_in = rnn_out[:, -1, :]
        out = self.prediction_head(mlp_in)

        return out

    def dump_to_checkpoint(model, filepath):
        checkpoint = {"state_dict": model.state_dict(), "config": model.config}
        torch.save(checkpoint, filepath)
        print(f"Model checkpoint saved to {filepath}")

    def load_from_checkpoint(filepath):
        checkpoint = torch.load(filepath)
        config = checkpoint["config"]
        model = SpatialDQN(**config)
        model.load_state_dict(checkpoint["state_dict"])
        print("Model loaded from checkpoint")
        return model


def make_mlp(layer_dims, activation_fn: ActivationType = ActivationType.RELU):
    layers = []

    for idx, dim in enumerate(layer_dims[:-1]):
        layers.append(nn.Linear(in_features=dim, out_features=layer_dims[idx + 1]))
        layers.append(activation_fn.build())

    return nn.Sequential(*layers[:-1])
