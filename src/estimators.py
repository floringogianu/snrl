""" Neural Network architecture for Atari games.
"""
from functools import partial
from itertools import chain
from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from src.conv_spectral_norm import spectral_norm_conv2d, Conv2dSpectralNorm
from src.linear_spectral_norm import spectral_norm, LinearSpectralNorm
from src.estimator_utils import conv2mat

__all__ = [
    "MLP",
    "AtariNet",
    "MinAtarNet",
    "RandomMinAtarNet",
    "get_feature_extractor",
    "get_head",
]


def hook_spectral_normalization(  # pylint: disable=bad-continuation
    spectral, layers, leave_smaller=False, lipschitz_k=1, flow_through_norm="linear"
):
    """Uses the convention in `spectral` to hook spectral normalization on
    modules in `layers`.

    Args:
        spectral (str): A string of negative indices. Ex.: `-1` or `-2,-3`.
                To hook spectral normalization only for computing the norm
                and not applying it on the weights add the identifier `L`.
                Ex.: `-1L`, `-2,-3,-4L`.
        layers (list): Ordered list of tuples of (module_name, nn.Module).
        leave_smaller (bool): If False divide by rho, if True by max(rho, 1)
        lipschitz_k (bool): the Lipschitz constant

    Returns:
        normalized: Layers
    """
    # Filter unsupported layers
    layers = [
        (n, m)
        for (n, m) in layers
        if isinstance(m, (nn.Conv2d, nn.Linear, SharedBiasLinear))
    ]
    N = len(layers)

    # Some convenient conventions
    if spectral == "":
        # log all layers, but do not apply
        spectral = ",".join([f"-{i}L" for i in range(N)])
    elif spectral == "full":
        # apply snorm everywhere
        spectral = ",".join([f"-{i}" for i in range(N)])
    else:
        spectral = str(spectral)  # sometimes this is just a number eg.: -3

    # For N=5, spectral="-2,-3L":   [('-2', True), ('-3L', False)]
    layers_status = [(i, "L" not in i) for i in spectral.split(",")]
    # For N=5, spectral="-2,-3L":   [(3, True), (2, False)]
    layers_status = [(int(i if s else i[:-1]) % N, s) for i, s in layers_status]

    hooked_layers = []
    conv_flow_through_norm = flow_through_norm in [True, "conv", "all"]
    linear_flow_through_norm = flow_through_norm in [True, "linear", "all"]

    for (idx, active) in layers_status:
        layer_name, layer = layers[idx]

        if isinstance(layer, nn.Conv2d):
            spectral_norm_conv2d(
                layer,
                active=active,
                leave_smaller=leave_smaller,
                lipschitz_k=lipschitz_k,
                flow_through_norm=conv_flow_through_norm,
            )
        elif isinstance(layer, (nn.Linear, SharedBiasLinear)):
            spectral_norm(
                layer,
                active=active,
                leave_smaller=leave_smaller,
                lipschitz_k=lipschitz_k,
                flow_through_norm=linear_flow_through_norm,
            )
        else:
            raise NotImplementedError(
                "S-Norm on {} layer type not implemented for {} @ ({}): {}".format(
                    type(layer), idx, layer_name, layer
                )
            )
        hooked_layers.append((idx, layer))

        print(
            "{} λ={} SNorm to {} @ ({}): {}".format(
                "Active " if active else "Logging", lipschitz_k, idx, layer_name, layer
            )
        )
    return hooked_layers


def get_feature_extractor(input_depth):
    """ Configures the default Atari feature extractor. """
    convs = [
        nn.Conv2d(input_depth, 32, kernel_size=8, stride=4),
        nn.Conv2d(32, 64, kernel_size=4, stride=2),
        nn.Conv2d(64, 64, kernel_size=3, stride=1),
    ]

    return nn.Sequential(
        convs[0],
        nn.ReLU(inplace=True),
        convs[1],
        nn.ReLU(inplace=True),
        convs[2],
        nn.ReLU(inplace=True),
    )


def get_head(hidden_size, out_size, shared_bias=False):
    """ Configures the default Atari output layers. """
    fc0 = nn.Linear(64 * 7 * 7, hidden_size)
    fc1 = (
        SharedBiasLinear(hidden_size, out_size)
        if shared_bias
        else nn.Linear(hidden_size, out_size)
    )
    return nn.Sequential(fc0, nn.ReLU(inplace=True), fc1)


def no_grad(module):
    """ Callback for turning off the gradient of a module.
    """
    try:
        module.weight.requires_grad = False
    except AttributeError:
        pass


def variance_scaling_uniform_(tensor, scale=0.1, mode="fan_in"):
    # type: (Tensor, float) -> Tensor
    r"""Variance Scaling, as in Keras.

    Uniform sampling from `[-a, a]` where:

        `a = sqrt(3 * scale / n)`

    and `n` is the number of neurons according to the `mode`.

    """
    # pylint: disable=protected-access,invalid-name
    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor)
    a = 3 * scale
    a /= fan_in if mode == "fan_in" else fan_out
    weights = nn.init._no_grad_uniform_(tensor, -a, a)
    # pylint: enable=protected-access,invalid-name
    return weights


class MLP(nn.Module):
    """ MLP estimator with variable depth and width for both C51 and DQN. """

    def __init__(  # pylint: disable=bad-continuation
        self, action_no, layers=None, support=None, spectral=None, **kwargs,
    ):
        super(MLP, self).__init__()

        self.action_no = action_no
        dims = [*layers, action_no]
        # create support and adjust dims in case of distributional
        if support is not None:
            # handy to make it a Parameter so that model.to(device) works
            self.support = nn.Parameter(torch.linspace(*support), requires_grad=False)
            dims[-1] = self.action_no * len(self.support)
        else:
            self.support = None
        # create MLP layers
        self.layers = nn.ModuleList(
            [nn.Linear(dims[i - 1], dims[i]) for i in range(1, len(dims))]
        )

        if spectral is not None:
            self.hooked_layers = hook_spectral_normalization(
                spectral, self.layers.named_children(), **kwargs,
            )

    def forward(self, x, probs=False, log_probs=False):
        assert not (probs and log_probs), "Can't output both p(s, a) and log(p(s, a))"

        x = x.view(x.shape[0], -1)  # usually it comes with a history dimension

        for module in self.layers[:-1]:
            x = F.relu(module(x), inplace=True)
        qs = self.layers[-1](x)

        # distributional RL
        # either return p(s,·), log(p(s,·)) or the distributional Q(s,·)
        if self.support is not None:
            logits = qs.view(qs.shape[0], self.action_no, len(self.support))
            if probs:
                return torch.softmax(logits, dim=2)
            if log_probs:
                return torch.log_softmax(logits, dim=2)
            qs_probs = torch.softmax(logits, dim=2)
            return torch.mul(qs_probs, self.support.expand_as(qs_probs)).sum(2)
        # or just return Q(s,a)
        return qs

    def get_spectral_norms(self):
        """ Return the spectral norms of layers hooked on spectral norm. """
        return {
            str(idx): layer.weight_sigma.item() for idx, layer in self.hooked_layers
        }


class SharedBiasLinear(nn.Linear):
    """ Applies a linear transformation to the incoming data: `y = xA^T + b`.
        As opposed to the default Linear layer it has a shared bias term.
        This is employed for example in Double-DQN.

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
    """

    def __init__(self, in_features, out_features):
        super(SharedBiasLinear, self).__init__(in_features, out_features, True)
        self.bias = Parameter(torch.Tensor(1))

    def extra_repr(self):
        return "in_features={}, out_features={}, bias=shared".format(
            self.in_features, self.out_features
        )


class AtariNet(nn.Module):
    """ Estimator used for ATARI games.
    """

    def __init__(  # pylint: disable=bad-continuation
        self,
        action_no,
        input_ch=1,
        hist_len=4,
        hidden_size=256,
        shared_bias=False,
        initializer="xavier_uniform",
        support=None,
        spectral=None,
        **kwargs,
    ):
        super(AtariNet, self).__init__()

        assert initializer in (
            "xavier_uniform",
            "variance_scaling_uniform",
        ), "Only implements xavier_uniform and variance_scaling_uniform."

        self.__action_no = action_no
        self.__initializer = initializer
        self.__support = None
        self.spectral = spectral
        if support is not None:
            self.__support = nn.Parameter(
                torch.linspace(*support), requires_grad=False
            )  # handy to make it a Parameter so that model.to(device) works
            out_size = action_no * len(self.__support)
        else:
            out_size = action_no

        # get the feature extractor and fully connected layers
        self.__features = get_feature_extractor(hist_len * input_ch)
        self.__head = get_head(hidden_size, out_size, shared_bias)

        self.reset_parameters()

        # We allways compute spectral norm except when None or notrace
        if spectral is not None:
            self.__hooked_layers = hook_spectral_normalization(
                spectral,
                chain(self.__features.named_children(), self.__head.named_children()),
                **kwargs,
            )

    def forward(self, x, probs=False, log_probs=False):
        # assert x.dtype == torch.uint8, "The model expects states of type ByteTensor"
        x = x.float().div(255)
        assert not (probs and log_probs), "Can't output both p(s, a) and log(p(s, a))"

        x = self.__features(x)
        x = x.view(x.size(0), -1)
        qs = self.__head(x)

        # distributional RL
        # either return p(s,·), log(p(s,·)) or the distributional Q(s,·)
        if self.__support is not None:
            logits = qs.view(qs.shape[0], self.__action_no, len(self.support))
            if probs:
                return torch.softmax(logits, dim=2)
            if log_probs:
                return torch.log_softmax(logits, dim=2)
            qs_probs = torch.softmax(logits, dim=2)
            return torch.mul(qs_probs, self.support.expand_as(qs_probs)).sum(2)
        # or just return Q(s,a)
        return qs

    @property
    def support(self):
        """ Return the support of the Q-Value distribution. """
        return self.__support

    def get_spectral_norms(self):
        """ Return the spectral norms of layers hooked on spectral norm. """
        return {
            str(idx): layer.weight_sigma.item() for idx, layer in self.__hooked_layers
        }

    def reset_parameters(self):
        """ Weight init.
        """
        init_ = (
            nn.init.xavier_uniform_
            if self.__initializer == "xavier_uniform"
            else partial(variance_scaling_uniform_, scale=1.0 / sqrt(3.0))
        )

        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                init_(module.weight)
                module.bias.data.zero_()

    @property
    def feature_extractor(self):
        """ Return the feature extractor. """
        return self.__features

    @property
    def head(self):
        """ Return the layers used as heads in Bootstrapped DQN. """
        return self.__head

    def div_grad_by_rho(self):
        """ Divide the already computed gradient by the spectral norm.
        """
        for module in self.modules():
            for _, hook in module._forward_pre_hooks.items():
                if isinstance(hook, (LinearSpectralNorm, Conv2dSpectralNorm)):
                    assert not hook._active
                    if hook._leave_smaller:
                        scale = max(module.weight_sigma.item() / hook._lipschitz_k, 1)
                    else:
                        scale = module.weight_sigma.item() / hook._lipschitz_k
                    with torch.no_grad():
                        module.weight_orig.grad.data /= scale


def set_up_layers(freeze, snorm, scale, layers):
    """ Normalize or scale certain layers and freeze them. """
    # Filter unsupported layers
    layers = [
        (n, m)
        for (n, m) in layers
        if isinstance(m, (nn.Conv2d, nn.Linear, SharedBiasLinear))
    ]
    N = len(layers)

    freeze = [int(lidx) % N for lidx in freeze.split(",")]
    in_chnls = [l.in_channels for n, l in layers if isinstance(l, nn.Conv2d)]
    x_shapes = [(ch, 10 - 2 * i, 10 - 2 * i) for i, ch in enumerate(in_chnls)]

    for lidx in freeze:
        lname, layer = layers[lidx]
        if snorm:
            if isinstance(layer, nn.Linear):
                rho = torch.svd(layer.weight.data).S.max()
                layer.weight.data /= rho
            elif isinstance(layer, nn.Conv2d):
                x_shape = x_shapes[lidx]
                circ = conv2mat(layer.weight.data, x_shape)
                rho = torch.svd(circ).S.max()
                layer.weight.data /= rho
        else:
            layer.weight.data *= scale

        layer.weight.requires_grad = False
        layer.bias.requires_grad = False

        print(
            "{} lidx={} @ ({}): {}".format(
                "S-Norm" if snorm else f"Scale={scale:0.1f}", lidx, lname, layer
            )
        )


class RandomMinAtarNet(nn.Module):
    """ An estimator with random layers that can be spectrally normalized or not.
    """

    def __init__(
        self,
        action_no,
        input_ch,
        freeze="-2,-3,-4",
        snorm=True,
        scale=1.0,
        hidden_size=128,
    ):
        super(RandomMinAtarNet, self).__init__()

        assert not (
            snorm and (scale != 1.0)
        ), "Cannot both normalize and scale the layers!"

        # hard-coded stuff
        self.spectral = None
        self.__support = None
        self.__initializer = "xavier_uniform"
        # init layers
        self.__features = nn.Sequential(
            nn.Conv2d(input_ch, 16, kernel_size=3, stride=1),  # -4
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, stride=1),  # -3
            nn.ReLU(inplace=True),
        )
        self.__head = nn.Sequential(
            nn.Linear(6 ** 2 * 16, hidden_size),  # -2
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, action_no),
        )
        self.reset_parameters()
        # mess-up layer
        set_up_layers(
            freeze,
            snorm,
            scale,
            chain(self.__features.named_children(), self.__head.named_children()),
        )

    def forward(self, x):
        # assert x.dtype == torch.uint8, "The model expects states of type ByteTensor"
        x = x.float()
        if x.ndimension() == 5:
            x = x.squeeze(1)  # drop the "history"

        x = self.__features(x)
        x = x.view(x.size(0), -1)
        return self.__head(x)

    def reset_parameters(self):
        """ Weight init.
        """
        init_ = (
            nn.init.xavier_uniform_
            if self.__initializer == "xavier_uniform"
            else partial(variance_scaling_uniform_, scale=1.0 / sqrt(3.0))
        )

        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                init_(module.weight)
                module.bias.data.zero_()

    @property
    def feature_extractor(self):
        """ Return the feature extractor. """
        return self.__features

    @property
    def head(self):
        """ Return the layers used as heads in Bootstrapped DQN. """
        return self.__head

    @property
    def support(self):
        """ Return the support of the Q-Value distribution. """
        return self.__support


class DeepLinearMinAtarNet(nn.Module):
    """ Estimator used for ATARI games.
    """

    def __init__(  # pylint: disable=bad-continuation
        self, action_no, input_ch=1, spectral=None, layer_dims=(64, 64, 64), **kwargs,
    ):
        super(DeepLinearMinAtarNet, self).__init__()
        self.__action_no = action_no
        self.__support = None
        self.spectral = spectral

        self.__features = nn.Sequential(
            nn.Conv2d(input_ch, 16, kernel_size=3, stride=1),  # -4
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, stride=1),  # -3
            nn.ReLU(inplace=True),
        )
        for m in self.__features.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.requires_grad = False
                # p.bias.requires_grad = False

        layers, in_sz = [], 16 * 6 * 6
        for out_sz in layer_dims:
            layers.append(nn.Linear(in_sz, out_sz))
            in_sz = out_sz
        layers.append(nn.Linear(in_sz, action_no))
        self.__dln = nn.Sequential(*layers)

        # We allways compute spectral norm except when None or notrace
        if spectral is not None:
            self.__hooked_layers = hook_spectral_normalization(
                spectral, self.__dln.named_children(), **kwargs,
            )

    def forward(self, x):
        x = x.float()
        if x.ndimension() == 5:
            x = x.squeeze(1)  # drop the "history"

        x = self.__features(x).flatten(start_dim=1)
        return self.__dln(x)

    @property
    def support(self):
        """ Return the support of the Q-Value distribution. """
        return self.__support

    def get_spectral_norms(self):
        """ Return the spectral norms of layers hooked on spectral norm. """
        return {
            str(idx): layer.weight_sigma.item() for idx, layer in self.__hooked_layers
        }


class MinAtarNet(nn.Module):
    """ Estimator used for ATARI games.
    """

    def __init__(  # pylint: disable=bad-continuation
        self,
        action_no,
        input_ch=1,
        support=None,
        spectral=None,
        initializer="xavier_uniform",
        layer_dims=((16,), (128,)),
        **kwargs,
    ):
        super(MinAtarNet, self).__init__()

        assert initializer in (
            "xavier_uniform",
            "variance_scaling_uniform",
        ), "Only implements xavier_uniform and variance_scaling_uniform."

        self.__action_no = action_no
        self.__initializer = initializer
        self.__support = None
        self.spectral = spectral
        if support is not None:
            self.__support = nn.Parameter(
                torch.linspace(*support), requires_grad=False
            )  # handy to make it a Parameter so that model.to(device) works
            out_size = action_no * len(self.__support)
        else:
            out_size = action_no

        # configure the net
        conv_layers, lin_layers = layer_dims
        feature_extractor, in_ch, out_wh = [], input_ch, 10
        for out_ch in conv_layers:
            feature_extractor += [
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1),
                nn.ReLU(inplace=True),
            ]
            in_ch = out_ch
            out_wh -= 2  # change this for a different kernel size or stride.
        self.__features = nn.Sequential(*feature_extractor)

        head, in_size = [], out_wh ** 2 * in_ch
        for hidden_size in lin_layers:
            head += [
                nn.Linear(in_size, hidden_size),
                nn.ReLU(inplace=True),
            ]
            in_size = hidden_size
        head.append(nn.Linear(in_size, out_size))
        self.__head = nn.Sequential(*head)

        self.reset_parameters()

        # We allways compute spectral norm except when None or notrace
        if spectral is not None:
            self.__hooked_layers = hook_spectral_normalization(
                spectral,
                chain(self.__features.named_children(), self.__head.named_children()),
                **kwargs,
            )

    def forward(self, x, probs=False, log_probs=False):
        assert not (probs and log_probs), "Can't output both p(s, a) and log(p(s, a))"
        # assert x.dtype == torch.uint8, "The model expects states of type ByteTensor"
        x = x.float()
        if x.ndimension() == 5:
            x = x.squeeze(1)  # drop the "history"

        x = self.__features(x)
        x = x.view(x.size(0), -1)
        qs = self.__head(x)

        # distributional RL
        # either return p(s,·), log(p(s,·)) or the distributional Q(s,·)
        if self.__support is not None:
            logits = qs.view(qs.shape[0], self.__action_no, len(self.support))
            if probs:
                return torch.softmax(logits, dim=2)
            if log_probs:
                return torch.log_softmax(logits, dim=2)
            qs_probs = torch.softmax(logits, dim=2)
            return torch.mul(qs_probs, self.support.expand_as(qs_probs)).sum(2)
        # or just return Q(s,a)
        return qs

    @property
    def support(self):
        """ Return the support of the Q-Value distribution. """
        return self.__support

    def get_spectral_norms(self):
        """ Return the spectral norms of layers hooked on spectral norm. """
        return {
            str(idx): layer.weight_sigma.item() for idx, layer in self.__hooked_layers
        }

    def div_grad_by_rho(self):
        """ Divide the already computed gradient by the spectral norm.
        """
        for module in self.modules():
            for _, hook in module._forward_pre_hooks.items():
                if isinstance(hook, (LinearSpectralNorm, Conv2dSpectralNorm)):
                    assert not hook._active
                    if hook._leave_smaller:
                        scale = max(module.weight_sigma.item() / hook._lipschitz_k, 1)
                    else:
                        scale = module.weight_sigma.item() / hook._lipschitz_k
                    with torch.no_grad():
                        module.weight_orig.grad.data /= scale

    def reset_parameters(self):
        """ Weight init.
        """
        init_ = (
            nn.init.xavier_uniform_
            if self.__initializer == "xavier_uniform"
            else partial(variance_scaling_uniform_, scale=1.0 / sqrt(3.0))
        )

        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                init_(module.weight)
                module.bias.data.zero_()

    @property
    def feature_extractor(self):
        """ Return the feature extractor. """
        return self.__features

    @property
    def head(self):
        """ Return the layers used as heads in Bootstrapped DQN. """
        return self.__head
