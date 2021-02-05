""" Linear spectral normalisation with tracking.
"""

import torch
from torch.nn.functional import normalize
from torch.nn.utils.spectral_norm import (
    SpectralNorm,
    SpectralNormStateDictHook,
    SpectralNormLoadStateDictPreHook,
)


class LinearSpectralNorm(SpectralNorm):
    def __init__(
        self,
        *args,
        active=True,
        leave_smaller=False,
        lipschitz_k=1,
        flow_through_norm=True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._active = bool(active)
        self._leave_smaller = bool(leave_smaller)  # do not touch the children, pervert!
        self._lipschitz_k = float(lipschitz_k)
        self._flow_through_norm = bool(flow_through_norm)

    def compute_weight(self, module, do_power_iteration):
        weight = getattr(module, self.name + "_orig")
        u = getattr(module, self.name + "_u")
        v = getattr(module, self.name + "_v")
        sigma = getattr(module, self.name + "_sigma")
        weight_mat = self.reshape_weight_to_matrix(weight)

        if do_power_iteration:
            with torch.no_grad():
                for _ in range(self.n_power_iterations):
                    # Spectral norm of weight equals to `u^T W v`, where `u` and `v`
                    # are the first left and right singular vectors.
                    # This power iteration produces approximations of `u` and `v`.
                    v = normalize(
                        torch.mv(weight_mat.t(), u), dim=0, eps=self.eps, out=v
                    )
                    u = normalize(torch.mv(weight_mat, v), dim=0, eps=self.eps, out=u)
                if self.n_power_iterations > 0:
                    # See above on why we need to clone
                    u = u.clone(memory_format=torch.contiguous_format)
                    v = v.clone(memory_format=torch.contiguous_format)

        _sigma = torch.dot(u, torch.mv(weight_mat, v))
        sigma.copy_(_sigma.data)

        if not self._flow_through_norm:
            _sigma = _sigma.detach()

        if self._active:
            if self._leave_smaller:
                weight = weight / max(_sigma / self._lipschitz_k, 1)
            else:
                weight = weight / (_sigma / self._lipschitz_k)
        else:
            weight = weight + 0

        return weight

    @staticmethod
    def apply(  # pylint: disable=bad-continuation,arguments-differ
        module,
        name,
        n_power_iterations,
        dim,
        eps,
        active,
        leave_smaller=False,
        lipschitz_k=1,
        flow_through_norm=True,
    ):
        for _k, hook in module._forward_pre_hooks.items():
            if isinstance(hook, SpectralNorm) and hook.name == name:
                raise RuntimeError(
                    "Cannot register two spectral_norm hooks on "
                    "the same parameter {}".format(name)
                )

        fn = LinearSpectralNorm(
            name,
            n_power_iterations,
            dim,
            eps,
            active=active,
            leave_smaller=leave_smaller,
            lipschitz_k=lipschitz_k,
            flow_through_norm=flow_through_norm,
        )
        weight = module._parameters[name]

        with torch.no_grad():
            weight_mat = fn.reshape_weight_to_matrix(weight)

            h, w = weight_mat.size()
            # randomly initialize `u` and `v`
            u = normalize(weight.new_empty(h).normal_(0, 1), dim=0, eps=fn.eps)
            v = normalize(weight.new_empty(w).normal_(0, 1), dim=0, eps=fn.eps)

        delattr(module, fn.name)
        module.register_parameter(fn.name + "_orig", weight)
        # We still need to assign weight back as fn.name because all sorts of
        # things may assume that it exists, e.g., when initializing weights.
        # However, we can't directly assign as it could be an nn.Parameter and
        # gets added as a parameter. Instead, we register weight.data as a plain
        # attribute.
        setattr(module, fn.name, weight.data)
        module.register_buffer(fn.name + "_u", u)
        module.register_buffer(fn.name + "_v", v)
        module.register_buffer(
            fn.name + "_sigma", torch.dot(u, torch.mv(weight_mat, v).detach())
        )
        # setattr(module, fn.name + "_sigma", fn.eps)

        module.register_forward_pre_hook(fn)
        module._register_state_dict_hook(SpectralNormStateDictHook(fn))
        module._register_load_state_dict_pre_hook(SpectralNormLoadStateDictPreHook(fn))
        return fn


def spectral_norm(
    module,
    name="weight",
    n_power_iterations=1,
    eps=1e-12,
    dim=None,
    active=True,
    leave_smaller=False,
    lipschitz_k=1,
    flow_through_norm=True,
):
    r"""Applies spectral normalization to a parameter in the given module.

    .. math::
        \mathbf{W}_{SN} = \dfrac{\mathbf{W}}{\sigma(\mathbf{W})},
        \sigma(\mathbf{W}) = \max_{\mathbf{h}: \mathbf{h} \ne 0} \dfrac{\|\mathbf{W} \mathbf{h}\|_2}{\|\mathbf{h}\|_2}

    Spectral normalization stabilizes the training of discriminators (critics)
    in Generative Adversarial Networks (GANs) by rescaling the weight tensor
    with spectral norm :math:`\sigma` of the weight matrix calculated using
    power iteration method. If the dimension of the weight tensor is greater
    than 2, it is reshaped to 2D in power iteration method to get spectral
    norm. This is implemented via a hook that calculates spectral norm and
    rescales weight before every :meth:`~Module.forward` call.

    See `Spectral Normalization for Generative Adversarial Networks`_ .

    .. _`Spectral Normalization for Generative Adversarial Networks`: https://arxiv.org/abs/1802.05957

    Args:
        module (nn.Module): containing module
        name (str, optional): name of weight parameter
        n_power_iterations (int, optional): number of power iterations to
            calculate spectral norm
        eps (float, optional): epsilon for numerical stability in
            calculating norms
        dim (int, optional): dimension corresponding to number of outputs,
            the default is ``0``, except for modules that are instances of
            ConvTranspose{1,2,3}d, when it is ``1``

    Returns:
        The original module with the spectral norm hook

    Example::

        >>> m = spectral_norm(nn.Linear(20, 40))
        >>> m
        Linear(in_features=20, out_features=40, bias=True)
        >>> m.weight_u.size()
        torch.Size([40])

    """
    if dim is None:
        if isinstance(
            module,
            (
                torch.nn.ConvTranspose1d,
                torch.nn.ConvTranspose2d,
                torch.nn.ConvTranspose3d,
            ),
        ):
            dim = 1
        else:
            dim = 0
    LinearSpectralNorm.apply(
        module,
        name,
        n_power_iterations,
        dim,
        eps,
        active,
        leave_smaller=leave_smaller,
        lipschitz_k=lipschitz_k,
        flow_through_norm=flow_through_norm,
    )
    return module
