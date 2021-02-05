""" Implements Spectral Normalization of Conv2d layers, following
    https://arxiv.org/abs/1804.04368 and the PyTorch implementation of
    torch.nn.utils.spectral_norm
"""
import torch
import torch.nn.functional as F
from torch.nn.utils.spectral_norm import (
    SpectralNorm,
    SpectralNormLoadStateDictPreHook,
    SpectralNormStateDictHook,
)


__all__ = ["spectral_norm_conv2d"]


class Conv2dSpectralNorm:
    """ Approximates spectral norm of kernel weights using power itteration. """

    _version = 1

    def __init__(  # pylint: disable=bad-continuation
        self,
        name,
        n_power_iterations=1,
        eps=1e-12,
        active=True,
        leave_smaller=False,
        lipschitz_k=1,
        flow_through_norm=True,
    ):
        assert n_power_iterations >= 0, "n_power_iterations should be positive."
        self.name = name
        self.n_power_iterations = n_power_iterations
        self.eps = eps
        self._active = bool(active)
        self._leave_smaller = bool(leave_smaller)  # do not touch the children, pervert!
        self._lipschitz_k = float(lipschitz_k)
        self._flow_through_norm = bool(flow_through_norm)

        if self._flow_through_norm:
            raise NotImplementedError("We don't have it yet.")

    def __call__(self, module, inputs):
        # eigenvetors u and v are lazy initialized.
        if getattr(module, f"{self.name}_u").ndim == 0:
            self._set_eigenvectors(module, inputs)

        # normalize weights and set them in-place.
        setattr(
            module,
            self.name,
            self.compute_weight(module, do_power_iteration=module.training),
        )

    def compute_weight(self, module, do_power_iteration):
        r"""Where the deed is done.
        """
        A = getattr(module, self.name + "_orig")  # this is the kernel
        u = getattr(module, self.name + "_u")  # left eigenvector
        v = getattr(module, self.name + "_v")  # right eigenvector
        sigma = getattr(module, self.name + "_sigma")  # sigma, of course
        eps = torch.tensor(self.eps, device=A.device)
        stride = module.stride
        padding = module.padding
        dilation = module.dilation

        if do_power_iteration:
            with torch.no_grad():
                for _ in range(self.n_power_iterations):
                    v_ = F.conv2d(
                        u, A, stride=stride, padding=padding, dilation=dilation
                    )
                    beta = torch.max(v_.norm(), eps)
                    v = torch.div(v_, beta, out=v)

                    u_ = F.conv_transpose2d(
                        v, A, stride=stride, padding=padding, dilation=dilation
                    )

                    # this is the largest eigenvalue
                    sigma.copy_(torch.max(u_.norm(), eps))
                    u = torch.div(u_, sigma, out=u)

                    # See above on why we need to clone
                    if self.n_power_iterations > 0:
                        u = u.clone(memory_format=torch.contiguous_format)
                        v = v.clone(memory_format=torch.contiguous_format)

        if self._active:
            if self._leave_smaller:
                A = A / max(sigma.item() / self._lipschitz_k, 1)
            else:
                A = A / (sigma.item() / self._lipschitz_k)
        else:
            A = A + 0

        return A

    @torch.no_grad()
    def _set_eigenvectors(self, module, inputs):
        """ This is called once, if the `u` and `v` buffers have not been
            yet registered. We don't do this in the `apply` static method
            because we need forward time information such as input size.
        """
        w = module._parameters[f"{self.name}_orig"]
        u_shape = torch.Size((1, *inputs[0].shape[-3:]))  # eg.: (1, 1, 84, 84)
        u = F.normalize(w.new_empty(u_shape).normal_(0, 1), dim=0, eps=self.eps)

        # we do this only to get the shape.
        v_shape = F.conv2d(u, w, stride=module.stride).shape
        v = F.normalize(w.new_empty(v_shape).normal_(0, 1), dim=0, eps=self.eps)

        sigma = getattr(module, self.name + "_sigma")  # right eigenvector
        sigma.copy_(torch.max(u.norm(), torch.full_like(u.norm(), self.eps)))

        # set the buffers
        getattr(module, self.name + "_u").resize_as_(u).copy_(u)
        getattr(module, self.name + "_v").resize_as_(v).copy_(v)

    @staticmethod
    def apply(
        module,
        name,
        n_power_iterations,
        eps,
        active,
        leave_smaller=False,
        lipschitz_k=1,
        flow_through_norm=False,
    ):
        r"""Because the normalization is dependent of the input size, we
        lazy initialize some of the objects.
        """

        for _, hook in module._forward_pre_hooks.items():
            if (
                isinstance(hook, (SpectralNorm, Conv2dSpectralNorm))
                and hook.name == name
            ):
                raise RuntimeError(
                    "Cannot register two spectral_norm hooks on "
                    "the same parameter {}".format(name)
                )
        fn = Conv2dSpectralNorm(
            name,
            n_power_iterations,
            eps,
            active=active,
            leave_smaller=leave_smaller,
            lipschitz_k=lipschitz_k,
            flow_through_norm=flow_through_norm,
        )
        weight = module._parameters[name]

        # we register the Parameters to a new attribute
        delattr(module, fn.name)
        module.register_parameter(f"{fn.name}_orig", weight)
        module.register_buffer(name + "_sigma", torch.ones_like(weight.sum()))
        module.register_buffer(name + "_u", weight.new_empty(()))
        module.register_buffer(name + "_v", weight.new_empty(()))

        # we keep the old attribute around, but as a simple tensor
        # this is required because other torch stuff assumes it exists.
        setattr(module, fn.name, weight.data)
        # setattr(module, fn.name + "_sigma", fn.eps)

        # hooks
        module.register_forward_pre_hook(fn)
        module._register_state_dict_hook(SpectralNormStateDictHook(fn))
        module._register_load_state_dict_pre_hook(SpectralNormLoadStateDictPreHook(fn))
        return fn

    def remove(self, module):
        weight = self.compute_weight(module, do_power_iteration=False)
        delattr(module, self.name)
        delattr(module, self.name + "_u")
        delattr(module, self.name + "_v")
        delattr(module, self.name + "_orig")
        delattr(module, self.name + "_sigma")
        module.register_parameter(self.name, torch.nn.Parameter(weight.detach()))


def spectral_norm_conv2d(  # pylint: disable=bad-continuation
    module,
    name="weight",
    n_power_iterations=1,
    eps=1e-12,
    active=True,
    leave_smaller=False,
    lipschitz_k=1,
    flow_through_norm=False,
):
    r"""Applies spectral normalization to parameters of Conv2d modules.

    It is still doing SVD power itteration, but considers the special
    structure of the convolution operator. In effect, this implementation
    approximates the spectral norm of the doubly block circulant matrix.

    Example::
        >>> TODO
    """

    Conv2dSpectralNorm.apply(
        module,
        name,
        n_power_iterations,
        eps,
        active,
        leave_smaller=leave_smaller,
        lipschitz_k=lipschitz_k,
        flow_through_norm=flow_through_norm,
    )
    return module


def main():
    device = torch.device("cpu")
    conv0 = torch.nn.Conv2d(1, 2, 3, stride=1)
    conv0 = torch.nn.utils.spectral_norm(conv0)
    conv0.to(device)

    for _ in range(10):
        z = conv0(torch.rand(1, 1, 7, 7).to(device))

    conv1 = torch.nn.Conv2d(1, 2, 3, stride=1)
    conv1 = torch.nn.utils.spectral_norm(conv1)
    conv1.to(device)
    z = conv1(torch.rand(1, 1, 7, 7).to(device))

    state_dict = conv0.state_dict()
    print(state_dict.keys())
    conv1.load_state_dict(state_dict)

    # assert torch.all(conv0.weight.eq(conv1.weight))
    assert torch.all(conv0.weight_orig.eq(conv1.weight_orig))
    assert torch.all(conv0.weight_u.eq(conv1.weight_u))
    assert torch.all(conv0.weight_v.eq(conv1.weight_v))


if __name__ == "__main__":
    main()
