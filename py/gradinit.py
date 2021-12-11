from torch import nn, Tensor, tensor
import types

FACTOR_PARAM_NAME = 'factor'

def _wrap_layer(module: nn.Module):
    old_forward = module.forward
    factor = nn.parameter.Parameter(
        tensor(1., requires_grad=True,
        dtype=module.weight.dtype,
        device=module.weight.device))
    module.register_parameter(FACTOR_PARAM_NAME, factor)
    module.weight.requires_grad = False
    if hasattr(module, 'bias'):
        module.bias.requires_grad = False
    def forward(self, input: Tensor):
        return factor * old_forward(input)
    module.forward = types.MethodType(forward, module)
    def unwrap():
        factor = module._parameters.pop(FACTOR_PARAM_NAME).detach()
        module.weight.multiply_(factor)
        module.weight.requires_grad = True
        if hasattr(module, 'bias'):
            module.bias.multiply_(factor)
            module.bias.requires_grad = True
        module.forward = old_forward
    return unwrap

WRAPPERS = {
    nn.Linear: _wrap_layer,
    nn.Bilinear: _wrap_layer,
    nn.Conv1d: _wrap_layer,
    nn.Conv2d: _wrap_layer,
    nn.Conv3d: _wrap_layer,
    nn.ConvTranspose1d: _wrap_layer,
    nn.ConvTranspose2d: _wrap_layer,
    nn.ConvTranspose3d: _wrap_layer,
    nn.BatchNorm1d: _wrap_layer,
    nn.BatchNorm2d: _wrap_layer,
    nn.BatchNorm3d: _wrap_layer
    }

class GradInit:
    def __init__(self, module: nn.Module):
        self._module = module
        self._unwrap = None

    def __enter__(self):
        self._unwrap = [
            WRAPPERS[type(submodule)](submodule) 
            for submodule in self._module.modules()
            if type(submodule) in WRAPPERS]
        return self._module

    def __exit__(self, type, value, traceback):
        self._do_unwrap()

    def __del__(self):
        self._do_unwrap()

    def _do_unwrap(self):
        if self._unwrap is not None:
            for unwrap in self._unwrap:
                unwrap()
            self._unwrap = None
