from functools import update_wrapper
from numbers import Number
import torch
import torch.nn.functional as F


def broadcast_all(*values):
    r"""
    Given a list of values (possibly containing numbers), returns a list where each
    value is broadcasted based on the following rules:
      - `torch.*Tensor` instances are broadcasted as per :ref:`_broadcasting-semantics`.
      - numbers.Number instances (scalars) are upcast to tensors having
        the same size and type as the first tensor passed to `values`.  If all the
        values are scalars, then they are upcasted to scalar Tensors.
    Args:
        values (list of `numbers.Number` or `torch.*Tensor`)
    Raises:
        ValueError: if any of the values is not a `numbers.Number` or
            `torch.*Tensor` instance
    """
    if not all(torch.is_tensor(v) or isinstance(v, Number) for v in values):
        raise ValueError('Input arguments must all be instances of numbers.Number or torch.tensor.')
    if not all(map(torch.is_tensor, values)):
        options = dict(dtype=torch.get_default_dtype())
        for value in values:
            if torch.is_tensor(value):
                options = dict(dtype=value.dtype, device=value.device)
                break
        values = [v if torch.is_tensor(v) else torch.tensor(v, **options)
                  for v in values]
    return torch.broadcast_tensors(*values)


def _standard_normal(shape, dtype, device):
    if torch._C._get_tracing_state():
        # [JIT WORKAROUND] lack of support for .normal_()
        return torch.normal(torch.zeros(shape, dtype=dtype, device=device),
                            torch.ones(shape, dtype=dtype, device=device))
    return torch.empty(shape, dtype=dtype, device=device).normal_()
