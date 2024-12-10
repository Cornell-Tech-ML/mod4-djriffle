from typing import Tuple

from . import operators
from .autodiff import Context
from .fast_ops import FastOps
from .tensor import Tensor
from .tensor_functions import Function, rand


# List of functions in this file:
# - avgpool2d: Tiled average pooling 2D
# - argmax: Compute the argmax as a 1-hot tensor
# - Max: New Function for max operator
# - max: Apply max reduction
# - softmax: Compute the softmax as a tensor
# - logsoftmax: Compute the log of the softmax as a tensor - See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
# - maxpool2d: Tiled max pooling 2D
# - dropout: Dropout positions based on random noise, include an argument to turn off


def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """Reshape an image tensor for 2D pooling

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.

    """
    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0

    new_height = height // kh
    new_width = width // kw

    reshaped = input.contiguous().view(batch, channel, new_height, kh, new_width, kw)

    output = reshaped.permute(0, 1, 2, 4, 3, 5).contiguous()
    output = output.view(batch, channel, new_height, new_width, kh * kw)

    return output, new_height, new_width


def avgpool2d(tensor: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Applies 2D average pooling over an input signal composed of several input planes.

    Args:
    ----
        tensor: Input tensor of shape (batch x channel x height x width)
        kernel: Tuple of (kernel_height, kernel_width) specifying size of pooling region

    Returns:
    -------
        Tensor of shape (batch x channel x new_height x new_width) where new_height and
        new_width are determined by the kernel size

    """
    output, new_height, new_width = tile(tensor, kernel)
    return (
        output.mean(4)
        .contiguous()
        .view(output.shape[0], output.shape[1], new_height, new_width)
    )


fastmax = FastOps.reduce(operators.max, -float("inf"))


def argmax(tensor: Tensor, dim: int) -> Tensor:
    """Applies 2D average pooling to an input signal composed of multiple input planes.

    Args:
        tensor (Tensor): A tensor of shape [batch, channel, height, width], representing the input data.
        kernel (Tuple[int, int]): A tuple specifying the pooling region size as (kernel_height, kernel_width).
        dim (int): The dimension along which to compute the maximum values.

    Returns:
        Tensor: A tensor of shape [batch, channel, new_height, new_width], where `new_height` and `new_width`
        are determined by the pooling kernel size.

    """
    max = fastmax(tensor, dim)
    return max == tensor


class Max(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, dim: Tensor) -> Tensor:
        """Forward pass for the max operation.

        Args:
            ctx: A context object used to save values required for the backward pass.
            a (Tensor): The input tensor over which to find maximum values.
            dim (int): The dimension along which to compute the maximum values.

        Returns:
            Tensor: A tensor containing the maximum values along the specified dimension.

        """
        ctx.save_for_backward(a, int(dim.item()))
        return fastmax(a, int(dim.item()))

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Backward pass for the max operation.

        Args:
            ctx: A context object containing saved tensors from the forward pass.
            grad_output (Tensor): The gradient of the loss with respect to the output tensor.

        Returns:
            Tuple[Tensor, float]: 
                - The gradient of the loss with respect to the input tensor.
                - The gradient of the loss with respect to the dimension (always 0.0).

        """
        input, dim = ctx.saved_values
        return grad_output * argmax(input, dim), 0.0


def max(tensor: Tensor, dim: int) -> Tensor:
    """Returns the maximum values along a specified dimension.

    Args:
        tensor (Tensor): The input tensor from which to find maximum values.
        dim (int): The dimension along which to compute the maximum values.

    Returns:
        Tensor: A tensor containing the maximum values along the specified dimension.

    """
    return Max.apply(tensor, tensor._ensure_tensor(dim))


def softmax(tensor: Tensor, dim: int) -> Tensor:
    """Applies the softmax function to the input tensor along a specified dimension.

    Args:
        tensor (Tensor): The input tensor on which to apply the softmax function.
        dim (int): The dimension along which to compute the softmax.

    Returns:
        Tensor: A tensor with the softmax function applied along the specified dimension.
        
    """
    return tensor.exp() / tensor.exp().sum(dim)


def logsoftmax(tensor: Tensor, dim: int) -> Tensor:
    """Applies the log softmax function to the input tensor along a specified dimension.

    Args:
        tensor (Tensor): The input tensor on which to apply the log softmax function.
        dim (int): The dimension along which to compute the log softmax.

    Returns:
        Tensor: A tensor with the log softmax function applied along the specified dimension.

    """
    return softmax(tensor, dim).log()


def maxpool2d(tensor: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Applies 2D max pooling to the input tensor.

    Args:
        tensor (Tensor): A tensor of shape [batch, channel, height, width], representing the input data.
        kernel (Tuple[int, int]): A tuple specifying the pooling window size as (kernel_height, kernel_width).

    Returns:
        Tensor: A tensor with max pooling applied, of shape [batch, channel, height // kernel_height, width // kernel_width].

    """
    batch, channel, _, _ = tensor.shape
    tiled, new_height, new_width = tile(tensor, kernel)

    return max(tiled, dim=4).contiguous().view(batch, channel, new_height, new_width)


def dropout(tensor: Tensor, p: float, ignore: bool = False) -> Tensor:
    """Applies dropout to the input tensor during training.

    This operation randomly zeroes out some elements of the input tensor with a probability `p`, using samples from a Bernoulli distribution. Each channel is zeroed out independently during every forward call.

    Args:
        tensor (Tensor): The input tensor to which dropout is applied.
        p (float): The probability of an element being zeroed out. Must be between 0 and 1.
        ignore (bool): If True, ignores dropout and returns the input tensor unchanged.

    Returns:
        Tensor: The input tensor with dropout applied.
        
    """
    if ignore or p == 0.0:
        return tensor
    
    if p == 1.0:
        # drop out everything
        return tensor.zeros()
    
    mask = rand(tensor.shape) > p
    return tensor * mask