from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar, Any
from numba import njit as _njit
from numba import prange
import numpy as np  # come back to this

from .tensor_data import (
    MAX_DIMS,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)
from .tensor_ops import MapProto, TensorOps

if TYPE_CHECKING:
    from typing import Callable, Optional

    from .tensor import Tensor
    from .tensor_data import Shape, Storage, Strides

# TIP: Use `NUMBA_DISABLE_JIT=1 pytest tests/ -m task3_1` to run these tests without JIT.

# This code will JIT compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.
Fn = TypeVar("Fn")


def njit(fn: Fn, **kwargs: Any) -> Fn:
    """Decorator to JIT compile functions with `njit`."""
    return _njit(inline="always", **kwargs)(fn)  # type: ignore


to_index = njit(to_index)
index_to_position = njit(index_to_position)
broadcast_index = njit(broadcast_index)


class FastOps(TensorOps):
    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """See `tensor_ops.py`"""
        # This line JIT compiles your tensor_map
        f = tensor_map(njit(fn))

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)
            f(*out.tuple(), *a.tuple())
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        """See `tensor_ops.py`"""
        f = tensor_zip(njit(fn))

        def ret(a: Tensor, b: Tensor) -> Tensor:
            c_shape = shape_broadcast(a.shape, b.shape)  # type: ignore
            out = a.zeros(c_shape)
            f(*out.tuple(), *a.tuple(), *b.tuple())
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        """See `tensor_ops.py`"""
        f = tensor_reduce(njit(fn))

        def ret(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = 1

            # Other values when not sum.
            out = a.zeros(tuple(out_shape))
            out._tensor._storage[:] = start

            f(*out.tuple(), *a.tuple(), dim)
            return out

        return ret

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        """Batched tensor matrix multiply ::

            for n:
              for i:
                for j:
                  for k:
                    out[n, i, j] += a[n, i, k] * b[n, k, j]

        Where n indicates an optional broadcasted batched dimension.

        Should work for tensor shapes of 3 dims ::

            assert a.shape[-1] == b.shape[-2]

        Args:
        ----
            a : tensor data a
            b : tensor data b

        Returns:
        -------
            New tensor data

        """
        # Make these always be a 3 dimensional multiply
        both_2d = 0
        if len(a.shape) == 2:
            a = a.contiguous().view(1, a.shape[0], a.shape[1])
            both_2d += 1
        if len(b.shape) == 2:
            b = b.contiguous().view(1, b.shape[0], b.shape[1])
            both_2d += 1
        both_2d = both_2d == 2

        ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))  # type: ignore
        ls.append(a.shape[-2])
        ls.append(b.shape[-1])
        assert a.shape[-1] == b.shape[-2]
        out = a.zeros(tuple(ls))

        tensor_matrix_multiply(*out.tuple(), *a.tuple(), *b.tuple())

        # Undo 3d if we added it.
        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        return out


# Implementations


def tensor_map(
    fn: Callable[[float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """NUMBA low_level tensor_map function. See tensor_ops.py for description.

    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * When out and in are stride-aligned, avoid indexing

    Args:
    ----
        fn: function mappings floats-to-floats to apply.

    Returns:
    -------
        Tensor map function.

    """

    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        # Check if strides and shapes are aligned
        if np.array_equal(out_strides, in_strides) and np.array_equal(
            out_shape, in_shape
        ):
            # If aligned, apply the function directly in parallel
            for i in prange(len(out)):
                out[i] = fn(in_storage[i])
        else:
            # Handle misaligned strides/shapes with broadcasting and indexing
            for i in prange(len(out)):
                out_index = np.empty(MAX_DIMS, dtype=np.int32)
                in_index = np.empty(MAX_DIMS, dtype=np.int32)

                to_index(i, out_shape, out_index)
                broadcast_index(out_index, out_shape, in_shape, in_index)

                out_position = index_to_position(out_index, out_strides)
                in_position = index_to_position(in_index, in_strides)

                # Apply the function and store the result
                out[out_position] = fn(in_storage[in_position])

    return njit(_map, parallel=True)  # type: ignore


def tensor_zip(
    fn: Callable[[float, float], float],
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:
    """NUMBA higher-order tensor zip function. See tensor_ops.py for description.

    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * When out, a, b are stride-aligned, avoid indexing

    Args:
    ----
        fn: function maps two floats to float to apply.

    Returns:
    -------
        Tensor zip function.

    """

    def _zip(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:
        # Check if all strides and shapes are aligned
        strides_aligned = (
            np.array_equal(out_strides, a_strides)
            and np.array_equal(out_strides, b_strides)
            and np.array_equal(out_shape, a_shape)
            and np.array_equal(out_shape, b_shape)
        )

        if strides_aligned:
            # If aligned, apply the function directly in parallel
            for i in prange(len(out)):
                out[i] = fn(a_storage[i], b_storage[i])
        else:
            # Handle misaligned strides/shapes with broadcasting and indexing
            for i in prange(len(out)):
                # Allocate index buffers
                out_index = np.empty(MAX_DIMS, dtype=np.int32)
                a_index = np.empty(MAX_DIMS, dtype=np.int32)
                b_index = np.empty(MAX_DIMS, dtype=np.int32)

                # Compute output, a, and b indices
                to_index(i, out_shape, out_index)
                broadcast_index(out_index, out_shape, a_shape, a_index)
                broadcast_index(out_index, out_shape, b_shape, b_index)

                # Compute positions based on strides
                out_position = index_to_position(out_index, out_strides)
                a_position = index_to_position(a_index, a_strides)
                b_position = index_to_position(b_index, b_strides)

                # Apply the function and store the result
                out[out_position] = fn(a_storage[a_position], b_storage[b_position])

    return njit(_zip, parallel=True)  # type: ignore


def tensor_reduce(
    fn: Callable[[float, float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """NUMBA higher-order tensor reduce function. See `tensor_ops.py` for description.

    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * Inner-loop should not call any functions or write non-local variables

    Args:
    ----
        fn: reduction function mapping two floats to float.

    Returns:
    -------
        Tensor reduce function

    """

    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
    ) -> None:
        for i in prange(len(out)):
            # Allocate index buffers
            out_index = np.empty(MAX_DIMS, dtype=np.int32)
            a_index = np.empty(MAX_DIMS, dtype=np.int32)

            # Compute initial indices and positions
            to_index(i, out_shape, out_index)
            broadcast_index(out_index, out_shape, a_shape, a_index)
            out_position = index_to_position(out_index, out_strides)
            a_position = index_to_position(a_index, a_strides)

            # Initialize the output with the first element
            out[out_position] = a_storage[a_position]

            # Reduce along the specified dimension
            for j in range(1, a_shape[reduce_dim]):
                a_index[reduce_dim] = j
                a_position = index_to_position(a_index, a_strides)
                out[out_position] = fn(out[out_position], a_storage[a_position])

    return njit(_reduce, parallel=True)


def _tensor_matrix_multiply(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    a_storage: Storage,
    a_shape: Shape,
    a_strides: Strides,
    b_storage: Storage,
    b_shape: Shape,
    b_strides: Strides,
) -> None:
    """NUMBA tensor matrix multiply function."""
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0

    for batch in prange(out_shape[0]):
        a_offset = batch * a_batch_stride
        b_offset = batch * b_batch_stride

        for row in range(a_shape[1]):
            for col in range(b_shape[2]):
                # Compute the dot product for the current row and column
                dot_product = 0.0
                for k in range(a_shape[2]):
                    a_index = a_offset + row * a_strides[1] + k * a_strides[2]
                    b_index = b_offset + k * b_strides[1] + col * b_strides[2]
                    dot_product += a_storage[a_index] * b_storage[b_index]

                # Store the result in the output tensor
                out_index = (
                    batch * out_strides[0] + row * out_strides[1] + col * out_strides[2]
                )
                out[out_index] = dot_product


tensor_matrix_multiply = njit(_tensor_matrix_multiply, parallel=True)
assert tensor_matrix_multiply is not None
