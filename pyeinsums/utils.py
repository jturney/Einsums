# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.

"""
This module contains extra utilities written in Python for the Einsums library.
"""

from __future__ import annotations

import functools

import typing
import random

from einsums import core

import numpy as np


def labeled_section(arg: typing.Union[str, typing.Callable]):
    """
    Decorates a function. Add a line to the timer report for profiling the function.

    :param arg: This decorator may take a string. This will be placed after the function name in the timer report.
    :type arg: Optional[str]

    .. versionadded:: 1.1.0
    """
    if isinstance(arg, str):

        def labeled_section_outer(func):
            @functools.wraps(func)
            def labeled_section_inner(*args, **kwargs):
                section = core.Section(f"{func.__name__} {arg}")
                retval = func(*args, **kwargs)
                del section
                return retval

            return labeled_section_inner

        return labeled_section_outer
    elif hasattr(arg, "__call__"):

        @functools.wraps(arg)
        def labeled_section_inner(*args, **kwargs):
            section = core.Section(f"{arg.__name__}")
            retval = arg(*args, **kwargs)
            del section
            return retval

        return labeled_section_inner
    else:
        raise TypeError("Argument to labeled_section not valid!")


def enumerate_many(*args, start=0):
    """
    Enumerate only takes one iterable argument, but it is sometimes useful to
    be able to enumerate multiple lists at once. This allows you to do that.
    If the iterables are of different lengths, then once one ends, it will emit
    :py:code:`None` in its variables. Once they all end, this will raise
    :py:code:`StopIteraton`.

    :param args: The iterators to iterate over.
    :return: Iterator of tuples. The last entry in the tuple is the index number. If an
    iterator ends early, then its entry will be filled with ``None``.

    .. versionadded:: 1.0.0
    """
    index = 0
    stop_cond = [False for arg in args]

    # Skip the first several indices. Basically copying the code below, but removing the yield.
    for i in range(start):
        while not all(stop_cond):
            out = tuple(index)
            for index, arg in enumerate(args):
                if stop_cond[index]:
                    out = out + tuple(None)
                else:
                    try:
                        out = out + tuple(next(arg))
                    except StopIteration:
                        stop_cond[index] = True
                        out = out + tuple(None)
            if all(stop_cond):
                return

    # Yield the outputs one by one until all the lists are exhausted.
    while not all(stop_cond):
        out = tuple(index)
        for index, arg in enumerate(args):
            if stop_cond[index]:
                out = out + tuple(None)
            else:
                try:
                    out = out + tuple(next(arg))
                except StopIteration:
                    stop_cond[index] = True
                    out = out + tuple(None)
        if all(stop_cond):
            return
        yield out
    return


class TensorIndices:
    """
    Iterator for moving through the indices of a tensor.

    .. versionadded:: 1.0.0
    """

    def __init__(self, other, **kwargs):
        if isinstance(other, (core.RuntimeTensor, core.RuntimeTensorView)):
            reverse = kwargs.reverse if "reverse" in kwargs else False
            self.__curr_index = 0 if not reverse else len(other)
            self.__strides = [0 for i in range(other.rank())]
            self.__reverse = reverse

            size = 1
            for i in reversed(range(other.rank())):
                self.__strides[i] = size
                size *= other.dim(i)
            self.__size = size
        else:
            self.__curr_index = other.__curr_index
            self.__strides = other.__strides
            self.__reverse = other.__reverse
            self.__size = other.__size

    def __iter__(self) -> TensorIndices:
        """
        Return the iterator for this iterator (itself).

        :return: The object itself.

        .. versionadded:: 1.0.0
        """
        return self

    def __reversed__(self) -> TensorIndices:
        """
        Return the iterator that goes in reverse.

        :return: The iterator that goes from the current index back to the beginning.

        .. versionadded:: 1.0.0
        """
        out = TensorIndices(self)
        out.__reverse = not self.__reverse
        return out

    def __next__(self) -> tuple[int]:
        """
        Get the next index and update the internal state.

        :return: The next index.
        :raises StopIteration: Raises this if there are no more indices.
        
        .. versionadded:: 1.0.0
        """
        if self.__curr_index < 0 or self.__curr_index >= self.__size:
            raise StopIteration

        out = [0 for i in range(len(self.__strides))]

        hold = self.__curr_index

        for i, stride in enumerate(self.__strides):
            out[i], hold = divmod(hold, stride)

        if self.__reverse:
            self.__curr_index -= 1
        else:
            self.__curr_index += 1

        return tuple(out)


__singles = [np.float32]
__doubles = [float, np.float64]
__complex_singles = [np.complex64]
__complex_doubles = [complex, np.complex128]


def create_tensor(*args, dtype=float):
    """
    Create a tensor. The arguments will be passed to the constructor.
    The data type can also be specified.

    :param args: The arguments to pass to the constructor.
    :param dtype: The data type to be stored. Can only be single or double precision real or complex floating points.

    .. versionadded:: 1.0.0
    """
    if dtype in __singles:
        return core.RuntimeTensorF(*args)
    if dtype in __doubles:
        return core.RuntimeTensorD(*args)
    if dtype in __complex_singles:
        return core.RuntimeTensorC(*args)
    if dtype in __complex_doubles:
        return core.RuntimeTensorZ(*args)
    raise ValueError(f"Can not create tensor with data type {dtype}!")


def tensor_factory(name: str, dims: list[int], dtype=float, method="einsums"):
    """
    Create either a NumPy array or an Einsums tensor. This function is mostly used in tests
    to ensure cross functionality.

    :param name: The name of the tensor. Ignored for NumPy tensors.
    :param dims: The dimensions of the tensor.
    :param dtype: The data type for the tensor.
    :param method: The kind of tensor to create. It should be either "einsums"  or "numpy'.
    :raises ValueError: if the method is not valid.

    .. versionadded:: 1.1.0
    """
    if method.lower() == "einsums":
        return create_tensor(name, dims, dtype=dtype)
    elif method.lower() == "numpy":
        return np.zeros(dims, dtype)
    else:
        raise ValueError(
            "Can only produce tensors when the method is 'einsums' or 'numpy'."
        )


def remove_complex(dtype):
    """
    Takes a datatype and gives the real equivalent.

    .. versionadded:: 1.1.0
    """
    if dtype in __complex_singles:
        return __singles[0]
    if dtype in __complex_doubles:
        return __doubles[0]
    return dtype


def is_complex(dtype):
    """
    Checks to see if the datatype is a complex datatype.

    .. versionadded:: 1.1.0
    """
    return (
        dtype in __complex_singles
        or dtype in __complex_doubles
        or isinstance(dtype, (core.RuntimeTensorC, core.RuntimeTensorZ))
        or (isinstance(dtype, str) and dtype[0] == "Z")
    )


def add_complex(dtype):
    """
    Takes a datatype and gives the complex equivalent.

    .. versionadded:: 1.1.0
    """
    if dtype in __singles:
        return __complex_singles[0]
    if dtype in __doubles:
        return __complex_doubles[0]
    return dtype


def create_random_numpy_array(dims: list[int], dtype=float):
    """
    Creates a new NumPy array and fills it with random data.

    :param dims: The dimensions of the array.
    :param dtype: The data type to store.
    :raises ValueError: if the data type is not a real or complex floating point type.

    .. versionadded:: 1.1.0
    """
    rng = np.random.default_rng()

    if dtype in __singles or dtype in __doubles:
        return rng.random(dims, dtype=dtype)
    elif dtype in __complex_singles:
        real_arr = rng.random(dims, dtype=np.float32)
        imag_arr = rng.random(dims, dtype=np.float32)

        return real_arr.astype(np.complex64) + 1.0j * imag_arr.astype(np.complex64)
    elif dtype in __complex_doubles:
        real_arr = rng.random(dims, dtype=np.float64)
        imag_arr = rng.random(dims, dtype=np.float64)

        return real_arr.astype(np.complex128) + 1.0j * imag_arr.astype(np.complex128)
    else:
        raise ValueError("The data type must be real or complex floating point.")


def create_random_tensor(name: str, dims: list[int], dtype=float):
    """
    Creates a new Einsums tensor and fills it with random data.

    :param name: The name of the tensor.
    :param dims: The dimensions of the array.
    :param dtype: The data type to store.
    :raises ValueError: if the data type is not a real or complex floating point type.

    .. versionadded:: 1.0.0
    """
    if dtype in __singles:
        return core.create_random_tensorF(name, dims)
    elif dtype in __doubles:
        return core.create_random_tensorD(name, dims)
    elif dtype in __complex_singles:
        return core.create_random_tensorC(name, dims)
    elif dtype in __complex_doubles:
        return core.create_random_tensorZ(name, dims)
    else:
        raise ValueError(f"Can not create random tensor with data type {dtype}!")


def random_tensor_factory(
    name: str, dims: list[int], dtype: type = float, method: str = "einsums"
):
    """
    Create either a NumPy array or an Einsums tensor and fills it with random data.
    This function is mostly used in tests to ensure cross functionality.

    :param name: The name of the tensor. Ignored for NumPy tensors.
    :param dims: The dimensions of the tensor.
    :param dtype: The data type for the tensor.
    :param method: The kind of tensor to create. It should be either "einsums"  or "numpy'.
    :raises ValueError: if the method is not valid.

    .. versionadded:: 1.1.0
    """
    if method == "einsums":
        return create_random_tensor(name, dims, dtype)
    elif method == "numpy":
        return create_random_numpy_array(dims, dtype)
    else:
        raise ValueError(
            "Can only produce tensors when the method is 'einsums' or 'numpy'."
        )


def create_random_definite(name: str, rows: int, mean=1.0, dtype=float):
    """
    Create a random positive definite tensor. If the ``mean`` parameter is
    negative, then this will create a random negative definite tensor.

    :param name: The name of the tensor.
    :param rows: The number of rows for the tensor. It is a square matrix, so this
    is also the number of columns.
    :param mean: The average eigenvalue. The eigenvalues will be distributed in a
    Boltzmann-Maxwell distribution to ensure that there will never be a zero eigenvalue.
    :param dtype: The data type to store.
    :return: A positive definite matrix.
    :raises ValueError: if the data type is not a real or complex floating point type.

    .. versionadded:: 1.1.0
    """
    if dtype in __singles:
        return core.create_random_definiteF(name, rows, mean)
    elif dtype in __doubles:
        return core.create_random_definiteD(name, rows, mean)
    elif dtype in __complex_singles:
        return core.create_random_definiteC(name, rows, mean)
    elif dtype in __complex_doubles:
        return core.create_random_definiteZ(name, rows, mean)
    else:
        raise ValueError(f"Can not create random tensor with data type {dtype}!")


def create_random_definite_numpy_array(rows: int, mean=1.0, dtype=float):
    """
    Create a random positive definite NumPy array. If the ``mean`` parameter is
    negative, then this will create a random negative definite tensor.

    :param rows: The number of rows for the tensor. It is a square matrix, so this
    is also the number of columns.
    :param mean: The average eigenvalue. The eigenvalues will be distributed in a
    Boltzmann-Maxwell distribution to ensure that there will never be a zero eigenvalue.
    :param dtype: The data type to store.
    :return: A positive definite matrix.
    :raises ValueError: if the data type is not a real or complex floating point type.

    .. versionadded:: 1.1.0
    """
    return np.array(create_random_definite("", rows, mean, dtype), dtype=dtype)


def random_definite_tensor_factory(
    name: str, rows: int, mean=1.0, dtype: type = float, method: str = "einsums"
):
    """
    Create a random positive definite NumPy array or Einsums tensor. If the ``mean`` parameter is
    negative, then this will create a random negative definite tensor.

    :param name: The name for the matrix.
    :param rows: The number of rows for the tensor. It is a square matrix, so this
    is also the number of columns.
    :param mean: The average eigenvalue. The eigenvalues will be distributed in a
    Boltzmann-Maxwell distribution to ensure that there will never be a zero eigenvalue.
    :param dtype: The data type to store.
    :param method: Which kind of tensor to create. Can be "einsums" or "numpy".
    :return: A positive definite matrix.
    :raises ValueError: if the method is not valid.

    .. versionadded:: 1.1.0
    """
    if method == "einsums":
        return create_random_definite(name, rows, mean, dtype)
    elif method == "numpy":
        return create_random_definite_numpy_array(rows, mean, dtype)
    else:
        raise ValueError(
            "Can only produce tensors when the method is 'einsums' or 'numpy'."
        )


def create_random_semidefinite(
    name: str, rows: int, mean=1.0, force_zeros=1, dtype=float
):
    """
    Create a random positive semidefinite tensor. If the ``mean`` parameter is
    negative, then this will create a random negative semidefinite tensor.
    The number of guaranteed zeros can be set. By default, at least one eigenvalue
    will be zero.

    :param name: The name of the tensor.
    :param rows: The number of rows for the tensor. It is a square matrix, so this
    is also the number of columns.
    :param mean: The average eigenvalue. The eigenvalues will be distributed in a
    Boltzmann-Maxwell distribution to ensure that there will never be a zero eigenvalue.
    :param force_zeros: The number of guaranteed zero eigenvalues to use.
    :param dtype: The data type to store.
    :return: A positive definite matrix.
    :raises ValueError: if the data type is not a real or complex floating point type.

    .. versionadded:: 1.1.0
    """
    if dtype in __singles:
        return core.create_random_semidefiniteF(name, rows, mean, force_zeros)
    elif dtype in __doubles:
        return core.create_random_semidefiniteD(name, rows, mean, force_zeros)
    elif dtype in __complex_singles:
        return core.create_random_semidefiniteC(name, rows, mean, force_zeros)
    elif dtype in __complex_doubles:
        return core.create_random_semidefiniteZ(name, rows, mean, force_zeros)
    else:
        raise ValueError(f"Can not create random tensor with data type {dtype}!")


def create_random_semidefinite_numpy_array(
    rows: int, mean=1.0, force_zeros=1, dtype=float
):
    """
    Create a random positive semidefinite NumPy array. If the ``mean`` parameter is
    negative, then this will create a random negative semidefinite tensor.
    The number of guaranteed zeros can be set. By default, at least one eigenvalue
    will be zero.

    :param rows: The number of rows for the tensor. It is a square matrix, so this
    is also the number of columns.
    :param mean: The average eigenvalue. The eigenvalues will be distributed in a
    Boltzmann-Maxwell distribution to ensure that there will never be a zero eigenvalue.
    :param force_zeros: The number of guaranteed zero eigenvalues to use.
    :param dtype: The data type to store.
    :return: A positive definite matrix.
    :raises ValueError: if the data type is not a real or complex floating point type.

    .. versionadded:: 1.1.0
    """
    return np.array(
        create_random_semidefinite("", rows, mean, force_zeros, dtype), dtype=dtype
    )


def random_semidefinite_tensor_factory(
    name: str,
    rows: int,
    mean=1.0,
    force_zeros=1,
    dtype: type = float,
    method: str = "einsums",
):
    """
    Create a random positive semidefinite NumPy array or Einsums tensor. If the ``mean`` parameter is
    negative, then this will create a random negative demidefinite tensor.
    The number of guaranteed zeros can be set. By default, at least one eigenvalue
    will be zero.

    :param name: The name for the matrix.
    :param rows: The number of rows for the tensor. It is a square matrix, so this
    is also the number of columns.
    :param mean: The average eigenvalue. The eigenvalues will be distributed in a
    Boltzmann-Maxwell distribution to ensure that there will never be a zero eigenvalue.
    :param force_zeros: The number of guaranteed zero eigenvalues to use.
    :param dtype: The data type to store.
    :param method: Which kind of tensor to create. Can be "einsums" or "numpy".
    :return: A positive definite matrix.
    :raises ValueError: if the method is not valid.

    .. versionadded:: 1.1.0
    """
    if method == "einsums":
        return create_random_semidefinite(name, rows, mean, force_zeros, dtype)
    elif method == "numpy":
        return create_random_semidefinite_numpy_array(rows, mean, force_zeros, dtype)
    else:
        raise ValueError(
            "Can only produce tensors when the method is 'einsums' or 'numpy'."
        )
