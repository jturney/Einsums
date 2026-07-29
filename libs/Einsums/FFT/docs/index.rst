..
    Copyright (c) The Einsums Developers. All rights reserved.
    Licensed under the MIT License. See LICENSE.txt in the project root for license information.

.. _modules_Einsums_FFT:

===
FFT
===

This module contains wrappers for performing the fast Fourier transform.

See the :ref:`API reference <modules_Einsums_FFT_api>` of this module for more
details.

Public API
----------

- :cpp:func:`~einsums::fft::fft` - performs the fast Fourier transform on the input tensor using the linked FFT library.
- :cpp:func:`~einsums::fft::ifft` - performs the inverse fast Fourier transform on the input tensor using the linked FFT library.
- :cpp:func:`~einsums::fft::fftfreq` - gets the frequency corresponding to each position in a frequency tensor.