"""
Module to calculate dynamic structure factor for 2D movies

Created on Wed Jul 5 22:48:12 2017

Daniel S. Seara
www.github.com/dsseara
daniel.seara@yale.edu
"""
import numpy as np
import scipy.signal as signal


def azimuthal_average(data, center=None, binsize=1.0, mask=None, weight=None):
    """
    Calculates the azimuthal average of a 2D array

    Parameters
    ----------
    data : array_like
        2D numpy array of numerical data
    center : array_like, optional
        1x2 numpy array the center of the image from which to measure the 
        radial profile from, in units of array index. Default is center of
        data array
    binsize : scalar, optional
        radial width of each annulus over which to average,
        in units of array index
    mask : array_like, optional
        2D array same size as data with 0s where you want to exclude data.
        Default is to not use a mask

    Returns
    -------
    radial_profile : array_like
        Radially averaged 1D array from data array

    Based on radialProfile found at:
    http://www.astrobetter.com/wiki/tiki-index.php?page=python_radial_profiles
    """
    data = np.asarray(data)
    # Get all the indices in x and y direction
    [y, x] = np.indices(data.shape)

    # Define the center from which to measure the radius
    if not center:
        center = np.array([(x.max() - x.min()) / 2, (y.max() - y.min()) / 2])

    # Get distance from all points to center
    r = np.hypot(x - center[0], y - center[1])

    if mask is None:
        mask = np.ones(data.shape, dtype='bool')

    if weight is None:
        weight = np.ones(data.shape)

    # Get the bins according to binsize
    nbins = int(np.round(r.max()) / binsize) + 1
    maxbin = nbins * binsize
    bins = np.linspace(0, maxbin, nbins + 1)

    bin_centers = (bins[1:] - bins[:-1]) / 2
    values = np.histogram(r, bins)[0]  # second element returns bins themselves
    radial_profile = np.histogram(r, bins, weights=data * mask * weight)[0] \
        / np.histogram(r, bins, weights=mask * weight)[0]

    return radial_profile, values, bin_centers


def azimuthal_average_3D(data, tdim=0, center=None, binsize=1.0, mask=None,
                         weight=None):
    """
    Takes 3D data and gets radial component of last two dimensions

    Parameters
    ----------
    data : array_like
        3D numpy array, 1 dimension is time, the other two are spatial.
        User specifies which dimension in temporal
    tdim : scalar, optional
        specifies which dimension of data is temporal. Options are 0, 1, 2.
        Defaults to 0
    center : array_like, optional
        1x2 numpy array the center of the image from which to measure the
        radial profile from, in units of array index. Default is center of
        data array
    binsize : scalar, optional
        radial width of each annulus over which to average,
        in units of array index
    mask : array_like, optional
        2D array same size as data with 0s where you want to exclude data.
        Default is to not use a mask

    Returns
    -------
    tr_profile : array_like
        2D, spatially radially averaged data over time.
        First dimesion is time, second is spatial

    See also: azimuthalAverage
    TO DO
    -----

    """

    data = np.asarray(data)

    # Put temporal axis first
    data = np.rollaxis(data, tdim)

    for frame, spatial_data in enumerate(data):
        radial_profile = azimuthal_average(spatial_data, center, binsize, mask, weight)
        if frame == 0:
            tr_profile = radial_profile
        else:
            tr_profile = np.vstack((tr_profile, radial_profile))

    return tr_profile


def image2array(image):
    """
    Turns a pims Frame object into a 3D numpy array.

    Parameters
    ----------
    image = The pims Frame object, size [M,N], with P frames

    Returns
    -------
    imageArr = 3D numpy array, size [P,M,N]
    """
    startFrame = image[0].frame_no
    endFrame = image[-1].frame_no + 1
    nframes = endFrame - startFrame
    imageArr = np.zeros([nframes, image[0].shape[0], image[0].shape[1]])

    for ii in range(0, nframes):
        imageArr[ii, :, :] = image[ii]

    return imageArr


def power_spectrum(data, spacings=None, window=None, onesided=True,
                   normalize=False):
    """
    Calculates the power spectrum of a shifted array

    Parameters
    ----------
    data : array_like
        nd numpy array of which to find the power spectrum
    spacings : array_like
        array or list of physical spacings between points in data in each dimension.
        Used to return grid in fourier space. Defaults to array of ones.
    window : string, float, or tuple, optional
        Type of window to create. Same as scipy.signal.get_window()
    onesided : bool
        boolean to return one-sided power spectrum only or not. Default True
    normalize : bool
        boolean to normalize the power spectrum by the size of

    Returns
    -------
    power_spectrum : array_like
        Power spectrum of data as nd numpy array
    fourier_grid : list
        list whose nth element is a numpy array of fourier coordinates of nth
        dimension of data

    See also
    --------
    scipy.signal.get_window(),
    """

    data = np.asarray(data)

    if window is not None:
        for dim, dim_data in enumerate(data):
            n = dim_data.size
            w = signal.get_window(window, dimsize)
            _nd_window(data, w)

    if spacings is None:
        spacings = np.ones(len(data.shape))
    else:
        spacings = np.asarray(spacings)

    fourier_grid = []

    for dim_size, dim_spacing in zip(data.shape, spacings):
        fourier_grid.append(2 * np.pi * np.fft.fftshift(np.fft.fftfreq(dim_size, dim_spacing)))

    data_fft = np.fft.fftshift(np.fft.fftn(data))

    if normalize:
        power_spectrum = np.abs(data_fft / data_fft.size)**2
    else:
        power_spectrum = np.abs(data_fft)**2

    if onesided:
        power_spectrum = _one_side(power_spectrum)
        for dim, array in enumerate(fourier_grid):
            fourier_grid[dim] = _one_side(array)

    return power_spectrum, fourier_grid


def _one_side(array, axis=0, whichHalf=1):
    """
    Gives back half of an array

    Parameters
    ----------
    array : array_like
        data to cut in half
    axis : scalar, optional
        Which axis to return one side of. Default 0
    whichHalf : scalar, optional
        1 for first half of array along given axis, 2 for second half.
        Default 1

    Returns
    -------
    halfArray : array_like
        same data as array, but only half as long along given axis
    """
    length = array.shape[axis]
    halfLength = np.floor(length / 2).astype(int)
    if whichHalf == 1:
        halfArray = np.rollaxis(np.rollaxis(array, axis)[:halfLength, ...],
                                0, axis + 1)
    elif whichHalf == 2:
        halfArray = np.rollaxis(np.rollaxis(array, axis)[halfLength:, ...],
                                0, axis + 1)

    return halfArray


def _nd_window(data, window):
    """
    Performs an in-place windowing on N-dimensional spatiotemporal-domain data.
    Done to mitigate boundary effects in the FFT.
    Adapted from: https://stackoverflow.com/questions/27345861/
                  extending-1d-function-across-3-dimensions-for-data-windowing

    Parameters
    ----------
    data : array_like
        nd input data to be windowed, modified in place.
    window : array_like
        1d window function, output from scipy.signal.get_window()

    Results
    -------
    data : array_like
        in place windowed version of input array, data

    See also
    --------
    scipy.signal.get_window()
    """

    for axis, axis_size in enumerate(data.shape):
        # set up shape for numpy broadcasting
        filter_shape = [1, ] * data.ndim
        filter_shape[axis] = axis_size
        window = window.reshape(filter_shape)
        # scale the window intensities to maintain image intensity
        np.power(window, (1.0 / data.ndim), output=window)
        data *= window


def dhoModel(w, Gamma0, I0, Gamma, I, Omega):
    """
    Defines the damped harmonic oscillator model for S(q,w)/S(q)

    This is to be used at a specific q, which can then be used to build all
    the parameters as functions of q

    S(q,w)         1/2 Gamma0(q)                   Omega(q) Gamma^2(q)
    ----- = I0(q)----------------- + I(q)-------------------------------------
     S(q)        w^2 + (Gamma0/2)^2      (w^2 - Omega^2(q))^2 + (w Gamma(q))^2
    """

    return I0 * 0.5 * Gamma0 / (w**2 + (0.5 * Gamma0)**2) \
        + I * Omega * Gamma**2 / ((w**2 - Omega**2)**2 + (w * Gamma)**2)


def dhoJac(w, Gamma0, I0, Gamma, I, Omega):
    """
    Define the jacobian of the damped harmonic oscillator model for S(q,w)/S(q)

    J[0] = d/dI0
    J[1] = d/dGamma0
    J[2] = d/dI
    J[3] = d/dGamma
    J[4] = d/dOmega
    """
    J = np.zeros(5)

    J[0] = 0.5 * Gamma0 / (w**2 + (0.5 * Gamma0)**2)
    J[1] = 0.5 * I0 * (w**2 - 0.75 * Gamma0**2) / (w**2 + (0.5 * Gamma0)**2)**2
    J[2] = Omega * Gamma**2 / ((w**2 - Omega**2)**2 + (w * Gamma)**2)
    J[3] = 2 * I * Omega * Gamma * (w**2 - Omega**2)**2 \
        / ((w**2 - Omega**2)**2 + w**2 * Gamma**2)**2
    J[4] = I * Gamma**2 * \
        ((w - 2 * Omega**2) * (w - 3 * Omega**2) + w**2 * Gamma**2) \
        / ((w**2 - Omega**2)**2 + (w * Gamma)**2)**2
    return J


def objectiveFunc(data, *args):
    return data - dhoModel(*args)
