"""
Module to calculate dynamic structure factor for 2D movies

Created on Wed Jul 5 22:48:12 2017

Daniel S. Seara
www.github.com/dsseara
daniel.seara@yale.edu
"""
import numpy as np


def azimuthalAverage(image, center=None, binsize=1.0, mask=None, weight=None,
                     returnAll=False):
    """
    Calculates the azimuthal average of a 2D array

    INPUT
    ------
    image = the 2D array
    center = the center of the image from which to measure
             the radial profile from. If center = None, uses
             the center of the 2D array itself
    binsize = radial width of each annulus over which to average,
              given in units of array index
    mask = 2D array same size as image with 0s where you want to

    OUTPUT
    ------
    gr = radially averaged 1D array

    Based on radialProfile found at:
    http://www.astrobetter.com/wiki/tiki-index.php?page=python_radial_profiles

    TO DO
    ------
    - Let user specify binsize
    """
    # Get all the indices in x and y direction
    [y, x] = np.indices(image.shape)

    # Define the center from which to measure the radius
    if not center:
        center = np.array([(x.max() - x.min()) / 2, (y.max() - y.min()) / 2])

    # Get distance from all points to center
    r = np.hypot(x - center[0], y - center[1])

    if mask is None:
        mask = np.ones(image.shape, dtype='bool')

    if weight is None:
        weight = np.ones(image.shape)

    # Get the bins according to binsize
    nbins = int(np.round(r.max()) / binsize) + 1
    maxbin = nbins * binsize
    bins = np.linspace(0, maxbin, nbins + 1)

    binCenters = (bins[1:] - bins[:-1]) / 2
    nr = np.histogram(r, bins)[0]  # second element returns bins themselves
    gr = np.histogram(r, bins, weights=image * mask * weight)[0] \
        / np.histogram(r, bins, weights=mask * weight)[0]

    if returnAll:
        return binCenters, nr, gr
    else:
        return gr


def azimuthalAverage3D(gt, tdim=0, **kwargs):
    """
    Takes 3D data and gets radial component of last two dimensions

    INPUT
    ------
    gt = 3D data, first dimension is time, second two are spatial

    OUTPUT
    ------
    grt = 2D data, first dimesion is time, second is the radially averaged
          spatial components

    TO DO
    -----

    """
    # Get all indices in x,y, and t directions
    if tdim == 0:
        [t, y, x] = gt.shape

        for tt in range(0, t):
            temp = gt[tt, :, :]
            tempr = azimuthalAverage(temp, **kwargs)
            if tt == 0:
                grt = tempr
            else:
                grt = np.vstack((grt, tempr))
    elif tdim == 1:
        [y, t, x] = gt.shape

        for tt in range(0, t):
            temp = gt[:, tt, :]
            tempr = azimuthalAverage(temp, **kwargs)
            if tt == 0:
                grt = tempr
            else:
                grt = np.vstack((grt, tempr))
    elif tdim == 2:
        [y, x, t] = gt.shape

        for tt in range(0, t):
            temp = gt[:, :, tt]
            tempr = azimuthalAverage(temp, **kwargs)
            if tt == 0:
                grt = tempr
            else:
                grt = np.vstack((grt, tempr))

    return grt


def image2array(image):
    """
    Turns a pims Frame object into a 3D numpy array.

    INPUT
    ------
    image = The pims Frame object, size [M,N], with P frames

    OUTPUT
    ------
    imageArr = 3D numpy array, size [P,M,N]
    """
    startFrame = image[0].frame_no
    endFrame = image[-1].frame_no + 1
    nframes = endFrame - startFrame
    imageArr = np.zeros([nframes, image[0].shape[0], image[0].shape[1]])

    for ii in range(0, nframes):
        imageArr[ii, :, :] = image[ii]

    return imageArr


def powerSpectrum(array, window=None, plot=False, onesided=False, norm=False):
    """
    Calculates the power spectrum of a shifted array

    INPUT
    ------
    array = Array of which to find the power spectrum

    OUTPUT
    ------
    pSpec = Power spectrum of array
    """
    q = np.fft.fftn(array)
    q = np.fft.fftshift(q)
    if norm is False:
        pSpec = np.abs(q)**2  # normalize by number of elements in the array
    else:
        pSpec = np.abs(q / q.size)**2  # normalize by numel in the array

    # if onesided:
    #     pSpec =

    return pSpec


def oneSide(array, whichHalf=1, axis=0):
    """
    Gives back half of an array
    """

    if 


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
