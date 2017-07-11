"""
Module to calculate dynamic structure factor for 2D movies

Created on Wed Jul 5 22:48:12 2017

Daniel S. Seara
www.github.com/dsseara
daniel.seara@yale.edu
"""
import numpy as np

__all__ = ['azimuthalAverage', 'azimuthalAverage3D']


def azimuthalAverage(g, center=None):
    """
    Calculates the azimuthal average of a 2D array
    
    INPUT
    ------
    g = the 2D array
    center = the center of the image from which to measure
             the radial profile from. If center = None, uses
             the center of the 2D array itself

    OUTPUT
    ------
    gr = radially averaged 1D array

    Based on radialProfile found at:
    http://www.astrobetter.com/wiki/tiki-index.php?page=python_radial_profiles
    """
    # Get all the indices in x and y direction
    [y, x] = np.indices(g.shape)

    # Define the center from which to measure the radius
    if not center:
        center = np.array([(x.max() - x.min())/2, (y.max() - y.min())/2])

    # Get distance from all points to center
    r = np.hypot(x-center[0], y-center[1])

    # Get indices of radii in ascending order and 1D
    inds = np.argsort(r.flat)
    # Sort both the radii and data, again in 1D
    rSorted = r.flat[inds]
    gSorted = g.flat[inds]

    # Assuming binsize of 1 pixel, find all radii rounded down to
    # nearest integer
    rInt = rSorted.astype(int)
    # Find out where the radius changes by an integer
    deltaR = rInt[1:] - rInt[:-1]
    rInd = np.where(deltaR)[0]

    # Find how many radii you have inside each band
    nr = rInd[1:] - rInd[:-1]

    csum = np.cumsum(gSorted, dtype=float)
    count = csum[rInd[1:]] - csum[rInd[:-1]]

    gr = count/nr
    return gr


def azimuthalAverage3D(gt,tdim = 0):
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
    - allow user input of which dimension is the time dimension

    """
    # Get all indices in x,y, and t directions
    [y, x, t] = gt.shape

    for tt in range(0, t):
        temp = gt[:, :, tt]
        tempr = azimuthalAverage(temp)
        if tt == 0:
            grt = tempr
        else:
            grt = np.vstack((grt,tempr))

    return grt


def image2array(image):
    """
    Turns a pims Frame object into a 3D numpy array. As it is, 
    only each frame is an array

    INPUT
    ------
    image = The pims Frame object, size [M,N], with P frames

    OUTPUT
    ------
    imageArr = 3D numpy array, size [M,N,P]
    """
    nframes = image[-1].frame_no + 1
    imageArr = image[0]
    for ii in range(1,nframes):
        imageArr = np.dstack((imageArr,image[ii]))

    return imageArr


def powerSpectrum(array):
    """
    Calculates the power spectrum of an shifted array

    INPUT
    ------
    array = Array of which to find the power spectrum

    OUTPUT
    ------
    pSpec = Power spectrum of array
    """
    q = np.fft.fftn(array)
    q = np.fft.fftshift(q)
    pSpec = np.abs(q)**2

    return pSpec