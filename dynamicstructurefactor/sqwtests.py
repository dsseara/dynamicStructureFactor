import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import dynamicstructurefactor.sqw as sqw

def _nd_window_test(window):
    '''
    Test of sqw._nd_window() helper function

    Parameters
    ----------
    window : string, float, or tuple, optional
        Type of window to use, defaults to 'boxcar' with shape of data, i.e.
        data stays the same. See `scipy.signal.get_window` for all windows
        allowed

    Example
    -------
    _nd_window_test('bartlett')
    '''
    data = np.ones((10, 10, 10))
    x = np.linspace(0, 100, 10)
    xx, yy, zz = np.meshgrid(x, x, x)
    data, win = sqw._nd_window(data, window)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xx, yy, zz, c=data)

    fig2, ax2 = plt.subplots()
    ax2.plot(np.squeeze(win[0]), 'o-')
    ax2.set_title('Window: {0}'.format(window))

    plt.show()

def azimuthal_average_test():
    '''
    Test the azimuthal averaging function on a radial sinc function

    Parameters
    ----------

    Results
    -------
        Two figures. First is of radial sinc function, second is it's azimuthal
        average.

    Example
    -------

    '''
    x = np.linspace(-2 * np.pi, 2 * np.pi, 200)
    dx = x[1] - x[0]
    xx, yy = np.meshgrid(x, x, indexing='ij')
    r = (xx**2 + yy**2)**0.5
    k = 2  # wavenumber
    sphericalWave = np.sin(k * r) / (k * r)
    radialProfile, r = sqw.azimuthal_average(sphericalWave)
    r *= dx

    fig, ax = plt.subplots(1, 2)
    ax[0].pcolor(xx / (2 * np.pi), yy / (2 * np.pi), sphericalWave)
    ax[0].set_aspect('equal')
    ax[0].set_xlabel(r'$x/2\pi$')
    ax[0].set_ylabel(r'$y/2\pi$')
    ax[0].set_title(r'Spherical Wave, $\frac{\sin(kr)}{kr}$')

    ax[1].plot(r / (2 * np.pi), radialProfile)
    ax[1].set_aspect('equal')
    ax[1].set_xlabel(r'$r/2\pi$')
    ax[1].set_ylabel(r'$I(r)$')
    ax[1].set_title('Radial Profile')

    plt.show()
    plt.tight_layout()


def azimuthal_average_circleMask_test():
    '''
    Test the azimuthal averaging function on a radial sinc function,
    using a mask for the circle inscribed in the data.

    Parameters
    ----------

    Results
    -------
        Two figures. First is of the masked radial sinc function,
        second is it's azimuthal average.

    Example
    -------

    '''
    x = np.linspace(-2 * np.pi, 2 * np.pi, 200)
    dx = x[1] - x[0]
    xx, yy = np.meshgrid(x, x, indexing='ij')
    r = (xx**2 + yy**2)**0.5
    k = 2  # wavenumber
    sphericalWave = np.sin(k * r) / (k * r)
    radialProfile, r = sqw.azimuthal_average(sphericalWave, mask='circle')
    r *= dx

    fig, ax = plt.subplots(1, 2)
    ax[0].pcolor(xx / (2 * np.pi), yy / (2 * np.pi), sphericalWave)
    ax[0].set_aspect('equal')
    ax[0].set_xlabel(r'$x/2\pi$')
    ax[0].set_ylabel(r'$y/2\pi$')
    ax[0].set_title(r'Spherical Wave, $\frac{\sin(kr)}{kr}$')

    ax[1].plot(r / (2 * np.pi), radialProfile)
    ax[1].set_aspect('equal')
    ax[1].set_xlabel(r'$r/2\pi$')
    ax[1].set_ylabel(r'$I(r)$')
    ax[1].set_title('Radial Profile')

    print(r/(2*np.pi))
    plt.show()
    plt.tight_layout()
