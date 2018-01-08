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


    '''
    data = np.ones((10, 10, 10))
    window = 'triangle'
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
