import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patheffects import PathPatchEffect, SimpleLineShadow, Normal
mu = 0.5
def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    from math import factorial
    
    try:
        window_size = np.abs(window_size)
        order = np.abs(order)
    except ValueError:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.asmatrix([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')

def truncate(array):
    for i in range(len(array)):
        if array[i] <= 0:
            array[i] = 0.15
        if array[i] >= 1:
            array[i] = 0.85

a = list(map(lambda x: float(format(x, '.10f')), np.random.normal(mu, 0.1, 10000)))
truncate(a)
hist, bin_edges = np.histogram(a, bins=30)
b= np.array([])
for (i, an) in enumerate(bin_edges[:-1]):
   b = np.append(b, list(map(lambda x: float(format(x, '.10f')), np.random.normal(1 - an, - abs(an - mu) + mu, hist[i]))))
b = b.flatten()
truncate(b)
final = []
for i in range(len(a)):
    final.append(a[i]/(a[i]+b[i]))
summary = 0
asd = 0
for x in final:
    if x <=0.5:
        summary+=1
    else:
        asd+=1
print(summary, asd)
fig = plt.figure(figsize=(10, 5), dpi=300)
ax = fig.add_subplot(1,1,1)  
ax.set_axisbelow(True)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.yaxis.grid(color='gray', linestyle='dashed')
ax.set_ylabel("PDF")
ax.set_xlabel(r'$\pi^{(n)}[1] = \frac{a}{a+b}$', fontsize=15)
n, bins, patches = ax.hist(final, bins=30, ec='#ef6c00', histtype='bar', color='#ef6c00', alpha=1, rwidth=0.75, path_effects=[SimpleLineShadow(shadow_color="black", linewidth=3, offset=(1, -3)), Normal()])
bins = bins[1:]
cm1 = n.cumsum()
# smooth the line (more appealing look)
cm1_hat = savitzky_golay(cm1, 15, 5)
ax2 = ax.twinx()
ax2.plot(bins, cm1_hat, lw=2, color="#8085e9", linestyle='--', path_effects=[SimpleLineShadow(shadow_color="black", linewidth=3, offset=(2, -2)), Normal()])
fillx = [x for x in bins if x <= 0.5]
cm1_hat_fill = cm1_hat[:len(fillx)]
#ax2.fill_between(fillx, 0,cm1_hat_fill, alpha=.5)
ax2.spines["right"].set_visible(False)
ax2.spines["left"].set_visible(False)
ax2.spines["top"].set_visible(False)
ax2.set_ylabel("CDF")
ax2.set_xlim(bins[1], 1)
ax2.hlines(y=summary, xmin=0, xmax=1, linewidth=2, color='r', linestyle=(0, (3, 5, 1, 5)))
ax2.set_ylim(0,10_000)
y_ticks = np.append(ax2.get_yticks(), summary)
ax2.set_yticks(y_ticks)
x_ticks = np.append(ax.get_xticks(), 0.5)
ax.set_xticks(x_ticks)
fig.tight_layout()
plt.savefig('./sim1.svg', format='svg',bbox_inches='tight', pad_inches = 0)
plt.savefig('./sim1.png', format='png',bbox_inches='tight', pad_inches = 0)
plt.savefig('./sim1.jpg', format='jpg',bbox_inches='tight', pad_inches = 0)
# plt.show()
