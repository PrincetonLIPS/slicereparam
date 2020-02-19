import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt

# slice sample from a 1D distribution
# Question -> do the slice interval steps need to be differentiated?

# should take in density to be evaluated as function, then draw samples

def doubling_interval(f, x0, y, w=1.0, p=5):
    """
    f - a function proportional to the density
    x0 - the current point
    y - the vertical level defining the slice
    w - estimate of the typical size of a slice
    p - integer limiting the size of a slice to 2^p w
    """

    # sample uniform
    u = npr.rand()

    # define left and right
    L = x0 - w*u
    R = L + w
    K = np.copy(p)
    while K > 0 and (y < f(L) or y < f(R)):
        v = npr.rand()
        if V < 0.5:
            L -= R-L
        else:
            R += R-L
        K -= 1.0

    return L, R

def stepping_interval(f, x0, y, w=1.0, m=10):
    """
    f - a function proportional to the density
    x0 - the current point
    y - the vertical level defining the slice
    w - estimate of the typical size of a slice
    m - integer limiting the size of a slice to m*w
    """
    u = npr.rand()
    L = x0 - w*u
    R = L+w
    v = npr.rand()
    J = np.floor(m*v)
    K = (m-1) - J

    while J > 0 and y < f(L):
        L -= w
        J -= 1

    while K > 0 and y < f(R):
        R += w
        K -= 1

    return L, R

def sample_interval_stepping(f, x, y, L, R, max_iters=25):
    """
    f - a function proportional to the density
    x0 - the current point
    y - the vertical level defining the slice
    L - left part of interval
    R - right part of interval
    """
    Lbar = np.copy(L)
    Rbar = np.copy(R)

    for i in range(max_iters):
        u = npr.rand()
        x1 = Lbar + u*(Rbar-Lbar)


        if y < f(x1):
            return x1

        if x1 < x:
            Lbar = np.copy(x1)
        else:
            Rbar = np.copy(x1)


def slice_sample_1d(f, num_samples=10):

    # now -> f(x) is proportional up to p(x)
    # exp(f(x)) is proportional to p(x)
    # so f is log, up to normalizing constant

    x = npr.randn()
    xs = []
    xs.append(x)

    for i in range(num_samples):

        # draw horizontal slice
        u = npr.rand()
        y = f(x) * u

        # get interval
        L, R = stepping_interval(f, x, y, w=1.0, m=5)

        # sample point
        x = sample_interval_stepping(f, x, y, L, R)
        xs.append(x)

    return xs


# test!
mu = 3.0
sig2 = 5.0
# f = lambda x : np.exp(-0.5 * (x - mu)**2 / sig2)
# f = lambda x : 1.0 / x * np.exp(-0.5 * (np.log(x) - mu)**2 / sig2)
def f(x):
    if x > 0:
        return 1.0 / x * np.exp(-0.5 * (np.log(x) - mu)**2 / sig2)
    else:
        return -np.inf

xs = slice_sample_1d(f, num_samples=50000)

# x_range = np.arange(np.min(xs)-1.0, np.max(xs)+1.0, 0.1)
x_range = np.arange(0.0, np.max(xs)+1.0, 0.1)
# pdf = 1.0 / np.sqrt(2.0*np.pi*sig2) * np.exp(-0.5 * (x_range - mu)**2 / sig2)
pdf = np.divide(1.0, x_range * np.sqrt(2.0*np.pi*sig2)) * np.exp(-0.5 * (np.log(x_range) - mu)**2 / sig2)

plt.ion()
plt.figure()
plt.hist(xs, 40, normed=True)
plt.plot(x_range, pdf, 'k', linewidth=1)
