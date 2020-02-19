import autograd.numpy as np
import autograd.numpy.random as npr
import matplotlib.pyplot as plt
from autograd import grad

# slice sampling for 1D gaussian is trivial
# inverse pdf is known

# define auxiliary function
# g_sqrt = lambda y : np.sqrt(-2.0 * sig2 * np.log(y * np.sqrt(2.0 * np.pi* sig2)))
def sqrt_term(y, mu, sig2):
    return np.sqrt(-2.0 * sig2 * np.log(y * np.sqrt(2.0 * np.pi* sig2)))

def gaussian_pdf(x, mu, sig2):
    return 1.0 / np.sqrt(2.0 * np.pi* sig2) * np.exp(-0.5 * (x - mu) **2 / sig2)

def slice_sample(mu, sig2, num_samples=10,x0=None):
    """
    x0 - initial x
    mu - mean
    sig2 - variance
    num_samples - number of samples
    """

    # xs = []
    xs = np.array([])
    if x0 is None:
        x = npr.randn()
    else:
        x = x0
    # xs.append(x)

    for i in range(num_samples):

        # sample uniform random variables
        u1 = npr.rand()
        u2 = npr.rand()

        # sample height
        y = gaussian_pdf(x, mu, sig2) * u1

        # get boundaries
        L = mu - sqrt_term(y, mu, sig2)
        R = mu + sqrt_term(y, mu, sig2)

        # sample new location
        x = L + (R-L) * u2
        xs = np.append(xs, x)

    return xs

sig2 = 1.0
def sample_x(mu, x):

    # x = mu / 2.0 + npr.rand() * mu
    x = mu + npr.randn()*np.sqrt(sig2)
    u1 = npr.rand()
    u2 = npr.rand()

    y = gaussian_pdf(x, mu, sig2)

    # sqrt_term = np.sqrt(-2.0 * sig2 * np.log(y * np.sqrt(2.0 * np.pi* sig2)))
    L = mu - sqrt_term(y, mu, sig2)
    R = mu + sqrt_term(y, mu, sig2)

    x_new = L + (R-L)*u2

    return x_new


xstar = 5.0
mu1 = -1.0
mu2 = -1.0

# def loss(mu, x):
#     x_new = sample_x(mu, x)
#     return (x_new - xstar )**2

def loss(mu, x, num_samples=1):
    x_new = slice_sample(mu, sig2, num_samples=num_samples, x0=x)
    return np.mean((x_new - xstar )**2)

def loss_reparam(mu):
    x_new = mu + npr.randn()*np.sqrt(sig2)
    return (x_new - xstar)**2

g1 = grad(loss)
g2 = grad(loss_reparam)

num_iters = 500
mu1s = []
loss1 = []
mu1s.append(mu1)
mu2s = []
loss2 = []
mu2s.append(mu2)
x = npr.randn()
for i in range(num_iters):
    x = sample_x(mu1, x)
    mu1 -= g1(mu1, x) * 0.01
    # mu1 -= g1(mu1) * 0.01
    mu2 -= g2(mu2) * 0.01
    loss1.append(loss(mu1, x))
    # loss1.append(loss(mu1))
    loss2.append(loss_reparam(mu2))
    mu1s.append(mu1)
    mu2s.append(mu2)

plt.ion()
plt.figure()
plt.subplot(211)
plt.plot(mu1s,'b')
plt.plot(mu2s,'r')
plt.plot([0,num_iters],[xstar, xstar],'k--',linewidth=0.5)
plt.subplot(212)
plt.plot(loss1,'b', label="Slice")
plt.plot(loss2,'r', label="Reparam")
plt.legend()
plt.tight_layout()

# different numbers of samples
mus = []
losses = []
num_samples = [1,5,10,25]
for s in num_samples:
    mu_s = []
    mu = -1.0
    loss_s = []
    for i in range(num_iters):
        x = sample_x(mu, x)
        mu -= g1(mu, x, num_samples=s) * 0.01
        mu_s.append(mu)
        loss_s.append(loss(mu, x, num_samples=s))

    mus.append(mu_s)
    losses.append(loss_s)

plt.figure()
plt.subplot(211)
plt.plot([0,num_iters],[xstar, xstar],'k--',linewidth=0.5)
for (i, s) in enumerate(num_samples):
    plt.plot(mus[i])
plt.subplot(212)
for (i, s) in enumerate(num_samples):
    plt.plot(losses[i], label=str(s))
plt.legend()
plt.tight_layout()
