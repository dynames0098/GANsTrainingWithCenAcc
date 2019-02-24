import scipy.stats  as spss
import matplotlib.pyplot as plt
import numpy as np


def kde(mu, tau, bbox=None, xlabel="", ylabel="", cmap='Blues', ax=None):
    values = np.vstack([mu, tau])
    kernel = spss.gaussian_kde(values)
    if not ax:
        _, ax = plt.subplots()
    ax.axis(bbox)
    ax.set_aspect(abs(bbox[1] - bbox[0]) / abs(bbox[3] - bbox[2]))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    xx, yy = np.mgrid[bbox[0]:bbox[1]:300j, bbox[2]:bbox[3]:300j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    f = np.reshape(kernel(positions).T, xx.shape)
    cfset = ax.contourf(xx, yy, f, cmap=cmap)
    return ax

