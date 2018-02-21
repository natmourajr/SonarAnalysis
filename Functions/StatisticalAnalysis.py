"""
    Python Functions
    numpy>=1.11.1
    scikit-learn>=0.17.1
"""

import numpy as np
from sklearn.neighbors import KernelDensity

# Estimating PDF
def EstPDF(data, bins=np.array([-1,0, 1]), mode='kernel', kernel='epanechnikov', kernel_bw=0.01, verbose=False):
    # kernels = 'epanechnikov','gaussian', 'tophat','exponential', 'linear', 'cosine'
    if mode == 'hist':
        if verbose:
            print 'EstPDF: Histogram Mode'
        [y,pts] = np.histogram(data,bins=bins,density=True)
        bins_centers = pts[0:-1]+np.diff(pts)
        pdf = y*np.diff(pts)
        return [pdf,bins_centers]
    if mode == 'kernel':
        if verbose:
            print 'EstPDF: Kernel Mode'
        if kernel is None:
            if verbose:
                print 'No kernel defined'
            return -1
        if kernel_bw is None:
            if verbose:
                print 'No kernel bandwidth defined'
            return -1
        kde = (KernelDensity(kernel=kernel,algorithm='auto',bandwidth=kernel_bw).fit(data))
        aux_bins = bins
        log_dens_x = (kde.score_samples(aux_bins[:, np.newaxis]))
        pdf = np.exp(log_dens_x)
        pdf = pdf/sum(pdf)
        bins_centers = bins
        return [pdf,bins_centers]

# Computing KL Divergence
def KLDiv(p, q, bins=np.array([-1,0,1]), mode='kernel', kernel='epanechnikov', kernel_bw=0.1, verbose=False):
    [p_pdf,p_bins] = EstPDF(p, bins=bins, mode=mode, kernel=kernel, kernel_bw=kernel_bw, verbose=verbose)
    [q_pdf,q_bins] = EstPDF(q, bins=bins, mode=mode, kernel=kernel, kernel_bw=kernel_bw, verbose=verbose)
    kl_values = []
    for i in range(len(p_pdf)):
        if p_pdf[i] == 0 or q_pdf[i] == 0 :
            kl_values = np.append(kl_values,0)
        else:
            kl_value = np.abs(p_pdf[i]*np.log10(p_pdf[i]/q_pdf[i]))
            if np.isnan(kl_value):
                kl_values = np.append(kl_values,0)
            else:
                kl_values = np.append(kl_values,kl_value)
    return [np.sum(kl_values),kl_values]

# https://gist.github.com/GaelVaroquaux/ead9898bd3c973c40429
'''
    Non-parametric computation of entropy and mutual-information
    Adapted by G Varoquaux for code created by R Brette, itself
    from several papers (see in the code).
    These computations rely on nearest-neighbor statistics
'''

from scipy.special import gamma,psi
from scipy import ndimage
from scipy.linalg import det
from numpy import pi

from sklearn.neighbors import NearestNeighbors

EPS = np.finfo(float).eps

def nearest_distances(X, k=1):
    '''
        X = array(N,M)
        N = number of points
        M = number of dimensions
        returns the distance to the kth nearest neighbor for every point in X
    '''
    knn = NearestNeighbors(n_neighbors=k)
    knn.fit(X)
    d, _ = knn.kneighbors(X) # the first nearest neighbor is itself
    return d[:, -1] # returns the distance to the kth nearest neighbor

def entropy(X, k=1):
    '''
        Returns the entropy of the X.
        Parameters
        ===========
        X : array-like, shape (n_samples, n_features)
        The data the entropy of which is computed
        k : int, optional
        number of nearest neighbors for density estimation
        Notes
        ======
        Kozachenko, L. F. & Leonenko, N. N. 1987 Sample estimate of entropy
        of a random vector. Probl. Inf. Transm. 23, 95-101.
        See also: Evans, D. 2008 A computationally efficient estimator for
        mutual information, Proc. R. Soc. A 464 (2093), 1203-1215.
        and:
        Kraskov A, Stogbauer H, Grassberger P. (2004). Estimating mutual
        information. Phys Rev E 69(6 Pt 2):066138.
    '''

    # Distance to kth nearest neighbor
    X = X[:,np.newaxis]
    r = nearest_distances(X, k) # squared distances
    n, d = X.shape
    volume_unit_ball = (pi**(.5*d)) / gamma(.5*d + 1)
    '''
        F. Perez-Cruz, (2008). Estimation of Information Theoretic Measures
        for Continuous Random Variables. Advances in Neural Information
        Processing Systems 21 (NIPS). Vancouver (Canada), December.
        return d*mean(log(r))+log(volume_unit_ball)+log(n-1)-log(k)
        '''
    return (d*np.mean(np.log(r + np.finfo(X.dtype).eps))
            + np.log(volume_unit_ball) + psi(n) - psi(k))

def mutual_information(variables, k=1):
    '''
        Returns the mutual information between any number of variables.
        Each variable is a matrix X = array(n_samples, n_features)
        where
        n = number of samples
        dx,dy = number of dimensions
        Optionally, the following keyword argument can be specified:
        k = number of nearest neighbors for density estimation
        Example: mutual_information((X, Y)), mutual_information((X, Y, Z), k=5)
        '''
    if len(variables) < 2:
        raise AttributeError("Mutual information must involve at least 2 variables")
    all_vars = np.hstack(variables)
    return (sum([entropy(X, k=k) for X in variables]) - entropy(all_vars, k=k))


def mutual_information_2d(x, y, sigma=1, normalized=False):
    """
        Computes (normalized) mutual information between two 1D variate from a
        joint histogram.
        Parameters
        ----------
        x : 1D array
        first variable
        y : 1D array
        second variable
        sigma: float
        sigma for Gaussian smoothing of the joint histogram
        Returns
        -------
        nmi: float
        the computed similariy measure
    """

    bins = (256, 256)

    jh = np.histogram2d(x, y, bins=bins)[0]

    # smooth the jh with a gaussian filter of given sigma
    ndimage.gaussian_filter(jh, sigma=sigma, mode='constant',
                            output=jh)

    # compute marginal histograms
    jh = jh + EPS
    sh = np.sum(jh)
    jh = jh / sh
    s1 = np.sum(jh, axis=0).reshape((-1, jh.shape[0]))
    s2 = np.sum(jh, axis=1).reshape((jh.shape[1], -1))

    # Normalised Mutual Information of:
    # Studholme,  jhill & jhawkes (1998).
    # "A normalized entropy measure of 3-D medical image alignment".
    # in Proc. Medical Imaging 1998, vol. 3338, San Diego, CA, pp. 132-143.
    if normalized:
        mi = ((np.sum(s1 * np.log(s1)) + np.sum(s2 * np.log(s2)))
              / np.sum(jh * np.log(jh))) - 1
    else:
        mi = ( np.sum(jh * np.log(jh)) - np.sum(s1 * np.log(s1))
              - np.sum(s2 * np.log(s2)))

    return mi
