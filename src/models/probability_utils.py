import typing

import random
import numpy as np
import scipy.stats as ss

from abc import ABC, abstractmethod
from collections.abc import Iterable


def set_seed_global(seed):
    random.seed(seed)
    np.random.seed(seed)


def length_safe_none(obj):
    return 1 if (obj is None or not indexable(obj)) else len(obj)


def has_method(obj, foo):
    return hasattr(obj, foo) and callable(getattr(obj, foo))


def indexable(obj):
    return has_method(obj, "__getitem__")


class IndexWrapper:
    """
    Makes sure an item is indexable for multiple indices. 
    If not, the __getitem__ will always return the same item.
    """

    def __init__(self, item) -> None:
        self.item = item
        # Checking the condition only once, instead for every call to __getitem__.
        if indexable(self.item):
            if len(self.item) > 1:
                self.lookup = lambda key: self.item[key]
            else:
                self.lookup = lambda key: self.item[0]
        else:
            self.lookup = lambda key: self.item

    def __getitem__(self, key):
        return self.lookup(key)


class AbstractDistribution(ABC):
    @abstractmethod
    def dim(self):
        """
        Return the dimensionality of x.
        """
        pass

    @abstractmethod
    def sample(self, theta, num=1):
        """
        Generate samples from model

        Args:
            theta: Parameters distribution is conditioned on (may be None). May be vector.
            num (int, optional): number of samples for each theta. Defaults to 1.
        """
        pass

    @abstractmethod
    def query_density(self, x, theta):
        """
        Query the density at the x.

        Args:
            x: iterable of samples.
            theta: iterable of parameters of matching size (or single parameter, 
            assuming same parameter theta for all x) , at which to query the density.
        """
        pass
    
    @abstractmethod
    def query_log2_density(self, x, theta):
        """
        Query the log2 density at the x.

        Args:
            x: iterable of samples.
            theta: iterable of parameters of matching size (or single parameter, 
            assuming same parameter theta for all x) , at which to query the density.
        """
        pass


class MultivariateNormalDistribution(AbstractDistribution):
    def __init__(self, mean, cov) -> None:
        self.d = ss.multivariate_normal(mean=mean, cov=cov)

    def dim(self):
        return self.d.dim

    def sample(self, theta=None, num=1):
        """
        Generate samples from model
        """
        N = length_safe_none(theta)
        return self.d.rvs(size=(N, num))

    def query_density(self, x, theta=None):
        """
        Query the density at the x.
        """
        return self.d.pdf(x)
    
    def query_log2_density(self, x, theta=None):
        return self.d.logpdf(x) / np.log(2)


class MixtureDistribution(AbstractDistribution):
    def __init__(self, ds, weights) -> None:
        # Weights are normalized to sum to 1
        self.weights = weights / np.sum(weights)
        self.ds = ds
        self.num_ds = len(self.weights)

    def dim(self):
        return self.ds[0].dim()

    def sample(self, theta=None, num=1):
        """
        Generate samples from model

        Args:
            theta: Theta of components distributions.
            num (int, optional): number of samples for each theta. Defaults to 1.
        """
        N = length_safe_none(theta)
        t = IndexWrapper(theta)
        mixture_idx = np.random.choice(
            self.num_ds, size=(N, num), replace=True, p=self.weights)

        samples = np.zeros(shape=(N, num, self.dim()))
        for i in range(N):
            # The ith row of sample is conditioned on theta.
            # The component is chosen based on the ith row of mixture_idx.
            # Number of samples should be 1 because we iterate over mixture_idx.
            samples[i] = np.stack([self.ds[j].sample(
                theta=t[i], num=1).flatten() for j in mixture_idx[i]], axis=0).reshape(samples[i].shape)
        return samples

    def query_density(self, x, theta=None):
        """
        Query the density at the x.

        Args:
            x: numpy array of shape Nxdim.
            theta: Numpy array of shape Nxk.

        Return:
            ndarray: Shape N, evaluating the pdf at each x given theta.
        """
        t = IndexWrapper(theta)
        y = np.zeros(shape=len(x))
        for i in range(self.num_ds):
            y += self.ds[i].query_density(x, t[i]) * self.weights[i]
        return y
    
    def query_log2_density(self, x, theta=None):
        t = IndexWrapper(theta)
        y = np.zeros(shape=len(x))
        y[:] = np.NINF
        for i in range(self.num_ds):
            y = np.logaddexp2(y, self.ds[i].query_log2_density(x, t[i]) + np.log2(self.weights[i]))
        return y


class GMMDistribution(AbstractDistribution):
    def __init__(self, comp_params, weights) -> None:
        # comp_params is iterable of (mean, cov) tuples
        self.comp_params = comp_params
        self.gmm = MixtureDistribution([MultivariateNormalDistribution(
            mean=comp[0], cov=comp[1]) for comp in comp_params], weights=weights)

    def dim(self):
        return self.gmm.dim()

    def sample(self, theta=None, num=1):
        """
        Generate samples from model

        Args:
            theta: None.
            num (int, optional): number of samples for each theta. Defaults to 1.
        """
        return self.gmm.sample(theta=theta, num=num)

    def query_density(self, x, theta=None):
        """
        Query the density at the x.

        Args:
            x: numpy array of shape (N, 2).
            theta: None.
        """
        return self.gmm.query_density(x=x, theta=theta)
    
    def query_log2_density(self, x, theta=None):
        return self.gmm.query_log2_density(x=x, theta=theta)
    
    
    @staticmethod
    def petal_gmm_model(base_cov, k_r, k_theta, n_sigma_to_cover, std_scale_by_radius, weight_scale_by_radius, weight_split_among_theta: bool):
        eig_vals, eig_vecs = np.linalg.eigh(base_cov)
        # When (N,) is multiplied by (N,N), each element in the LHS multiplies all entries in the corresponding columns. 
        # This achieves scaling each eigenvector (which are ordered as columns of eig_vecs) by its corresponding eigen value.
        scaled_eigvecs = (np.sqrt(eig_vals) * n_sigma_to_cover / k_r) * eig_vecs
        # scaled_eigvecs = np.matmul(np.diag(np.sqrt(eig_vals) / k_r), eig_vecs)
        # 1, k_t, 2k_t, 3k_t, ...
        num_components = np.array([max(i * k_theta, 1) for i in range(k_r)])
        
        comp_weights = np.copy(weight_scale_by_radius)
        if weight_split_among_theta:
            comp_weights /= num_components  # Splitting weights among all components at the same radius
        comp_weights /= np.sum(comp_weights)  # Normalizing probabilities to sum to 1
        
        components = []
        weights = []
        
        for i in range(k_r):
            scaled_cov_root = scaled_eigvecs * std_scale_by_radius[i]
            scaled_cov = np.matmul(scaled_cov_root, scaled_cov_root.T)
            if i == 0:
                components.append((np.array([0, 0]), scaled_cov))
                weights.append(comp_weights[i])
            else:
                mean_angles = np.linspace(start=0, stop=2 * np.pi, num=k_theta * i, endpoint=False)
                means = np.array([np.cos(mean_angles), np.sin(mean_angles)]) * i  # taking i radii away in the direction based on theta
                means = np.matmul(scaled_eigvecs, means)  # means[:, i] is the mean of the i'th component
                for j in range(means.shape[1]):
                    components.append((means[:, j], scaled_cov))
                    weights.append(comp_weights[i])
        
        return GMMDistribution(comp_params=components, weights=weights)
    
    
