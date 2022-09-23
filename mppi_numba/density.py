#!/usr/bin/env python3
"""
Utils for generating samples from a made-up density such as Gaussian mixture
"""
import numpy as np


class Density(object):
    
    def __init__(self, sample_bounds, pmf_bounds, sample_fn, num_samples=1e4):
        self.sample_bounds = sample_bounds # bound within which to sample
        self.pmf_bounds = pmf_bounds # bound within which to extract the PMF
        self.sample_fn = sample_fn
        self.num_samples = num_samples # num samples for estimating different values

        self.sample_initialized = False
        self.initialize_samples(self.num_samples)
        self.sample_initialized = False
        self.samples = None
            
    def initialize_samples(self, num_samples):
        self.samples = self.sample(num_samples)
        self.sample_initialized = True

    def mean(self, samples=None):
        if samples is not None:
            return np.mean(samples)
        else:
            if not self.sample_initialized:
                self.initialize_samples(self.num_samples)
            return np.mean(self.samples)

    def var(self, samples=None):
        if samples is not None:
            return np.var(samples)
        else:
            if not self.sample_initialized:
                self.initialize_samples(self.num_samples)
            return np.var(self.samples)
    
    def cvar(self, alpha, front=True, samples=None):
        assert alpha>0 and alpha<=1.0, "Alpha must be in (0,1]"
            
        p = alpha*100.0 if front else (1.0-alpha)*100.0
        if samples is None:
            if not self.sample_initialized:
                self.initialize_samples(self.num_samples)
            samples = self.samples

        thres = np.percentile(samples, p)
        if front:
            mask = samples<thres
        else:
            mask = samples>thres
        assert np.sum(mask)>0
        return np.mean(samples[mask]), thres
    
    def sample(self, num):
        return self.sample_fn(num)

    def get_pmf(self, num_bins, include_min_max=True):
        # For convenience, mass for min and max values are added as separate bins
        if not self.sample_initialized:
            self.initialize_samples(self.num_samples)
        vrange = self.pmf_bounds
        nums, edges = np.histogram(self.samples, num_bins, range=vrange, density=True)

        bin_width = (vrange[1]-vrange[0])/num_bins
        values = np.arange(vrange[0], vrange[1], bin_width) + bin_width/2

        if include_min_max:
            # Insert minimum value at the beginning (useful for 0 traction elements)
            values = np.insert(values, 0, vrange[0])
            nums = np.insert(nums, 0, 0)
            # Insert max value at the end (useful for nominal model that attains max values)
            values = np.append(values, vrange[1])
            nums = np.append(nums, 0)

        # Return (values, pmf)
        return values, nums/np.sum(nums)


class GaussianMixture(Density):

    def __init__(self, sample_bounds, pmf_bounds, weights, means, stds, num_samples=1e3):
        assert sum(weights)==1
        assert len(weights)==len(means)==len(stds)
        assert len(sample_bounds)==2
        assert len(pmf_bounds)==2
        assert sample_bounds[1]>=sample_bounds[0]
        assert pmf_bounds[1]>=pmf_bounds[0]
        assert pmf_bounds[0]<=sample_bounds[0] and pmf_bounds[1]>=sample_bounds[1]
        self.num_components = len(weights)
        
        def sample_fn(num):
           # Sample from mixture of Gaussian truncated between range
            num_sampled = 0
            data = []
            # indices = np.arange(len(weights))
            while num_sampled < num:
                idx = np.random.choice(self.num_components, p=weights)
                sample = np.random.normal(loc=means[idx], scale=stds[idx])
                if sample >= sample_bounds[0] and sample <= sample_bounds[1]:
                    data.append(sample)
                    num_sampled += 1 
            return np.asarray(data)
        
        super().__init__(sample_bounds, pmf_bounds, sample_fn, num_samples)