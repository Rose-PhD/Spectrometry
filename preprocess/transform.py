import numpy as np
from abc import ABC, abstractmethod
from scipy.stats import zscore

class Transform(ABC):
    """Abstract class for all transformation performed on the data"""

    @abstractmethod
    def forward(self, *args, **kwargs):
        raise NotImplementedError('Child must implement this method')
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    
    def fit_transform(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    

class Range_Clip(Transform):
    """Clips wavelength ranges for high end spectrometer"""
    
    def __init__(self, lower_bound: int, upper_bound: int= None):
        self.lower_bound = lower_bound
        if upper_bound is not None:
            assert lower_bound < upper_bound, f'Lower bound {lower_bound} should be less than upper bound {upper_bound}'

        self.upped_bound = upper_bound
        self._tot_kept = 0
        self._done_flag = False
        self._tot_removed = 0

    
    def get_stats(self):
        """Returns the stats for the removed and maintained ranges"""

        assert self._done_flag, f'Run the forward method firs, to pre-compute stats'
        return {'kept': self._tot_kept, 'Removed': self._tot_removed}
    
    def forward(self, x_data: np.ndarray, wavelength_range: np.ndarray):
        """CLips the data to wavelength range specified by upper bound and lower bound"""

        mask = wavelength_range >= self.lower_bound
        if self.upped_bound is not None:
            mask = mask & (wavelength_range <= self.upped_bound)

        self._tot_kept += mask.sum()
        self._tot_removed += (~mask).sum()
        self._done_flag = True

        x_clipped = x_data[: ,mask]
        wavelength_clipped = wavelength_range[mask]
        assert x_clipped.shape[-1] == (wavelength_clipped.shape)[-1], f'Inconsistent shaped'
        
        return x_clipped, wavelength_clipped


class Zscore_Outlier_Filter(Transform):
    """Removes outliers from the data using the z_score keeping values under outlier boundary"""
    def __init__(self, boundary:int = 3):
        self._tot_outliers = 0
        self.boundary = boundary
    
    def forward(self, x_data):
        """Computes z_score and removes values under outlier boundary"""

        z_score = np.abs(zscore(x_data))
        mask = (z_score < self.boundary).all(axis= 1)

        x_clean = x_data[mask]
        self._tot_outliers += (~mask).sum()

        return x_clean
         
        
    def get_outlier_count(self):
        assert not self._tot_outliers == 0, f'Run the forward method to precompute the outliers'
        return self._tot_outliers 


class Bound_Outlier_Filter(Transform):
    """Filters out values that result from incorrect sensor values"""

    def __init__(self, lower_bound: float =0, upper_bound: float=1.0):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self._tot_removed = 0

    def forward(self, x_data: np.ndarray):
        """Computes the  mask and removes values in tensor that are below or abocve boundary"""
        
        mask = ((x_data >= self.lower_bound) & (x_data <= self.upper_bound)).all(axis=1)
        x_cleaned = x_data[mask]
        
        self._tot_removed += (~mask).sum()
        return x_cleaned
    
    def get_removed_count(self):
        """Returns the number of removed rows with outliers"""
        assert not self._tot_removed == 0, f'Call forward method to compute the removed count'
        return self._tot_removed
        


    

