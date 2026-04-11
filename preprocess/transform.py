import numpy as np
from abc import ABC, abstractmethod

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

         



    

