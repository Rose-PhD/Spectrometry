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
    
    def __init__(self, lower_bound: int, wavelength: np.ndarray, upper_bound: int= None, ):
        self.lower_bound = lower_bound
        if upper_bound is not None:
            assert lower_bound < upper_bound, f'Lower bound {lower_bound} should be less than upper bound {upper_bound}'

        self.upped_bound = upper_bound
        self._tot_kept = 0
        self._done_flag = False
        self._tot_removed = 0
        self.wavelength = wavelength

    
    def get_stats(self):
        """Returns the stats for the removed and maintained ranges"""

        assert self._done_flag, f'Run the forward method firs, to pre-compute stats'
        return {'kept': self._tot_kept, 'Removed': self._tot_removed}
    
    def forward(self, x_data: np.ndarray):
        """CLips the data to wavelength range specified by upper bound and lower bound"""

        wavelength_range = self.wavelength
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


def merge(x, y):
    batch, dim_x, dim_y = *x.shape, y.shape[-1]
    data = np.zeros((batch, dim_x + dim_y))
    data[:, :dim_x] = x
    data[:, dim_x: ] = y
    return data, dim_x, dim_y


class Bound_Outlier_Filter(Transform):
    """
    Transform that removes large sensor readings for the scan coder
    """

    def __init__(self, lower_bound:float=-0.15, upper_bound: float= 2.0, verbose=False):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.verbose = verbose

    
    def forward(self, x: np.ndarray, y: np.ndarray):
        """
        Removes outliers based on the given lower and upper bound

        Arg:
            x: array of features
            y: array of target features
        """
        lower_bound = self.lower_bound
        upper_bound = self.upper_bound
        
        # Merge the data for easy filtering 
        data, dim_x, _ = merge(x, y)

        mask = ((data[:, :dim_x] >= lower_bound) & (data[:, :dim_x] <= upper_bound)).all(axis=1)
        data_v2 = data[mask]
        if self.verbose:
            print(f'Removed: {data.shape[0] - data_v2.shape[0]}')

        return data_v2[:, :dim_x], data_v2[:, dim_x:]

    
class Smooth_Signal(Transform):
    """
    Transform used to smooth signals
    """

    def __init__(self, window_size: int=5) -> None:
        """
        Arg:
            window_size: kernel size i.e. x_1, x_2, ..., x_n to consider during signal smoothing

        Return:
            None
        """
        self.window_size = window_size

    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Arg:
            x: array of signal to smoothen, can be of form (batch, dim) or (dim,)

        Return:
            out: np.ndarray -> array of smoothened signal with same size as the input array ie. x_input
        """
        n_dim = x.ndim
        out = None

        kernel = np.ones(self.window_size) / self.window_size

        def convolve_1d(x_1d, kernel_):

            return np.convolve(x_1d, kernel_, mode='same')


        if n_dim > 1:
            out = np.array([
                convolve_1d(x_i, kernel) for x_i in x
            ])
        else:
            out = convolve_1d(x, kernel)

        return  out


        


        


    

