"""
This file implements the data loader wrapper for all devices
"""
import random
import numpy as np
from abc import abstractmethod
from data.dataset import Dataset


class DataLoader:

    def __init__(self, batch_size, random):
        self.batch_size = batch_size
        self.random = random

    @abstractmethod
    def __iter__(self):
        """Yields batchs for model training"""
        raise NotImplementedError('Must be implemented by child class')


class SpectralDataLoader(DataLoader):

    def __init__(self, dataset: Dataset, batch_size: int, random=False):
        super().__init__(batch_size, random)
        self.dataset = dataset

    def __iter__(self):
        """Creates a generator for batching spectral data"""
        _len = len(self.dataset)
        indices = list(range(_len))
        if self.random:
            random.shuffle(indices)

        temp_data_buffer = list()
        for i in range(0, _len, self.batch_size):
            selected_indices = indices[i: i + self.batch_size]
            for index in selected_indices:
                data_v = self.dataset[index]
                temp_data_buffer.append(data_v)
            yield SpectralDataLoader.make_contiguous(temp_data_buffer)
 
    @staticmethod
    def make_contiguous(data_buff: list):
        """Concates the intermediate data into continous format"""
        return np.vstack(data_buff)
        


        