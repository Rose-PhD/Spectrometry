from abc import abstractmethod
import numpy as np
from buaiir_spectra.data.dataset import SpectralDataset


class DataLoader:

    @abstractmethod
    def __iter__(self):
        raise NotImplemented("Subclasses must implement this method")

class SpectralDataLoader(DataLoader):

    def __init__(self, dataset: SpectralDataset, batch_size: int, shuffle: bool =False, permutate: bool =False, permute_weeks: bool=False):
        """
        Computes iterable batches around the dataset

        Arg:
            dataset: Dataset -> object of dataset to be loaded
            batch_size: int -> size of the batches of the data to be loaded
            shuffle: bool -> flags whether to shuffle data or not
            permutate: bool -> flags whether to shuffle content of batches
            permute_weeks: bool -> flags whether to shuffle weeks or not

        Returns:
            None 
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.permutate = permutate
        self.permutate_weeks= permute_weeks

        # labels
        self.labels = dataset.labels

    def __iter__(self):
        
        label_indices = np.arange(len(self.labels))
        if self.shuffle:
            label_indices = np.random.permutation(label_indices)
            self.labels = [self.labels[item] for item in label_indices]

        # Extract the indices
        pos = []
        for label in self.labels:
            pos_i = self.dataset.meta_data[self.dataset.meta_data['search_label'] == label].index.values
            pos.extend(pos_i)
        
        pos = np.array(pos)
        
        # Permutate the underlying indices
        if self.permutate_weeks:
            pos = np.random.permutation(pos)
        self.indices = pos

        for i in range(0, len(self.dataset), self.batch_size):
            selected_pos = self.indices[i: i + self.batch_size]
            
            temp_buffer_x = []
            temp_buffer_y = []

            for index in selected_pos:
                x, y = self.dataset[index]
                temp_buffer_x.append(x)
                temp_buffer_y.append(y)
            
            temp_buffer_x = np.vstack(temp_buffer_x)
            temp_buffer_y = np.vstack(temp_buffer_y)

            if self.permutate:
                p_indices = np.random.permutation(len(temp_buffer_x))
                temp_buffer_x = temp_buffer_x[p_indices]
                temp_buffer_y = temp_buffer_y[p_indices]

            yield temp_buffer_x, temp_buffer_y

if __name__ == '__main__':
    from buaiir_spectra.data.dataset import SpectralDataset
    from buaiir_spectra.utils.device import Device

    DATA_PATH = '/home/wilfred/Datasets/spectra_data'

    dataset = SpectralDataset(DATA_PATH, Device.SCAN_CODER)
    dataloader = SpectralDataLoader(dataset, batch_size=4)

    print(dataset.meta_data[dataset.meta_data['raw_count'] <6])

