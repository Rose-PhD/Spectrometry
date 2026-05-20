from abc import abstractmethod
import numpy as np
from buaiir_spectra.data.dataset import SpectralDataset


class DataLoader:

    @abstractmethod
    def __iter__(self):
        raise NotImplemented("Subclasses must implement this method")
    

class SpectralDataLoader(DataLoader):

    def __init__(self, dataset: SpectralDataset, batch_size: int, shuffle=False, permutate=False, permute_weeks=False):
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
            pos_i = self.dataset.meta_data[self.dataset.meta_data['search_label'] == label].values
            pos.extend(pos_i)
        
        pos = np.array(pos)
        
        # permutate the underlying indices
        if self.permutate_weeks:
            pos = np.random.permutation(pos)
        self.indices = pos

        for i in range(0, len(self.dataset), self.batch_size):
            selected_pos = self.indices[i: i + self.batch_size]


        


if __name__ == '__main__':
    from buaiir_spectra.data.dataset import SpectralDataset
    from buaiir_spectra.utils.device import Device

    DATA_PATH = '/home/wilfred/Datasets/spectra_data'

    dataset = SpectralDataset(DATA_PATH, Device.BIO_SCIENCE)
    dataloader = SpectralDataLoader(dataset, batch_size=4)

    print(dataset.labels)

