from abc import abstractmethod
from buaiir_spectra.data.dataset import SpectralDataset


class DataLoader:

    @abstractmethod
    def __iter__(self):
        raise NotImplemented("Subclasses must implement this method")
    

class SpectralDataLoader(DataLoader):

    def __init__(self, dataset: SpectralDataset, batch_size: int, shuffle=False, permutate=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.permutate = permutate

        # labels
        self.labels = dataset.labels

    
    def __iter__(self):
        pass


if __name__ == '__main__':
    from buaiir_spectra.data.dataset import SpectralDataset
    from buaiir_spectra.utils.device import Device

    DATA_PATH = '/home/wilfred/Datasets/spectra_data'

    dataset = SpectralDataset(DATA_PATH, Device.BIO_SCIENCE)
    dataloader = SpectralDataLoader(dataset, batch_size=4)

    print(dataset.labels)

