import numpy as np
from buaiir_spectra.data.datase_deprecatedt import SpectralDataset
from buaiir_spectra.utils.device import Device 
from buaiir_spectra.utils.config_info import SEED

np.random.seed(SEED)

class SpectralDataLoader:

    def __init__(self, dataset: SpectralDataset, batch_size:int = 1, shuffle: bool = False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.len = len(dataset)
        self.indices = np.arange(self.len)
        if shuffle:
            self.indices = np.random.permutation(self.len)

    def __iter__(self):
        for i in range(0, self.len, self.batch_size):
            selected_pos = self.indices[i: i+ self.batch_size]

            temp_x = []
            temp_images = []
            temp_y = []
            temp_labels = []

            for index in selected_pos:
                # Read the data
                try:
                    x, images, y, labels = self.dataset[index]
                except:
                    print('*'*100)
                    print(f'Error comes from {index}')
                    print(f'Y Values: {y}')
                    print('*'*100)
                temp_x.append(x)
                temp_y.append(y)
                temp_images.append(images)
                temp_labels.extend(labels)
            
            yield np.vstack(temp_x), np.vstack(temp_images), np.vstack(temp_y), temp_labels