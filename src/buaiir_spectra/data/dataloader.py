from abc import abstractmethod
import numpy as np
from buaiir_spectra.data.dataset import SpectralDataset
from buaiir_spectra.utils.device import Device


class DataLoader:

    @abstractmethod
    def __iter__(self):
        raise NotImplemented("Subclasses must implement this method")

class SpectralDataLoader(DataLoader):
    
    def __init__(self, dataset: SpectralDataset, batch_size: int, shuffle: bool=False, permutate: bool =False, permutate_weeks: bool=False, permutate_plants:bool= False):
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
        self.permutate_weeks= permutate_weeks
        self.permutate_plants = permutate_plants

        # labels
        self.labels = dataset.labels
        self.pos, self.indices = self._compute_indices()
    
    
    def _compute_indices(self):
        """
        Computes the indices used during data loading

        Arg:
            None
        
        Return:
            None
        """
        # Retrive the number of weeks
        n_weeks = len(self.dataset.weeks.keys()) + 1
        pos = []
        
        for label in self.labels:
            res_array = np.full(n_weeks, np.nan)
            result = self.dataset.meta_data.loc[self.dataset.meta_data['search_label'] == label, ['week']]

            tuples_ = list(zip(result.index, result['week']))
            # fill in data in correct position
            for _index, week in tuples_:
                res_array[week] = _index
            
            pos.append(res_array)
        
        # clean pos array and drop week 0
        pos = np.array(pos)
        pos = pos[:, 1:]

        if self.permutate_plants:
            pos = np.random.permutation(pos)
        

        indices = pos.transpose()

        if self.permutate_weeks:
            indices = np.random.permutation(indices)
        
        indices = indices.flatten()
        indices = indices[~np.isnan(indices)]

        if self.permutate:
            indices = np.random.permutation(indices)

        return pos, indices
            

    def __iter__(self):

        n_data = len(self.indices)
        batch_size = self.batch_size

        for i in range(0, n_data, batch_size):
            selected_pos = self.indices[i: i + batch_size]
            
            temp_buffer_x = []
            temp_buffer_y = []

            for index in selected_pos:
                x, y = self.dataset[index]

                if self.dataset.device == Device.LOW_COST:
                    for i in range(2):
                        temp_buffer_x.append(x[: , i, :])

                    for i in range(3):
                        temp_buffer_y.append(y[:, i, :])
                else:
                    temp_buffer_x.append(x)
                    temp_buffer_y.append(y)
            
            print(f'Selected position: ', selected_pos)

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

    dataset = SpectralDataset(DATA_PATH, Device.LOW_COST)
    dataloader = SpectralDataLoader(dataset, batch_size=4)

    # print(dataset.meta_data[dataset.meta_data['raw_count'] <3])

