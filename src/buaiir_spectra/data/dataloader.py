from abc import abstractmethod
import numpy as np
from buaiir_spectra.data.dataset import Dataset
from buaiir_spectra.utils.device import Device
from typing import List, Tuple


class DataLoader:

    @abstractmethod
    def __iter__(self):
        raise NotImplemented("Subclasses must implement this method")

class SpectralDataLoader(DataLoader):
    
    def __init__(self, dataset: Dataset, batch_size: int, permutate: bool =False, permutate_weeks: bool=False, permutate_plants:bool= False):
        """
        Computes iterable batches around the dataset

        Arg:
            dataset: Dataset -> object of dataset to be loaded
            batch_size: int -> size of the batches of the data to be loaded
            permutate: bool -> shuffles the order both plants and weeks
            permutate_plants: bool -> shuffles the order of plants only
            permutate_weeks: bool -> shuffles the oder of weeks only

        Returns:
            None 
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.permutate = permutate
        self.permutate_weeks= permutate_weeks
        self.permutate_plants = permutate_plants

        # labels
        self.labels = dataset.labels
        self.unshuffled_pos = None
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

        # keep a copy of unshuffled order
        self.unshuffled_pos = np.array(pos)

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
    
    def load_data_of(self, label: str | int):
        """
        Loads all data for a single label for analysis

        """
        label = label.strip().upper() if isinstance(label, str) else label

        if isinstance(label, str) and not label in self.labels:
            raise ValueError(
                f"{label} not among the labels consider choosing from "
                f'{list(self.labels)}'
            )
        
        if isinstance(label, int) and not label in range(0, len(self.labels)):
            raise IndexError(
                f"Label index is out of range"
                f"Max supported is {len(self.labels)}"
            )
        
        label_index = self.labels.index(label) if isinstance(label, str) else label
        label_pos = self.unshuffled_pos[label_index, :]

        # drop Nan values
        label_pos = label_pos[~np.isnan(label_pos)]

        # call the dataloader to load all the data
        return self._load_single(label_pos)


    def _load_single(self, selected_pos: List[int]) -> Tuple[np.ndarray]:
        """
        Loads data for a subset of selected indices

        Arg:
            selected_pos: List[int] -> collections of selected indices

        Return:
            Tuple[np.ndarray] -> List of temporay buffer of (Features, targets)
        """
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
        
        return np.vstack(temp_buffer_x), np.vstack(temp_buffer_y)

        
            

    def __iter__(self):

        n_data = len(self.indices)
        batch_size = self.batch_size

        for i in range(0, n_data, batch_size):
            selected_pos = self.indices[i: i + batch_size]

            temp_buffer_x, temp_buffer_y = self._load_single(selected_pos)

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

