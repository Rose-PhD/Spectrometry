from abc import abstractmethod
import numpy as np
from buaiir_spectra.data.datase_deprecatedt import Dataset
from buaiir_spectra.utils.device import Device
from typing import List, Tuple


class DataLoader:

    @abstractmethod
    def __iter__(self):
        raise NotImplemented("Subclasses must implement this method")

class SpectralDataLoaderBase(DataLoader):
    
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
    
    def load_data_of_disease_class(self, disease_class: str):
        """

        Loads data of  a single disease class e.g. BLB, MSV, MLN

        Arg:
            disease_class: str -> identifier of the disease
        """

        meta_data = self.dataset.meta_data
        classes_ = list(self.dataset.disease_class_codes.keys())

        # Normalize 
        if isinstance(disease_class, str): 
            disease_class = disease_class.strip().upper()
            
            if disease_class not in classes_:
                raise ValueError(
                    f"Disease class not supported"
                    f"Choose from {classes_}"
                )
        
        elif isinstance(disease_class, int):
            values = list(self.dataset.disease_class_codes.values())
            
            if disease_class not in values:
                raise ValueError(
                    f"Disease code not supported"
                    f"Chosse from {list(class_indices)}"
                )
            disease_class = classes_[values.index(disease_class)]

        indices = meta_data[meta_data['disease_class'] == disease_class].index.values
        return self._load_single(indices)


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

        if self.dataset.load_with_images:
            temp_buffer_img = []


        for index in selected_pos:
            try:
                out_idx = self.dataset[index]
                x, y = out_idx[0], out_idx[1]
            except:
                print(index)
                raise

            if self.dataset.load_with_images:
                img = out_idx[-1]
            
            temp_buffer_x.append(x)
            temp_buffer_y.append(y)
            
            if self.dataset.load_with_images:
                temp_buffer_img.append(img)
        
        # create a dynamic contianer
        out = [temp_buffer_x, temp_buffer_y]

        if self.dataset.load_with_images:
            out.append(temp_buffer_img)

        
        return tuple([np.vstack(buffer_idx) for buffer_idx in out])
        


    def __iter__(self):

        n_data = len(self.indices)
        batch_size = self.batch_size

        if batch_size == -1:
            batch_size = n_data

        for i in range(0, n_data, batch_size):
            selected_pos = self.indices[i: i + batch_size]

            out_tuple = self._load_single(selected_pos)
            temp_buffer_x, temp_buffer_y  = out_tuple[0], out_tuple[1]

            if self.dataset.load_with_images:
                temp_buffer_img = out_tuple[-1]


            if self.permutate:
                p_indices = np.random.permutation(len(temp_buffer_x))
                temp_buffer_x = temp_buffer_x[p_indices]
                temp_buffer_y = temp_buffer_y[p_indices]
                
                if self.dataset.load_with_images:
                    temp_buffer_img = temp_buffer_img[p_indices]
                    
            
            if self.dataset.load_with_images:
                yield temp_buffer_x, temp_buffer_y, temp_buffer_img
            else:
                yield temp_buffer_x, temp_buffer_y


class SpectralDataLoader_v1(SpectralDataLoaderBase):

    # Lable Buffers
    labels_train = None
    labels_test = None
    labels_val = None
    
    # Count Buffers
    counts_train = None
    counts_test = None
    counts_val = None


    is_init_correctly = False
    is_test_ratio_dfn = False

    def __init__(self, dataset: Dataset, batch_size: int = 4, permutate = False, permutate_weeks = False, permutate_plants = False, split: str = None,
             val_ratio: float = None, test_ratio: float = None
        ):

        super().__init__(dataset, batch_size, permutate, permutate_weeks, permutate_plants)

        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
    
        self.split = self._valid_split(split)

        if split is not None:
            
            self._valid_init(split)
            self._valid_val_ratio(val_ratio)

            if split == 'train':
                self._reset_state()
                self._set_label_state()
                self._set_iterable_indices()

                if not self.test_ratio == None:
                    self.__class__.is_test_ratio_dfn = True
            
            elif split == 'test':
                if not self.__class__.is_test_ratio_dfn:
                    raise TypeError(
                        f'Dataloader was intiated to laod train and val only, specifiy test_ratio in creation of train_loader'
                    )

            self._set_iterable_indices()


    @staticmethod
    def build_label_dict(labels, diseases):

        valid_labels = {}
        valid_count = {}

        for disease in diseases:
            valid = [label for label in labels if disease in label]
            valid_labels[disease] = valid
            valid_count[disease] = len(valid)

        return valid_labels, valid_count
    
    @staticmethod
    def get_label_splits(d_classes, d_counts, val_ratio=0.2):

        counts = list(d_counts.values())
        counts = np.array(counts)

        val_len = counts.sum() * val_ratio

        val_sizes = np.floor(counts * val_len / counts.sum())

        mask = val_sizes < 2
        val_sizes[mask] = 2

        mask = val_sizes > 2
        val_sizes[mask] = 2
    

        train_sizes =  counts - val_sizes

        train_labels = []
        val_labels = []

        for i, d_class in enumerate(d_classes.keys()):
            val_len_idx = int(val_sizes[i])
            train_len_idx = int(train_sizes[i])

            indices = np.random.permutation(val_len_idx + train_len_idx)
            labels = d_classes[d_class]

            train_pos_idx = indices[:train_len_idx]
            test_pos_idx = indices[train_len_idx:]

            train_labels_idx = [labels[i] for i in train_pos_idx]
            test_labels_idx = [labels[i] for i in test_pos_idx]

            train_labels.extend(train_labels_idx)
            val_labels.extend(test_labels_idx)

        return train_labels, val_labels, train_sizes, val_sizes
    
    def _set_label_state(self):

        d_classes, d_counts = self.build_label_dict(
            self.labels,
            self.dataset.disease_class_codes
        )

        labels_train, labels_val, counts_train, counts_val = self.get_label_splits(
            d_classes, 
            d_counts, 
            val_ratio= self.val_ratio
        )

        self.__class__.labels_train = labels_train
        self.__class__.labels_val = labels_val
        self.__class__.counts_train = counts_train
        self.__class__.counts_val = counts_val

        if not self.test_ratio == None:

            d_classes, d_counts = self.build_label_dict(
                self.__class__.labels_train,
                self.dataset.disease_class_codes
            ) 

            labels_train, labels_test, counts_train, counts_test = self.get_label_splits(
                d_classes,
                d_counts,
                val_ratio=self.test_ratio
            )

            self.__class__.labels_train= labels_train 
            self.__class__.labels_test = labels_test
            self.__class__.counts_train = counts_train
            self.__class__.counts_test = counts_test


    def _set_iterable_indices(self):

        if self.split == 'train':
            self.labels = self.__class__.labels_train

        
        elif self.split == 'val':
            self.labels = self.__class__.labels_val

        else:
            self.labels = self.__class__.labels_test

        pos, indices  = self._compute_indices()

        self.indices = indices
        self.pos = pos
    


    def _reset_state(self):

        self.__class__.labels_train = None
        self.__class__.labels_test = None
        self.__class__.labels_val = None
        self.__class__.counts_train = None
        self.__class__.counts_val = None
        self.__class__.counts_test = None
        self.__class__.is_test_ratio_dfn= None

    
    @staticmethod
    def _valid_split(split: str):
        if split is None:
            return None
        valid = ['train', 'test', 'val']
        split = split.strip().lower()

        if split not in valid:
            raise ValueError(
                f'Incorrect split identify, choose from  {valid}'
            )
        return split
    

    def _valid_init(self, split):

        if self.__class__.is_init_correctly:
            return 
        
        elif split == 'train':
            self.__class__.is_init_correctly = True

        else:
            raise TypeError(
                f'Expected split of type `train` first but found {split} '
            )

    def _valid_val_ratio(self, val_ratio):

        if self.split == 'train' and val_ratio is None:
            raise ValueError(
                f'Specify the split ratio using val_ratio argument'
            )
    
    
    def __len__(self):

        return len(self.indices)
    

class SpectralDataLoader(SpectralDataLoader_v1):
    """
    Implements protocol 2 for data loading i.e. only eliminate data at weekly level
    """
    train_idx = None
    val_idx = None
    test_idx = None

    def __init__(self, dataset, batch_size = 4, permutate=False, permutate_weeks=False, \
                 permutate_plants=False, split = None, val_ratio = None, test_ratio = None,\
                use_week_mask_protocol=False
            ):
        super().__init__(dataset, batch_size, permutate, permutate_weeks, permutate_plants, split, val_ratio, test_ratio)
        self.use_week_mask_protocol = use_week_mask_protocol

        if self.use_week_mask_protocol:
            if self.split == 'train':
                self.__class__.train_idx = []
                self.__class__.val_idx = []
                self.__class__.test_idx  = []

                self._compute_indices_()
                self.indices = self.__class__.train_idx
            
            elif self.split == 'val':
                self.indices = self.__class__.val_idx
            
            else:
                self.indices = self.__class__.test_idx
    
    
    def _compute_indices_(self):
        """
        Compute the data for training, testing and validation

        Arg:
            None
        Return:
            None
        
        """
        n_data = len(self.dataset.meta_data)
        val_ratio = self.val_ratio
        test_ratio = self.test_ratio

        if test_ratio is None:
            test_ratio = 0 # reset to zero for None test ratios

        val_size = int(n_data * val_ratio)
        test_size = int(n_data * test_ratio)
        train_size = n_data - (val_size + test_size)

        indices = np.arange(n_data)
        
        if self.permutate:
            indices = np.random.permutation(n_data)
        
        train_idx = indices[:train_size]
        val_idx = indices[train_size: train_size + val_size]
        test_idx = indices[train_size + val_size: ]

        self.__class__.train_idx = train_idx
        self.__class__.val_idx = val_idx
        self.__class__.test_idx = test_idx



if __name__ == '__main__':
    from buaiir_spectra.data.datase_deprecatedt import SpectralDataset
    from buaiir_spectra.utils.device import Device

    DATA_PATH = '/home/wilfred/Datasets/spectra_data'

    dataset = SpectralDataset(DATA_PATH, Device.SCAN_CODER, load_with_images=True)
    dataloader = SpectralDataLoader(dataset, batch_size=2)

    x, y, img = next(iter(dataloader))

    print(x.shape, y.shape, img.shape)




