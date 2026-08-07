import os
import re
import joblib
from joblib import Parallel, delayed, load
import numpy as np
from buaiir_spectra.utils.device import Device
from typing import List
from pathlib import Path
from huggingface_hub import snapshot_download
from huggingface_hub.utils import logging
import warnings
from importlib.resources import files
from buaiir_spectra.utils.config_info import SEED, HF_TOKEN
warnings.filterwarnings("ignore")
logging.set_verbosity_warning()

np.random.seed(SEED)
META_PATH = files("buaiir_spectra").joinpath("wavelength_and_cols_ds.pkl")
state_dict = joblib.load(META_PATH)




class load_spectra:

    def __init__(self, device: Device, shuffle: bool=False, no_files_per_load: int =-1, load_with_images: bool=False):
        """
        Loads device specific batches which are too big to fit in memory. Depending on the resources  a 
        specific number of files or entire data can be fitted into memory

        Arg:
            device: Device -> target device whose batches are to be loaded
            shuffle: bool ->  randomly shuffle the batches before loading to ensure better model generalability during training
            no_files_per_load: int -> identifies the total batches to load into memory passed on the memory capacity of the host device
            load_with_images: bool -> specifies whether to load with images or not
        
        Return:
            None
        """
        self.device = device
        self.shuffle = shuffle
        self.no_files_per_load = no_files_per_load
        self.load_with_images= load_with_images
        self.wavelength = state_dict[device.name]['wavelenght']
        self.target_cols = state_dict[device.name]['target_columns']

        # load and sort files
        self.sorted_files = self._load_files()
        self.selection_pos = np.arange(len(self.sorted_files))

        if self.shuffle:
            self.selection_pos = np.random.permutation(len(self.sorted_files))

    def _load_files(self):
        """
        Populates the file path of all the batches for a given device

        Arg:
            None
        Return:
            None
        """
        BATCH_PATHS = self.load_device_data(self.device)
        return sorted(BATCH_PATHS, key=lambda f: self._sort_key(f, self.device))
    
    @staticmethod
    def _sort_key(filename, device):
        """
        Arranges the batches in the correct order to enable consistent and predictable data loading

        Arg:
            filename: str -> file name to sort
            device: Device -> target device whose files are to be sorted
        
        Return:
            matched group if true else return -infinite
        """
        filename = filename.name 
        pattern = rf'{device.name}_(\d+)_batch\.pkl'
        match = re.search(pattern, filename)
        if match:
            return int(match.group(1))
        return float('inf')
    
    def __call__(self):
        """
        Loads all the data
        """

        if self.no_files_per_load != -1:
            raise TypeError(f"This is only callable if loading all the data in memory")
        
        return next(self.__iter__())

    def __iter__(self):
        """
        Provides an iterator for loading images

        Arg:
            None
        Return:
            None
        """
        if self.no_files_per_load == -1:
            # load all data if -1 flag is provided
            self.no_files_per_load = len(self.sorted_files)
        
    
        for i in range(0, len(self.sorted_files), self.no_files_per_load):
            selected_pos_idx = self.selection_pos[i: i+self.no_files_per_load]


            selected_files = [self.sorted_files[i] for i in selected_pos_idx]

            
            results = Parallel(n_jobs=-1)(
                    delayed(load)(file_path)
                    for file_path in selected_files
                )

            if self.no_files_per_load != 1:
                temp_x = []
                temp_y = []
                temp_images = []
                temp_labels = []
                
                for state_dict in results:
                    x = state_dict['x']
                    y = state_dict['y']
                    images = state_dict['images']
                    labels = state_dict['labels']
                    
                    temp_x.append(x)
                    temp_y.append(y)
                    temp_images.append(images)
                    temp_labels.extend(labels)

                if self.load_with_images:
                    yield np.vstack(temp_x), np.vstack(temp_images), np.vstack(temp_y), temp_labels 
                
                else:
                    yield np.vstack(temp_x), np.vstack(temp_y), temp_labels

            else:
                if self.load_with_images:
                    yield results[0]['x'], results[0]['images'], results[0]['y'], results['labels'] 
                
                else:
                    yield results[0]['x'], results[0]['y'], results['labels']

    @staticmethod
    def load_device_data(device: Device) -> List[str]:
        """
        Downloads and caches device specificd data from hugging face

        Arg:
            device: Target device whose data is to be loaded

        Return:
            List: list of path to all the cached data loadable using joblib
        """

        REPO_ID = "wilfredk/labeled_data"

        repo_path = snapshot_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            token=HF_TOKEN,
            allow_patterns=f"{device.name}/*"
        )

        files = []
        for file in Path(repo_path).rglob("*"):
            if file.is_file():
                if device.name not in file.parts:
                    continue
                files.append(file)
        
        return files









    

 

    




    

    
