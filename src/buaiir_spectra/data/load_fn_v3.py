import numpy as np
import re
import pandas as pd
import joblib
from buaiir_spectra.utils.device import Device
from buaiir_spectra.utils.config_info import SEED, HF_TOKEN, REPO_ID
from pathlib import Path
from importlib.resources import files
from huggingface_hub import snapshot_download
from joblib import Parallel, delayed, load


np.random.seed(SEED)
META_PATH = files("buaiir_spectra").joinpath("wavelength_and_cols_ds.pkl")
GLOBAL_META_DICT = joblib.load(META_PATH)


class load_spectral:

    def __init__(self, device: Device, batch_size=-1, shuffle=False):
        self.device = device
        self.batch_size = batch_size
        self.wavelength = GLOBAL_META_DICT[device]['wavelenght']

        sorted_files = self.load_files()
        indices = np.arange(len(sorted_files))

        if shuffle:
            indices = np.random.permutation(indices)
        
        self.sorted_files = sorted_files
        self.indices = indices


    
    @staticmethod
    def load_device_data(device: Device):
        repo = snapshot_download(repo_id=REPO_ID, repo_type="dataset", token=HF_TOKEN, allow_patterns=f"{device.name}/*")

        files = []
        for file in Path(repo).rglob("*"):
            if file.is_file():
                if device.name not in file.parts:
                    continue
                files.append(file)

        return files


    def load_files(self):

        def sort_key(filename, device):
            filename = filename.name
            pattern = rf'{device.name}_(\d+)'
            match = re.search(pattern, filename)

            if match:
                return int(match.group(1))
            return -1

        BATCH_PATHS = self.load_device_data(self.device)
        return sorted(BATCH_PATHS, key=lambda f: sort_key(f, self.device))
    


    def __call__(self):
        if not self.batch_size == -1:
            raise TypeError(f'Only callable if loading all data in memory i.e. set batch_size=-1')
        return next(self.__iter__())


    def __iter__(self):
        batch_size = self.batch_size
        if self.batch_size == -1:
            batch_size = len(self.indices)

        for i in range(0, len(self.sorted_files), batch_size):
            selected_pos = self.indices[i: i + batch_size]
            selected_files = [self.sorted_files[i] for i in selected_pos]

            results = Parallel(n_jobs=-1, prefer="processes")(
                delayed(joblib.load)(file_path) for file_path in selected_files
            )

            X, Images, Y = [], [], []
            for loaded_data in results:
                x = loaded_data['X']
                y = loaded_data['y']
                img = loaded_data['images']

                X.append(x)
                Y.append(y)
                Images.append(img)

            
            yield np.vstack(X), np.vstack(Images), pd.concat(Y, axis=0)







        


