import os
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
import pandas as pd 
import numpy as np
import ast


data_path = Path("../spectral_data")
WEEK_NAME_INDEX = -1
WEEK_NO_INDEX = -5


class Device(Enum):
    BIO_SCIENCE = 0
    SCAN_CODER = 1
    LOW_COST = 2

class FileFormats(Enum):
    csv = ".csv"
    png = ".png"
    json = ".json"


class Dataset(ABC):
    """Abstract Dataclass from all dataset objects for loading data"""

    @abstractmethod
    def __len__(self, device):
        """Returns length of dataset"""
        raise NotImplementedError('Child class must implement __len__ method')
    

    def __getitem__(self, index):
        """Returns a single item"""
        raise NotImplementedError("Child class must implement __getitem__ method")
    

class SpectralDataset(Dataset):
    """Loads spectral dataset based on device code provided"""

    def __init__(self, data_path: str):
        self.data_path = data_path
    
        self.tracked_xlsx = [] # additional meta data
        self.tracked_csvs = []
        self.tracked_jsons = [] # irrelevant for computation

        self.high_end_csvs = []
        self.high_end_pngs = [] # irrelevant for computation

        self.low_cost_csvs = []
        self.low_cost_imgs = []

        self.high_end_raw_files = []
        self.high_end_calculation_files = []
        
        # Loading data
        self._load_fn()
        self.scan_corder_data = self._load_scan_corder_data()
        self.extract_high_end_raw_calculations()

    def __len__(self):
        """Returns length of all devices in order (Bio Science, Scan coder, Low cost)"""
        pass

    def len(self, device:Device):
        """Returns the lens of dataset based on the device code specified"""
        if device == Device.SCAN_CODER:
            return self.scan_corder_data.shape[0]
        elif device == Device.BIO_SCIENCE:
            return len(self.high_end_csvs)
        
        elif device == Device.LOW_COST:
            return len(self.low_cost_csvs)
        else:
            raise ValueError("Unsupported device")
    
    def __getitem__(self, index, device: Device):
        """Generic functions that dynamically determines the device to load for preprocessing"""
        if device == Device.LOW_COST:
            return self.get_low_cost_item(index, device)
        
        elif device == Device.BIO_SCIENCE:
            return self.get_high_end_item(index, device)
        
        elif device == Device.SCAN_CODER:
            return None
        else:
            raise ValueError("Unsupported device code provided")

    def clean_low_cost_cols(self, df, index):
        """Cleans and formats cols for the low cost device"""
        df_cols = df.columns
        target_cols = ['spectral_1', 'spectral_2', 'calibration']
        if 'wavelength' in df_cols:
            target_cols.extend(['wavelength'])

        extracted_vals = []
        for col in target_cols:
            data_v = ast.literal_eval(df[col][0])
            if isinstance(data_v,dict):
                data_v = data_v['intensity']
            data_v = np.array(data_v)
            extracted_vals.append(data_v)
        avg_spectral = (extracted_vals[0] + extracted_vals[1]) / 2
        wavelength = extracted_vals[-1] if 'wavelength' in target_cols else None
        return wavelength, avg_spectral

    def get_low_cost_item(self, index, device: Device):
        """Loads and preprocesses data for the low cost"""
        assert index <= 0 <= self.len(Device.LOW_COST), f'Index out of range, maximum supporte for low cost device is {self.len(Device.LOW_COST)}'
        spectral_url, img_url = self.low_cost_csvs[index], self.low_cost_imgs[index]
        specimen_df = pd.read_csv(spectral_url)
        specimen_df = self.clean_low_cost_cols(specimen_df, index)
        return specimen_df[0], specimen_df[-1], img_url
    
    def extract_high_end_raw_calculations(self):
        file_hash = dict()
        for reading in self.high_end_csvs:
            dynamic_label = reading.split('/')[-1]
            if 'calculations' in dynamic_label:
                label = dynamic_label.split('_')[0]
            else:
                label = dynamic_label.split('.')[0]
            
            if label not in file_hash:
                file_hash[label] = {'raw': None, 'calculations': None}

            if 'calculations' in dynamic_label:
                file_hash[label]['calculations'] = reading

            elif dynamic_label.endswith('.csv'):
                file_hash[label]['raw'] = reading
        
        _keys = list(file_hash.keys())
        for key in _keys:
            self.high_end_raw_files.append(file_hash[key]['raw'])
            self.high_end_calculation_files.append(file_hash[key]['calculations'])

        self.high_end_csvs, indices = SpectralDataset.remove_none(self.high_end_calculation_files)
        self.high_end_calculation_files, _ = SpectralDataset.remove_none(self.high_end_raw_files, indices)

    
    @staticmethod
    def identify_none(data: list):
        index_of_nones = []
        for i, x in enumerate(data):
            if x is None:
                index_of_nones.append(i)
        return index_of_nones
    
    @staticmethod
    def remove_none(data: list, indices= None):
        if indices is None:
            indices = SpectralDataset.identify_none(data)
        for index in sorted(indices, reverse=True):
            del data[index]
        return data, indices


    @staticmethod
    def extract_week(path):
        """Extracts the week from the scan corder directory"""
        return path.split('/')[WEEK_NAME_INDEX][WEEK_NO_INDEX]

    def get_index_of_latest_week(self):
        """Used to get index of latest tracked week by the scan corder"""
        current_week = SpectralDataset.extract_week(self.tracked_csvs[0])
        latest_index = 0
        for i, week_dir in enumerate(self.tracked_csvs):
            week = SpectralDataset.extract_week(week_dir)
            if week > current_week:
                latest_index = i
        return latest_index, self.tracked_csvs[latest_index]

    def _load_scan_corder_data(self):
        _, latest_week = self.get_index_of_latest_week()
        return pd.read_csv(latest_week)

    @staticmethod
    def clean_high_end_cols(df: pd.DataFrame):
        """Cleans the pandas columns of high end spectrometer"""
        return df.columns.str.strip().str.lower().str.replace(' ', '_')
    
    @staticmethod
    def convert_high_end_cols(df: pd.DataFrame):
        cols = df.columns
        no_cols = len(cols)
        for i in range(no_cols):
            idx = -(i + 1)
            if idx == -1:
                df[cols[idx]] = df.iloc[:, idx].str.replace("%", "").astype(np.float32)
            else:
                df[cols[idx]] = df.iloc[:, idx].astype(np.float32)
        return df
    
    def get_high_end_item(self, index):
        assert 0 <= index <= self.len(device=Device.BIO_SCIENCE), f'Index out of range max({self.len(device=Device.BIO_SCIENCE)})'
        
        raw_dir = self.high_end_raw_files[index]
        calculations_dir = self.high_end_calculation_files[index]

        raw_readings = pd.read_csv(raw_dir, skiprows=6, nrows=3648)
        calibration_values = pd.read_csv(raw_dir, skiprows=3662)

        raw_readings.columns = SpectralDataset.clean_high_end_cols(raw_readings)
        calibration_values.columns = SpectralDataset.clean_high_end_cols(calibration_values)

        raw_readings = SpectralDataset.convert_high_end_cols(raw_readings)
        return raw_readings, calibration_values        


    def _load_fn(self):
        """ Load pipeline for all devices. Populates all meta information available"""
        for root in os.listdir(data_path):
            root_dir = os.path.join(data_path, root)
            if root.endswith(".csv"):
                self.high_end_csvs.append(week_dir)
                continue
            for week in os.listdir(root_dir):
                week_dir = os.path.join(root_dir, week)
                if week.endswith(".xlsx"):
                    self.tracked_xlsx.append(week_dir)
                    continue
                elif week.endswith(".csv"):
                    self.high_end_csvs.append(week_dir)
                    continue
                for device in os.listdir(week_dir):
                    device_dir = os.path.join(week_dir, device)
                    for cat in os.listdir(device_dir):
                        cat_dir = os.path.join(device_dir, cat)
                        if cat.endswith(".csv"):
                            self.tracked_csvs.append(cat_dir)
                            continue
                        for disease_cat in os.listdir(cat_dir):
                            disease_cat_dir = os.path.join(cat_dir, disease_cat)
                            if disease_cat.endswith(".json"):
                                self.tracked_jsons.append(disease_cat_dir)
                                continue
                            for point in os.listdir(disease_cat_dir):
                                point_dir  = os.path.join(disease_cat_dir, point)
                                for specimen in os.listdir(point_dir):
                                    specimen_dir = os.path.join(point_dir, specimen)
                                    if specimen.endswith(".csv"):
                                        self.high_end_csvs.append(specimen_dir)
                                        continue
                                    elif specimen.endswith(".png"):
                                        self.high_end_pngs.append(specimen_dir)
                                        continue
                                    for sub in os.listdir(specimen_dir):
                                        sub_dir = os.path.join(specimen_dir, sub)
                                        if sub.endswith(".jpg"):
                                            self.low_cost_imgs.append(sub_dir)
                                        elif "data" in sub and sub.endswith(".csv"):
                                            self.low_cost_csvs.append(sub_dir)
                                        elif sub.endswith(".png"):
                                            self.high_end_pngs.append(sub_dir)
                                        elif sub.endswith(".csv") and "data" not in sub:
                                            self.high_end_csvs.append(sub_dir)                         



