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

    @classmethod
    def get_devices(cls):
        """Returns the names of the devices in the enum"""
        return list(cls)
    
    @classmethod
    def get_device_names(cls):
        return [device.name for device in cls]



class FileFormats(Enum):
    """Data file formats"""
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

    def __init__(self, data_path: str, device: Device):
        """Shared state across all devices"""
        self.data_path = data_path
        self.device = device
        self.wavelength = None
        self.DONE_COMPUTING_WAVELENGTH = False
        self.tracked_xlsx = []

        """Dynamic state declaration"""
        if device == Device.SCAN_CODER: 
            self.tracked_csvs = []

        elif device == Device.BIO_SCIENCE:
            self.high_end_csvs = []
            self.high_end_raw_files = []
            self.high_end_calculation_files = []
        elif device == Device.LOW_COST:
            self.low_cost_csvs = []
            self.low_cost_imgs = []
            

        """ Dynamic load functions"""
        self._load_fn()
        if device == Device.SCAN_CODER:
            # self.scan_corder_data = self._load_scan_corder_data()
            self.scan_corder_data = self.get_scan_corder_data()
        elif device == Device.BIO_SCIENCE:
            self.extract_high_end_raw_calculations()
        

    def __len__(self):
        """Computest the length of the state obj of the current device"""
        _len = None
        
        if self.device == Device.BIO_SCIENCE:
            _len = len(self.high_end_raw_files)
        elif self.device == Device.SCAN_CODER:
            _len = self.len_scan_corder_data()
        elif self.device == Device.LOW_COST:
            _len = len(self.low_cost_csvs) 
        return _len

    
    def __getitem__(self, index):
        """Generic function for extraction of item from data"""
        if self.device == Device.SCAN_CODER:
            return self.get_scan_corder_item(index)
        elif self.device == Device.LOW_COST:
            return self.get_low_cost_item(index)
        else:
            return self.get_high_end_item(index)

        
    def len_scan_corder_data(self):
        """Computes the lenght of scan corder data"""
        return self.scan_corder_data.shape[0]

    def get_scan_corder_item(self, index):
        """Returns a scan corder reading by index"""
        LOW_TARGET_INDEX_BOUND = 3
        _len = self.len_scan_corder_data()
        assert 0 <= index <= _len, f'Index out of range max({_len})'
        _values = self.scan_corder_data.iloc[index][LOW_TARGET_INDEX_BOUND: ].values
        return np.array(_values).astype(np.float32)


    def clean_low_cost_cols(self, df, index):
        """Cleans and formats cols for the low cost device"""
        df_cols = df.columns
        target_cols = ['spectral_1', 'spectral_2', 'calibration']
        _includes_wavelength = 'wavelength' in df_cols
   
        if  _includes_wavelength and not self.DONE_COMPUTING_WAVELENGTH:
            target_cols.extend(['wavelength'])

        extracted_vals = []
        for col in target_cols:
            data_v = ast.literal_eval(df[col][0])
            if isinstance(data_v,dict):
                data_v = data_v['intensity']
            data_v = np.array(data_v)
            extracted_vals.append(data_v)
        # avg_spectral = (extracted_vals[0] + extracted_vals[1]) / 2
        if not self.DONE_COMPUTING_WAVELENGTH and _includes_wavelength:
            self.wavelength = extracted_vals[-1].astype(np.float32)
        return np.vstack([extracted_vals[0], extracted_vals[1]]).astype(np.float32)

    def get_low_cost_item(self, index):
        """Loads low cost spectral item"""
        _len = len(self)
        assert index <= index <= _len, f'Index out of range,  max({_len})'
        spectral_url, img_url = self.low_cost_csvs[index], self.low_cost_imgs[index]
        specimen_df = pd.read_csv(spectral_url)
        return self.clean_low_cost_cols(specimen_df, index)
        

    @staticmethod
    def get_label(reading):
        """Concatenates the plant label and period for unique labels"""
        PERIOD_INDEX, NAME_INDEX = 3, -1
        reading = reading.split('/')
        _period, dynamic_name = reading[PERIOD_INDEX], reading[NAME_INDEX]
        dynamic_label = dynamic_name
        _period = _period.split('_')[-1][-1]
        if 'calculation' in dynamic_name:
            dynamic_name = dynamic_name.split('_')[0]
        else:
            dynamic_name = dynamic_name.split('.')[0]
        return dynamic_name + _period, dynamic_label

    @staticmethod
    def get_week_for_scan_corder_reading(reading):
        """Extracts the scan corder week"""
        return reading.split('/')[-1].split('.')[0][-1]
    
    def get_scan_corder_data(self):
        """Concatenates the tracked csvs for the scan corder from the data"""
        dfs = []
        NULL_COL = 'Token ID'
        for week_data in self.tracked_csvs:
            df = pd.read_csv(week_data).drop(columns=NULL_COL)
            _period = SpectralDataset.get_week_for_scan_corder_reading(week_data)
            df['week'] = _period
            dfs.append(df)
        return pd.concat(dfs)
    
    def extract_high_end_raw_calculations(self):
        file_hash = dict()
        for reading in self.high_end_csvs:
            label, dynamic_label = SpectralDataset.get_label(reading)
            if label not in file_hash:
                file_hash[label] = {'raw': None, 'calculation': None}
            
            if 'calculation' in dynamic_label:
                file_hash[label]['calculation'] = reading
            elif dynamic_label.endswith('.csv'):
                file_hash[label]['raw'] = reading
            
            else:
                raise ValueError('Unsupported file format')
        _keys = list(file_hash.keys())
        for key in _keys:
            self.high_end_raw_files.append(file_hash[key]['raw'])
            self.high_end_calculation_files.append(file_hash[key]['calculation'])
        
        self.high_end_calculation_files, indices = SpectralDataset.remove_none(self.high_end_calculation_files)
        self.high_end_raw_files, _ = SpectralDataset.remove_none(self.high_end_raw_files, indices)

    
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
        """Loads a single item for high end spectrometer"""
        _len = len(self)
        assert 0 <= index <= _len, f'Index out of range max({_len})'
        
        raw_dir = self.high_end_raw_files[index]
        calculations_dir = self.high_end_calculation_files[index]

        raw_readings = pd.read_csv(raw_dir, skiprows=6, nrows=3648)
        raw_readings.columns = SpectralDataset.clean_high_end_cols(raw_readings)
    
        """comment out for computation of calibration values"""
        # calibration_values = pd.read_csv(raw_dir, skiprows=3662)
        # calibration_values.columns = SpectralDataset.clean_high_end_cols(calibration_values)

        raw_readings = SpectralDataset.convert_high_end_cols(raw_readings)
        if not self.DONE_COMPUTING_WAVELENGTH:
            self.wavelength = np.array(raw_readings['wavelength'].values).astype(np.float32)
        
        return raw_readings[raw_readings.columns[-1]].values.astype(np.float32)
   


    def _load_fn(self):
        """ Load pipeline for all devices. Populates all meta information available"""
        for root in os.listdir(data_path):
            root_dir = os.path.join(data_path, root)
            if root.endswith(".csv"):
                if self.device == Device.BIO_SCIENCE:
                    self.high_end_csvs.append(week_dir)
                continue
            for week in os.listdir(root_dir):
                week_dir = os.path.join(root_dir, week)
                if week.endswith(".xlsx"):
                    self.tracked_xlsx.append(week_dir)
                    continue
                elif week.endswith(".csv"):
                    if self.device == Device.BIO_SCIENCE:
                        self.high_end_csvs.append(week_dir)
                    continue
                for device in os.listdir(week_dir):
                    device_dir = os.path.join(week_dir, device)
                    for cat in os.listdir(device_dir):
                        cat_dir = os.path.join(device_dir, cat)
                        if cat.endswith(".csv"):
                            if self.device == Device.SCAN_CODER:
                                self.tracked_csvs.append(cat_dir)
                            continue
                        for disease_cat in os.listdir(cat_dir):
                            disease_cat_dir = os.path.join(cat_dir, disease_cat)
                            if disease_cat.endswith(".json"):
                                continue
                            for point in os.listdir(disease_cat_dir):
                                point_dir  = os.path.join(disease_cat_dir, point)
                                for specimen in os.listdir(point_dir):
                                    specimen_dir = os.path.join(point_dir, specimen)
                                    if specimen.endswith(".csv"):
                                        if self.device == Device.BIO_SCIENCE:
                                            self.high_end_csvs.append(specimen_dir)
                                        continue
                                    elif specimen.endswith(".png"):
                                        continue
                                    for sub in os.listdir(specimen_dir):
                                        sub_dir = os.path.join(specimen_dir, sub)
                                        if sub.endswith(".jpg"):
                                            if self.device == Device.LOW_COST:
                                                self.low_cost_imgs.append(sub_dir)
                                            continue
                                        elif "data" in sub and sub.endswith(".csv"):
                                            if self.device == Device.LOW_COST:
                                                self.low_cost_csvs.append(sub_dir)
                                            continue
                                        elif sub.endswith(".png"):
                                            continue
                                        elif sub.endswith(".csv") and "data" not in sub:
                                            if self.device == Device.BIO_SCIENCE:
                                                self.high_end_csvs.append(sub_dir)
                                            continue                         



