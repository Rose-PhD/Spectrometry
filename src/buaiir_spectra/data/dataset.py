import os
import re
import ast
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
from joblib import Parallel, delayed
from PIL import Image
from enum import Enum
import concurrent.futures
from huggingface_hub import snapshot_download
from buaiir_spectra.utils.config_info import RAW_ACCESS
import joblib
from importlib import resources


SEED = 42
DATA_PATH = Path('/home/wilfred/Downloads/spectra_data_clean')
WAVLENGTH_PATH = resources.files("buaiir_spectra").joinpath("wavelength_and_cols_ds.pkl")
np.random.seed(SEED)


# SETUP MACROS FOR FASTER DATALOADING
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"

def download_rawfiles(des_path: str= "./spectra_data") -> Path:
    snapshot_download(
        repo_id="wilfredk/raw_dataset",
        repo_type="dataset",
        local_dir=des_path,
        token=RAW_ACCESS,
        max_workers=1
    )
    return Path(des_path)


class B_Device:
    
    @staticmethod
    def get_week_groups(root):
        week_groups = defaultdict(list)
        week_pattern = re.compile(r'^[BCM](\d+)')
      
        for file in root.rglob("*"):
            if file.is_file() and "Spectrometer" in file.parts:
                m = week_pattern.match(file.stem)
                if m:
                    week = int(m.group(1))
                    week_groups[week].append(file)
        return dict(sorted(week_groups.items()))

    @staticmethod
    def get_crop_groups(week_groups, week_value):
        crop_groups = defaultdict(list)
        for file in week_groups[week_value]:
            parts = file.parts
            crop = parts[parts.index("Spectrometer") + 1].capitalize()
            crop_groups[crop].append(file)
        return crop_groups
    
    @staticmethod
    def get_reading_group(crop_group, crop):
        reading_groups = defaultdict(list)
        for file in crop_group[crop]:
            reading = file.parent.name 
            reading_groups[reading].append(file)
    
        return reading_groups

    @staticmethod
    def get_label_groups(reading_group, group_code):
        label_pattern = re.compile(r'^(.*?)(?=Ra\d+[bg])')
        label_groups = defaultdict(list)
        for file in reading_group[group_code]:
            m = label_pattern.match(file.stem)
            if m:
                label = m.group(1)
                label_groups[label].append(file)
        return dict(sorted(label_groups.items()))

    @staticmethod
    def get_label_data(group_labels, label):
        reading_pattern = re.compile(r'(Ra\d+[a-zA-Z])')
        reading_groups = defaultdict(list)
        for file in group_labels[label]:
            m = reading_pattern.search(file.stem)
            if m:
                reading = m.group(1) 
                if file.suffix == '.csv' and "_calculations" not in file.stem:
                    reading_groups[reading].append(file)
        return dict(sorted(reading_groups.items()))

    @staticmethod
    def get_all_labels(group_labels):
        dfs = []
        for label in group_labels.keys():
            readings = B_Device.get_label_data(group_labels, label)
            df  = pd.DataFrame({'R': readings.keys(),  'Files': readings.values()})
            df['Label'] = label
            dfs.append(df)
        df = pd.concat(dfs, axis=0)
        col_order = ['R','Label', 'Files']
        df = df[col_order]
        return df, col_order

    @staticmethod
    def get_all_R_data(reading_groups):
        dfs = []
        for R in reading_groups.keys():
            group_labels = B_Device.get_label_groups(reading_groups, R)
            all_labels, cols_order = B_Device.get_all_labels(group_labels)
            dfs.append(all_labels)
    
        df = pd.concat(dfs, axis=0)
        df = df[cols_order]
        return df, cols_order
    
    @staticmethod
    def get_all_crop_data(crop_group):
        dfs = []
        def get_crop_code(crop):
            crop_pattern = re.compile(r"^([A-Za-z])")
            match = crop_pattern.search(crop)
            return match.group(1)
             
        for crop in sorted(crop_group.keys()):
            reading_group = B_Device.get_reading_group(crop_group, crop)
            df, cols_order = B_Device.get_all_R_data(reading_group)
            df['Crop'] = get_crop_code(crop)
            dfs.append(df)
        df = pd.concat(dfs, axis=0).reset_index(drop=True)
        cols_order = ['Crop'] + cols_order
        df = df[cols_order]
        return df, cols_order
            
    @staticmethod
    def get_all_data(week_groups):
        dfs = []
    
        def get_disease_class(labels):
            class_pattern = re.compile(r"^[A-Z]([A-Z]+)\d+$")
            return labels.str.extract(class_pattern, expand=False)
        for week in sorted(week_groups.keys()):
            crop_groups = B_Device.get_crop_groups(week_groups, week)
            df, cols_order = B_Device.get_all_crop_data(crop_groups)
            df['Week'] = week
            dfs.append(df)
    
        df = pd.concat(dfs, axis=0).reset_index(drop=True)
        df["Label"] = df["Label"].str.replace( r"^([BCM])\d+([A-Z]+\d+)$", r"\1\2",regex=True)
        df["Class"] = get_disease_class(df["Label"])
        cols_order = ['Week', "Class"] + cols_order
        df = df[cols_order]
        return df

    
    @staticmethod
    def class_counts(week, final_df):
        count = (pd.Series(final_df.loc[final_df["Week"] == week, "Label"].unique())
            .str.replace(r"\d+$", "", regex=True)
            .value_counts()
            .sort_index())
        return count

    @staticmethod
    def read_csv(path: str):
        headers = []
        sections  = []
        current_section = []
        number_pattern = re.compile(r"^\s*-?\d+(\.\d+)?([eE]-?\d+)?%?\s*$")
        
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    if current_section:
                        sections.append(current_section)
                        current_section= []
                        continue
    
                    continue
                first_token = line.split(",", 1)[0]
                if not number_pattern.match(first_token):
                    headers.append(line)
    
                else:
                    current_section.append(line)
        if current_section:
            sections.append(current_section)
    
        RAW_SPECTRAL_INDEX = -3
        raw_lines = sections[RAW_SPECTRAL_INDEX]
        
        array = np.vstack([
            np.fromstring(
                line.replace("%", ""),
                sep=",",
                dtype=np.float32
            )
            for line in raw_lines
        ])
    
        columns = headers[RAW_SPECTRAL_INDEX].split(",")
        return  pd.DataFrame(array, columns=columns)

    @staticmethod
    def merge(primal_df, img_df):
        extracted_nums = primal_df['R'].str.extract(r'^Ra(\d+)', expand=False).astype(int)
        img_df = img_df.rename(columns= {'R': 'R_new'})
        def map_r_code(num):
            if num == 1:
                return 'R1'
            elif num == 2:
                return 'R2'
            else:
                return 'R3' 
        
        primal_df['R_new'] = extracted_nums.map(map_r_code)
        primal_df = pd.merge(primal_df, img_df, how='left', on=['Week','Crop', 'Class', 'Label', 'R_new'])
        return primal_df

    @staticmethod
    def load(val: pd.Series):
        x_files = val['Files']
        img = val['Images']
        try:
            img = Image.open(img).resize((224, 224), resample=Image.Resampling.LANCZOS)
            img = np.asarray(img)
            y = val.drop(['Images', 'Files', 'R_new'])

        except:
            print(f'Operation failed at: \n', val)
            raise
    
        num_data = len(x_files)
        # 
        
        if num_data > 1:
            x = []
            for file in x_files:
                x_i = B_Device.read_csv(file).iloc[:, -1].values
                x.append(x_i)
    
            y_values = [y.values for _  in range(num_data)]
            img = [img[None, :] for _ in range(num_data)]
    
            x = np.vstack(x).astype(np.float32)
            y = pd.DataFrame(data=y_values, columns= y.index.to_list())
            img = np.vstack(img)
        else:
            x = B_Device.read_csv(x_files[0]).iloc[:, -1].values

        # extract unique for ndims > 3
        if img.ndim > 3:
            x = x[0]
            y = y.iloc[0, :]
            img = img[0]
    
        return x, img, y


# COMPUTATION PIPELINE FOR THE LOW COST DEVICE
class L_Device:
    
    @staticmethod
    def get_week_groups(root):
        week_groups = defaultdict(list)
        week_pattern = re.compile(r"spectral_data_week(\d+)")
      
        for file in root.rglob("*"):
            if file.is_file() and "Low-cost" in file.parts:
                m  = week_pattern.search(str(file))
                if m:
                    week = int(m.group(1))
                    week_groups[week].append(file)
        week_groups = dict(sorted(week_groups.items()))
        return week_groups
        
    @staticmethod
    def get_crop_groups(week_groups, week_value):
        crop_groups = defaultdict(list)
        crop_pattern = re.compile(r"Reading\s+\d+/([^/]+)/")
        week_group = week_groups[week_value]
        for file in week_group:
            match = crop_pattern.search(file.as_posix())
            if match:
                crop = match.group(1).capitalize()
                crop_groups[crop].append(file)
    
        return crop_groups
        
    @staticmethod
    def get_reading_groups(crop_groups, crop):
        reading_pattern = re.compile(r"(Reading\s+\d+)")
        reading_groups = defaultdict(list)
    
        files = crop_groups[crop]
    
        for file in files:
            match = reading_pattern.search(file.as_posix())
            if match:
                reading = match.group(1)
                reading_groups[reading].append(file)
    
        return dict(sorted(reading_groups.items()))
    
    @staticmethod
    def get_disease_groups(reading_groups, disease_code):
        disease_groups = defaultdict(list)
        disease_pattern = re.compile(r"Reading\s+\d+/[^/]+/([^/]+)/")
        files = reading_groups[disease_code]
    
        for file in files:
            match = disease_pattern.search(file.as_posix())
            if match:
                disease = match.group(1).upper()
                disease_groups[disease].append(file)
    
        return disease_groups
        
    @staticmethod
    def get_label_groups(disease_groups, disease_code):
        crop_number_pattern = re.compile(r"scan_[A-Z]\d+_HLT(\d+)_")
        crop_number_groups = defaultdict(list)
        files = disease_groups[disease_code]
    
        for file in files:
            match = crop_number_pattern.search(file.as_posix())
            if match:
                crop_number = int(match.group(1))
                crop_number_groups[crop_number].append(file)
    
        return crop_number_groups

    @staticmethod
    def get_separate_modalities(label_groups, label):
        spectra_files = []
        image_files = []
        files = label_groups[label]
        for file in files:
            if file.name == "data.csv":
                spectra_files.append(file)
            elif file.name == "leaf_image.jpg":
                image_files.append(file)
    
        df = pd.DataFrame({'Files': spectra_files, 'Images': image_files})
        df['Label'] = label
        col_order = ['Label', 'Files', 'Images']
        df = df[col_order]
        return df, col_order
    
    @staticmethod
    def get_all_label_data(label_groups):
        dfs= []
        for label in sorted(label_groups.keys()):
            df, col_order = L_Device.get_separate_modalities(label_groups, label)
            dfs.append(df)
        df = pd.concat(dfs, axis=0).reset_index(drop=True)
        return df, col_order
        
    @staticmethod
    def get_all_disease_data(disease_groups):
        dfs = []
        for disease_code in disease_groups.keys():
            label_groups = L_Device.get_label_groups(disease_groups, disease_code)
            df, col_order = L_Device.get_all_label_data(label_groups)
            df['Class'] = disease_code
            dfs.append(df)
    
        df = pd.concat(dfs, axis=0).reset_index(drop=True)
        col_order = ['Class'] + col_order
        df = df[col_order]
        return df, col_order
        
    @staticmethod
    def get_all_reading_data(reading_groups):
        dfs = []
        def get_R_code(R):
            reading_pattern = re.compile(r"Reading\s+(\d+)")
            match = reading_pattern.search(R)
            return  'R' + str(match.group(1))
    
        for R in reading_groups.keys():
            disease_groups = L_Device.get_disease_groups(reading_groups, R)
            df, col_order = L_Device.get_all_disease_data(disease_groups)
            df['R'] = get_R_code(R)
            dfs.append(df)
    
        df = pd.concat(dfs, axis=0)
        col_order = ['R'] + col_order
        df = df[col_order]
        return df, col_order
    
    @staticmethod
    def get_all_crop_data(crop_groups):
        dfs = []
        def get_crop_code(crop):
            crop_pattern = re.compile(r"^([A-Za-z])")
            match = crop_pattern.search(crop)
            return match.group(1)
        for crop in crop_groups:
            reading_groups = L_Device.get_reading_groups(crop_groups, crop)
            df, col_order = L_Device.get_all_reading_data(reading_groups)
            df['Crop'] = get_crop_code(crop)
            dfs.append(df)
    
        df = pd.concat(dfs, axis=0).reset_index(drop=True)
        col_order = ['Crop'] + col_order
        df = df[col_order]
        return df, col_order
            
    @staticmethod
    def get_all_data(week_groups):
        results = []
        def process_week(week):
            crop_groups = L_Device.get_crop_groups(week_groups, week)
            df, col_order = L_Device.get_all_crop_data(crop_groups)
            df['Week'] = week
            return df, col_order
    
        results = Parallel(n_jobs=-1, prefer="processes")(
            delayed(process_week)(week)
            for week in week_groups.keys()
        )
        dfs = [df for df, _ in results]
        df = pd.concat(dfs, axis=0, ignore_index=True)
        col_order = ['Week'] + results[0][1] 
        df['Label']  = df['Crop'] + df['Class'] + df['Label'].astype(str)
        df = df[col_order]
        return df

    @staticmethod
    def load(val: pd.Series):
        x_f = pd.read_csv(val['Files'])
        image_f = val['Images']
        y_f = val.drop(['Files', 'Images'])
    
        spec_1 = ast.literal_eval(x_f.loc[:, 'spectral_1'][0])
        spec_2 = ast.literal_eval(x_f.loc[:, 'spectral_2'][0])
    
        if isinstance(spec_1, dict):
            spec_1 = spec_1['intensity']
            spec_2 = spec_2['intensity']
            calibration = ast.literal_eval(x_f.loc[: "calibration"][0])["intensity"]
        else:
            calibration = ast.literal_eval(x_f.loc[:, "calibration"][0])

        # Type conversion
        calibration = np.array(calibration, dtype=np.float32)
        spec_1 = np.array(spec_1, dtype=np.float32)
        spec_2 = np.array(spec_2, dtype=np.float32)

        # calibration
        EPS =   1e-6
        EPS_1 = np.mean(spec_1) / np.maximum(np.mean(calibration), EPS)
        EPS_2 = np.mean(spec_2) / np.maximum(np.mean(calibration), EPS)

        spec_1 = (spec_1 + calibration) / np.maximum(spec_1 - calibration, EPS_1)
        spec_2 = (spec_2 + calibration) / np.maximum(spec_2 - calibration, EPS_2)
    
        spec_1 = spec_1[None, :]
        spec_2 = spec_2[None,: ]
    
        x = np.vstack([spec_1, spec_2])
        y = pd.DataFrame(data= [y_f.values, y_f.values], columns= y_f.index.to_list())

        try:
            img = Image.open(image_f).resize((244, 244), resample= Image.Resampling.LANCZOS)
            img = np.asarray(img)[None, :]
            img = np.vstack([img, img])
        except:
            print(f'Operation failed at: \n', val)
            raise
    
        return x, img, y

# COMPUTATION PIPELINE FOR THE SCAN CORDER DEVICE
class S_Device:
    
    @staticmethod
    def get_week_files(root):
        files = []
        for file in root.rglob("*"):
            if file.is_file() and "Scancorder" in file.parts and file.suffix == '.csv':
                    files.append(file)
        week_pattern = re.compile(r"spectral_data_week(\d+)")
        files = sorted(files, key=lambda p: int(week_pattern.search(p.as_posix()).group(1)))
        return files
        
    @staticmethod
    def extract_sample_info(series):
        sample_pattern = re.compile(
            r"^(?P<Crop>[A-Z])\d+\s+"
            r"(?P<Class>[A-Z]+)(?P<No>\d+)\s+"
            r"(?P<Reading>R[a-z])\s+"
            r"(?P<Weight>\d+[a-z])$"
        )
        info = series.str.extract(sample_pattern, expand=True)
        info["No"] = info["No"].astype(int)
        info["R"] = info["Reading"] + info["Weight"]
        info['Label'] = info['Crop'] + info['Class'] + info['No'].astype(str)
        
        return info[["Crop", "Class", "R", "Label"]]
    @staticmethod
    def get_week_data(files, week):
        df = pd.read_csv(files[week]).dropna(axis=1).drop(columns=['UUID', 'Timestamp'])
        meta_df = S_Device.extract_sample_info(df['Sample ID'])
        df = pd.concat([meta_df, df], axis=1).drop(columns=['Sample ID'])
        
        week_pattern = re.compile(r"spectral_data_week(\d+)")
        week = int(week_pattern.search(files[week].as_posix()).group(1))
        df['Week'] = week 
        col_order = ['Week'] + [c for c in df.columns if c != 'Week' and 'Unnamed' not in c]
        df = df[col_order]
        return df
    
    @staticmethod
    def get_all_data(files):
        dfs = Parallel(n_jobs=-1, prefer='processes')(
            delayed(S_Device.get_week_data)(files, week)
            for week in range(len(files))
        )
    
        df =  pd.concat(dfs, axis=0, ignore_index=True)
        return df


    @staticmethod
    def load(val: pd.Series):
        x = val[5:17].values.astype(np.float32)
        img_f = val['Images']
        cols = val.index.to_list()
        cols_to_drop = [cols[i] for i in np.r_[5:17].astype(int)]
        
        y = val.drop(cols_to_drop).drop(['Images', 'R_new'])
        try:
            img = Image.open(img_f).resize((224, 224), resample=Image.Resampling.LANCZOS)
            img = np.asarray(img)
        except:
            print(f'Operation failed at: \n', val)
            raise
    
        return x, img, y

    @staticmethod
    def merge_imgs(primal_df, img_df):
        R_mapping = {
            'Ra1g': 'R1', 
            'Ra1b': 'R1',
            'Rb1g': 'R2', 
            'Rb1b': 'R2',
            'Rc1g': 'R3', 
            'Rc1b': 'R3'
        }
        primal_df['R_new'] = primal_df['R'].map(R_mapping)
        img_df = img_df.rename(columns={'R': 'R_new'})
        primal_df = pd.merge(primal_df, img_df, how='left', on= ['Week', 'Crop', 'Class', 'Label', 'R_new'])
        
        return primal_df


class Device(Enum):
    # Device wrapper to avoid not support device errors 
    BIO_SCIENCE = 0
    SCAN_CORDER =  1
    LOW_COST  = 2

    @classmethod
    def get_devices(cls):
        return [member.name for member in cls]

    @classmethod
    def get_devices(cls):
        return list(cls)

# BUILDING ACCESS TO WAVELENGTH DATA
wavelength = joblib.load(WAVLENGTH_PATH)
device_keys = wavelength.keys() # corrupted keys
correct_keys = Device.get_devices() # clean keys

device_dict = {correct_key_pos: device_key_pos for correct_key_pos, device_key_pos in zip(correct_keys, device_keys)} # build key maps

# define wavlenght 
def get_wavelength(device: Device, wavelength=wavelength, device_dict=device_dict):
    return wavelength[device_dict[device]]['wavelenght']


class SpectralDataset:

    def __init__(self, root, device: Device):
        # Compute Expert data
        lab_df_path = os.path.join(root, 'laboratory_data.xlsx')
        lab_df = pd.read_excel(lab_df_path)

        # Load Low cost as base case to extract Images
        week_groups = L_Device.get_week_groups(root)
        final_df = L_Device.get_all_data(week_groups)
        filter_labels = ['Week', 'Crop', 'Class', 'Label']

        if not device == Device.LOW_COST:

            # Extract images files --> now safe to overwrite low cost object
            img_df = final_df.loc[:, filter_labels + ['R', 'Images']]
            
            if device == Device.SCAN_CORDER:
                files = S_Device.get_week_files(root)   # Get Unique weeks
                final_df = S_Device.get_all_data(files) # Load all data on weekly basis
                final_df = pd.merge(final_df, lab_df, how='left', on=filter_labels) # Merge with Expert data
                final_df = S_Device.merge_imgs(final_df, img_df)  # Merge with Images
                
            elif device == Device.BIO_SCIENCE:
                week_groups = B_Device.get_week_groups(root) # Get all Unique weeks
                final_df = B_Device.get_all_data(week_groups) # Load all data on weekly basis
                final_df = pd.merge(final_df, lab_df, how='left', on= filter_labels) # Merge with Expert data
                final_df = B_Device.merge(final_df, img_df) # Merge with Images

        else:
            final_df = pd.merge(final_df, lab_df, how='left', on=filter_labels) # Merge with Expert data only if LOW COST
            
        # Keep Buffers for futher computations 
        self.final_df = final_df 
        self.device = device
        self.wavelength = get_wavelength(device)
            
    def __len__(self):
        return len(self.final_df)


    def __getitem__(self, index):

        val = self.final_df.iloc[index, :] 
        # Switch load function based on device
        if self.device == Device.LOW_COST:
            return L_Device.load(val)

        elif self.device == Device.BIO_SCIENCE:
            return B_Device.load(val)

        elif self.device == Device.SCAN_CORDER:
            return S_Device.load(val)
        else:
            raise ValueError(f'Device not supported')


class SpectralDataLoader:
    
    def __init__(self, dataset: SpectralDataset, batch_size=4, shuffle=False, random_seed=42):
        self.dataset = dataset
        self.batch_size = batch_size
        # compute the size of dataset
        num_data = len(dataset)
        indices = np.arange(num_data)
        np.random.seed(random_seed)
        if shuffle:
            indices = np.random.permutation(indices)

        self.indices = indices
        self.num_data = num_data


    def __len__(self):
        return self.num_data // self.batch_size

    def __iter__(self):
        max_workers = os.cpu_count() * 2
        for i in range(0, self.num_data, self.batch_size):
            selected_pos = self.indices[i: i + self.batch_size]

            X, Images, Y = [], [], []

            def load_single_item(index):
                x, img, y = self.dataset[index]
                if img.ndim < 4:
                    img = img[None, :]
                return x, img, y

            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                results = executor.map(load_single_item, selected_pos)

            # Unpkac the parallized results
            for x, img, y in results:
                X.append(x)
                Images.append(img)
                Y.append(y)

            # Process target data (Y)
            if isinstance(Y[0], pd.Series):
                cols = Y[0].index.to_list()
                Y_values = [y.values for y in Y]
                Y = pd.DataFrame(data = Y_values, columns = cols)

            else:
                Y = pd.concat(Y, axis=0)

            yield np.vstack(X), np.vstack(Images), Y
                        