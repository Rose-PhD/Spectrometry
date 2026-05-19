import os 
import pandas as pd
import numpy as np
import ast
from pathlib import Path
import re
from buaiir_spectra.data.dataset import Dataset
from buaiir_spectra.utils.device import Device
from typing import List, Tuple



class SpectralDataset_v2(Dataset):
    def __init__(self, data_path, device: Device, label_extra_week: int= 9) -> None:
        """
        Class wrapper for loading spectral data for specific device

        Args:
            data_path: root to source of data
            device: target device whose data is required
            label_extra_week: week to consider will loading sample labels
        """
        # Dataset configurations                              
        self.data_path = data_path
        self.device = device
        self.label_extra_week = label_extra_week
        
        # Lables & meta data                                
        self.labels = None  
        self.meta_data = None

        # Data containers                                
        self.weeks = dict()                                 
        self.expert_files = list()                         
        
        # Load states
        self._load_fn()                                     
        self.weeks = dict(sorted(self.weeks.items()))        
        self._load_meta_data()

        # Data alignment Configurations
        if self.device == Device.LOW_COST:
            self.MLN_SHIFT = 5                              
            self.MSV_SHIFT = 0                              
            self.CMD_SHIFT = 5                              
            self.CBB_SHIFT = 0 
        
        # Load expert states after alignment                             
        self.populate_expert_readings()
        self.create_disease_embeddings()
        self.create_plant_embeddings()

    
    @staticmethod
    def _normalize_to_spectra_data(path: str, root_name:str = "spectra_data") -> str:
        """
        Extracts new path from idx of spectra_data 

        Args:
            path:str -> directory where data is store or downloaded to 
            root_name: str -> key to search for in the given path

        Return:
            new_path: str -> updated path 
        """
        path = Path(path).resolve()
        parts = path.parts

        # check for root name
        if root_name not in parts:
            raise ValueError(f'{root_name} not found in path')
        
        idx = parts.index(root_name)
        new_path = Path(*parts[idx:])
        return new_path
        


    @staticmethod
    def get_week(path: str) -> None:
        """
        Extract the week from specimen data path

        Args:
            path -> str: path containing label from which the week is extracted
        
        Returns:
            None
        """
        match = re.search(r'week(\d+)', path.lower())
        if match:
            return int(match.group(1))
        raise ValueError(f'Week not found in path: {path}')
    

    @staticmethod
    def extract_low_cost_label(path: str) -> None:
        """
        Extracts labels for low cost device

        Args:
            path -> str: path to file system where low cost reading is stored
        """
        label = path.split('/')[-2].replace('_', '').replace('scan', '')
        return label[:label.find('Ra')]

    def populate_data_by_week_BIO_SCEINCE(self, parent_dir, specimen_data):
        """Extracts and populates data on a weekly basis"""
        specimen_data_dir = os.path.join(parent_dir, specimen_data)
        week = self.get_week(specimen_data_dir)

        # Handl creation of new weeks
        if week not in self.weeks:
            self.weeks[week] = {}
        
        # Extract label from specimen data path
        label = None
        if 'calculation' in specimen_data:
            label = specimen_data.split('_')[0]
        else:
            label = specimen_data.split('.')[0]
        truncated_label = label[:label.find('Ra')]
        if '#' in truncated_label:
            truncated_label = truncated_label[:truncated_label.find('#')]

        # Handle creation of new labels
        if truncated_label not in self.weeks[week].keys():
            self.weeks[week][truncated_label] = {'raw': [], 'calculations': []}
        
        # Handle addition of calculation files and raw files
        if 'cal' in specimen_data:
            self.weeks[week][truncated_label]['calculations'].append(specimen_data_dir)
        else:
            self.weeks[week][truncated_label]['raw'].append(specimen_data_dir)

    
    def populate_data_by_week_LOW_COST(self, parent_dir, specimen_data):
        """Extracts and populates weekly data for Low cost device"""
        specimen_data_dir = os.path.join(parent_dir, specimen_data)
        for leaf_data in os.listdir(specimen_data_dir):
            leaf_data_dir = os.path.join(specimen_data_dir, leaf_data)
            
            week = self.get_week(leaf_data_dir)
            # Handle creation of new weeks
            if week not in self.weeks:
                self.weeks[week] = dict()

            leaf_label = self.extract_low_cost_label(leaf_data_dir)
            # Handle creation of new leaf labels
            if leaf_label not in self.weeks[week].keys():
                self.weeks[week][leaf_label] = {'raw': [], 'img': []}
            
            # Handle storage of leaf and image data
            if leaf_data_dir.endswith('.jpg'):
                self.weeks[week][leaf_label]['img'].append(leaf_data_dir)
            else:
                self.weeks[week][leaf_label]['raw'].append(leaf_data_dir)

    
    def _load_fn(self):
        """Functional Router for device data loading based defined Device type"""
        for root in os.listdir(self.data_path):
            root_dir = os.path.join(self.data_path, root)
            if root_dir.endswith('.xlsx'):
                self.expert_files.append(root_dir)
                continue
    
            for device in os.listdir(root_dir):
                device_dir = os.path.join(root_dir, device)
                for data in os.listdir(device_dir):
                    data_dir = os.path.join(device_dir, data)

                    # Handle case for LOW_COST
                    if self.device == Device.LOW_COST:
                        """Impl data loading for LOW COST DEVICE"""
                        
                        if 'Reading' in data:
                            for reading in os.listdir(data_dir):
                                reading_dir = os.path.join(data_dir, reading)
                                for disease_class in os.listdir(reading_dir):
                                    disease_class_dir = os.path.join(
                                        reading_dir,
                                        disease_class
                                    )
                                    for specimen in os.listdir(disease_class_dir):
                                        # call the populate data by week function
                                        self.populate_data_by_week_LOW_COST(
                                            disease_class_dir,
                                            specimen
                                        )
                    elif self.device == Device.SCAN_CODER:
                        """Impl data loading for the SCAN_CORDER"""

                        # Filters out json files
                        if data_dir.endswith('.csv'):
                            week = self.get_week(data_dir)
                            if week not in self.weeks:
                                self.weeks[week] = None
                            
                            self.weeks[week] = data_dir

                    elif self.device == Device.BIO_SCIENCE:
                        """Impls data loading for BIO_SCIENCE DEVICE"""
                        if 'Reading' not in data_dir and not data_dir.endswith('.csv'): 
                            for disease_category in os.listdir(data_dir):
                                # filter out jsons
                                if disease_category.endswith('.json'):
                                    continue

                                disease_category_dir = os.path.join(
                                    data_dir, 
                                    disease_category
                                )

                                for reading in os.listdir(disease_category_dir):
                                    reading_dir = os.path.join(
                                        disease_category_dir, 
                                        reading
                                    )
                                    # Handle case of extra folder Hierarch in R
                                    if 'R' in reading:
                                        for point in os.listdir(reading_dir):
                                            point_dir = os.path.join(
                                                reading_dir, 
                                                point
                                            )
                                            for specimen_data in os.listdir(point_dir):
                                                # Skip image files
                                                if specimen_data.endswith('.png'):
                                                    continue
                                                self.populate_data_by_week_BIO_SCEINCE(
                                                    point_dir,
                                                    specimen_data
                                                )
                                    else:
                                        # Extract specimen data
                                        for specimen_data in os.listdir(reading_dir):
                                            if specimen_data.endswith('.png'):
                                                continue
                                            self.populate_data_by_week_BIO_SCEINCE(
                                                reading_dir,
                                                specimen_data
                                            )
    @staticmethod
    def _generate_sort_key_LOW_COST(label: str):
        """Generates sort key from give label for LOW_COST_DEVICE"""
        first_letter = label[0]
        last_number = int(re.findall(r'\d+$', label)[0]) # extract last numer
        return (first_letter, last_number)
    
    @staticmethod
    def _extract_disease_class_BIOSCIENCE(label):
        """Extracts the disease class from BIO SCIENCE label"""
        match = re.search(r'^[A-Z]\d+([A-Z]+)\d+$', label)
        return match.group(1) if match else None


    def _create_device_meta_data_BIO_SCINECE(self):
        """Populates meta data from BIO SCIENCE DEVICE"""
        temp_buffer = {}
        # sort the weeks
        for week in self.weeks.keys():
            smp_temp = {}
            for key in self.weeks[week].keys():
                # extract raw and calculation data files
                specimen_data_dirs = self.weeks[week][key]['raw']
                calculation_data_dirs = self.weeks[week][key]['calculations']

                # Count number of files in each case
                raw_count = len(specimen_data_dirs)
                calculation_count = len(calculation_data_dirs)

                # create state for temporary object
                smp_temp[key] = {
                    'raw_count': raw_count,
                    'calculation_count': calculation_count,
                    'specimen_data_dirs': specimen_data_dirs,
                    'calculation_data_dirs': calculation_data_dirs
                }
            temp_buffer[week] = smp_temp
            
            # Populate the information in pandas dataframe
            rows = []
            for week, labels in temp_buffer.items():
                for label, values in labels.items():
                    # Exract plant types, plant number
                    plant_type, plant_number = self._generate_sort_key_LOW_COST(label)
                    disease_class = self._extract_disease_class_BIOSCIENCE(label)
                    rows.append({
                        'week': week,
                        'label': label,
                        'plant_type': plant_type,
                        'disease_class': disease_class,
                        'plant_number': plant_number,
                        'raw_count': values['raw_count'],
                        'calculation_count': values['calculation_count'],
                        'specimen_data_dirs': values['specimen_data_dirs'],
                        'calculation_data_dirs': values['calculation_data_dirs']
                    
                    })
        # update underlying states
        self.meta_data = pd.DataFrame(rows)
        self.meta_data = self.meta_data.sort_values(by = [
            'week',
            'plant_type',
            'disease_class',
            'plant_number'
        ]).reset_index(drop=True)
        # Building a unique sample index 
        self.meta_data['search_label'] = (
            self.meta_data['plant_type'].astype(str) +
            self.meta_data['disease_class'].astype(str) +
            self.meta_data['plant_number'].astype(str)
        )
        self.labels = list(self.meta_data[self.meta_data['week'] == self.label_extra_week]['search_label'].unique())



    def _create_device_meta_data_LOW_COST(self):
        """Populates meta data from LOW COST DEVICE"""
        temp_buff = {}
        for week in self.weeks.keys():
            smp_temp = {}
            for key in self.weeks[week].keys():
                specimen_data_dirs = self.weeks[week][key]['raw']
                img_data_dirs = self.weeks[week][key]['img']

                raw_count = len(specimen_data_dirs)
                img_count = len(img_data_dirs)
            
                smp_temp[key] = {
                    'raw_count': raw_count, 
                    'img_count': img_count,
                    'specimen_data_dirs': specimen_data_dirs,
                    'img_data_dirs': img_data_dirs 
                }
            
            temp_buff[week] = smp_temp

        # Populate information in a pandas dataframe
        rows = []
        for week, labels in temp_buff.items():
            for label, values in labels.items():
                
                # extract the sort keys and disease_category
                plant_type, plant_number = self._generate_sort_key_LOW_COST(label)
                specimen_data_dir = values['specimen_data_dirs']

                disease_cat = specimen_data_dir[0].split('/')[-3]


                rows.append({
                    'week': week,
                    'label': label,
                    'plant_type': plant_type,
                    'disease_class': disease_cat,
                    'plant_number': plant_number,
                    'raw_count': values['raw_count'],
                    'img_count': values['img_count'],
                    'specimen_data_dirs': specimen_data_dir,
                    'img_data_dirs': values['img_data_dirs']
                })

        # update the underlying states
        self.meta_data = pd.DataFrame(rows)
        self.meta_data = self.meta_data.sort_values(by = [
            'week',
            'plant_type',
            'disease_class',
            'plant_number'
        ]).reset_index(drop=True)
        # Building a unique sample index 
        self.meta_data['search_label'] = (
            self.meta_data['plant_type'].astype(str) +
            self.meta_data['disease_class'].astype(str) +
            self.meta_data['plant_number'].astype(str)
        )
        self.labels = list(self.meta_data[self.meta_data['week'] == self.label_extra_week]['search_label'].unique())

    
    def _extract_other_mata_info_SCAN_CORDER(self, label):
        """Extracts the sort key, plant type and disease type"""
        # keep only first two sections
        cleaned = ' '.join(label.split()[:2])
        match = re.search(r'^([A-Z])(\d+)\s+([A-Z]+)(\d+)$', cleaned)
        if match:
            first_letter = match.group(1)
            # first_number = int(match.group(2))
            disease = match.group(3)
            second_number = int(match.group(4))
            return (cleaned, first_letter,  disease, second_number)
        return None

    
    def _create_device_meta_data_SCAN_CORDER(self):
        temp_df_buffer = []

        # read all weekly data
        for week in self.weeks.keys():
            pd_smp = pd.read_csv(self.weeks[week])
            pd_smp['week'] = week
            temp_df_buffer.append(pd_smp)
        
        # clean and update the underlying states
        self.meta_data = pd.concat(temp_df_buffer)
        self.meta_data = self.meta_data.drop(columns=['UUID', 'Token ID', 'Timestamp']).rename(columns={'Sample ID': 'label'})

        # Extract additional meta information
        self.meta_data[['clean_label', 'plant_type',  'disease_class', 'plant_number']] = (
            self.meta_data['label']
            .apply(self._extract_other_mata_info_SCAN_CORDER)
            .apply(pd.Series)
        )
        self.meta_data = self.meta_data.sort_values(by = [
            'week',
            'plant_type',
            'disease_class',
            'plant_number'
        ]).reset_index(drop=True)
        # Building a unique sample index 
        self.meta_data['search_label'] = (
            self.meta_data['plant_type'].astype(str) +
            self.meta_data['disease_class'].astype(str) +
            self.meta_data['plant_number'].astype(str)
        )
        self.labels = list(self.meta_data[self.meta_data['week'] == self.label_extra_week]['search_label'].unique())
        self.meta_data = self._group_data_by_label_SCAN_CORDER() # compute and return batched scan corder data

        

    def _load_meta_data(self):
        """Routing function for populating meta information for each device"""
        if self.device == Device.LOW_COST:
            self._create_device_meta_data_LOW_COST()
        elif self.device == Device.BIO_SCIENCE:
            self._create_device_meta_data_BIO_SCINECE()
        
        elif self.device == Device.SCAN_CODER:
            self._create_device_meta_data_SCAN_CORDER()
        else:
            raise ValueError(f'Unsupported device')


    def __len__(self):
        """Return the length of the dataset for the underlying device"""
        return self.meta_data.shape[0]

    def get_specimen_count(self):
        if self.device == Device.SCAN_CODER:
            return len(self)
        else:
            total_files = self.meta_data['raw_count'].sum()
            if self.device == Device.LOW_COST:
                NO_OF_READING_PER_FILE = 2
                total_files = total_files * NO_OF_READING_PER_FILE
            return total_files
        
    
    def _group_data_by_label_SCAN_CORDER(self):
        """Bundles SCAN CORDER data to avoid duplication in the dataframe"""
        grouped_samples = {}
        self.wavelength_cols = [
            '394', '445', '451', '517', '573',
            '589', '591', '615', '632', '849',
            '867', '946'
        ]

        for (week, label), group in self.meta_data.groupby(['week', 'search_label']):
            grouped_samples[(week, label)] = {
                'week': week,
                'search_label': label,
                'plant_type': group['plant_type'].iloc[0],
                'disease_class': group['disease_class'].iloc[0],
                'plant_number': group['plant_number'].iloc[0],
                'raw_count': len(group),
                'specimen_reading': group[
                    self.wavelength_cols
                ].values.tolist()
            }

        # Convert dict back to pandas dataframe
        grouped_df = pd.DataFrame(grouped_samples.values())  
        return grouped_df
    
    @staticmethod
    def _rename_df(df: pd.DataFrame, new_col_names: List[str] = ['plant_number', 'score', 'titer_1', 'l_1', 'titer_2', 'l_2', 'titer_3', 'l3']):
        """Renames columns to a consistent format"""
        col_names = list(df.columns)
        rename_dict = {old_col: new_col for old_col, new_col in zip(col_names, new_col_names)}
        return df.rename(columns=rename_dict)
    
    def _get_expert_file_MAIZE(self, df_path: str):
        """
        Cleans the expert files for MAIZE records and return MLN,  & MSV dataframes

        Args:
            df_path -> path to where expert files are stored

        Returns (Tuple):
            ML  N_df -> panda.DataFrame of MLN readings
            MSV_df -> panda.DataFrame of MSV readins
        """

        maize_df = pd.read_excel(df_path)
        week_mask = maize_df.iloc[:, 0].astype(str).str.match(r'^\s*week\s+\d+\s*$', case=False, na=False) # get row postion of WEEKS in df
        maize_df['week'] = maize_df.iloc[:, 0].where(week_mask).str.extract(r'(\d+)', expand=False)
        maize_df['week'] = maize_df['week'].ffill()         # perform forward fill
        maize_df['week'] = maize_df['week'].fillna(1)       # fill the first week with 1 

        
        maize_df = maize_df.drop(columns={'week 1', 'Disease description'})     # drop week 1 and Disease description
        split_idx = maize_df.columns.get_loc('Symptom description.1')           # splitting index to extract MLN and MSV dfs

        # Create MLN split and populate meta data
        mln_df = maize_df.iloc[:, :split_idx+1]
        mln_df['week'] = maize_df['week'].astype('int64')
        mln_df['disease_class'] = 'MLN'

        # Create MSV df split and populate meta data
        msv_df = maize_df.iloc[:, split_idx + 1:]
        msv_df['DAY'] = maize_df['DAY']
        msv_df['score'] = maize_df['score']
        msv_df['disease_class'] = 'MSV'
        msv_df['week'] = msv_df['week'].astype('int64')

        # Rearrange the column position for msv_df to use shared renaming function
        msv_df = msv_df[['DAY', 'score', 'MSV1', 'Symptom description (A= asymptomatic, S=Symptom)', 'MLN2', 'Unnamed: 13', 'MLN3.1', 'Unnamed: 15', 'week', 'disease_class']]

        # Rename MLN and MSV df
        mln_df = self._rename_df(mln_df)
        msv_df = self._rename_df(msv_df)
        return (mln_df, msv_df)
    
    
    def _get_expert_files_CASSAVA_CMD(self, df_path:str):
        """Extracts expert readings for CMD only from the source file"""
        week_col = 'CBSD'
        cmd_df = pd.read_excel(df_path, sheet_name='CMD')

        # build a mask for rows with weeks of type str
        cmd_df['week'] = cmd_df[week_col].astype(str).str.extract(r'WEEK\s*(\d+)', expand=False)
        cmd_df['week'] = cmd_df['week'].ffill()         # apply a forward fill

        # drop rows with week rows
        cmd_df = cmd_df[~cmd_df[week_col].astype(str).str.contains(r'WEEK', case=False, na=False)]
        cmd_df = cmd_df.reset_index(drop=True)          # rest index
        new_col_names = ['plant_number', 'score', 'titer_1', 'l_1' ,'titer_2','l_2', 'titer_3', 'l3', 'week']
        cmd_df = self._rename_df(cmd_df, new_col_names)

        # extracting plant numbers only
        cmd_df['plant_number'] = cmd_df['plant_number'].astype(str).str.extract(r'(\d+)')[0].astype(int)
        cmd_df['week'] = cmd_df['week'].fillna(1)   # fill the first week of NaNs with 1
        cmd_df['week'] = cmd_df['week'].astype('int64')  # convert week to int64 from obj type
        cmd_df['disease_class'] = 'CMD'             # fill in disease class
        return cmd_df
    
    def _get_expert_files_CASSAVA_CBB(self, df_path:str):
        """Extracts expert readings for CBB only from the source file"""
        week_col = 'Cassava bacterial blight'
        cbb_df = pd.read_excel(df_path, sheet_name='CBB')

        week_mask = cbb_df.iloc[:, 0].astype(str).str.match(r'^\s*week\s*\d+', case=False, na=False)    # create mask for rows with weeks type str
        cbb_df['week'] = cbb_df.iloc[:, 0].where(week_mask).str.extract(r'(\d+)', expand=False)         # apply mask
        cbb_df['week'] = cbb_df['week'].ffill()     # forward fill the weeks
        cbb_df = cbb_df[~week_mask].copy()          # drop cols with week strs
        cbb_df = cbb_df.drop(0).reset_index(drop=True) # drop column 0 with custom labels

        # format column names
        new_col_names = ['plant_number', 'description', 'score' ,'titer_1', 'l_1' ,'titer_2','l_2', 'titer_3', 'l2', 'week']
        cbb_df = self._rename_df(cbb_df, new_col_names)

        # extract plant numbers only
        cbb_df['plant_number'] = cbb_df['plant_number'].astype(str).str.extract(r'(\d+)')[0].astype(int)
        cbb_df['disease_class'] = 'CBB'                # fill disease class
        cbb_df = cbb_df.drop(columns=['description']) # drop description column to ensure consistency
        cbb_df['week'] = cbb_df['week'].astype('int64')
        
        return cbb_df
    
    def _get_expert_file_CASSAVA(self, df_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Wrapper function that computes and returns expert files for CMD, and CBB

        Arg:
            df_path -> str: path to where expert files for cassav is stored

        Return (Tuple):
            CMD_df -> pd.DataFrame: dataframe containing chemical meta data for CMD
            CBB_df -> pd.DataFrame: dataframe containing chemical meta data fro CBB
        """
        cmd_df = self._get_expert_files_CASSAVA_CMD(df_path)
        cbb_df = self._get_expert_files_CASSAVA_CBB(df_path)
        
        return (cmd_df, cbb_df)
    
    def _get_expert_file_BEANS(self, df_path:str) -> Tuple[pd.DataFrame]:
        """
        Cleans and formats beans expert readings

        Arg:
            df_path-> str: path to where the expert file csv is stored
        
        Return:
            dataframe -> pd.Dataframe: with consistent column alignment with the other expert files
        """

        beans_df = pd.read_excel(df_path)
        beans_df = beans_df.dropna(how='all').reset_index(drop=True) # drop rows with only NaNs
        
        # Annotate with appropriate week numbers
        first_col = beans_df.columns[0]
        date_mask = beans_df[first_col].astype(str).str.match(r'^\s*\d{1,2}(st|nd|rd|th)?\s+[A-Za-z]+\s*$', case=False, na=False)

        beans_df.loc[date_mask, 'week'] = range(1, date_mask.sum() + 1) # create week numbers from detected dates
        beans_df['week'] = beans_df['week'].ffill().fillna(0) # forward fill downwards and replace NaN with 0
        beans_df = beans_df[~date_mask].reset_index(drop=True) # remove date rows

        # Ensure weekly data alignment i.e. starting from 1 not 0
        beans_df['week'] = beans_df['week'] + 1

        # Handle cases of missing plant_numbers following the sequeeze of 1..5
        plant_col = 'Plant No'
        last_value = None
        for idx in beans_df.index:
            current = beans_df.loc[idx, plant_col] 
            # if current value exits, update tracker
            if pd.notna(current):
                last_value = int(current)
            # if Nan, continue sequence
            else:
                if last_value is None:
                    last_value = 1
                else:
                    last_value = (last_value % 5) + 1
                
                beans_df.loc[idx, plant_col] = last_value
        
        # drop empty columns
        beans_df = beans_df.dropna(axis=1, how='all')
        beans_df.columns = ['plant_number', 'disease_description', 'plant_1', 'plant_2', 'plant_3', 'elisa_score_1', 'elisa_score_2', 'elisa_score_3', 'week']
        beans_df = beans_df.drop(columns=['disease_description'])

        # compute score as mean of plant_* in df
        cols_to_average= [f'plant_{i+1}' for i in range(3)]
        beans_df['score'] = beans_df[cols_to_average].mean(axis=1).astype(int)
        beans_df = beans_df.drop(columns=cols_to_average) # drop plant columns

        # making l_* columns to ensure consistency with other expert reading files
        beans_df['l_1'], beans_df['l_2'], beans_df['l_3'] = (np.nan for _ in range(3))
        beans_df['disease_class'] = 'BLB' # disease_class to be updated

        # renaming colums
        elisa_rename_keys = {f'elisa_score_{i+1}': f'titer_{i+1}' for i in range(3)}
        beans_df = beans_df.rename(columns=elisa_rename_keys)

        # reordering the columns for consistent merger
        beans_df = beans_df[['plant_number', 'score', 'titer_1', 'l_1', 'titer_2', 'l_2', 'titer_3', 'l_3', 'week', 'disease_class']]
        return beans_df

    
    def populate_expert_readings(self) -> None:
        """
        Wrapper function, loads all expert files and merges them with the meta data file
        """
        expert_dfs = []
        for file in self.expert_files:
            if 'cassava' in file and '~' not in file:
                cmd_df, cbb_df = self._get_expert_file_CASSAVA(file) # extract CMD and CBB files

                if self.device == Device.LOW_COST:
                    # shift cmd_df to align with low_cost meta_data
                    cmd_df['plant_number'] = cmd_df['plant_number'] + self.CMD_SHIFT
                             

                    # shift cbb_df to align with low cost meta_data
                    cbb_df['plant_number'] = cbb_df['plant_number'] + self.CBB_SHIFT
                    mask_3 = cbb_df['plant_number'] == 3
                    mask_4 = cbb_df['plant_number'] == 4
                    mask_5 = cbb_df['plant_number'] == 5

                    col = cbb_df.columns.difference(['plant_number']) # remove plant_number from cols to be swapped
                    temp_3 = cbb_df.loc[mask_3, col].copy()
                    temp_4 = cbb_df.loc[mask_4, col].copy()
                    temp_5 = cbb_df.loc[mask_5, col].copy()

                    # Performing the column swaps
                    cbb_df.loc[mask_3, col] = temp_5.values
                    cbb_df.loc[mask_4, col] = temp_3.values
                    cbb_df.loc[mask_5, col] = temp_4.values
                expert_dfs.extend([cmd_df, cbb_df])


            elif 'Maize' in file:
                mln_df, msv_df = self._get_expert_file_MAIZE(file)  # extract MLN, MSV files
                
                if self.device == Device.LOW_COST:
                    # Peform label alignment
                    mln_df['plant_number'] = mln_df['plant_number'] + self.MLN_SHIFT
                    msv_df['plant_number'] = msv_df['plant_number'] + self.MSV_SHIFT
                expert_dfs.extend([mln_df, msv_df])
            elif 'Beans' in file:
                blb_df = self._get_expert_file_BEANS(file)
                expert_dfs.append(blb_df)


        # merge the meta_data with expert readings
        merge_df = pd.concat(expert_dfs)
        self.meta_data = pd.merge(self.meta_data, merge_df, how='left')  
 

    def get_weekly_data(self, week: int):
        """Returns data for a specific week"""
        return self.meta_data[self.meta_data['week'] == week]
    

    @staticmethod
    def read_single_BIOSCIENCE(path: str) -> List[pd.DataFrame]:
        """
        Reads Raw data, peak wavelength information, and calibration for BIO_SCIENCE

        Arg:
            path: str -> path of file to read
        
        Returns:
            RAW_DATA_FILE: pd.DataFrame -> Contains the raw spectral readings
            PEAK_WAVELENGTH_FILE: pd.DataFrame -> Contains the peak wavelength
            CALIBRATION_FILE: pd.DataFrame -> Contains calibration data
        """

        # Data Containers
        headers = []
        interval_state = []
        global_state = []

        # Pattern matching
        number_pattern = re.compile(r"^\s*-?\d+(\.\d+)?([eE]-?\d+)?%?\s*$")

        # Determine if Numeric or Non numeric
        def _is_numeric_row(line):
            
            parts = line.split(",")
            return all(number_pattern.match(p) for p in parts)


        with open(path, 'r') as f:
            for line in f:
                line = line.strip("\n")
                if line == '':
                    # update global state and reset interval data
                    global_state.append(interval_state)
                    interval_state = []
                    continue
                if not _is_numeric_row(line):
                    headers.append(line)
                    continue
                interval_state.append(line)

        # Percentage dropping function
        drop_prec = lambda s: re.sub(r"[%'']", "", s)
        to_array = lambda s: np.array(ast.literal_eval(f"[{s}]"))
        to_b_array = lambda S: np.array([to_array(drop_prec(s)) for s in S]).astype(np.float32)

        def make_compartible(index) -> pd.DataFrame:
            """
            Organizes the data into a dataframe match readable to the user
            """
            array = to_b_array(global_state[index])
            cols = headers[index].split(',')

            return pd.DataFrame(data=array, columns=cols)
        
        
        # Reference index
        RAW_SPECTRA_INDEX = -3
        PEAK_VALUES_INDEX = -2
        CALIBRATION_INDEX = -1

        return make_compartible(RAW_SPECTRA_INDEX), make_compartible(PEAK_VALUES_INDEX), make_compartible(CALIBRATION_INDEX) 
    
    def read_one_BIOSCIENCE(self, index) -> iter:
        """
        Yields the spectra data files for the indexed object

        Arg:
            index -> position to tagert file
        """
        for file in self.meta_data.loc[index, 'specimen_data_dirs']:
            yield self.read_single_BIOSCIENCE(file)


    def read_one_SCAN_CODER(self, index: int) -> iter:
        """
        Yields the specimen files for the indexed file

        Arg:
            index: int -> position to the target file
        """
        for file in self.meta_data.loc[index, 'specimen_reading']:
            yield np.array(file, dtype=np.float32)


    @staticmethod
    def read_single_LOW_COST(path: str) -> Tuple[np.ndarray]:
        """
        Reads raw files 1 & 2, band energy files 1 & 2 files of LOW COST device

        Arg:
            path: str -> directory to file location
        
        Returns:
            spectral_1: np.ndarray-> spectral reading 1
            spectral_2: np.ndarray -> spectral reading 2
            band_energy_1: np.ndarray -> band energy computed from spectral reading 1
            band_energy_2: np.ndarray -> band energy computed from spectral reading 2
        """
        df0 = pd.read_csv(path)
        spectral_1 = ast.literal_eval(df0['spectral_1'][0])

        if isinstance(spectral_1, dict):
            spectral_2 = ast.literal_eval(df0['spectral_2'])
            band_energy_1 = spectral_1['bandEnergy']
            band_energy_2 = spectral_2['bandEnergy']
            spectral_1 = spectral_1['intensity']
            spectral_2 = spectral_2['intensity']
        else:
            spectral_2 = ast.literal_eval(df0['spectral_2'][0])
            band_energy_1 = ast.literal_eval(df0['band_energy_spectral_1'][0])
            band_energy_2 = ast.literal_eval(df0['band_energy_spectral_2'][0])
        
        return (
            np.array(spectral_1, dtype=np.float32), 
            np.array(spectral_2, dtype=np.float32), 
            np.array(band_energy_1, dtype=np.float32),
            np.array(band_energy_2, dtype=np.float32)
        )
    
    def read_one_LOW_COST(self, index: int) -> iter:
        """
        Yields the spectral dataset for the indexed object

        Arg:
            index: int -> position to target file to be loaded
        
        Returns:
            iter -> Iterator that yield the spectral files
        """
        for file in self.meta_data.loc[index, 'specimen_data_dirs']:
            yield self.read_single_LOW_COST(file)
            

    def read_one(self, index: int):
        """
        Wrapper function that yields the specimen file for the indexed file

        Arg:
            index: int -> position to the target if
        """
        if self.device == Device.BIO_SCIENCE:
            return self.read_one_BIOSCIENCE(index)
        
        elif self.device == Device.LOW_COST:
            return self.read_one_LOW_COST(index)
        
        else:
            return self.read_one_SCAN_CODER(index)
        
    def create_disease_embeddings(self):
        """
        Computes the embedding for the disease classes

        """
        disease_classes = sorted(self.meta_data['disease_class'].unique())
        disease_classes.remove('HLT')

        # Force HLT to be embedded as one
        final_class = ['HLT']
        final_class.extend(disease_classes)

        self.disease_class_codes = {d : i for i, d in enumerate(final_class)}

    def create_plant_embeddings(self):
        """
        Computes the embeddings for the plants in the dataset
    
        """
        plant_types = sorted(self.meta_data['plant_type'].unique())
        self.plant_type_codes = {c: i for i, c in enumerate(plant_types)}

    
    def __getitem__(self, index: int) -> pd.Series:
        """
        Extracts the meta data of a single readings

        Args:
            index: int -> index of record to be read

        Returns:
            data_obj: pd.Series -> meta data for the indexed record
        """
        # Repulicate titer, week, disease, class, and expert score for the n readings
        # columns drops l_1, l_2, l_3, img_count, 
        # target vector titer_readings
        # feature vector: raw_reading(7), week, plant_type

        n_data = self.meta_data.loc[index, 'raw_count']
        out_dim = None
        n_tiles = 1

        if self.device == Device.BIO_SCIENCE:
            spectral_range_size = 3648
            peak_wavelength_size = 0 # Peak wavelength are inconsistent
            out_dim = spectral_range_size + peak_wavelength_size
        
        elif self.device == Device.SCAN_CODER:
            spectral_range_size = 12
            out_dim = spectral_range_size
        else:
            spectral_range_size = 381
            band_energy = 6
            out_dim = spectral_range_size + band_energy
            n_tiles += 1

        # Targe features
        N_titer_readings = 3
        Disease_class_size = 1
        week_size = 1
        expert_score_size = 1

        # Create Buffer arrays
        if self.device == Device.LOW_COST:
            # add a channel Dimension to x_out
            N_samples = 2
            x_out = np.zeros((n_data, N_samples, out_dim))
        else:
            x_out = np.zeros((n_data, out_dim))

        y_out = np.zeros(N_titer_readings + Disease_class_size + week_size + expert_score_size)

        # Fill in the target data
        target_cols = [f'titer_{i}' for i in range(1, 4)]
        target_cols.extend(['score', 'week', 'disease_class'])
        n_cols = len(target_cols)

        for i in range(n_cols):
            if i == n_cols - 1:
                y_out[i] = self.disease_class_codes[self.meta_data.loc[index, target_cols[i]]]
            else:
                y_out[i] = self.meta_data.loc[index, target_cols[i]]

        if self.device == Device.LOW_COST:
            y_out = np.tile(y_out, (2, n_data, 1))
        
        else:
            y_out = np.tile(y_out, (n_data , 1))
    

        # Iterate over single data loader and batch the data
        for i, file in enumerate(self.read_one(index)):
            if self.device == Device.SCAN_CODER:
                x_out[i, :] = file
            elif self.device == Device.LOW_COST:
                spectral_1, spectra_2, band_energy_1, band_energy_2 = file
                x_out[i, 0, :spectral_range_size] = spectral_1
                x_out[i, 0, spectral_range_size:] = band_energy_1
                x_out[i, 1, :spectral_range_size] = spectra_2
                x_out[i, 1, spectral_range_size:] = band_energy_2
            
            else:
                raw_data, _ , _ = file # calibration data not required for -> _
                x_out[i, :spectral_range_size] = raw_data[raw_data.columns[-1]].values
                
        return x_out, y_out
    

class SpectralDataset(SpectralDataset_v2):
    """
    Class wrapper that loads spectra data of the specified device, current supporting BIO_SCIENCE, SCAN_CODER, LOW_COST 
    
    """
   
    def __init__(self, data_path: str, device: Device, label_extra_week: int=9):
        """
        Args:
            data_path-> str: path to where data is stored
            device -> Device: whose data should be loaded
            label_extra_week: int -> week to consider when loading labels, i.e. should be complete
        """
        super().__init__(data_path, device, label_extra_week)

        # Handle distribution if the device is not low cost
        if not device == Device.LOW_COST:
            dataset_lc = SpectralDataset_v2(data_path, device=Device.LOW_COST)
            images_df = dataset_lc.meta_data.loc[:,  ['week', 'plant_type', 'plant_number', 'disease_class', 'img_count', 'img_data_dirs']]
            
            # apply mapping shifts for alignment
            mapping = {
                ('C', 'CMD'): 5,
                ('C', 'CBB'): 0,
                ('C', 'HLT'): 10,
                ('M', 'MLN'): 5,
                ('M', 'MSV'): 0,
                ('M', 'HLT'): 10,
                ('B', 'BLB'): 0,
                ('B', 'BRD'): 0,
                ('B', 'HLT'): 0,
            }

            for (plant_c, disease), shift in mapping.items():
                mask = (
                    (images_df['plant_type'] == plant_c) &
                    (images_df['disease_class'] == disease)
                )
                images_df.loc[mask, 'plant_number'] -= shift

            # perform merging
            self.meta_data = self.meta_data.merge(images_df, how='left')
            


if __name__ == '__main__':
    from buaiir_spectra.utils.device import Device
    from buaiir_spectra.data.dataset_v2 import SpectralDataset

    data_path = '/home/wilfred/Datasets/spectra_data'
    dataset = SpectralDataset(data_path, Device.LOW_COST)

    DISEASE_CLASS= 'CMD'
    filtered = dataset.meta_data[dataset.meta_data['disease_class'] ==DISEASE_CLASS]
    print(filtered)



    




   




    

    









        




        

    



