import os 
from data.dataset import Device
import pandas as pd
import numpy as np
import re
from data.dataset import Dataset
from typing import List, Tuple

DATA_PATH = 'spectral_data' 

"""
TODO:
    Merging the chemical reading data with device meta data
    Ensure consistent merging with Low-cost device
    Redistribute the images to each device
"""

class SpectralDataset_v2(Dataset):
    def __init__(self, data_path, device: Device, label_extra_week: int= 9):
        """
        Class wrapper for loading spectral data for specific device

        Args:
            data_path: root to source of data
            device: target device whose data is required
            label_extra_week: week to consider will loading sample labels
        """
        self.data_path = data_path
        self.device = device                                # target device for data being loaded
        self.labels = None                                  # labels for selection
        self.weeks = dict()                                 # dict of labels and files
        self.expert_files = list()                          # state for expert scores
        self.label_extra_week = label_extra_week
        self.meta_data = None                               # state for clean meta data
                           
        
        
        self._load_fn()                                     # load states
        self.weeks = dict(sorted(self.weeks.items()))       # sort weeks 
        self._load_meta_data()
        if self.device == Device.LOW_COST:
            self.MLN_SHIFT = 5                              # shift to align with 6-10 plant_numbers
            self.MSV_SHIFT = 0                              # mantian 1-5 plant_number
            self.CMD_SHIFT = 5                              # shift to align with 6-10 plant_numbers
            self.CBB_SHIFT = 0                              # maintain 1-5 plant number
        # if self.device == Device.BIO_SCIENCE or self.device == Device.SCAN_CODER:
        #     self.populate_expert_readings() 
        self.populate_expert_readings()
        


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
                disease_cat = specimen_data_dir[0].split('/')[5]

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
        beans_df['disease_class'] = np.nan # disease_class to be updated

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
                expert_dfs.extend([cmd_df, cbb_df])
                
                if self.device == Device.LOW_COST:
                    # shift cmd_df to align with low_cost meta_data
                    cmd_df['plant_number'] = cmd_df['plant_number'] + self.CMD_SHIFT
                    mask_7 = cmd_df['plant_number'] == 7
                    mask_8 = cmd_df['plant_number'] == 8

                    col = cmd_df.columns.difference(['plant_number']) # remove plant_number from cols to be swapped
                    temp_7 = cmd_df.loc[mask_7, col].copy()           
                    temp_8 = cmd_df.loc[mask_8, col].copy()           
                    cmd_df.loc[mask_7, col] = temp_8.values          
                    cmd_df.loc[mask_8, col] = temp_7.values           

                    # shift cbb_df to align with low cost meta_data
                    cbb_df['plant_number'] = cbb_df['plant_number'] + self.CBB_SHIFT
                    mask_3 = cbb_df['plant_number'] == 3
                    mask_4 = cbb_df['plant_number'] == 4
                    mask_5 = cbb_df['plant_number'] == 5

                    col = cbb_df.columns.difference(['plant_number']) # remove plant_number from cols to be swapped
                    temp_3 = cbb_df[mask_3, col].copy()
                    temp_4 = cbb_df[mask_4, col].copy()
                    temp_5 = cbb_df[mask_5, col].copy()

                    # Performing the column swaps
                    cbb_df.loc[mask_3, col] = temp_5.values
                    cbb_df.loc[mask_4, col] = temp_3.values
                    cbb_df.loc[mask_5, col] = temp_4.values


            elif 'Maize' in file:
                mln_df, msv_df = self._get_expert_file_MAIZE(file)  # extract MLN, MSV files
                
                if self.device == Device.LOW_COST:
                    # Peform label alignment
                    mln_df['plant_number'] = mln_df['plant_number'] + self.MLN_SHIFT
                    msv_df['plant_number'] = msv_df['plant_number'] + self.MSV_SHIFT

                expert_dfs.extend([mln_df, msv_df])


        # merge the meta_data with expert readings
        merge_df = pd.concat(expert_dfs)
        self.meta_data = pd.merge(self.meta_data, merge_df, how='left')  
 

    def get_weekly_data(self, week: int):
        """Returns data for a specific week"""
        return self.meta_data[self.meta_data['week'] == week]
    

    
    def __getitem__(self):
        """Extracts a single item to be batched"""
        pass
    
        
if __name__ == '__main__':
    import os 
    from data.dataset import Device

    # investigate the population of data to the dataframe
    dataset = SpectralDataset_v2(DATA_PATH, device=Device.LOW_COST)

    # WEEK = int(input('Target week: '))
    DISEASE_CLASS = input("Enter disease class: ")

    os.system('clear')

    # filtered_df = dataset.meta_data[(dataset.meta_data['week'] == WEEK) & (dataset.meta_data['disease_class'] == DISEASE_CLASS)]
    filtered_df = dataset.meta_data[dataset.meta_data['disease_class'] == DISEASE_CLASS]
    print(f'Disease class: {DISEASE_CLASS}', sep='\t| ')
    print(filtered_df.head(50))

    # Testing the beans extraction pipeline
    print(f'Labels')
    print(dataset.labels)

   




    

    









        




        

    



