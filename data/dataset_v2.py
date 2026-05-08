import os 
from data.dataset import Device
import pandas as pd
import re
from data.dataset import Dataset

DATA_PATH = 'spectral_data' 

"""
TODO:
    If device is not LOW_COST load low cost images to facilitate access to image data
    This applies to SCAN_CODER and BIO_SCIENCE
    This is to enable working with spectral dataset and images simultenously
"""

class SpectralDataset_v2(Dataset):
    def __init__(self, data_path, device: Device, label_extra_week: int= 4):
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


    @staticmethod
    def get_week(path):
        """Extract the week from specimen data path"""
        match = re.search(r'week(\d+)', path.lower())
        if match:
            return int(match.group(1))
        raise ValueError(f'Week not found in path: {path}')
    

    @staticmethod
    def extract_low_cost_label(path):
        """Extracts labels for low cost device"""
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
        self.labels = list(self.meta_data[self.meta_data['week'] == self.label_extra_week]['label'])




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
        self.labels = list(self.meta_data[self.meta_data['week'] == self.label_extra_week]['label'])

    
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
        self.labels = list(self.meta_data[self.meta_data['week'] == self.label_extra_week]['clean_label'].unique())

        

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

    def get_weekly_data(self, week: int):
        """Returns data for a specific week"""
        return self.meta_data[self.meta_data['week'] == week]

    
    def __getitem__(self):
        """Extracts a single item to be batched"""
        pass
    

        
if __name__ == '__main__':
    import os 
    from data.dataset import Device

    os.system('clear')
    for device in Device.get_devices():
        dataset = SpectralDataset_v2(DATA_PATH, device=device)
        print(f'Device: {device.name}', f'No of labels: {len(dataset.labels)}', f'Dataset length: {len(dataset)}',sep='\n')
        print(f'Number of readings: {dataset.get_specimen_count()}')
        print(f'Unique weeks: {list(dataset.weeks.keys())}')
        WEEK = 11
        if device == Device.SCAN_CODER:
            print(dataset.get_weekly_data(WEEK).head(50))
        else:
            print(dataset.get_weekly_data(WEEK))
    
        print("."*200)




        

    



