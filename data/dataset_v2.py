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

class SpectralDataset_v2:
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
                self.weeks[week][leaf_label] = {'raw': None, 'img': None}
            
            # Handle storage of leaf and image data
            if leaf_data_dir.endswith('.jpg'):
                self.weeks[week][leaf_label]['img'] = leaf_data_dir
            else:
                self.weeks[week][leaf_label]['raw'] = leaf_data_dir

    
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

    def _create_device_meta_data_BIO_SCINECE(self):
        """Populates meta data from BIO SCIENCE DEVICE"""
        temp_buffer = {}
        # sort the weeks
        for week in self.weeks.keys():
            smp_temp = {}
            for key in self.weeks[week].keys():
                raw_count = len(self.weeks[week][key]['raw'])
                calculation_count = len(self.weeks[week][key]['calculations'])

                # create state for temporary object
                smp_temp[key] = {
                    'raw_count': raw_count,
                    'calculation_count': calculation_count
                }
            temp_buffer[week] = smp_temp
            
            # Populate the information in pandas dataframe
            rows = []
            for week, labels in temp_buffer.items():
                for label, values in labels.items():
                    rows.append({
                        'week': week,
                        'label': label,
                        'raw_count': values['raw_count'],
                        'calculation_count': values['calculation_count']
                    })
        # update underlying states
        self.meta_data = pd.DataFrame(rows)
        self.labels = list(self.meta_data[self.meta_data['week'] == self.label_extra_week]['label'])

        

    def _create_device_meta_data_LOW_COST(self):
        """Populates meta data from LOW COST DEVICE"""
        temp_buff = {}
        for week in self.weeks.keys():
            smp_temp = {}
            for key in self.weeks[week].keys():
                has_raw = 0
                has_img = 0
                if self.weeks[week][key]['raw'] is not None:
                    has_raw += 1
                
                if self.weeks[week][key]['img'] is not None:
                    has_img += 1
                smp_temp[key] = {'raw_count': has_raw, 'img_count': has_img}
            
            temp_buff[week] = smp_temp

        # Populate information in a pandas dataframe
        rows = []
        for week, labels in temp_buff.items():
            for label, values in labels.items():
                rows.append({
                    'week': week,
                    'label': label,
                    'raw_count': values['raw_count'],
                    'img_count': values['img_count']
                })

        # update the underlying states
        self.meta_data = pd.DataFrame(rows)
        self.labels = list(self.meta_data[self.meta_data['week'] == self.label_extra_week]['label'])

    
    def _create_device_meta_data_SCAN_CORDER(self):
        temp_df_buffer = []

        # read all weekly data
        for week in self.weeks.keys():
            pd_smp = pd.read_csv(self.weeks[week])
            pd_smp['week'] = week
            temp_df_buffer.append(pd_smp)
        
        self.meta_data = pd.concat(temp_df_buffer)
        self.labels = list(
            self.meta_data[self.meta_data['week'] == self.label_extra_week]['Sample ID']
        )
        

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
        
if __name__ == '__main__':
    import os 
    from data.dataset import Device

    os.system('clear')
    for device in Device.get_devices():
        dataset = SpectralDataset_v2(DATA_PATH, device=device)
        print(f'Device: {device.name}', f'Data size: {dataset.meta_data.shape}', f'No of labels: {len(dataset.labels)}', sep='\t| ')
        print(f'Unique weeks: {dataset.weeks.keys()}')
        print(f'Data Labels to be sorted: {dataset.labels}')
        print("="*190)


        

    



