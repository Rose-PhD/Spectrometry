from buaiir_spectra.data.datase_deprecatedt import SpectralDataset
from buaiir_spectra.utils.device  import Device
import joblib



global_state = {}
target_names = ['titer_avg', 'score', 'plant_type', 'week', 'disease_class']
DATA_PATH = '/home/wilfred/Datasets/spectra_data'


for device in Device.get_devices():
    ds = SpectralDataset(DATA_PATH, device)
    global_state[device.name] = {'target_columns': target_names, 'wavelenght': ds.wavelength}

joblib.dump(global_state, "src/buaiir_spectra/wavelength_and_cols_ds.pkl")




    

