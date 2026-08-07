import joblib
from importlib.resources import files
from buaiir_spectra.utils.device import Device


META_PATH = files("buaiir_spectra").joinpath("wavelength_and_cols_ds.pkl")
state_dict = joblib.load(META_PATH)

for device in Device.get_devices():
    print(state_dict[device])

