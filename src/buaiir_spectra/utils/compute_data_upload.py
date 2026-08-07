from buaiir_spectra.data.datase_deprecatedt import SpectralDataset
from buaiir_spectra.utils.device import Device
from buaiir_spectra.data.dataloader_deprecated_v2 import SpectralDataLoader
import numpy as np
import joblib
import os

DATA_PATH  = '/home/wilfred/Datasets/spectra_data'

device = Device.SCAN_CODER
ds = SpectralDataset(DATA_PATH, device=device, load_with_images=True)
dl = SpectralDataLoader(ds, batch_size=1)

def save_batch(temp_x, temp_y, temp_images, temp_labels, wavelength, device_name, batch_identifier, compress=3):
    """Helper function to cleanly stack and serialize data."""
    x = np.vstack(temp_x)
    y = np.vstack(temp_y)
    images = np.vstack(temp_images)

    state = {
        'x': x, 
        'y': y, 
        'images': images, 
        'labels': list(temp_labels),  # Create a copy to prevent reference bugs
        'wavelength': wavelength
    }
    
    # Ensure save directory exists
    save_dir = f'src/buaiir_spectra/data_splits/{device_name}'
    os.makedirs(save_dir, exist_ok=True)
    
    file_path = f'{save_dir}/{device_name}_{batch_identifier}_batch.pkl'
    print(f'Saving data to {file_path}.......')
    joblib.dump(state, file_path, compress=compress)

def load_split(dl: SpectralDataLoader, device: Device, wavelength) -> None:
    temp_x, temp_y, temp_images, temp_labels = [], [], [], []

    for i, batch in enumerate(dl):
        print(f'Working with {i+1} batch ', end=": ")
        x_idx, image_idx, y_idx, labels = batch
        print(f'x_shape: {x_idx.shape}', f'y_shape: {y_idx.shape}', f'Image_shape: {image_idx.shape}', sep=' | ')

        temp_x.append(x_idx)
        temp_y.append(y_idx)
        temp_images.append(image_idx)
        temp_labels.extend(labels)

        # Save every 2 batches
        if (i + 1) % 2 == 0:
            save_batch(temp_x, temp_y, temp_images, temp_labels, wavelength, device.name, f'{i+1}')
            print(f'Done saving {i+1}_batch, resetting states for next half batch.....\n')

            # Clean memory safely
            temp_x.clear()
            temp_y.clear()
            temp_images.clear()
            temp_labels.clear()
    
    # CORRECTED: Save leftover data if total batch count is odd
    if len(temp_x) > 0:
        print("Saving remaining leftover data...")
        save_batch(temp_x, temp_y, temp_images, temp_labels, wavelength, device.name, 'last', compress=0)
        print('Done saving last batch.')

dls = [dl]
if __name__ == '__main__':
    for dl in dls:
        print(f'Working with data for {device.name}')
        print('*' * 50)
        load_split(dl, device, ds.wavelength)
