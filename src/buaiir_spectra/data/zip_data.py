import numpy as np
import joblib
from buaiir_spectra.data_v2.dataset import SpectralDataset, SpectralDataLoader, Device


# from buaiir_spectra.data.dataset import SpectralDataset
# from buaiir_spectra.utils.device import Device
# from buaiir_spectra.data.dataloader_v2 import SpectralDataLoader






# DATA_PATH  = '/home/wilfred/Datasets/spectra_data'
DATA_PATH = '/home/wilfred/Downloads/spectra_data'

device = Device.LOW_COST
ds = SpectralDataset(DATA_PATH, device=device, load_with_images=True)
dl = SpectralDataLoader(ds, batch_size=1)


def load_split(dl: SpectralDataLoader, device: Device) -> None:
    """
    Loads the data of a specific batch dataloader

    Arg:
        device: Device whose data is to be loaded
        split: target folder train, test or val
        dl: Iterable data loader whose data is to be extracted

    Return:
        None
    """
    
    temp_x, temp_y, temp_images, temp_labels = [], [], [], []

    for i, batch in enumerate(dl):
        print(f'Working with {i+1} batch ', end=": ")
        x_idx, image_idx, y_idx, labels = batch
        print(f'x_shape: {x_idx.shape}',f'y_shape: {y_idx.shape}', f'Image_shape: {image_idx.shape}', sep=' | ', end='\n')

        temp_x.append(x_idx)
        temp_y.append(y_idx)
        temp_images.append(image_idx)
        temp_labels.extend(labels)


        if (i +1) % 2 == 0:
            x = np.vstack(temp_x)
            y = np.vstack(temp_y)
            images = np.vstack(temp_images)

            state = {'x': x, 'y': y, "images": images, 'labels': temp_labels, 'wavelength': ds.wavelength}
            f'Saving {i+1} batch (x, y, images, wavelength)  data to disk.......'
            joblib.dump(state, f'src/buaiir_spectra/{device.name}/{device.name}_{i+1}_batch.pkl', compress=3)
            print(f'Done saving {i+1}_batch, resetting states for next half batch.....')

            del x
            del y
            del images
            del labels

            # reset states
            temp_x.clear()
            temp_y.clear()
            temp_images.clear()
            temp_labels.clear()
            state.clear()
    
    if temp_x is not None or temp_y is not None or temp_images is not None:
        x = np.vstack(temp_x)
        y = np.vstack(temp_y)
        images = np.vstack(temp_images)

        state = {'x': x, 'y': y, 'images': images, 'labels': temp_labels, 'wavelength': ds.wavelength}
        f'Saving Last half batch to disk.....'
        joblib.dump(state, f'src/buaiir_spectra/{device.name}/{device.name}_last_batch.pkl')



dls = [dl]
if __name__ == '__main__':

    for dl in dls:
        print(f'Working with data for {device.name}')
        print('*'*200)
        load_split(dl, device)






