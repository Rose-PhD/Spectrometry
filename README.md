## BUAIIR Spectra

**BUAIIR Spectra** is a Python library designed to simplify data loading and batching for spectral analysis tasks using an open-source dataset collected and maintained by BUAIIR.

The library provides a clean interface for creating `Dataset` and `DataLoader` objects, with built-in preprocessing and conversion utilities compatible with common machine learning frameworks.

---

## Dataset Overview

The spectral dataset was collected using three different devices:

| Device Name     | Spectral Range(nm) |
| --------------- | ------------------ |
| BIO_SCIENCE     | 3648               |
| SCAN CODER      | 12                 |
| LOW COST device | 381                |

For each device, spectral data was collected across three crop types:

| Crop    | Total Samples | Class Breakdown     |
| ------- | ------------- | ------------------- |
| Beans   | 15            | 5 HLT, 5 BRD, 5 BLB |
| Maize   | 15            | 5 HLT, 5 MSV, 5 MLN |
| Cassava | 15            | 5 HLT, 5 CMD, 5 CBB |

Each crop type was subjected to controlled inoculation with viral and bacterial diseases, resulting in multiple classification labels per crop.

### Class Definitions

Each crop contains a healthy control class plus disease-specific classes:

| Crop    | Class Code | Description              |
| ------- | ---------- | ------------------------ |
| Beans   | HLT        | Healthy / Control        |
| Beans   | BLB        | Bean Bacterial Blight    |
| Beans   | BRD        | Bean Rust Disease        |
| Maize   | HLT        | Healthy / Control        |
| Maize   | MSV        | Maize Streak Virus       |
| Maize   | MLN        | Maize Lethal Necrosis    |
| Cassava | HLT        | Healthy / Control        |
| Cassava | CMD        | Cassava Mosaic Disease   |
| Cassava | CBB        | Cassava Bacterial Blight |

### Data Collection Period

The dataset was collected over a period of **15 weeks**, with repeated sampling across all classes, crops, and devices.

---

## Data Loading

The library provides utilities for loading spectral data into structured machine learning pipelines.

It supports:

- Dataset object creation
- DataLoader batching
- Standardized preprocessing and conversion
- Compatibility with deep learning frameworks such as PyTorch

Example usage:

## Installation

You can install the **buaiir-spectra** library using pip:

```bash
pip install buaiir-spectra
```

### Requirements

Make sure you have Python 3.8+ installed. The library is designed to work with common scientific Python packages such as NumPy and PyTorch.

Dataloading per device using load_spectra function

```python
from buaiir_spectra.data.load_fn import load_spectra

# Load data for BIO_SCIENCE device
lf = load_spectra(device=Device.BIO_SCIENCE, shuffle=False, no_files_per_load=4, load_with_images=True)

for batch in lf:
    x, y, images = next(iter(lf))
    print(f'x_shape: {x.shape}',f'y_shape: {y.shape}', f'image_shape: {images.shape}')



# Load data for the SCAN_CODER device
lf = load_spectra(device=Device.SCAN_CODER, shuffle=False, no_files_per_load=4, load_with_images=True)

# Load data for the LOW_COST device
lf = load_spectra(device=Device.LOW_COST, shuffle=False, no_files_per_load=4, load_with_images=True)

# Loading only spectra data without images
lf = load_spectra(device=Device.LOW_COST, shuffle=False, no_files_per_load=4, load_with_images=False)

    for batch in lf:
        x, y = next(iter(lf))
        print(f'x_shape: {x.shape}',f'y_shape: {y.shape}')

# To load all the data in-memory (Note is too huge, partial loading can be achieved by reducing the no_files_per_load)
lf = load_spectra(device=Device.SCAN_CODER, shuffle=False, no_files_per_load=-1, load_with_images=True)
x, y, images = next(iter(lf))

```

Nature of Target(y)
The target returns a tuple **(titer_value, expert_score, week, disease_class)** where each element describes a specific aspect of the spectral and laboratory observation collected from each plant sample.

| Feature       | Type    | Description                                                                            |
| ------------- | ------- | -------------------------------------------------------------------------------------- |
| titer_value   | Float   | Ground truth measurement collected from each plant, aligned with the spectral reading. |
| expert_score  | Integer | Visual severity score assigned by an agricultural expert based on observable symptoms. |
| week          | Integer | Week of data collection during the 15-week sampling period.                            |
| disease_class | Float   | Label representing the disease type or health status of the plant sample.              |

Feature matrix (x) currently contains only the calibrated wavelength reading of each device

Dynamic data loading

```python
from buaiir_spectra.data.dataset import SpectralDataset
from buaiir_spectra.utils.device import Device

# Path to where data is store
DATA_PATH = '/home/wilfred/Datasets/spectra_data'

for test_device in Device.get_devices():
    dataset = SpectralDataset(DATA_PATH, device=test_device) # dataset
    x, y = dataset[0] # load sample data
    print(f'Prinitng shapes for device: {test_device.name}')
    print(x.shape, y.shape)

```

Properties of dataset

```python
from buaiir_spectra.data.dataset import SpectralDataset
from buaiir_spectra.utils.device import Device

# create Dataset object for LOW COST only
dataset = SpectralDataset('/home/usr/Datasets/spectra_data', Device.SCAN_CODER)

# Get wavelength range for the device
wavelength = dataset.wavelength
print(wavelength)

# Get disease class codes used in batching
disease_classes = dataset.disease_class_codes
print(f'Supported disease classes {disease_classes}')

# Get plant_type codes used in the batching
plant_types = dataset.plant_type_codes
print(f'Supported crop types: {plant_types}')

```

Data batching

```python
from buaiir_spectra.data.dataset import SpectralDataset
from buaiir_spectra.utils.device import Device
from buaiir_spectra.data.dataloader import SpectralDataLoader

# Path to where you dataset is store: adjust accordingly
DATA_PATH = '/home/usr/Datasets/spectra_data'

dataset = SpectralDataset(data_path=DATA_PATH, device=Device.SCAN_CODER)

# creat the dataloader
dataloader = SpectralDataLoader(dataset, batch_size=4)

# iterate over the batchs
for batch in dataloader:
    # extract the x_batch and y_batch
    x_batch, y_batch = batch

    # print the shape of the batches
    print(x_batch.shape, y_batch.shape)

```

Parameters for data wrangling provided by Dataloader

```python
from buaiir_spectra.data.dataset import SpectralDataset
from buaiir_spectra.data.dataloader import SpectralDataLoader
from buaiir_spectra.utils.device import Device


# Path to where you dataset is store: adjust accordingly
DATA_PATH = '/home/usr/Datasets/spectra_data'

# Loading data for BIO SCINCE
dataset = SpectralDataset(data_path=DATA_PATH, device=Device.BIO_SCIENCE)

# Creating a dataloader with plant labels shuffled
dataloader_with_shuffled_plants = SpectralDataLoader(dataset, batch_size=40, permutate_plants= True)

# Creating a dataloader with weeks shuffled
dataloader_with_shuffled_weeks = SpectralDataLoader(dataset, batch_size=40, permutate_weeks= True)

# Creating a dataloader with completely shuffled data, best for regularization
dataloader_with_shuffled_weeks = SpectralDataLoader(dataset, batch_size=40, permutate=True)

```

Extracting label specific or week specific data

```python
from buaiir_spectra.data.dataset import SpectralDataset
from buaiir_spectra.data.dataloader import SpectralDataLoader
from buaiir_spectra.utils.device import Device

DATA_PATH = '/home/wilfred/Datasets/spectra_data'

# Loading data for BIO SCINCE
dataset = SpectralDataset(data_path=DATA_PATH, device=Device.SCAN_CODER)
dataloader = SpectralDataLoader(dataset, batch_size=150)


# Load data for only a single disease class e.g CMD
x, y = dataloader.load_data_of_disease_class('CMD')

# Load data for a specific label across all weeks
x_1, y_1 = dataloader.load_data_of('BBLB1')

# Checking all supported labels
supported_labels = dataloader.labels
print(supported_labels)

```
