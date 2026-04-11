# Spectrometry

A Python package for loading, preprocessing, and analyzing multi-spectral sensor data for plant disease detection. Supports three different spectral sensor hardware types across longitudinal multi-week measurements.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Devices](#devices)
- [Data Organization](#data-organization)
- [Module Documentation](#module-documentation)
  - [data/dataset.py](#datadatasetpy)
  - [data/data_loader.py](#datadata_loaderpy)
  - [preprocess/transform.py](#preprocesstransformpy)
  - [utils/load_fns.py](#utilsload_fnspy)
- [Usage](#usage)
- [Disease Labels](#disease-labels)
- [Dependencies](#dependencies)

---

## Overview

This package provides a unified interface for reading spectral data collected from three different sensor devices deployed in plant disease research. It handles device-specific file formats, batch loading for ML pipelines, and preprocessing transformations including wavelength range clipping and outlier removal.

The data covers multiple plant varieties (beans, cassava, maize) measured across weeks 1–6, with each plant labeled for disease state.

---

## Project Structure

```
spectrometry/
├── data/
│   ├── __init__.py             # Module init (no exports)
│   ├── dataset.py              # SpectralDataset and Device enum
│   └── data_loader.py          # SpectralDataLoader for batch iteration
├── preprocess/
│   ├── __init__.py             # Exports Transform, Range_Clip
│   └── transform.py            # Transform, Range_Clip, Zscore_Outlier_Filter, Bound_Outlier_Filter
├── utils/
│   ├── __init__.py             # Module init (no exports)
│   └── load_fns.py             # load_data, plot_spectral, display_img, join_path
├── tests/
│   ├── module_tests.ipynb                    # Integration tests for all devices
│   ├── scan_coder_visualization.ipynb        # SCAN_CODER device validation
│   ├── high_end_spectral_visualization.ipynb # BIO_SCIENCE device validation
│   └── low_cost.ipynb                        # LOW_COST device validation
├── spectral_data/
│   └── Multi-spectral sensor data/
│       ├── spectral_data_week1/
│       ├── spectral_data_week2/
│       ├── spectral_data_week3/
│       ├── spectral_data_week5/
│       └── spectral_data_week6/
├── pyproject.toml
└── requirements.txt
```

---

## Installation

```bash
pip install -e .
```

Requires Python >= 3.9. Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Devices

Three sensor hardware types are supported, represented by the `Device` enum:

| Enum Member          | Value | Description                        |
| -------------------- | ----- | ---------------------------------- |
| `Device.BIO_SCIENCE` | `0`   | High-end laboratory spectrometer   |
| `Device.SCAN_CODER`  | `1`   | Compolytics Scancorder (mid-range) |
| `Device.LOW_COST`    | `2`   | Low-cost multispectral sensor      |

---

## Data Organization

Raw data is organized by week, then by device, then by plant variety and disease label:

```
spectral_data/Multi-spectral sensor data/
└── spectral_data_week{N}/
    └── {device}/
        └── {plant_variety}/
            └── {disease_label}/
                └── {point}/
                    └── {specimen}/
                        └── {data_files}
```

**File types per device:**

- **SCAN_CODER:** `.csv` files with wavelength columns and a `Token ID` metadata column
- **BIO_SCIENCE:** Paired raw CSV files (3648-row intensity readings) and calculation CSVs
- **LOW_COST:** `.csv` files with dict/list-encoded `spectral_1`, `spectral_2`, `calibration`, `wavelength` columns; `.jpg` images

---

## Module Documentation

### data/dataset.py

Core dataset abstraction for loading spectral readings from all three devices.

---

#### `Device` (Enum)

Enumerates the three supported sensor hardware types.

```python
class Device(Enum):
    BIO_SCIENCE = 0
    SCAN_CODER = 1
    LOW_COST = 2
```

**Methods:**

| Method               | Returns     | Description                       |
| -------------------- | ----------- | --------------------------------- |
| `get_devices()`      | `list`      | Returns all `Device` enum members |
| `get_device_names()` | `list[str]` | Returns device name strings       |

---

#### `FileFormats` (Enum)

Enumerates supported data file extensions.

```python
class FileFormats(Enum):
    csv = "csv"
    png = "png"
    json = "json"
```

---

#### `Dataset` (Abstract Base Class)

Abstract interface that all dataset implementations must follow.

**Methods:**

| Method        | Signature         | Description                        |
| ------------- | ----------------- | ---------------------------------- |
| `__len__`     | `(device) -> int` | Returns number of items in dataset |
| `__getitem__` | `(index) -> Any`  | Returns a single data item         |

---

#### `SpectralDataset(Dataset)`

Main dataset loader. Handles directory traversal, file discovery, and item-level data reading for all three device types.

**Constructor:**

```python
SpectralDataset(data_path: str, device: Device)
```

Scans `data_path` recursively to discover device-specific files. For `SCAN_CODER`, concatenates all CSVs into a single DataFrame at init time. For `BIO_SCIENCE`, pairs raw and calculation files.

**State attributes:**

| Attribute                    | Type           | Description                              |
| ---------------------------- | -------------- | ---------------------------------------- |
| `data_path`                  | `str`          | Root directory for spectral data         |
| `device`                     | `Device`       | Target device type                       |
| `wavelength`                 | `np.ndarray`   | Lazily computed wavelength values        |
| `DONE_COMPUTING_WAVELENGTH`  | `bool`         | Flag preventing wavelength recomputation |
| `tracked_csvs`               | `list`         | CSV paths (SCAN_CODER only)              |
| `tracked_xlsx`               | `list`         | XLSX file paths                          |
| `high_end_csvs`              | `list`         | All BIO_SCIENCE CSV paths                |
| `high_end_raw_files`         | `list`         | BIO_SCIENCE raw reading files            |
| `high_end_calculation_files` | `list`         | BIO_SCIENCE calculation files            |
| `low_cost_csvs`              | `list`         | LOW_COST CSV paths                       |
| `low_cost_imgs`              | `list`         | LOW_COST image paths                     |
| `scan_corder_data`           | `pd.DataFrame` | Concatenated SCAN_CODER readings         |

**Methods:**

---

##### `__len__() -> int`

Returns the number of spectral readings available. Routes by device type:

- `SCAN_CODER`: row count of `scan_corder_data`
- `BIO_SCIENCE`: count of `high_end_raw_files`
- `LOW_COST`: count of `low_cost_csvs`

---

##### `__getitem__(index: int) -> np.ndarray`

Returns a single spectral reading as a `float32` numpy array. Dispatches to the appropriate device-specific getter.

---

##### `get_scan_corder_item(index: int) -> np.ndarray[float32]`

Extracts one row from the in-memory `scan_corder_data` DataFrame.

- Skips the first 3 metadata columns
- On the first call, computes `wavelength` from column headers and sets `DONE_COMPUTING_WAVELENGTH`
- Returns a 1D `float32` array of spectral intensities

---

##### `get_low_cost_item(index: int) -> np.ndarray[float32]`

Reads a LOW_COST device CSV file by index and returns parsed spectral data.

- Calls `clean_low_cost_cols()` to parse encoded column values
- Returns a 2D array of shape `(2, num_wavelengths)` stacking `spectral_1` and `spectral_2`

---

##### `get_high_end_item(index: int) -> np.ndarray[float32]`

Reads a BIO_SCIENCE raw spectrometer file by index.

- Skips 6-row header, reads 3648 spectral rows
- Standardizes column names via `clean_high_end_cols()`
- Converts values to `float32` via `convert_high_end_cols()`, handling `%` suffix in last column
- Returns the last column (calibrated intensity values) as a 1D `float32` array

---

##### `get_scan_corder_data() -> pd.DataFrame`

Concatenates all tracked SCAN_CODER CSV files into a single DataFrame.

- Drops the `Token ID` metadata column
- Adds a `week` column extracted from each file's path
- Returns the merged DataFrame

---

##### `extract_high_end_raw_calculations() -> None`

Pairs BIO_SCIENCE raw and calculation CSV files discovered during directory traversal.

- Populates `high_end_raw_files` and `high_end_calculation_files`
- Removes unmatched/missing entries via `remove_none()`

---

##### `clean_low_cost_cols(df: pd.DataFrame, index: int) -> np.ndarray[float32]`

Parses LOW_COST device CSV columns that contain Python dict/list string representations.

- Extracts `spectral_1`, `spectral_2`, `calibration`, and optionally `wavelength` columns
- Returns a 2D array stacking `spectral_1` and `spectral_2`

---

##### `_load_fn() -> None`

Recursively traverses `data_path` to discover and categorize all data files.

- Populates device-specific file lists: `tracked_csvs`, `high_end_csvs`, `low_cost_csvs`, `low_cost_imgs`
- Filters by file extension and directory structure conventions

---

##### `get_label(reading: str) -> tuple[str, str]` (static)

Parses a file path to extract the plant label and measurement period.

- Returns `(plant_name + period, full_filename)`

---

##### `get_week_for_scan_corder_reading(reading: str) -> str` (static)

Extracts the week number (last digit) from a SCAN_CODER file path.

---

##### `extract_week(path: str) -> str` (static)

Extracts the week character from a directory path segment.

---

##### `identify_none(data: list) -> list[int]` (static)

Returns the indices of all `None` values in a list.

---

##### `remove_none(data: list, indices: list[int] = None) -> tuple[list, list[int]]` (static)

Removes `None` entries from a list.

- Returns `(cleaned_list, removed_indices)`

---

##### `clean_high_end_cols(df: pd.DataFrame) -> pd.Index` (static)

Standardizes BIO_SCIENCE column names: strip whitespace, lowercase, replace spaces with underscores.

---

##### `convert_high_end_cols(df: pd.DataFrame) -> pd.DataFrame` (static)

Converts all BIO_SCIENCE DataFrame columns to `float32`.

- Special handling for the last column: strips `%` characters before conversion

---

### data/data_loader.py

Batch iteration over a `Dataset` for use in ML training pipelines.

---

#### `DataLoader` (Abstract Base Class)

Abstract interface for batch loaders.

**Constructor:**

```python
DataLoader(batch_size: int, random: bool)
```

**Methods:**

| Method     | Returns     | Description               |
| ---------- | ----------- | ------------------------- |
| `__iter__` | `Generator` | Yields batches (abstract) |

---

#### `SpectralDataLoader(DataLoader)`

Iterates over a `SpectralDataset` in batches, with optional index shuffling.

**Constructor:**

```python
SpectralDataLoader(dataset: Dataset, batch_size: int, random: bool = False)
```

**Methods:**

---

##### `__iter__() -> Generator`

Yields batches as contiguous 2D numpy arrays.

- Builds index list `[0, 1, ..., len(dataset)-1]`
- If `random=True`, shuffles indices in place before iteration
- Slices indices by `batch_size`, collects items, calls `make_contiguous()`
- Yields arrays of shape `(batch_size, num_features)`

---

##### `make_contiguous(data_buff: list[np.ndarray]) -> np.ndarray` (static)

Vertically stacks a list of 1D or 2D arrays into a single contiguous 2D array using `np.vstack()`.

---

### preprocess/transform.py

Transformation classes for cleaning and filtering spectral data before model training.

---

#### `Transform` (Abstract Base Class)

Abstract interface all transforms must implement.

**Methods:**

| Method          | Signature                  | Description                                         |
| --------------- | -------------------------- | --------------------------------------------------- |
| `forward`       | `(*args, **kwargs) -> Any` | Core transformation logic (abstract)                |
| `__call__`      | `(*args, **kwargs) -> Any` | Calls `forward()`; enables `transform(data)` syntax |
| `fit_transform` | `(*args, **kwargs) -> Any` | Alias for `forward()`; scikit-learn-compatible API  |

---

#### `Range_Clip(Transform)`

Clips spectral data to a specified wavelength range, removing readings outside the bounds.

**Constructor:**

```python
Range_Clip(lower_bound: int, upper_bound: int = None)
```

Validates that `lower_bound < upper_bound` when both are provided.

**Methods:**

---

##### `forward(x_data: np.ndarray, wavelength_range: np.ndarray) -> tuple[np.ndarray, np.ndarray]`

Applies wavelength range clipping to spectral data.

- Builds a boolean mask: `wavelength >= lower_bound` (and `<= upper_bound` if set)
- Applies mask column-wise to `x_data` and element-wise to `wavelength_range`
- Tracks kept/removed wavelength point counts
- Returns `(clipped_data, clipped_wavelengths)`

---

##### `get_stats() -> dict`

Returns clipping statistics from the last `forward()` call.

- Keys: `'kept'` (wavelength points retained), `'Removed'` (wavelength points discarded)
- Raises `AssertionError` if called before `forward()`

---

#### `Zscore_Outlier_Filter(Transform)`

Removes outlier rows using a z-score threshold across all spectral features.

**Constructor:**

```python
Zscore_Outlier_Filter(boundary: int = 3)
```

Default boundary of 3 standard deviations from the mean.

**Methods:**

---

##### `forward(x_data: np.ndarray) -> np.ndarray`

Filters rows where any feature value exceeds the z-score boundary.

- Computes `np.abs(zscore(x_data))` across all features
- Keeps only rows where all feature z-scores are below `boundary`
- Accumulates removed row count in `_tot_outliers`

---

##### `get_outlier_count() -> int`

Returns total number of outlier rows removed across all `forward()` calls.

- Raises `AssertionError` if called before `forward()`

---

#### `Bound_Outlier_Filter(Transform)`

Removes rows where any sensor value falls outside a specified numeric range. Useful for filtering physically invalid sensor readings.

**Constructor:**

```python
Bound_Outlier_Filter(lower_bound: float = 0, upper_bound: float = 1.0)
```

Default range `[0.0, 1.0]` matches the normalized sensor output range.

**Methods:**

---

##### `forward(x_data: np.ndarray) -> np.ndarray`

Filters rows where any value is outside `[lower_bound, upper_bound]`.

- Builds mask: `(x_data >= lower_bound) & (x_data <= upper_bound)`
- Requires all columns to pass: `.all(axis=1)`
- Accumulates removed row count in `_tot_removed`

---

##### `get_removed_count() -> int`

Returns total number of rows removed due to bound violations.

- Raises `AssertionError` if called before `forward()`

---

### utils/load_fns.py

Helper utilities for directory traversal, plotting, and image display.

---

##### `join_path(parent_pth: str, child_pth: str) -> str`

Lambda wrapper around `os.path.join()`. Joins two path components.

---

##### `load_data(data_path: str, spectral_buffer: list, image_buffer: list) -> None`

Traverses the LOW_COST device directory structure and collects file paths.

- Recursively descends: `week → class → plant → label → specimen_dir → files`
- Appends `.csv` files to `spectral_buffer`
- Appends `.jpg` files to `image_buffer`
- Raises `ValueError` for unsupported file extensions

---

##### `plot_spectral(_range: np.ndarray, values: np.ndarray) -> None`

Plots a single spectral signature curve.

- X-axis: wavelength values (labeled "Wavelength (400–700nm)")
- Y-axis: intensity values (labeled "Normalized Intensity")
- Title: "Spectral characteristic"

---

##### `display_img(img_path: str) -> None`

Lambda wrapper around `matplotlib.image.imshow()`. Reads and displays an image file.

---

## Usage

### Loading a dataset

```python
from data.dataset import SpectralDataset, Device

dataset = SpectralDataset("spectral_data/Multi-spectral sensor data", Device.SCAN_CODER)
print(len(dataset))       # number of readings
item = dataset[0]         # np.ndarray[float32] of spectral intensities
```

### Batch iteration

```python
from data.data_loader import SpectralDataLoader

loader = SpectralDataLoader(dataset, batch_size=32, random=True)
for batch in loader:
    # batch.shape == (32, num_wavelengths)
    pass
```

### Applying transforms

```python
from preprocess.transform import Range_Clip, Bound_Outlier_Filter, Zscore_Outlier_Filter

# Clip to visible spectrum
clip = Range_Clip(lower_bound=400, upper_bound=700)
x_clipped, wavelengths_clipped = clip(x_data, dataset.wavelength)
print(clip.get_stats())   # {'kept': N, 'Removed': M}

# Remove physically invalid sensor values
bound_filter = Bound_Outlier_Filter(lower_bound=0.0, upper_bound=1.0)
x_clean = bound_filter(x_clipped)

# Remove statistical outliers
z_filter = Zscore_Outlier_Filter(boundary=3)
x_final = z_filter(x_clean)
```

### Plotting

```python
from utils.load_fns import plot_spectral

plot_spectral(dataset.wavelength, dataset[0])
```

---

## Disease Labels

| Label | Disease                 |
| ----- | ----------------------- |
| `HLT` | Healthy                 |
| `BLB` | Bacterial Leaf Blight   |
| `BRD` | Bacterial Red/Brown Dot |
| `CBB` | Cassava Brown Streak    |
| `CMD` | Cassava Mosaic Disease  |
| `MSV` | Maize Streak Virus      |
| `MLN` | Maize Lethal Necrosis   |

---

## Dependencies

| Package              | Version       | Purpose                              |
| -------------------- | ------------- | ------------------------------------ |
| numpy                | 2.4.4         | Numerical arrays and operations      |
| pandas               | 3.0.2         | DataFrame loading and manipulation   |
| scipy                | 1.17.1        | Z-score computation                  |
| matplotlib           | 3.10.8        | Visualization                        |
| scikit-learn         | 1.8.0         | KMeans, PCA, and other ML algorithms |
| torch                | 2.11.0        | Deep learning models                 |
| xgboost              | 3.2.0         | Gradient boosting                    |
| jupyter / jupyterlab | 7.5.5 / 4.5.6 | Notebook environment                 |
