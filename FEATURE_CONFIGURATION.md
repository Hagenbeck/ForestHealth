# Using default_features.json to Configure Features

This guide explains how to use `default_features.json` to specify which features the `FeatureService` should generate from your satellite imagery data.

## Overview

The `default_features.json` file defines a set of features that are automatically calculated from your multi-dimensional satellite data when using `FeatureService`. Each feature represents a different statistical or spatial analysis applied to specific bands of your imagery.

You can define your own feature set by either:
* editing the `default_features.json`
* creating the FeatureService with a custom featuresetting

The consideration intervals are set like usual python slicing, with the start value being inclusive and the end value being exclusive.  
`start = 0` and `end = 12` would mean the first 12 values.  
`start = -12` and `end = null` means the last 12 values.  

## File Structure

The `default_features.json` file contains a JSON object with a single `features` array:

```json
{
    "features": [
        { /* feature objects */ }
    ]
}
```

Each feature in the array is a JSON object that specifies:
- **type**: The type of feature to calculate (determines available parameters)
- **band_id**: Which band of data to analyze (required for most features)
- Additional parameters depending on the feature type

## Available Feature Types

### Temporal Features (Time-based Analysis)

#### 1. **raw**
Returns the raw band data without any processing.

```json
{
    "type": "raw",
    "band_id": 5,
    "consideration_interval_start": null,
    "consideration_interval_end": null
}
```

**Parameters:**
- `band_id` (required): Band index to extract
- `consideration_interval_start` (optional): Start time index
- `consideration_interval_end` (optional): End time index

---

#### 2. **mean**
Calculates the mean value across time periods for a specific band.

```json
{
    "type": "mean",
    "band_id": 5,
    "consideration_interval_start": -12,
    "consideration_interval_end": null
}
```

**Parameters:**
- `band_id` (required): Band index
- `consideration_interval_start` (optional): Start of time window (negative = relative to end)
- `consideration_interval_end` (optional): End of time window

**Example:** `consideration_interval_start: -12` means "from 12 months ago to the present"

---

#### 3. **std**
Calculates the standard deviation across time periods.

```json
{
    "type": "std",
    "band_id": 0
}
```

**Parameters:**
- `band_id` (required): Band index
- `consideration_interval_start` (optional): Start of time window
- `consideration_interval_end` (optional): End of time window

---

#### 4. **deseasonalized_diff**
Calculates differences between time points at a fixed lag (typically 12 months for yearly comparison).

```json
{
    "type": "deseasonalized_diff",
    "band_id": 5,
    "lag": 12,
    "consideration_interval_start": null,
    "consideration_interval_end": null
}
```

**Parameters:**
- `band_id` (required): Band index
- `lag` (default: 12): Time lag for difference calculation (e.g., 12 for month-over-month yearly)
- `consideration_interval_start` (optional): Start of time window
- `consideration_interval_end` (optional): End of time window

---

#### 5. **deseasonalized_diff_specific_month**
Calculates year-over-year differences for a specific month of the year.

```json
{
    "type": "deseasonalized_diff_specific_month",
    "band_id": 2,
    "month": 8,
    "lag": 12
}
```

**Parameters:**
- `band_id` (required): Band index
- `month` (required): Month number (0-11, where 0=January, 8=September)
- `lag` (default: 12): Time lag (typically 12 for year-over-year)
- `consideration_interval_start` (optional): Start of time window
- `consideration_interval_end` (optional): End of time window

---

#### 6. **difference_in_mean_between_intervals**
Compares the mean values of two different time intervals.

```json
{
    "type": "difference_in_mean_between_intervals",
    "band_id": 6,
    "interval_one_start": 0,
    "interval_one_end": 11,
    "interval_two_start": -12,
    "interval_two_end": -1
}
```

**Parameters:**
- `band_id` (required): Band index
- `interval_one_start` (default: 0): Start of first interval
- `interval_one_end` (default: 11): End of first interval
- `interval_two_start` (default: -12): Start of second interval
- `interval_two_end` (default: -1): End of second interval

**Example:** Compare the first 12 months (interval 1) with the previous 12 months (interval 2)

---

### Spatial Features (Neighborhood-based Analysis)

Spatial features analyze patterns within a local neighborhood around each pixel using a sliding window.

#### 1. **spatial_cv**
Calculates the local coefficient of variation (std/mean ratio).

```json
{
    "type": "spatial_cv",
    "band_id": 5,
    "window_size": 5,
    "consideration_interval_start": null,
    "consideration_interval_end": null
}
```

**Parameters:**
- `band_id` (required): Band index
- `window_size` (default: 5): Size of the neighborhood window (5x5, 7x7, etc.)
- `consideration_interval_start` (optional): Start of time window
- `consideration_interval_end` (optional): End of time window

---

#### 2. **spatial_std**
Calculates the local standard deviation within a window.

```json
{
    "type": "spatial_std",
    "band_id": 3,
    "window_size": 5
}
```

**Parameters:**
- `band_id` (required): Band index
- `window_size` (default: 5): Size of the neighborhood window
- `consideration_interval_start` (optional): Start of time window
- `consideration_interval_end` (optional): End of time window

---

#### 3. **spatial_std_difference**
Calculates the standard deviation of the difference between two time interval means.

```json
{
    "type": "spatial_std_difference",
    "band_id": 5,
    "window_size": 5,
    "interval_one_start": 0,
    "interval_one_end": 11,
    "interval_two_start": -12,
    "interval_two_end": -1
}
```

**Parameters:**
- `band_id` (required): Band index
- `window_size` (default: 5): Size of the neighborhood window
- `interval_one_start` (default: 0): Start of first interval
- `interval_one_end` (default: 11): End of first interval
- `interval_two_start` (default: -12): Start of second interval
- `interval_two_end` (default: -1): End of second interval

---

#### 4. **spatial_range**
Calculates the local range (max - min) within a window, also known as peak-to-peak.

```json
{
    "type": "spatial_range",
    "band_id": 4,
    "window_size": 5
}
```

**Parameters:**
- `band_id` (required): Band index
- `window_size` (default: 5): Size of the neighborhood window
- `consideration_interval_start` (optional): Start of time window
- `consideration_interval_end` (optional): End of time window

---

#### 5. **spatial_edge_strength**
Calculates edge strength using Sobel gradient magnitude to identify sharp transitions.

```json
{
    "type": "spatial_edge_strength",
    "band_id": 5,
    "sigma": 1.0,
    "consideration_interval_start": null,
    "consideration_interval_end": null
}
```

**Parameters:**
- `band_id` (required): Band index
- `sigma` (default: 1.0): Gaussian smoothing parameter (must be > 0)
- `consideration_interval_start` (optional): Start of time window
- `consideration_interval_end` (optional): End of time window

---

## Understanding Index Conventions

### Band IDs
Band IDs are zero-indexed based on the order of bands in your data. For example:
- Band 0: First band
- Band 5: Sixth band
- etc.

### Time Indices
Time indices follow Python conventions:
- **Positive indices**: Counted from the start (0 = first month, 1 = second month, etc.)
- **Negative indices**: Counted from the end (-1 = last month, -12 = 12 months ago, etc.)
- **None/null**: Represents the boundary (start or end of available data)

**Examples:**
- `consideration_interval_start: 0, consideration_interval_end: 11` → First 12 months
- `consideration_interval_start: -12, consideration_interval_end: -1` → Last 12 months
- `consideration_interval_start: -12, consideration_interval_end: null` → Last 12 months to present
- `consideration_interval_start: null, consideration_interval_end: null` → All available data

---

## Using FeatureService with default_features.json

### Loading Default Features

When you create a `FeatureService` without specifying custom features, it automatically loads from `default_features.json`:

```python
from src.data_processing.feature_service import FeatureService
import numpy as np

# Your satellite data with shape (index, time, bands)
raw_data = np.load("path/to/data.npy")

# Creates FeatureService with default features from default_features.json
service = FeatureService(raw_data)

# Generate features
features_df = service.calculate_features_for_monthly_data()
```

### Using Custom Features

To use custom features instead of defaults, create a `FeatureSetting` and pass it to `FeatureService`:

```python
from src.data_processing.feature_service import FeatureService
from src.pydantic_models.feature_setting import FeatureSetting
import numpy as np

# Define custom features
custom_settings = FeatureSetting(
    features=[
        {
            "type": "mean",
            "band_id": 5
        },
        {
            "type": "spatial_std",
            "band_id": 3,
            "window_size": 7
        },
        {
            "type": "deseasonalized_diff_specific_month",
            "band_id": 5,
            "month": 8
        }
    ]
)

raw_data = np.load("path/to/data.npy")

# Creates FeatureService with custom features
service = FeatureService(raw_data, feature_settings=custom_settings)

features_df = service.calculate_features_for_monthly_data()
```

### Loading Custom Features from JSON

You can also load custom feature configurations from a JSON file:

```python
import json
from pathlib import Path
from src.data_processing.feature_service import FeatureService
from src.pydantic_models.feature_setting import FeatureSetting
import numpy as np

# Load custom features from JSON
config_path = Path("path/to/custom_features.json")
config_data = json.loads(config_path.read_text())
custom_settings = FeatureSetting(**config_data)

raw_data = np.load("path/to/data.npy")
service = FeatureService(raw_data, feature_settings=custom_settings)

features_df = service.calculate_features_for_monthly_data()
```

---

## Examples

### Example 1: Forest Health Monitoring (Default)
Monitor vegetation health and detect changes:

```json
{
    "features" : [
        {
            "type": "mean",
            "band_id": 5,
            "consideration_interval_start": -12
        },
        {
            "type": "mean",
            "band_id": 6,
            "consideration_interval_start": -12
        },
        {
            "type": "difference_in_mean_between_intervals",
            "band_id": 6,
            "interval_one_start": 0,
            "interval_one_end":11,
            "interval_two_start": -12,
            "interval_two_end":-1
        },
        {
            "type": "std",
            "band_id": 0
        },
        {
            "type": "spatial_std_difference",
            "band_id": 5,
            "window_size": 5,
            "interval_one_start": 0,
            "interval_one_end":11,
            "interval_two_start": -12,
            "interval_two_end":-1
        }
    ]
}
```

### Example 2: Quick Start - Minimal Vegetation Analysis
Get started with the most essential features for vegetation monitoring without complexity:

```json
{
    "features" : [
        {
            "type": "mean",
            "band_id": 5,
            "consideration_interval_start": -12
        },
        {
            "type": "deseasonalized_diff_specific_month",
            "band_id": 5,
            "month": 8
        },
        {
            "type": "spatial_std",
            "band_id": 5,
            "window_size": 5
        }
    ]
}
```
**Mean**: Average NDVI over the last 12 months - overall vegetation health
**Deseasonalized Difference for a specific month**: September NDVI year-over-year change - detect vegetation decline or growth
**Spatial STD**: Local vegetation variability - identify areas with heterogeneous forest conditions