# Manual and Explanation: Interactive Spot Detection and Threshold Estimation (Napari + Big-FISH)

## Overview

The notebook **0.0-getThreshold&Spots-GUI.ipynb** implements an **interactive GUI-based pipeline** for detecting RNA spots in time-lapse 3D microscopy data using **Napari** and the **Big-FISH** library.
The workflow allows a user to:

1. Select a cell dataset from disk
2. Visualize raw image stacks or maximum intensity projections
3. Define voxel and object (spot) physical dimensions
4. Automatically estimate an optimal detection threshold
5. Manually verify and refine the threshold
6. Visualize detected spots directly on the image data

The entire pipeline is controlled via **MagicGUI widgets** embedded in Napari.

---

## Dependencies and Initialization

### Imported Libraries

* **Core scientific stack:** `numpy`, `pandas`, `matplotlib`
* **Visualization:** `napari`, `magicgui`
* **Image I/O:** `tifffile`, `dask.image.imread`, `imaris_ims_file_reader`
* **Spot detection:** `bigfish.stack`, `bigfish.detection`
* **Utilities:** `kneed` (knee / elbow detection), `pathlib`

### Custom Modules

* `buildReferenceSpotFromImages`: builds reference spot profiles
* `runBigfishDetection.saveSpotsNPZ`: persists detected spots

### Version Reporting

The notebook prints Big-FISH and NumPy versions to ensure reproducibility.

---

## Global Configuration

| Variable          | Purpose                                 |
| ----------------- | --------------------------------------- |
| `mIdentifier`     | Prefix identifying valid cell folders   |
| `MaxTimePoint`    | Minimum number of frames per valid cell |
| `VOXEL_RADIUS`    | Physical voxel size (z, y, x) in nm     |
| `OBJECT_RADIUS`   | Expected spot radius (z, y, x) in nm    |
| `DEFAULT_CHOICES` | Dropdown initialization state           |
| `DEBUG`           | Verbose logging toggle                  |

---

## GUI Workflow (User Manual)

### 1. Choose Home Folder

**Widget:** `choose_home_folder`

* Select the directory containing cell subfolders.
* Automatically scans for folders matching `mIdentifier` and minimum frame count.
* Populates a dropdown with valid cells.

**Side effect:**
Creates or loads `thresholds.csv` to store per-cell thresholds.

---

### 2. Select Cell

Triggered automatically when the dropdown changes.

* Loads or initializes a **threshold table**:

  * Cell name
  * Detection threshold
  * Transcription site count
* Table edits are persisted to disk.

---

### 3. Load Cell Movie

**Widget:** `getFolderName` → `showMovie`

* Loads all `.tif` files in the selected cell folder.
* Displays either:

  * Full 4D stack (time × z × y × x), or
  * Maximum Intensity Projection (MIP) over z
* Image is displayed in Napari as `stackCell`.

---

### 4. Set Physical Parameters

**Widget:** `setVoxelandObjectParameters`

Defines:

* `VOXEL_RADIUS`: physical voxel dimensions (nm)
* `OBJECT_RADIUS`: expected RNA spot size (nm)

These parameters directly affect LoG filtering and spot detection sensitivity.

---

### 5. Automatic Threshold Estimation

**Widget:** `getThreshold`

* Subsamples frames using a user-defined sampling rate.
* Runs Big-FISH spot detection across sampled images.
* Computes:

  * Spot counts vs. threshold
  * Elbow / knee point using:

    * Big-FISH breaking point
    * Kneedle algorithm

**Output:**

* Elbow plot (embedded Matplotlib widget)
* Suggested automatic threshold

---

### 6. Manual Threshold Verification

**Widget:** `recordThreshold`

* User selects a threshold via slider.
* A subsampled movie is displayed.
* Detected spots are overlaid as a Napari `Points` layer.

**Detection steps per frame:**

1. LoG filtering
2. Local maxima detection
3. Intensity thresholding

This step allows visual confirmation of detection quality.

---

### 7. Visual Output

* Detected spots are rendered as white-outlined points.
* Raw image layers are removed to reduce clutter.
* Multiple thresholds can be tested iteratively.

---

### 8. Reset Controls

| Button                | Function                            |
| --------------------- | ----------------------------------- |
| **Analyse new cell**  | Clears layers for next cell         |
| **Analyse new movie** | Fully resets GUI state and dropdown |

---

## Internal Helper Functions

### `formatSpotsShape`

* Converts per-frame spot lists into a single Napari-compatible array:

  ```
  [time_index, z, y, x]
  ```

### Event Handlers

* `.called.connect` and `.changed.connect` callbacks synchronize:

  * GUI state
  * File system
  * Viewer layers

---

## Expected Directory Structure

```
home_folder/
├── cell_001/
│   ├── t000.tif
│   ├── t001.tif
│   └── ...
├── cell_002/
│   └── ...
└── thresholds.csv
```

---

## Intended Use Case

This notebook is designed for **interactive exploratory analysis**, not batch processing.
It is particularly suited for:

* smFISH / RNA spot detection
* Parameter tuning
* Quality control prior to automated pipelines

---

## Summary

This code provides a tightly integrated **Napari + Big-FISH GUI** that bridges automated spot detection with human-in-the-loop validation. The modular widget structure ensures reproducibility while preserving flexibility in biological image analysis.

---

If you want, I can also:

* Convert this into **repository-ready README documentation**
* Add a **flow diagram**
* Rewrite it as a **methods section for a paper**
* Produce a **short “quick-start” version**

