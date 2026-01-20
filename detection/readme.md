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


# Manual and Explanation: Interactive Mask Generation and Segmentation (Napari + Big-FISH)

## Overview

This notebook implements an **interactive Napari-based GUI** for **image segmentation and mask generation** using the **Big-FISH** library and `scikit-image`.
It is designed to process **single multi-dimensional microscopy files** (e.g. `.tif`) identified by a naming convention (`blur_`) and to generate, visualize, and persist **segmentation masks** (e.g. transcription site or region masks).

The workflow supports:

* Interactive file selection
* Image visualization (full stack and projection)
* Threshold-based segmentation
* Morphological refinement
* Mask reuse and persistence

---

## Dependencies and Initialization

### Core Libraries

* **Numerical & data handling:** `numpy`, `pandas`
* **Visualization:** `napari`, `matplotlib`
* **Image I/O:** `tifffile`, `dask.image.imread`, `imaris_ims_file_reader`
* **Segmentation & morphology:** `bigfish.segmentation`, `bigfish.stack`, `skimage`

### GUI Framework

* **magicgui** for interactive widgets
* **napari** for real-time image and label visualization

### Version Logging

The notebook prints Big-FISH and NumPy versions to ensure environment consistency.

---

## Global Configuration

| Variable                        | Purpose                                                      |
| ------------------------------- | ------------------------------------------------------------ |
| `mIdentifier`                   | Filename prefix used to identify valid image files (`blur_`) |
| `VOXEL_RADIUS`, `OBJECT_RADIUS` | Reserved for future physical calibration                     |
| `DEFAULT_CHOICES`               | Initial dropdown placeholder                                 |
| `SET_TABLE`, `CELL_CHOICE`      | GUI state tracking                                           |
| `DEBUG`                         | Enables verbose logging                                      |

---

## GUI Workflow (User Manual)

### 1. Choose Home Folder

**Widget:** `choose_home_folder`

* Select a directory containing `.tif` files.
* Files are filtered by:

  * Prefix defined by `mIdentifier`
  * Must be a file (not a directory)

The dropdown is automatically populated with valid image files.

---

### 2. Select Image File

Triggered automatically upon dropdown selection.

* Updates internal state to track the active file.
* Associates the selected image with existing threshold metadata if available.
* Displays a warning if expected threshold information is missing.

---

### 3. Load Image

**Widgets:** `getFolderName` → `showMovie`

Once a file is selected:

* The image is loaded using Dask for efficiency.
* Two layers are displayed in Napari:

  * **Maximum intensity projection** (`stackCell`)
  * **Full image stack** (`full`)

This allows both overview inspection and detailed exploration.

---

### 4. Segment Image and Generate Mask

**Widget:** `process_image`

This is the core segmentation step.

#### User-controlled parameters:

* **Threshold:** Intensity cutoff for segmentation
* **Kernel size (dilation):** Expands segmented regions
* **Kernel size (erosion):** Refines boundaries
* **Smoothness:** Controls boundary regularization

#### Processing pipeline:

1. Intensity thresholding
2. Removal of small artifacts
3. Morphological erosion
4. Morphological dilation
5. Instance labeling
6. Boundary smoothing

#### Mask reuse:

* If a mask already exists on disk, it is loaded instead of recomputed.

**Output:**
A labeled mask layer displayed in Napari.

---

### 5. Save Mask to Disk

**Widget:** `saveMask`

* Saves the segmentation mask as a NumPy array.
* Output directory:

  ```
  <parent_folder>/masks_tx/
  ```
* File name matches the original image name.

This enables reproducible downstream analysis without re-segmentation.

---

### 6. Reset Controls

| Button                | Action                             |
| --------------------- | ---------------------------------- |
| **Analyse new cell**  | Clears all viewer layers           |
| **Analyse new movie** | Clears layers and resets GUI state |

---

## Expected Directory Structure

```
project_root/
├── images/
│   ├── blur_001.tif
│   ├── blur_002.tif
│   └── ...
├── masks_tx/
│   ├── blur_001.npy
│   └── blur_002.npy
```

---

## Intended Use Case

This notebook is intended for:

* Interactive mask generation
* Rapid parameter tuning for segmentation
* Manual quality control
* Preprocessing prior to quantitative spot analysis

It is optimized for **human-in-the-loop segmentation**, not batch automation.

---

## Summary

This notebook provides a robust, GUI-driven segmentation workflow that integrates Big-FISH morphology tools with Napari’s interactive visualization. It emphasizes reproducibility, visual validation, and efficient reuse of segmentation results.

---

If you would like, I can also:

* Produce a **short quick-start guide**
* Convert this into a **README.md**
* Add a **parameter tuning guide**
* Align terminology with your **detection notebook** for consistency

