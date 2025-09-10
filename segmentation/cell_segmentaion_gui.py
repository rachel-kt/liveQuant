#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 10 19:53:17 2025

@author: rachel
"""

import os
import math
import tifffile
import numpy as np
import pandas as pd
from copy import deepcopy
from dask.array.image import imread as imr

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QFileDialog,
    QProgressBar, QInputDialog, QLabel, QMessageBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

from cellpose import models
from cellpose.io import imread, save_to_png
from cellpose.utils import remove_edge_masks
from bigfish.multistack import match_nuc_cell

from cropFunctions import *
from joblib import Parallel, delayed
from tqdm import tqdm

# -------------------- helper functions

def process_frame_mask(
    tempmaskFolder,
    imageName,
    timep,
    new_Proj_label,
    centroidLabelDF_Projection,
    remove_edge_masks,
    match_nuc_cell,
    getCentroidAndOrientationImage,
    makeParameterDf,
):
    """
    Process mask for frame t, match nuclei to projection, relabel mismatches.

    Args:
        tempmaskFolder (str): Folder containing temporary masks.
        imageName (str): Name of the image file.
        timep (int): Frame index.
        new_Proj_label (ndarray): Projection label image.
        centroidLabelDF_Projection (DataFrame): Projection centroids & labels.
        remove_edge_masks (function): Function to remove edge masks.
        match_nuc_cell (function): Function to match nuclei with projection.
        getCentroidAndOrientationImage (function): Extract centroids/orientations.
        makeParameterDf (function): Convert centroid/orientation data to DataFrame.

    Returns:
        centroidLabelDF_tCorr (DataFrame): Corrected nuclei info for frame t.
        centroidLabelDF_pCorr (DataFrame): Corrected projection info.
        noNucleiFinal (ndarray): Final list of unique nucleus labels.
    """
    print('running process frame mask')
    # Load mask path for frame t
    maskPathHome = os.path.join(tempmaskFolder, imageName.replace('.', '_')) + f"_t{timep:03}_cp_masks.png"
    if not os.path.isfile(maskPathHome):
        return None, None, None

    imageMaskt = imread(maskPathHome)
    labelImaget = label(imageMaskt)
    new_t_label = remove_edge_masks(labelImaget, change_index=True)

    # Match nucleus at frame t to projection
    newLablet, newLableAll = match_nuc_cell(new_t_label, new_Proj_label, single_nuc=True, cell_alone=False)

    noNucleiAfterMatch = np.unique(newLableAll)
    noNucleiAfterMatch = np.delete(noNucleiAfterMatch, np.where(noNucleiAfterMatch == 0))

    if noNucleiAfterMatch.size == 0:
        return None, None, None

    # Compute centroids
    test_t = getCentroidAndOrientationImage(newLablet)
    test_p = getCentroidAndOrientationImage(newLableAll)

    centroidLabelDF_t = makeParameterDf(test_t)
    centroidLabelDF_p = makeParameterDf(test_p)

    # Add newLabel column
    centroidLabelDF_p['newLabel'] = 0
    centroidLabelDF_t['newLabel'] = 0

    # Assign new labels based on nearest centroid in projection
    for iii in range(len(centroidLabelDF_p)):
        dist = (centroidLabelDF_Projection['x'] - centroidLabelDF_p['x'][iii])**2 + \
               (centroidLabelDF_Projection['y'] - centroidLabelDF_p['y'][iii])**2
        newLabel = centroidLabelDF_Projection.iloc[np.argmin(dist)]['label']
        centroidLabelDF_p.iloc[iii, 9] = newLabel
        idxs = np.where(centroidLabelDF_t.iloc[:, 2] == centroidLabelDF_p['label'][iii])[0][0]
        centroidLabelDF_t.iloc[idxs, 9] = newLabel

    # Relabel mismatched nuclei
    mismatches = centroidLabelDF_t[centroidLabelDF_t.iloc[:, 2] != centroidLabelDF_t.iloc[:, 9]]
    mismatchesp = centroidLabelDF_p[centroidLabelDF_p.iloc[:, 2] != centroidLabelDF_p.iloc[:, 9]]

    for ff, ffp in zip(mismatches.index, mismatchesp.index):
        newLablet[np.where(newLablet == centroidLabelDF_t.iloc[ff, 2])] *= -1
        newLableAll[np.where(newLableAll == centroidLabelDF_t.iloc[ff, 2])] *= -1

    for ff, ffp in zip(mismatches.index, mismatchesp.index):
        newLablet[np.where(newLablet == centroidLabelDF_t.iloc[ff, 9])] = centroidLabelDF_t.iloc[ff, 9]
        newLableAll[np.where(newLableAll == centroidLabelDF_p.iloc[ffp, 2])] = centroidLabelDF_p.iloc[ffp, 9]

    # Get final nuclei list
    noNucleiFinal = np.unique(newLableAll)
    noNucleiFinal = np.delete(noNucleiFinal, np.where(noNucleiFinal == 0))

    # Corrected centroid data
    test_tCorr = getCentroidAndOrientationImage(newLablet)
    test_pCorr = getCentroidAndOrientationImage(newLableAll)

    centroidLabelDF_tCorr = makeParameterDf(test_tCorr)
    centroidLabelDF_pCorr = makeParameterDf(test_pCorr)

    return centroidLabelDF_tCorr, centroidLabelDF_pCorr, noNucleiFinal


def crop_nuclei_by_projection(
    imageFile, timep, noNucleiA, centroidLabelDF_Projection,
    moviePath, useTimeMaxProjection=False,
    extensionMov='.tif', centroidListTPoint=None
):
    for nuclei in noNucleiA:
        sizeList = pd.DataFrame(
            centroidLabelDF_Projection[['label', 'sizex', 'sizey']], dtype=np.int64
        )
        imageName = os.path.basename(os.path.normpath(moviePath))
        minr = centroidLabelDF_Projection.loc[centroidLabelDF_Projection['label'] == nuclei, 'minr'].values[0]
        minc = centroidLabelDF_Projection.loc[centroidLabelDF_Projection['label'] == nuclei, 'minc'].values[0]
        maxr = centroidLabelDF_Projection.loc[centroidLabelDF_Projection['label'] == nuclei, 'maxr'].values[0]
        maxc = centroidLabelDF_Projection.loc[centroidLabelDF_Projection['label'] == nuclei, 'maxc'].values[0]

        bx0 = (minc, maxc, maxc, minc, minc)
        by0 = (minr, minr, maxr, maxr, minr)

        cellTimeSeriesPath = os.path.join(moviePath, f'cell_{nuclei}')
        if not os.path.exists(cellTimeSeriesPath):
            os.makedirs(cellTimeSeriesPath)

        if useTimeMaxProjection:
            bx, by = np.array(bx0), np.array(by0)
        else:
            matchIdx = np.where(centroidListTPoint[:, 2] == nuclei)[0][0]
            bx, by = getCropEstimates(bx0, by0, centroidListTPoint[matchIdx][1], centroidListTPoint[matchIdx][0])
            bx, by = np.array(bx), np.array(by)

        bx[bx < 0] = 0
        bx[bx > 1024] = imageFile.shape[-1]
        by[by < 0] = 0
        by[by > 1024] = imageFile.shape[-1]
        bx, by = tuple(np.array(bx)), tuple(np.array(by))

        cellExt = f"_cell_{nuclei}_t{timep:03}{extensionMov}"
        sizex = sizeList[sizeList['label'] == nuclei].sizex.values[0]
        sizey = sizeList[sizeList['label'] == nuclei].sizey.values[0]

        croppedImage = imageFile[
            :, math.floor(by[0]):math.floor(by[0]) + sizex,
            math.floor(bx[0]):math.floor(bx[0]) + sizey
        ]
        cellFileName = os.path.join(cellTimeSeriesPath, imageName + cellExt)

        with tifffile.TiffWriter(cellFileName, imagej=True) as tif:
            tif.write(croppedImage)


def plot_nuclei_positions(
    timep, 
    newLableAll, 
    newLablet, 
    imsQ, 
    noNucleiFinal, 
    centroidLabelDF_tCorr, 
    centroidLabelDF_Projection, 
    homeFolder, 
    imageName
):
    """
    Plots nuclei positions overlayed on images with bounding boxes.
    
    Parameters:
    -----------
    timep : int
        Current time point.
    newLableAll : np.ndarray
        Background image.
    newLablet : np.ndarray
        Overlay image.
    imsQ : str
        Position identifier.
    noNucleiFinal : list
        List of nuclei labels to plot.
    centroidLabelDF_tCorr : pd.DataFrame
        DataFrame containing centroid coordinates at current time.
    centroidLabelDF_Projection : pd.DataFrame
        DataFrame containing projected centroid coordinates and bounding boxes.
    homeFolder : str
        Path to save the plot.
    imageName : str
        Name of the image (used in filename).
    """

    shouldIplot = timep == 0

    if shouldIplot:
        fig, ax = plt.subplots()
        ax.imshow(newLableAll, cmap=plt.cm.gray)
        ax.imshow(newLablet, alpha=0.3)
        ax.text(15, 50, f'position {imsQ}', color='white')

    for kk in noNucleiFinal:
        # Get current nucleus centroid
        yt = centroidLabelDF_tCorr.loc[centroidLabelDF_tCorr['label'] == kk, 'x'].values[0]
        xt = centroidLabelDF_tCorr.loc[centroidLabelDF_tCorr['label'] == kk, 'y'].values[0]
        nuc = kk

        # Get projected nucleus info
        proj = centroidLabelDF_Projection[centroidLabelDF_Projection['label'] == nuc]
        y00 = proj['x'].values[0]
        x00 = proj['y'].values[0]
        minr, minc = proj['minr'].values[0], proj['minc'].values[0]
        maxr, maxc = proj['maxr'].values[0], proj['maxc'].values[0]

        # Original bounding box
        bx0 = (minc, maxc, maxc, minc, minc)
        by0 = (minr, minr, maxr, maxr, minr)

        # Centered bounding box around current centroid
        maxccRef = np.max([abs(minr - maxr), abs(minc - maxc)])
        half_size = np.round(maxccRef) // 2
        nminc_ = xt - half_size
        nminr_ = yt - half_size
        nmaxc_ = xt + half_size
        nmaxr_ = yt + half_size

        bxt = (nminc_, nmaxc_, nmaxc_, nminc_, nminc_)
        byt = (nminr_, nminr_, nmaxr_, nmaxr_, nminr_)

        if shouldIplot:
            ax.plot(bx0, by0, '-b', linewidth=0.7)
            ax.plot(bxt, byt, '-w', linewidth=0.7)
            ax.text(x00, y00, str(nuc), color='white')
            ax.text(xt, yt, str(nuc), color='green')

    if shouldIplot:
        save_path = f"{homeFolder}/crops_{imageName.replace('.', '_')}.png"
        plt.savefig(save_path)
        plt.close(fig)  # Close the figure to free memory
        
        
def createMask(timep, movieFile, filename, liveCellModel):
    """
    Create a mask for a given timepoint and save to PNG.
    """
    filename = filename + str(f"_t{timep:03}")
    imageFile = movieFile[timep]
    imgs = np.max(imageFile, axis=0)
    masks, flows, styles = liveCellModel.eval(imgs, diameter=None, channels=[[0,0]])
    save_to_png(imgs, masks, flows, filename)
    return 1  # return completed frame index

    
def get_centroid_lists(noNucleiFinal, centroidLabelDF_tCorr, centroidLabelDF_Projection):
    """
    Build centroid lists for projection and frame t.

    Args:
        noNucleiFinal (ndarray): Final list of nucleus labels.
        centroidLabelDF_tCorr (DataFrame): Corrected nuclei DataFrame for frame t.
        centroidLabelDF_Projection (DataFrame): Projection DataFrame with centroids.

    Returns:
        centroidListProjection (ndarray): [y, x, label] for projection centroids.
        centroidListTPoint (ndarray): [y, x, label] for frame t centroids.
    """
    centroidListProjection = []
    centroidListTPoint = []

    for kk in noNucleiFinal:
        # extract corrected t-frame centroid
        y = centroidLabelDF_tCorr.loc[centroidLabelDF_tCorr['label'] == kk, 'x'].values[0]
        x = centroidLabelDF_tCorr.loc[centroidLabelDF_tCorr['label'] == kk, 'y'].values[0]
        nuc = centroidLabelDF_tCorr.loc[centroidLabelDF_tCorr['label'] == kk, 'label'].values[0]

        # match to projection centroid
        y00 = centroidLabelDF_Projection.loc[centroidLabelDF_Projection['label'] == nuc, 'x'].values[0]
        x00 = centroidLabelDF_Projection.loc[centroidLabelDF_Projection['label'] == nuc, 'y'].values[0]

        centroidListProjection.append([y00, x00, nuc])
        centroidListTPoint.append([y, x, nuc])

    return np.array(centroidListProjection), np.array(centroidListTPoint)


def process_frame_mask(
    tempmaskFolder,
    imageName,
    timep,
    new_Proj_label,
    centroidLabelDF_Projection,
    remove_edge_masks,
    match_nuc_cell,
    getCentroidAndOrientationImage,
    makeParameterDf,
):
    """
    Process mask for frame t, match nuclei to projection, relabel mismatches.

    Args:
        tempmaskFolder (str): Folder containing temporary masks.
        imageName (str): Name of the image file.
        timep (int): Frame index.
        new_Proj_label (ndarray): Projection label image.
        centroidLabelDF_Projection (DataFrame): Projection centroids & labels.
        remove_edge_masks (function): Function to remove edge masks.
        match_nuc_cell (function): Function to match nuclei with projection.
        getCentroidAndOrientationImage (function): Extract centroids/orientations.
        makeParameterDf (function): Convert centroid/orientation data to DataFrame.

    Returns:
        centroidLabelDF_tCorr (DataFrame): Corrected nuclei info for frame t.
        centroidLabelDF_pCorr (DataFrame): Corrected projection info.
        noNucleiFinal (ndarray): Final list of unique nucleus labels.
    """

        
    # Load mask path for frame t
    maskPathHome = os.path.join(tempmaskFolder, imageName.replace('.', '_')) + f"_t{timep:03}_cp_masks.png"
    if not os.path.isfile(maskPathHome):
        return None, None, None


    imageMaskt = imread(maskPathHome)
    labelImaget = label(imageMaskt)
    new_t_label = remove_edge_masks(labelImaget, change_index=True)


    ## Match  nucleus at frame t to projection.

    newLablet, newLableAll = match_nuc_cell(new_t_label, new_Proj_label, single_nuc=True, cell_alone=False)

    noNucleiAfterMatch = np.unique(newLableAll)
    noNucleiAfterMatch = np.delete(noNucleiAfterMatch,np.where(noNucleiAfterMatch == 0))

    if noNucleiAfterMatch.size == 0:
        return None, None, None

    # Compute centroids

    test_t = getCentroidAndOrientationImage(newLablet)
    test_p = getCentroidAndOrientationImage(newLableAll)

    centroidLabelDF_t = makeParameterDf(test_t)
    centroidLabelDF_p = makeParameterDf(test_p)

    # Add newLabel column
    centroidLabelDF_p['newLabel'] = 0
    centroidLabelDF_t['newLabel'] = 0

    # Assign new labels based on nearest centroid in projection
    for iii in range(len(centroidLabelDF_p)):
        dist = (centroidLabelDF_Projection['x']-centroidLabelDF_p['x'][iii])**2+(centroidLabelDF_Projection['y']-centroidLabelDF_p['y'][iii])**2
        newLabel = centroidLabelDF_Projection.iloc[np.argmin(dist)]['label']
        centroidLabelDF_p.iloc[iii,9] = newLabel
        idxs = np.where(centroidLabelDF_t.iloc[:,2]==centroidLabelDF_p['label'][iii])[0][0]
        centroidLabelDF_t.iloc[idxs,9] = newLabel

    #---------------------- Find Mismatch and Relabel -------------------------#

    ## Relabel Mismatched Nuclei

    mismatches = centroidLabelDF_t[centroidLabelDF_t.iloc[:,2]!=centroidLabelDF_t.iloc[:,9]]
    mismatchesp = centroidLabelDF_p[centroidLabelDF_p.iloc[:,2]!=centroidLabelDF_p.iloc[:,9]]


    # Relabel mismatched nuclei

    for ff, ffp in zip(mismatches.index, mismatchesp.index):
        newLablet[np.where(newLablet == centroidLabelDF_t.iloc[ff,2])] *=-1
        newLableAll[np.where(newLableAll == centroidLabelDF_t.iloc[ff,2])] *=-1

    # assign correct labels

    for ff, ffp in zip(mismatches.index, mismatchesp.index):
        newLablet[np.where(newLablet == centroidLabelDF_t.iloc[ff,2]*-1)] = centroidLabelDF_t.iloc[ff,9]
        newLableAll[np.where(newLableAll == centroidLabelDF_p.iloc[ffp,2]*-1)] = centroidLabelDF_p.iloc[ffp,9]

    # Get final nuclei list
    noNucleiFinal = np.unique(newLableAll)
    noNucleiFinal = np.delete(noNucleiFinal,np.where(noNucleiFinal == 0))

    # Corrected centroid data
    test_tCorr = getCentroidAndOrientationImage(newLablet)
    test_pCorr = getCentroidAndOrientationImage(newLableAll)

    centroidLabelDF_tCorr = makeParameterDf(test_tCorr)
    centroidLabelDF_pCorr = makeParameterDf(test_pCorr)


    return centroidLabelDF_tCorr, centroidLabelDF_pCorr, noNucleiFinal

# ------------------- Worker Threads ------------------- #


class MaskCreationThread(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal()

    def __init__(self, liveCellModel, movieFile, filename, n_jobs=8):
        super().__init__()
        self.liveCellModel = liveCellModel
        self.movieFile = movieFile
        self.filename = filename
        self.n_jobs = n_jobs

    def run(self):
        total_frames = len(self.movieFile)
        self.completed = 0  # track progress
        
        # Run jobs in parallel, collect completed indices
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(createMask)(timep, self.movieFile, self.filename, self.liveCellModel)
            for timep in tqdm(range(total_frames))
        )

        # Update progress in order after completion
        for i, _ in enumerate(results, 1):
            self.completed = i
            print(self.completed)
            self.progress.emit(int(self.completed / total_frames * 100))

        self.finished.emit()

class MainProcessThread(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal()

    def __init__(self, sessionNames, homeFolder, useTimeMaxProjection):
        super().__init__()
        self.sessionNames = sessionNames
        self.homeFolder = homeFolder
        self.useTimeMaxProjection = useTimeMaxProjection

    def run(self):
        total_sessions = len(self.sessionNames)
        for s_idx, session in enumerate(self.sessionNames):
            moviePath = session
            imageName = os.path.basename(os.path.normpath(moviePath))
            maskFolder = os.path.join(self.homeFolder, 'tProjections')
            maskFilename = f"T_MAX_{imageName.replace('.', '_')}_cp_masks.png"
            maskpath = os.path.join(maskFolder, maskFilename)
            pathToTimeFrames = os.path.join(moviePath, '*.tif')
            movieFile = imr(pathToTimeFrames)

            for t, imageFile in enumerate(movieFile):
                # ------------------ Load T projection -------------------- #
                maskImageAll = imread(maskpath)
                labelImageAll = label(maskImageAll)
                new_Proj_label = remove_edge_masks(labelImageAll, change_index=True)
                new_Proj_label2 = deepcopy(new_Proj_label)
                noNucleiA = np.unique(new_Proj_label)
                noNucleiA = np.delete(noNucleiA, np.where(noNucleiA == 0))

                centroidListProjection = []
                coordsNuc = getCentroidAndOrientationImage(new_Proj_label)
                centroidLabelDF_Projection = makeParameterDf(coordsNuc)
                centroidLabelDF_Projection['sizex'] = np.max([centroidLabelDF_Projection['maxr'].apply(np.ceil)-centroidLabelDF_Projection['minr'].apply(np.floor),
                                                              centroidLabelDF_Projection['maxc'].apply(np.ceil)-centroidLabelDF_Projection['minc'].apply(np.floor)], axis=0)
                centroidLabelDF_Projection['sizey'] = centroidLabelDF_Projection['sizex']  # square crops

                if self.useTimeMaxProjection:
                    finalList = noNucleiA
                    centroidListTPoint = None
                else:
                    # ------------------ Process masks for each frame ------------------ #
                    tempmaskFolder = os.path.join(moviePath, 'tempMasks')
                    centroidLabelDF_tCorr, centroidLabelDF_pCorr, finalList = process_frame_mask(
                        tempmaskFolder,
                        imageName,
                        t,
                        new_Proj_label,
                        centroidLabelDF_Projection,
                        remove_edge_masks,
                        match_nuc_cell,
                        getCentroidAndOrientationImage,
                        makeParameterDf
                    )
                    if finalList is None:
                        continue  # Skip empty frames
                    # finalList = noNucleiFinal

                    # ------------------ Get centroid lists for cropping ------------------ #
                    centroidListProjection, centroidListTPoint = get_centroid_lists(finalList, centroidLabelDF_tCorr, centroidLabelDF_Projection)

                # ------------------ Crop nuclei by projection ------------------ #
                crop_nuclei_by_projection(
                    imageFile, t, finalList, centroidLabelDF_Projection, moviePath,
                    useTimeMaxProjection=self.useTimeMaxProjection,
                    extensionMov='.tif',
                    centroidListTPoint=centroidListTPoint
                )

                # ------------------ Update progress ------------------ #
                self.progress.emit(int(((s_idx + t/len(movieFile)) / total_sessions) * 100))

        self.finished.emit()



# ------------------- Main GUI ------------------- #

class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Cell Segmentation GUI")
        self.setGeometry(100, 100, 400, 300)
        
        self.liveCellModel = None
        self.homeFolder = None
        self.sessionNames = []
        self.identifier = 'order'
        self.useTimeMaxProjection = True
        self.n_jobs = 5  # default number of threads


        # Layout
        layout = QVBoxLayout()
        self.label = QLabel("Select an action:")
        layout.addWidget(self.label)

        self.btnLoadModel = QPushButton("Load liveCellModel")
        self.btnLoadModel.clicked.connect(self.load_liveCellModel)
        layout.addWidget(self.btnLoadModel)

        self.btnLoadSessions = QPushButton("Load Experiment Folder")
        self.btnLoadSessions.clicked.connect(self.load_experiment_folder)
        layout.addWidget(self.btnLoadSessions)

        self.btnCreateMasks = QPushButton("Create Masks")
        self.btnCreateMasks.clicked.connect(self.create_masks)
        layout.addWidget(self.btnCreateMasks)

        self.btnToggleProjection = QPushButton("Toggle Time Projection Mask (Currently: True)")
        self.btnToggleProjection.clicked.connect(self.toggle_projection)
        layout.addWidget(self.btnToggleProjection)

        self.btnRunProcess = QPushButton("Run Main Process")
        self.btnRunProcess.clicked.connect(self.run_main_process)
        layout.addWidget(self.btnRunProcess)

        self.progressBar = QProgressBar()
        self.progressBar.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.progressBar)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    # ------------------- Button Functions ------------------- #
    def load_liveCellModel(self):
        model_file, _ = QFileDialog.getOpenFileName(self, "Select Cellpose Model File")
        if model_file:
            self.liveCellModel = models.CellposeModel(pretrained_model=model_file)
            QMessageBox.information(self, "Model Loaded", f"Loaded model: {model_file}")

    def load_experiment_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Home Folder")
        if folder:
            self.homeFolder = folder
            identifier, ok = QInputDialog.getText(self, "Identifier", "Enter identifier for sessions:")
            if ok and identifier:
                self.identifier = identifier
            self.sessionNames = [os.path.join(self.homeFolder, i) for i in os.listdir(self.homeFolder)
                                 if self.identifier in i and os.path.isdir(os.path.join(self.homeFolder, i))]
            self.sessionNames.sort()
            QMessageBox.information(self, "Sessions Loaded", f"{len(self.sessionNames)} sessions found.")

    def create_masks(self):
        if not self.liveCellModel or not self.sessionNames:
            QMessageBox.warning(self, "Warning", "Load model and sessions first!")
            return
        
        # Ask user for number of threads
        n_jobs, ok = QInputDialog.getInt(
            self,
            "Set Number of Threads",
            "Enter number of parallel threads:",
            value=self.n_jobs,  # default
            min=1,
            max=64
        )
        
        if not ok:
            return  # user cancelled
        self.n_jobs = n_jobs  # save chosen value

        for session in self.sessionNames:
            moviePath = session
            imageName = os.path.basename(os.path.normpath(moviePath))
            pathToTimeFrames = os.path.join(moviePath, '*.tif')
            movieFile = imr(pathToTimeFrames)
            tempmaskFolder = os.path.join(moviePath, 'tempMasks')
            if not os.path.exists(tempmaskFolder):
                os.makedirs(tempmaskFolder)

            filename = os.path.join(tempmaskFolder, imageName.replace('.', '_'))
            self.maskThread = MaskCreationThread(self.liveCellModel, movieFile, filename, n_jobs=self.n_jobs)
            self.maskThread.progress.connect(self.progressBar.setValue)
            self.maskThread.finished.connect(lambda: QMessageBox.information(self, "Done", "Mask creation finished."))
            self.maskThread.start()

    def toggle_projection(self):
        self.useTimeMaxProjection = not self.useTimeMaxProjection
        self.btnToggleProjection.setText(f"Toggle Time Projection Mask (Currently: {self.useTimeMaxProjection})")

    def run_main_process(self):
        if not self.sessionNames or not self.homeFolder:
            QMessageBox.warning(self, "Warning", "Load sessions first!")
            return

        self.processThread = MainProcessThread(self.sessionNames, self.homeFolder, self.useTimeMaxProjection)
        self.processThread.progress.connect(self.progressBar.setValue)
        self.processThread.finished.connect(lambda: QMessageBox.information(self, "Done", "Main process finished."))
        self.processThread.start()


# ------------------- Run App ------------------- #

if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    window = App()
    window.show()
    sys.exit(app.exec_())
