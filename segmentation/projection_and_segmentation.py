#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  9 17:16:02 2025

@author: rachel
"""

import os
import numpy as np
import tifffile
import napari
import matplotlib.pyplot as plt
from qtpy.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton,
    QFileDialog, QMessageBox, QInputDialog
)
from cellpose import models
from cellpose.io import imread, save_to_png, masks_flows_to_seg, imsave
from dask.array.image import imread as imr
from magicgui import magicgui, magic_factory, widgets

class CellposeQtApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Cellpose Time Projection + Segmentation")
        self.model_path = None
        self.liveCellModel = None
        self.tiff_folder = None
        self.pathToTProjections = None

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout()
        central_widget.setLayout(layout)

        # Buttons
        self.btn_load_model = QPushButton("Load Cellpose Model")
        self.btn_load_model.clicked.connect(self.load_model)
        layout.addWidget(self.btn_load_model)

        self.btn_run_segmentation = QPushButton("Run Segmentation on TIFF Folder")
        self.btn_run_segmentation.clicked.connect(self.run_segmentation)
        self.btn_run_segmentation.setEnabled(False)
        layout.addWidget(self.btn_run_segmentation)

        self.btn_view_napari = QPushButton("View Results in Napari")
        self.btn_view_napari.clicked.connect(self.view_results)
        self.btn_view_napari.setEnabled(True)
        layout.addWidget(self.btn_view_napari)

    def load_model(self):
        model_path, _ = QFileDialog.getOpenFileName(self, "Select Cellpose pretrained model")
        if model_path:
            self.model_path = model_path
            self.liveCellModel = models.CellposeModel(pretrained_model=self.model_path)
            QMessageBox.information(self, "Model Loaded", f"Loaded Cellpose model:\n{self.model_path}")
            self.btn_run_segmentation.setEnabled(True)

    def run_segmentation(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select folder with TIFF sequences")
        if not folder_path:
            return
        self.tiff_folder = folder_path
        self.pathToTProjections = os.path.join(self.tiff_folder, 'tProjections')
        os.makedirs(self.pathToTProjections, exist_ok=True)

        identifier, ok = QInputDialog.getText(self, "Identifier", "Enter identifier for movie folders:")
        if not ok:
            return

        homeFolder = [os.path.join(self.tiff_folder, i) for i in os.listdir(self.tiff_folder)
                      if identifier in i and os.path.isdir(os.path.join(self.tiff_folder,i))]
        homeFolder.sort()

        for sessionName in homeFolder:
            pathToTimeFrames = os.path.join(sessionName, '*.tif')
            movieName = os.path.basename(sessionName)
            timeStack = imr(pathToTimeFrames)
            maxImage = np.max(timeStack, axis=1)
            TimeProjection = np.max(maxImage, axis=0)

            # Save time projection
            time_proj_file = os.path.join(self.pathToTProjections, f'T_MAX_{movieName.replace(".", "_")}.tif')
            with tifffile.TiffWriter(time_proj_file, imagej=True) as tif:
                tif.write(TimeProjection)

            # Segmentation using Cellpose
            imgs = imread(time_proj_file)
            channels = [[0, 0]]
            masks, flows, styles = self.liveCellModel.eval(imgs, diameter=None, channels=channels)

            # Save segmentation results
            base_name = os.path.join(self.pathToTProjections, f'T_MAX_{movieName.replace(".", "_")}')
            masks_flows_to_seg(imgs, masks, flows, base_name, 1, channels)
            save_to_png(imgs, masks, flows, base_name)

        QMessageBox.information(self, "Segmentation Done", "Time projections and segmentation labels saved!")
        self.btn_view_napari.setEnabled(True)

    def view_results(self):
        from magicgui import magicgui
        from cellpose.io import imsave
    
        # Select folder with TIFF sequences
        folder_path = QFileDialog.getExistingDirectory(self, "Select folder with TIFF sequences")
        if not folder_path:
            return
    
        # Automatically locate the tProjections folder
        pathToTProjections = os.path.join(os.path.dirname(folder_path), 'tProjections')
        print(pathToTProjections)
        if not os.path.exists(pathToTProjections):
            QMessageBox.warning(self, "No tProjections", f"No tProjections folder found in {folder_path}")
            return
    
        movieName = os.path.basename(folder_path)
        pathToTimeFrames = os.path.join(folder_path, '*.tif')
        timeStack = imr(pathToTimeFrames)
        maxImage = np.max(timeStack, axis=1)
    
        # Load the segmentation label image if it exists
        label_image_name = os.path.join(pathToTProjections, f'T_MAX_{movieName.replace(".", "_")}_cp_masks.png')
        label_image = None
        if os.path.exists(label_image_name):
            label_image = imread(label_image_name)
        else:
            print(f"No segmentation label image found: {label_image_name}")
    
        viewer = napari.Viewer()
        viewer.add_image(maxImage, colormap='green')
        if label_image is not None:
            viewer.add_labels(label_image, name='segmentation', opacity=0.3)
    
        # ---------------------------
        # MagicGUI button to save current segmentation
        # ---------------------------
        @magicgui(call_button="Save Current Segmentation")
        def save_mask():
            if 'segmentation' not in viewer.layers:
                QMessageBox.warning(None, "No Label Layer", "No 'segmentation' layer found to save!")
                return
            mask_data = viewer.layers['segmentation'].data
            os.makedirs(os.path.dirname(label_image_name), exist_ok=True)
            imsave(label_image_name, mask_data)
            QMessageBox.information(None, "Saved", f"Segmentation saved to:\n{label_image_name}")
            print(f"Segmentation saved to {label_image_name}")
    
        viewer.window.add_dock_widget(save_mask, area='right')
        napari.run()
    


if __name__ == "__main__":
    app = QApplication([])
    window = CellposeQtApp()
    window.show()
    app.exec_()
