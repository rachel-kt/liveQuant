#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  9 12:31:41 2025

@author: rachel
"""


import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import os
import numpy as np
import tifffile
from aicsimageio import AICSImage
import bioformats  # make sure python-bioformats is installed
from imaris_ims_file_reader.ims import ims
from dask.array.image import imread as imr
import napari
from qtpy.QtWidgets import QMainWindow, QWidget, QHBoxLayout
# from your_module import ims   # uncomment and replace with your IMS loader


# ---------------------------
# Helper functions
# ---------------------------
def get_session_name():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(initialdir="/", title="Please select a movie")
    root.destroy()
    return file_path

def choose_home_folder():
    root = tk.Tk()
    root.withdraw()
    folder_path = filedialog.askdirectory(initialdir="/", title="Please select a folder")
    root.destroy()
    return folder_path


# ---------------------------
# Writers with progress update
# ---------------------------
def writeTimePoints(sessionName, total_frames=10, zstack_number=5,
                    progress_callback=None, offset=0):
    movieFileName = sessionName.split('.dv')[0].split('/')[-1]
    extension = '.tif'
    pathToSaveFrames = os.path.join(sessionName.split('.dv')[0])
    os.makedirs(pathToSaveFrames, exist_ok=True)

    for timePoint in range(total_frames):
        newimage = bioformats.load_image(sessionName, c=None, z=0, t=timePoint)
        sp = newimage.shape
        newimage = newimage.reshape(1, sp[0], sp[1])

        for z in range(1, zstack_number):
            image = bioformats.load_image(sessionName, c=None, z=z, t=timePoint)
            image = image.reshape(1, sp[0], sp[1])
            newimage = np.concatenate([newimage, image], axis=0).astype('uint16')

        with tifffile.TiffWriter(
            os.path.join(pathToSaveFrames, movieFileName + f"_t{timePoint:03}" + extension),
            imagej=True
        ) as tif:
            tif.write(newimage)

        if progress_callback:
            progress_callback(offset + timePoint + 1)


def writeTimePointsIMS(imagePath, progress_callback=None, offset=0):
    imsData = ims(imagePath)
    outfolder = os.path.splitext(imagePath)[0]
    os.makedirs(outfolder, exist_ok=True)

    total_frames = imsData.shape[0]
    for timePoint in range(total_frames):
        tFrame = imsData[timePoint, :, :, :, :]
        tFrameShape = tFrame.shape
        tFrame = tFrame.reshape(1, 1, *tFrameShape).astype('uint16')
        pathToSaveFrames = os.path.join(outfolder, os.path.basename(imagePath).split(".ims")[0])
        with tifffile.TiffWriter(pathToSaveFrames + f"_t{timePoint:03}.tif", imagej=True) as tif:
            tif.write(tFrame)

        if progress_callback:
            progress_callback(offset + timePoint + 1)


def writeTimePointsTIF(imagePath, progress_callback=None, offset=0):
    cells = AICSImage(imagePath)
    _, totalTimes, _, zstack, xdim, ydim = cells.shape
    outfolder = os.path.splitext(imagePath)[0]
    os.makedirs(outfolder, exist_ok=True)

    imsData = cells.dask_data[0][:, 0, :, :, :]
    for timePoint in range(totalTimes):
        tFrame = imsData[timePoint].compute()
        tFrameShape = tFrame.shape
        tFrame = tFrame.reshape(1, 1, *tFrameShape).astype('uint16')
        pathToSaveFrames = os.path.join(outfolder, os.path.basename(imagePath).split(".tif")[0])
        with tifffile.TiffWriter(pathToSaveFrames + f"_t{timePoint:03}.tif", imagej=True) as tif:
            tif.write(tFrame)

        if progress_callback:
            progress_callback(offset + timePoint + 1)


# ---------------------------
# Tkinter GUI
# ---------------------------
class App:
    def __init__(self, root):
        self.root = root
        self.sessionName = None
        self.movieFormat = None
        self.sessionNames = []  # for bulk
        self.totalSteps = 0

        self.btn_select = tk.Button(root, text="Select File", command=self.select_file)
        self.btn_select.pack(pady=5)

        self.btn_run = tk.Button(root, text="Run Processing", command=self.run_processing, state=tk.DISABLED)
        self.btn_run.pack(pady=5)

        self.btn_bulk = tk.Button(root, text="Bulk Processing", command=self.bulk_processing)
        self.btn_bulk.pack(pady=5)
        
        self.btn_view = tk.Button(root, text="View TIFF Sequence", command=self.view_tiff_sequence)
        self.btn_view.pack(pady=5)

        self.progress = ttk.Progressbar(root, orient="horizontal", length=300, mode="determinate")
        self.progress.pack(pady=10)

    def select_file(self):
        self.sessionName = get_session_name()
        if self.sessionName:
            self.movieFormat = os.path.splitext(self.sessionName)[1].lower()
            self.btn_run.config(state=tk.NORMAL)
            messagebox.showinfo("File Selected", f"Selected: {self.sessionName}")

    def update_progress(self, current):
        self.progress["value"] = current
        self.root.update_idletasks()

    def run_processing(self):
        if not self.sessionName:
            messagebox.showwarning("No file", "Please select a file first")
            return

        self.progress["value"] = 0
        if self.movieFormat == ".dv":
            writeTimePoints(self.sessionName, progress_callback=self.update_progress)
        elif self.movieFormat == ".ims":
            writeTimePointsIMS(self.sessionName, progress_callback=self.update_progress)
        elif self.movieFormat == ".tif":
            writeTimePointsTIF(self.sessionName, progress_callback=self.update_progress)
        else:
            messagebox.showerror("Error", f"Unsupported file type: {self.movieFormat}")
            return

        messagebox.showinfo("Done", "Processing completed!")

    def bulk_processing(self):
        homeFolder = choose_home_folder()
        if not homeFolder:
            return

        all_files = os.listdir(homeFolder)
        extensions = [".dv", ".ims", ".tif"]
        found_format = None
        for ext in extensions:
            if any(f.endswith(ext) for f in all_files):
                found_format = ext
                break

        if not found_format:
            messagebox.showerror("Error", "No supported files found in this folder")
            return

        self.movieFormat = found_format
        self.sessionNames = [
            os.path.join(homeFolder, f)
            for f in all_files if f.endswith(found_format) and os.path.isfile(os.path.join(homeFolder, f))
        ]

        # calculate total steps across all files
        self.totalSteps = 0
        for f in self.sessionNames:
            if found_format == ".dv":
                self.totalSteps += 10  # <-- replace with actual total_frames if available
            elif found_format == ".ims":
                self.totalSteps += ims(f).shape[0]
            elif found_format == ".tif":
                self.totalSteps += AICSImage(f).shape[1]

        self.progress["maximum"] = self.totalSteps
        self.progress["value"] = 0
        current_step = 0

        for f in self.sessionNames:
            try:
                if self.movieFormat == ".dv":
                    writeTimePoints(f, progress_callback=self.update_progress, offset=current_step)
                    current_step += 10
                elif self.movieFormat == ".ims":
                    frames = ims(f).shape[0]
                    writeTimePointsIMS(f, progress_callback=self.update_progress, offset=current_step)
                    current_step += frames
                elif self.movieFormat == ".tif":
                    frames = AICSImage(f).shape[1]
                    writeTimePointsTIF(f, progress_callback=self.update_progress, offset=current_step)
                    current_step += frames
            except (KeyError, OSError):
                continue

        messagebox.showinfo("Done", "Bulk processing completed!")
        
    
    def view_tiff_sequence(self):
        homeFolder0 = choose_home_folder()
        if not homeFolder0:
            return

        # sessionName = os.path.basename(homeFolder0)
        pathToTimeFrames = os.path.join(homeFolder0, "*.tif")
        print("Loading TIFF stack:", pathToTimeFrames)

        # movieName = os.path.basename(homeFolder0)
        timeStack = imr(pathToTimeFrames)  # lazy load with dask
        print('image in memory')
        maxImage = np.max(timeStack, axis=1)
        
        print('max projection obtained')

        class MultiViewer(QMainWindow):
            def __init__(self, timeStack, maxImage, label_image=None):
                super().__init__()
                self.setWindowTitle("TIFF Sequence + Max Projection")

                central_widget = QWidget()
                layout = QHBoxLayout()
                central_widget.setLayout(layout)
                self.setCentralWidget(central_widget)

                # Viewer 1: timeStack
                self.viewer1 = napari.Viewer(show=False)
                ts_layer = self.viewer1.add_image(timeStack, colormap="green", name="TIFF Sequence")
                # if label_image is not None:
                #     self.viewer1.add_labels(label_image, name="Mask")

                # Viewer 2: max projection
                self.viewer2 = napari.Viewer(show=False)
                mp_layer = self.viewer2.add_image(maxImage, colormap="magenta", name="Max Projection")

                # Embed viewers into same window
                layout.addWidget(self.viewer1.window._qt_window)
                layout.addWidget(self.viewer2.window._qt_window)

                # ---------------------------
                # Synchronize time scrolling using current_step events
                # ---------------------------
                self._syncing = False  # prevent recursive update

                def sync_viewer1(event):
                    if not self._syncing:
                        self._syncing = True
                        current_coords = event.value
                        t_1 = np.array(current_coords)[0]
                        current_coords_2 = np.array(self.viewer2.dims.current_step)
                        current_coords_2[0] = t_1
                        self.viewer2.dims.current_step = tuple(current_coords_2)
                        self._syncing = False

                def sync_viewer2(event):
                    if not self._syncing:
                        self._syncing = True
                        current_coords = event.value
                        t_2 = np.array(current_coords)[0]
                        current_coords_1 = np.array(self.viewer1.dims.current_step)
                        current_coords_1[0] = t_2
                        self.viewer1.dims.current_step = tuple(current_coords_1)
                        self._syncing = False

                self.viewer1.dims.events.current_step.connect(sync_viewer1)
                self.viewer2.dims.events.current_step.connect(sync_viewer2)

        # Launch both viewers
        multi = MultiViewer(timeStack, maxImage)
        multi.show()
        napari.run()



if __name__ == "__main__":
    root = tk.Tk()
    root.title("Movie Frame Extractor")
    app = App(root)
    root.mainloop()
