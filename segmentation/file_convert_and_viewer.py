import os
import numpy as np
import tifffile
from aicsimageio import AICSImage
import bioformats  # make sure python-bioformats is installed
from imaris_ims_file_reader.ims import ims
from dask.array.image import imread as imr
import napari
from qtpy.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QProgressBar, QFileDialog, QMessageBox
)
from qtpy.QtCore import Qt

# ---------------------------
# Writers with progress update
# ---------------------------
# (Keep your existing writeTimePoints, writeTimePointsIMS, writeTimePointsTIF functions here)
# ...

# ---------------------------
# QtPy GUI
# ---------------------------
class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Movie Frame Extractor")
        self.sessionName = None
        self.movieFormat = None
        self.sessionNames = []  # for bulk
        self.totalSteps = 0

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout()
        central_widget.setLayout(layout)

        # Buttons
        self.btn_select = QPushButton("Select File")
        self.btn_select.clicked.connect(self.select_file)
        layout.addWidget(self.btn_select)

        self.btn_run = QPushButton("Run Processing")
        self.btn_run.clicked.connect(self.run_processing)
        self.btn_run.setEnabled(False)
        layout.addWidget(self.btn_run)

        self.btn_bulk = QPushButton("Bulk Processing")
        self.btn_bulk.clicked.connect(self.bulk_processing)
        layout.addWidget(self.btn_bulk)

        self.btn_view = QPushButton("View TIFF Sequence")
        self.btn_view.clicked.connect(self.view_tiff_sequence)
        layout.addWidget(self.btn_view)

        # Progress bar
        self.progress = QProgressBar()
        self.progress.setOrientation(Qt.Horizontal)
        layout.addWidget(self.progress)

    # ---------------------------
    # File selection
    # ---------------------------
    def select_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Please select a movie", "/", 
                                                   "All Supported Files (*.dv *.ims *.tif)")
        if file_path:
            self.sessionName = file_path
            self.movieFormat = os.path.splitext(file_path)[1].lower()
            self.btn_run.setEnabled(True)
            QMessageBox.information(self, "File Selected", f"Selected: {self.sessionName}")

    def update_progress(self, current):
        self.progress.setValue(current)
        QApplication.processEvents()  # Force GUI update
        
        
    # ---------------------------
    # Writers with progress update
    # ---------------------------
    def writeTimePoints(self,sessionName, total_frames=10, zstack_number=5,
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


    def writeTimePointsIMS(self,imagePath, progress_callback=None, offset=0):
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


    def writeTimePointsTIF(self, imagePath, progress_callback=None, offset=0):
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
    # Processing
    # ---------------------------
    def run_processing(self):
        if not self.sessionName:
            QMessageBox.warning(self, "No file", "Please select a file first")
            return

        self.progress.setValue(0)
        if self.movieFormat == ".dv":
            self.writeTimePoints(self.sessionName, progress_callback=self.update_progress)
        elif self.movieFormat == ".ims":
            self.writeTimePointsIMS(self.sessionName, progress_callback=self.update_progress)
        elif self.movieFormat == ".tif":
            self.writeTimePointsTIF(self.sessionName, progress_callback=self.update_progress)
        else:
            QMessageBox.critical(self, "Error", f"Unsupported file type: {self.movieFormat}")
            return

        QMessageBox.information(self, "Done", "Processing completed!")

    def bulk_processing(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Please select a folder", "/")
        if not folder_path:
            return

        all_files = os.listdir(folder_path)
        extensions = [".dv", ".ims", ".tif"]
        found_format = next((ext for ext in extensions if any(f.endswith(ext) for f in all_files)), None)

        if not found_format:
            QMessageBox.critical(self, "Error", "No supported files found in this folder")
            return

        self.movieFormat = found_format
        self.sessionNames = [
            os.path.join(folder_path, f)
            for f in all_files if f.endswith(found_format) and os.path.isfile(os.path.join(folder_path, f))
        ]

        # calculate total steps across all files
        self.totalSteps = 0
        for f in self.sessionNames:
            if found_format == ".dv":
                self.totalSteps += 10
            elif found_format == ".ims":
                self.totalSteps += ims(f).shape[0]
            elif found_format == ".tif":
                self.totalSteps += AICSImage(f).shape[1]

        self.progress.setMaximum(self.totalSteps)
        self.progress.setValue(0)
        current_step = 0

        for f in self.sessionNames:
            try:
                if self.movieFormat == ".dv":
                    self.writeTimePoints(f, progress_callback=self.update_progress, offset=current_step)
                    current_step += 10
                elif self.movieFormat == ".ims":
                    frames = ims(f).shape[0]
                    self.writeTimePointsIMS(f, progress_callback=self.update_progress, offset=current_step)
                    current_step += frames
                elif self.movieFormat == ".tif":
                    frames = AICSImage(f).shape[1]
                    self.writeTimePointsTIF(f, progress_callback=self.update_progress, offset=current_step)
                    current_step += frames
            except (KeyError, OSError):
                continue

        QMessageBox.information(self, "Done", "Bulk processing completed!")

    # ---------------------------
    # View TIFF sequence
    # ---------------------------
    def view_tiff_sequence(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Please select a folder", "/")
        if not folder_path:
            return

        pathToTimeFrames = os.path.join(folder_path, "*.tif")
        timeStack = imr(pathToTimeFrames)
        maxImage = np.max(timeStack, axis=1)

        class MultiViewer(QMainWindow):
            def __init__(self, timeStack, maxImage):
                super().__init__()
                self.setWindowTitle("TIFF Sequence + Max Projection")

                central_widget = QWidget()
                layout = QHBoxLayout()
                central_widget.setLayout(layout)
                self.setCentralWidget(central_widget)

                # Viewer 1
                self.viewer1 = napari.Viewer(show=False)
                self.viewer1.add_image(timeStack, colormap="green", name="TIFF Sequence")

                # Viewer 2
                self.viewer2 = napari.Viewer(show=False)
                self.viewer2.add_image(maxImage, colormap="magenta", name="Max Projection")

                layout.addWidget(self.viewer1.window._qt_window)
                layout.addWidget(self.viewer2.window._qt_window)

                self._syncing = False

                def sync_viewer1(event):
                    if not self._syncing:
                        self._syncing = True
                        t_1 = event.value[0]
                        coords2 = np.array(self.viewer2.dims.current_step)
                        coords2[0] = t_1
                        self.viewer2.dims.current_step = tuple(coords2)
                        self._syncing = False

                def sync_viewer2(event):
                    if not self._syncing:
                        self._syncing = True
                        t_2 = event.value[0]
                        coords1 = np.array(self.viewer1.dims.current_step)
                        coords1[0] = t_2
                        self.viewer1.dims.current_step = tuple(coords1)
                        self._syncing = False

                self.viewer1.dims.events.current_step.connect(sync_viewer1)
                self.viewer2.dims.events.current_step.connect(sync_viewer2)

        multi = MultiViewer(timeStack, maxImage)
        multi.show()
        napari.run()


if __name__ == "__main__":
    app_qt = QApplication([])
    window = App()
    window.show()
    app_qt.exec_()
