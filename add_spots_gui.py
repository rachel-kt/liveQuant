#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 11 14:17:15 2025

@author: rachel
"""

import sys
import os
from copy import deepcopy
from pathlib import Path
import numpy as np
from qtpy.QtWidgets import QApplication, QMainWindow, QWidget, QHBoxLayout, QPushButton, QVBoxLayout, QFileDialog
import napari
from magicgui import magicgui
from dask.array.image import imread as imr

# global settings
VOXEL_RADIUS = (0, 0, 0)
OBJECT_RADIUS = (0, 0, 0)
DEFAULT_CHOICES = ['None']
SET_TABLE = 0
CELL_CHOICE = 0
DEBUG = False
mIdentifier = 'cell'  # example identifier to filter folders
MaxTimePoint = 1       # minimal number of files in folder to include
recorded_points = []
recorded_viewer2 =[]
class MultiViewer(QMainWindow):
    
    def update_cursor(self, layer, event):
        """Update cursor positions in both viewers, handling 4D -> 3D."""
        pos = layer.world_to_data(event.position)
        pos_int = tuple(np.round(pos).astype(int))
    
        # Viewer1: full 4D point
        self.cursor1.data = [pos_int]
    
        # Viewer2: drop Z dimension (axis=1)
        if len(pos_int) == 4:
            self.cursor2.data = [(pos_int[0], pos_int[2], pos_int[3])]  # t, y, x
        else:
            self.cursor2.data = [pos_int]

        
    def add_sync_points_layer(self, viewer1, viewer2):
        # Add an empty points layer for cursor in both viewers
        cursor1 = self.viewer1.add_points(
            [], size=10, face_color='transparent', edge_color='red', name='cursor_4D'
        )
        cursor2 = self.viewer2.add_points(
            [], size=10, face_color='transparent', edge_color='red', name='cursor_3D'
        )
        return cursor1, cursor2
    

    def move_z_down(self, viewer1):
        """Move one step down along the z-axis."""
        z_index = self.viewer1.dims.current_step[1]
        # Decrement, but not below 0
        if z_index > 0:
            self.viewer1.dims.current_step = (self.viewer1.dims.current_step[0], z_index - 1, self.viewer1.dims.current_step[2], self.viewer1.dims.current_step[3])


    def move_z_up(self, viewer1):
        """Move one step up along the z-axis."""
        # Get current z index
        z_index = self.viewer1.dims.current_step[1]
        # Increment, but don't exceed max
        if z_index < self.viewer1.layers[0].data.shape[1] - 1:
            self.viewer1.dims.current_step = (self.viewer1.dims.current_step[0], z_index + 1, self.viewer1.dims.current_step[2], self.viewer1.dims.current_step[3])

    def select_previous_layer(self, viewer1):
        """Select the previous layer in layers_to_cycle."""
        layers_to_cycle = [self.viewer1.layers[n].name for n in np.arange(len(self.viewer1.layers))]
        current_layer = list(self.viewer1.layers.selection)[0].name
        if current_layer in layers_to_cycle:
            idx = layers_to_cycle.index(current_layer)
            new_idx = (idx + 1) % len(layers_to_cycle)
            self.viewer1.layers.selection.clear()
            self.viewer1.layers.selection.add(self.viewer1.layers[new_idx])
        else:
            # If current layer is not in the list, select the first
            self.viewer1.active_layer = layers_to_cycle[0]
            
    def select_next_layer(self, viewer1):
        """Select the previous layer in layers_to_cycle."""
        layers_to_cycle = [self.viewer1.layers[n].name for n in np.arange(len(self.viewer1.layers))]
        current_layer = list(self.viewer1.layers.selection)[0].name
        if current_layer in layers_to_cycle:
            idx = layers_to_cycle.index(current_layer)
            new_idx = (idx - 1) % len(layers_to_cycle)
            self.viewer1.layers.selection.clear()
            self.viewer1.layers.selection.add(self.viewer1.layers[new_idx])
        else:
            # If current layer is not in the list, select the first
            self.viewer1.active_layer = layers_to_cycle[0]
    
    def add_point_on_click(self,layer, event):
        global recorded_points, recorded_viewer2
        if event.type == 'mouse_press':
            # Get coordinates in data space
            coords = layer.world_to_data(event.position)
            coords = np.round(coords).astype(int)

            # Extract current t and z from the viewer
            t_index = self.viewer1.dims.current_step[0]  # time dimension
            z_index = self.viewer1.dims.current_step[1]  # z dimension
            y, x = coords[2], coords[3]  # image coordinates
            
            # Record the point
            new_point = [t_index, z_index, y, x]
            mipPoints = [t_index, y, x]
            
            recorded_points.append(new_point)

            recorded_viewer2.append(mipPoints)
            # print(layer)
            
            if 'spots' in self.viewer1.layers:
                self.viewer1.layers.remove('spots')
            # points_layer.add([new_point])
            # Add point with (t, z, y, x) coordinates
            points_layer = self.viewer1.add_points(
                recorded_points, name="spots", size=10, face_color="transparent", edge_color="red"
            )
            self.viewer1.layers.selection.clear()
            self.viewer1.layers.selection.add(self.viewer1.layers[0])
            
            
            if 'spots' in self.viewer2.layers:
                self.viewer2.layers.remove('spots')
            # points_layer.add([new_point])
            # Add point with (t, z, y, x) coordinates
            points_layer2 = self.viewer2.add_points(
                recorded_viewer2, name="spots", size=10, face_color="transparent", edge_color="red"
            )
            self.viewer2.layers.selection.clear()
            self.viewer2.layers.selection.add(self.viewer2.layers[0])
            
            print(f"Added point at (t,z,y,x): {(t_index, z_index, y, x)}")
    
    
    def save_spots_as_npz(self):
        """Save the spots layer as an .npz file."""
        spots_layer = self.viewer1.layers["spots"]
        
        if spots_layer is not None:
            points = np.array(spots_layer.data)  # Get points from the 'spots' layer
            file_name, _ = QFileDialog.getSaveFileName(self, 'Save Spots', '', 'NPZ files (*.npz)')
            if file_name:
                np.savez(file_name+'_spots', points=points, allow_pickle=True)
                print(f"Saved {len(points)} points to {file_name}")
        else:
            print("No 'spots' layer found!")

    def __init__(self, timeStack, maxImage):
        super().__init__()
        self.setWindowTitle("Spot Addition Interface")

        # central widget with vertical layout
        central_widget = QWidget()
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        # layout for viewers
        viewer_layout = QHBoxLayout()
        main_layout.addLayout(viewer_layout)

        # Viewer 1
        self.viewer1 = napari.Viewer(show=False)
        self.viewer1.add_image(timeStack, colormap="green", name="TIFF Sequence")

        # Viewer 2
        self.viewer2 = napari.Viewer(show=False)
        self.viewer2.add_image(maxImage, colormap="magenta", name="Max Projection")

        # add viewers to layout
        viewer_layout.addWidget(self.viewer1.window._qt_window)
        viewer_layout.addWidget(self.viewer2.window._qt_window)

        # sync flag
        self._syncing = False
        self.viewer1.bind_key("Down", lambda event: self.move_z_down(self.viewer1))
        self.viewer1.bind_key("Up", lambda event: self.move_z_up(self.viewer1))
        self.viewer1.bind_key("PageUp", lambda event: self.select_previous_layer(self.viewer1))
        self.viewer1.bind_key("PageDown", lambda event: self.select_next_layer(self.viewer1))
        
        self.viewer1.layers.selection.clear()
        self.viewer1.layers.selection.add(self.viewer1.layers[0])

        # sync callbacks
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

        # setup magicgui for folder selection
        self.setup_folder_gui(main_layout)
        
        self.cursor1, self.cursor2 = self.add_sync_points_layer(self.viewer1, self.viewer2)
        self.viewer1.layers[1].mouse_move_callbacks.append(self.update_cursor)
        self.viewer1.layers[0].mouse_drag_callbacks.append(self.add_point_on_click)
        
        # Add Save button for saving the spots layer
        save_button = QPushButton("Save Spots as NPZ")
        save_button.clicked.connect(self.save_spots_as_npz)
        main_layout.addWidget(save_button)

    def setup_folder_gui(self, parent_layout):
        """MagicGUI widget to choose base folder and select cell subfolder."""
        global DEFAULT_CHOICES

        @magicgui(
            auto_call=True,
            main_window=False,
            persist=False,
            base_folder={"label": "Choose home folder:", 'mode': 'd'},
            dropdown=dict(widget_type="Select", choices=DEFAULT_CHOICES, label="Choose cell to analyse"),
            layout="vertical",
        )
        def choose_home_folder(dropdown, base_folder=Path.home()):
            if DEBUG:
                print("calling choose home folder")
                print("dropdown:", dropdown)
                print("base_folder:", base_folder)
            return base_folder, dropdown

        # connect the change event to populate dropdown
        @choose_home_folder.changed.connect
        def choose_cell_on_file_change(event=None):
            global SET_TABLE, CELL_CHOICE
            if DEBUG:
                print("calling populate cell choices")
            if choose_home_folder.call_count == 1 and SET_TABLE == 0:
                SET_TABLE = 1
            hm = Path(choose_home_folder.base_folder.value)
            if hm.exists() and hm.is_dir():
                cellChoices = [
                    i for i in os.listdir(hm)
                    if mIdentifier in i and len(os.listdir(os.path.join(hm, i))) >= MaxTimePoint
                ]
                cellChoices.sort()
                if cellChoices:
                    choose_home_folder.dropdown.choices = cellChoices
                    DEFAULT_CHOICES[:] = deepcopy(cellChoices)
                    choose_home_folder.dropdown.value = cellChoices[CELL_CHOICE]

        # add the magicgui widget to the main layout
        parent_layout.addWidget(choose_home_folder.native)

        # create Load button
        load_button = QPushButton("Load Selected Cell")
        parent_layout.addWidget(load_button)
        
        
        def load_selected_cell():
            selected_folder = choose_home_folder.dropdown.value
            base_folder = Path(choose_home_folder.base_folder.value)
            if selected_folder and base_folder.exists():
                folder_path = base_folder / selected_folder[0]
                pathToTimeFrames = os.path.join(folder_path, "*.tif")
                timeStack = imr(pathToTimeFrames)
                maxImage = np.max(timeStack, axis=1)  # axis=0 for time axis
                # clear old layers and add new
                self.viewer1.layers.clear()
                self.viewer2.layers.clear()
                self.image_layer1 = self.viewer1.add_image(timeStack, colormap="green", name="TIFF Sequence")
                self.image_layer2 = self.viewer2.add_image(maxImage, colormap="magenta", name="Max Projection")
                self.cursor1, self.cursor2 = self.add_sync_points_layer(self.viewer1, self.viewer2)
                self.viewer1.layers[1].mouse_move_callbacks.append(self.update_cursor)
                # self.viewer1.bind_key('z', self.scroll_z_with_key, wheel=True)
                self.image_layer1.mouse_drag_callbacks.append(self.add_point_on_click)
                print(f"Loaded {selected_folder} into viewers.")
                

        load_button.clicked.connect(load_selected_cell)

        self.folder_gui = choose_home_folder
        

if __name__ == "__main__":
    # example initial data
    timeStack = np.random.rand(20,10, 128, 128)
    maxImage = timeStack.max(axis=1)

    app = QApplication(sys.argv)
    multi = MultiViewer(timeStack, maxImage)
    multi.show()
    napari.run()

