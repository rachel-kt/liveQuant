#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 11:29:55 2025

@author: rachel
"""

import os



import numpy as np


from copy import deepcopy
from dask.array.image import imread as imr
import bigfish.stack as stack
from bigfish.multistack import match_nuc_cell

from cellpose import models
from cellpose.utils import remove_edge_masks
from cellpose.io import imread, save_to_png, masks_flows_to_seg

import tkinter as tk
from tkinter import filedialog

from cropFunctions import *
from tqdm import tqdm
from joblib import Parallel, delayed
import time

start_time = time.time()

os.chdir('/home/rachel/Documents/bigFishLive/segmentation/')

def choose_home_folder():
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    file_path = filedialog.askdirectory(initialdir= "/", title='Please select a directory')  # Open file dialog
    root.destroy()  # Close the tkinter window
    return file_path

def choose_file():
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    file_path = filedialog.askopenfilename(initialdir= "/", title='Please select a movie')  # Open file dialog
    root.destroy()  # Close the tkinter window
    return file_path

def createMask(timep, movieFile, filename):
    filename = filename+str(f"_t{timep:03}")
    imageFile = movieFile[timep]
    imgs = np.max(imageFile, axis=0)

    # channels = [[0,0]]
    masks, flows, styles = liveCellModel.eval(imgs, diameter=None, channels=[[0,0]])
    # masks_flows_to_seg(imgs, masks, flows, filename, 1, channels)
    save_to_png(imgs, masks, flows, filename)



liveCellModel = models.CellposeModel(pretrained_model='/home/rachel/Documents/bigFishLive/liveCellModel/HelaLiveCell_09')#/overnight/tProjections/models/vera_5/')

identifier = 'ACY'

homeFolder = choose_home_folder()

print("Chosen home folder:", homeFolder)
sessionNames = [os.path.join(homeFolder,i) for i in os.listdir(homeFolder) if identifier in i and os.path.isdir(os.path.join(homeFolder,i))]
sessionNames.sort()

i = sessionNames[0]

moviePath=i
imsQ = i.split('/')[-1].split('_F')[-1]
imageName = i.split('/')[-1]
movieExtension = '.tif'
maskFolder = homeFolder+'/tProjections/'
maskpath = maskFolder+'T_MAX_'+imageName.replace('.','_')+'_cp_masks.png'
maskImageAll = imread(maskpath)
pathToTimeFrames = moviePath+'/*.tif'

movieFile = imr(pathToTimeFrames)

tempmaskFolder = os.path.join(i, 'tempMasks')
os.makedirs(tempmaskFolder, exist_ok=True)
filename = os.path.join(tempmaskFolder,imageName.replace('.','_'))


Parallel(n_jobs=8)(delayed(createMask)(i,movieFile, filename) for i in tqdm(range(len(movieFile))))

print(time.time()-start_time)
