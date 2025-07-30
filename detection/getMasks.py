from cellpose.io import imread, save_to_png, masks_flows_to_seg

def getMasks(data):
    print('Working on timepoint: ',time)
    nimg = len(imgs)
    channels = [[0,0]]
    movieName='max_'+str(f"{time:03}")
    # Segmentation

    masks, flows, styles = liveCellModel.eval(imgs, diameter=None, channels=channels)

    # Save segmentation Results

    masks_flows_to_seg(imgs, masks, flows,pathToTProjections+'T_MAX_'+movieName.replace('.','_'),  1, channels)
    save_to_png(imgs, masks, flows, pathToTProjections+'T_MAX_'+movieName.replace('.','_'))