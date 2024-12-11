#%%
import nd2reader
import matplotlib.pyplot as plt 
import numpy as np 
import cv2
import numpy as np
from nd2reader import ND2Reader
from matplotlib.colors import Normalize
from matplotlib import cm 
from PIL import Image, ImageEnhance
from skimage import (
    filters, measure, morphology, segmentation
)
from skimage.data import protein_transport
from scipy import ndimage as ndi
import pandas as pd 
import seaborn as sns 
#%%
path = "/Users/nwong/Documents/s_elongatus/images/awakenings/"
img1 = "/Users/nwong/Documents/s_elongatus/images/awakenings/healthy_001.nd2"
img2 = "/Users/nwong/Documents/s_elongatus/images/awakenings/starved_001.nd2"


# %%


def get_frames(image_path):
    frames = []
    with ND2Reader(image_path) as images:
        for image in images:
            frames.append(image)
    return frames[::-1] 

def plot_slide(frames):
    fig,axs = plt.subplots(ncols=len(frames), dpi=800)
    for i,frame in enumerate(frames):
        axs[i].imshow(frame, cmap='magma')
        axs[i].set_xticks([])
        axs[i].set_yticks([])
    fig.tight_layout()

def make_segmentation(img_array):
    # SEGMENT
    smooth = filters.gaussian(img_array, sigma=1)
    
    thresh_value = np.percentile(smooth, 99.5)  # Use a percentile
    thresh_value = filters.threshold_otsu(smooth)
    thresh = smooth > thresh_value

    #from skimage.filters import threshold_local
    #block_size = 11  # Adjust based on image size
    #adaptive_thresh = threshold_local(smooth, block_size)
    #thresh = smooth > adaptive_thresh

    fill = ndi.binary_fill_holes(thresh)
    clear = segmentation.clear_border(fill)
    dilate = morphology.binary_dilation(clear)

    erode = morphology.binary_erosion(clear)
    mask = np.logical_and(dilate, ~erode)
    return mask 

def flr_channel(frame):
    flr = filters.gaussian(np.asarray(frame), sigma=1)
    #thresh_value = np.percentile(flr, 99.5)
    #flr = flr > thresh_value
    return flr 

def phase_channel(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BAYER_BG2GRAY)
    ret,thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    foreground = cv2.erode(thresh, None, iterations=2)
    bgt = cv2.dilate(thresh, None, iterations=3)
    ret,background = cv2.threshold(bgt,1,128,1)
    marker = cv2.add(foreground, background)

    marker32 = np.uint8(marker)
    # Set unknown regions to -1 (watershed requires unknown regions to be marked as -1)
    marker32[background == 0] = -1
    marker32[foreground > 0] = 1  # Mark the foreground with 1

    phase = marker32.copy()
    return phase 

def apply_mask(mask, channel):
    return np.where(mask, channel, 0)


def channel_mean_intensity(channel_img, mask):
    # Ensure mask has distinct labels for each segmented region
    labeled_mask, _ = ndi.label(mask)

    props = measure.regionprops_table(
        labeled_mask,
        intensity_image=channel_img,
        properties=('label', 'area', 'intensity_mean')
    )
    
    # Convert the dictionary to a Pandas DataFrame
    props_df = pd.DataFrame(props)
    return props_df

def calc_single_cell_flr(frames):
    
    phase = phase_channel(frames[0])
    flr = flr_channel(frames[1])

    mask = make_segmentation(phase)

    props_c = channel_mean_intensity(flr, mask)
    
    return phase, flr, mask, props_c


if __name__ == "__main__":
    frames1 = get_frames(img1)
    frames2 = get_frames(img2)
    phase1, flr1, mask1, props_c1 = calc_single_cell_flr(frames1)
    phase2, flr2, mask2, props_c2 = calc_single_cell_flr(frames2)

    fig,axs=plt.subplots(nrows=2, ncols=3, dpi=800)


    axs[0,0].imshow(phase1, cmap='gray')
    axs[0,0].set_title('Phase', fontsize=15)
    axs[0,1].imshow(mask1, cmap='binary')
    axs[0,1].set_title('Segmentation', fontsize=15)
    axs[0,2].imshow(flr1, cmap='inferno')
    axs[0,2].set_title('Cy5', fontsize=15)

    axs[1,0].imshow(phase2, cmap='gray')
    axs[1,1].imshow(mask2, cmap='binary')
    axs[1,2].imshow(flr2, cmap='inferno')
    axs[0,0].set_ylabel("Healthy", fontsize=15)
    axs[1,0].set_ylabel("Chlorotic", fontsize=15)
    fig.tight_layout()



    for i,a in enumerate(axs.ravel()):
        if i!=3 or i!=6:
            a.set_xticks([])
            a.set_yticks([])
    fig.tight_layout()


    fig,axs = plt.subplots(sharey=True, dpi=800)
    props_c1['treatment'] = ["Healthy"]*len(props_c1)
    props_c2['treatment'] = ["Chlorotic"]*len(props_c2)
    df = pd.concat((props_c1, props_c2))
    sns.swarmplot(df,
                y="intensity_mean",
                x="treatment",
                ax=axs,
                size=6,
                color="gray",
                edgecolor='black',
                linewidth=1
                )
    axs.grid(axis='y')

    axs.set_title('Healthy', fontsize=15)
    axs.set_title("Chlorophyll Per Cell", fontsize=20)
    axs.set_ylabel("Flr Intensity", fontsize=15)
    axs.set_xlabel("Treatment", fontsize=15)
    fig.tight_layout(w_pad=0)
    # %%

    # %%
