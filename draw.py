from tqdm import tqdm
import pandas as pd
import numpy as np
from data import Data
import matplotlib.pyplot as plt
from gptkalman import Kalman

chip_ids = [2, 4, 5, 6 ]
#chip_ids = [ 4, 5, 6, 8]
train_sheet_num = 15
#Arch = [128, 32, 8, 2, 1]
pcolors = ['b', 'g', 'm', 'r', 'y', 'p']
#pcolors = ['b', 'g', 'm', 'r', 'y']

mtitlesz = 60
stitlesz = 56
labelsz = 52
ticklabelsz = 36
linewidth = 4
ssz = 16


# draw paper figures
fig, axes = plt.subplots(1, 1, figsize = (80, 26))

labels = axes.get_xticklabels() + axes.get_yticklabels()
[label.set_fontsize(ticklabelsz) for label in labels]

for i, chip_id in enumerate(tqdm(chip_ids)):
    AgingData = Data(chip_id, train_sheet_num)
    RawDataIterative, RawAgingData = AgingData.get_iterative_data()
    AgingData = RawAgingData * 100
    FilterData = Kalman(chip_id, train_sheet_num)
    KalmanData = FilterData.fliter()
    draw_x = np.array(range(len(KalmanData)))
    axes.plot(draw_x, AgingData, color = "blue", label='mean data',linewidth = linewidth)
    axes.plot(draw_x, KalmanData, color = "green", label='Kalman filter data',linewidth = linewidth)
    axes.set_xlabel('Time', fontsize = labelsz, labelpad = 30)
    axes.set_ylabel('Aging Factors', fontsize = labelsz, labelpad = 30)
    plt.suptitle('filter effert', fontsize = mtitlesz, x = 0.5, y = 1.03)
    handles, labels = axes.get_legend_handles_labels()
    axes.legend(handles, labels, loc='upper left', bbox_to_anchor=(1, 1), fontsize=labelsz)
    plt.tight_layout()
    plt.savefig('../imgfilter/chip' + str(chip_id) + '.png')
    axes.cla()