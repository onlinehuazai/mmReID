import numpy as np
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import cv2

def plot_CFAR(RDM_dB, RDM_mask):
    for i in range(RDM_dB.shape[0]):
        for j in range(RDM_dB.shape[1]):
            if RDM_mask[i, j] == 0:
                RDM_dB[i, j] = -100

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(RDM_dB, cmap='coolwarm', interpolation='nearest',
                   extent=[-len(RDM_dB[0]) // 2, len(RDM_dB[0]) // 2, len(RDM_dB), 0])

    plt.savefig('CFAR.png', pad_inches=0, dpi=200, bbox_inches='tight')
    plt.show()


def clutterRemoval(input_val, axis=1):
    """Perform basic static clutter removal by removing the mean from the input_val on the specified doppler axis.
    Args:
        input_val (ndarray): Array to perform static clutter removal on. Usually applied before performing doppler FFT.
            e.g. [num_chirps, num_vx_antennas, num_samples], it is applied along the first axis.
        axis (int): Axis to calculate mean of pre-doppler.
    Returns:
        ndarray: Array with static clutter removed.
    """
    # 对轴进行重排序，静态杂波去除要在chirp维度上进行
    reordering = np.arange(len(input_val.shape))
    reordering[0] = axis
    reordering[axis] = 0
    input_val = input_val.transpose(reordering)

    # Apply static clutter removal
    mean = input_val.mean(0)
    output_val = input_val - np.expand_dims(mean, axis=0)
    out = output_val.transpose(reordering)
    return out
