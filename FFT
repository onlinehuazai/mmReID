import numpy as np
from scipy.fft import fft, fftshift
from scipy.signal.windows import hamming
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import distance



def rangeFFT(data_radar, V_numRX, numADCSamples=256, idxProcChirp=128):
    range_win = hamming(numADCSamples)
    range_profile = np.zeros((numADCSamples, idxProcChirp, V_numRX), dtype=np.complex64)
    for k in range(V_numRX):
        for m in range(idxProcChirp):
            temp = data_radar[:, m, k] * range_win
            range_profile[:, m, k] = fft(temp, numADCSamples)
    # 返回 距离 FFT 的结果  维度 ADC * chirp * 天线
    return range_profile



def DopplerFFT(range_profile_clutterRemoval, V_numRX, numADCSamples=256, idxProcChirp=128):
    speed_profile = np.zeros((numADCSamples, idxProcChirp, V_numRX), dtype=np.complex64)
    doppler_win = hamming(idxProcChirp)
    for k in range(V_numRX):
        for n in range(numADCSamples):
            temp = range_profile_clutterRemoval[n, :, k] * doppler_win
            speed_profile[n, :, k] = fftshift(fft(temp, idxProcChirp))

    return speed_profile



def specific_range(range_profile, range_min, range_max):
    # select specific area of ADCSamples (containing signal responses)
    idxADCSpecific = [i for i in range(range_min, range_max)]
    specific_range_profile = range_profile[idxADCSpecific, :, :]
    return specific_range_profile



def specific_Doppler(speed_profile, idxProcChirp=128, spped_K=64):
    idxDoppleSpecific = [i for i in range(idxProcChirp//2 - spped_K//2, idxProcChirp//2 + spped_K//2)]
    speed_profile = speed_profile[:, idxDoppleSpecific, :]
    return speed_profile


def angleFFT(AOAInput, lamb, d_l, fft_size=180):
    # print(AOAInput.shape) 
    num_detected_obj = AOAInput.shape[0]
    azimuth_ant_padded = np.zeros(shape=(num_detected_obj, fft_size), dtype=np.complex_)
    azimuth_ant_padded[:, :AOAInput.shape[1]] = AOAInput
    # Process azimuth information
    azimuth_fft = fftshift(np.fft.fft(azimuth_ant_padded, axis=1))
    
    
    wx_list = []
    k_max_list=[]
    for obj_fft in azimuth_fft:
        k_max = np.argmax(np.abs(obj_fft))
        if k_max == 0 or k_max == len(obj_fft) - 1:
            k_max = 1 if k_max == 0 else len(obj_fft) - 2
        y1 = np.abs(obj_fft[k_max - 1])
        ym = np.abs(obj_fft[k_max])
        y2 = np.abs(obj_fft[k_max + 1])
        delta = (y2 - y1) / (2 * ym - y1 - y2)
        k_max_interp = k_max - delta
        k_max_list.append(k_max_interp)
        fw = (k_max_interp - fft_size // 2) / fft_size
        wx = np.arcsin(fw * lamb / d_l)
        wx_list.append(wx)
        
    wx_list=np.array(wx_list)
    return wx_list
