from ReadData import getadcDataFromDCA1000, ADC_frame
from CFAR import ca_cfar
from FFT import DopplerFFT, rangeFFT, angleFFT, specific_range, specific_Doppler
from IntraFrame import statistical_outlier_removal_intra_frame
from InterFrame import statistical_outlier_removal_inter_frame
from utils import clutterRemoval, MediaFilter, kalman_filter_on_lists, smooth_savgol_filter, plot_trajectory,adjust_x_values
from utils import plot_CFAR
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


c = 3.0e8
B = 3997.04e6            
K_S = 68.654e12          
T = B / K_S              
Tc = 40e-3               
fs = 5e6                 
f0 = 77e9                
lamb = c / f0
d_l = lamb / 2
idxProcChirp = 128
numADCSamples = 256
numRX = 4
numTX = 3
numLanes = 2
numChirp = idxProcChirp * 3


numGuard = 3            
numTrain = numGuard * 2  
P_fa = 1e-5              
SNR_OFFSET = -6 


path=""
filesize, adcData = getadcDataFromDCA1000(
    fileName=path, numLanes=numLanes, numRX=numRX, numADCSamples=numADCSamples, idxProcChirp=idxProcChirp)

numFrame = filesize // idxProcChirp // numADCSamples // numRX // numTX // numLanes

bottom_Frame = []
upper_Frame = []
coors_3D = []
coors_2D = []
for idxFrame in range(0, numFrame):
    frame = adcData[:, numChirp * (idxFrame):numChirp*(idxFrame+1), 0:numADCSamples]
    
    outputframe,outputframe2 = ADC_frame(
        frame, numRX=numRX, numADCSamples=numADCSamples, idxProcChirp=idxProcChirp)
    bottom_Frame.append(outputframe)
    upper_Frame.append(outputframe2)


for idxFrame in range(0, numFrame):
    R_list = []
    range_Avg = 0
    angle_Avg = 0
    range_x = 0
    range_y = 0
    range_z = 0
    data_radar1 = bottom_Frame[idxFrame]
    data_radar2 = upper_Frame[idxFrame]
    range_profile1 = rangeFFT(data_radar1, numADCSamples=numADCSamples, idxProcChirp=idxProcChirp, V_numRX=numRX*2)
    range_profile2 = rangeFFT(data_radar2, numADCSamples=numADCSamples, idxProcChirp=idxProcChirp, V_numRX=numRX)

    range_profile_clutterRemoval1 = clutterRemoval(range_profile1, axis=1)
    range_profile_clutterRemoval2 = clutterRemoval(range_profile2, axis=1)
    
    
    speed_profile1 = DopplerFFT(range_profile_clutterRemoval1,V_numRX=8,numADCSamples=numADCSamples, idxProcChirp=idxProcChirp)
    speed_profile2 = DopplerFFT(range_profile_clutterRemoval2,V_numRX=4,numADCSamples=numADCSamples, idxProcChirp=idxProcChirp)



    speed_profile1 = specific_range(speed_profile1, range_min=0, range_max=96)
    speed_profile2 = specific_range(speed_profile2, range_min=0, range_max=96)
    speed_profile1 = specific_Doppler(speed_profile1, idxProcChirp=128, spped_K=96)
    speed_profile2 = specific_Doppler(speed_profile2, idxProcChirp=128, spped_K=96)


    magnitudes = np.sqrt(np.real(speed_profile1) ** 2 + np.imag(speed_profile1) ** 2)

    total_magnitudes = np.mean(magnitudes, axis=2)

    RDM_dB = 10 * np.log10(total_magnitudes / np.max(total_magnitudes))


    RDM_mask, cfar_ranges, cfar_dopps, K = ca_cfar(RDM_dB, numGuard, numTrain, P_fa, SNR_OFFSET)

    angle_profile1 = angleFFT(
        speed_profile1[cfar_ranges, cfar_dopps, :], lamb=lamb, d_l=d_l, fft_size=180)
    
    angle_profile2 = speed_profile1[cfar_ranges, cfar_dopps, 2:6] * np.conj(speed_profile2[cfar_ranges, cfar_dopps, :])  # (N,  4)
    mean_phase_diff = np.mean(angle_profile2, axis=1)  

    

    wz = np.arctan2(mean_phase_diff.imag, mean_phase_diff.real)/np.pi
    elevation_angle = np.arcsin(wz)

    
    RDM_mask = RDM_mask.astype(bool)
    # plot_CFAR(RDM_dB, RDM_mask)
    magnitude_profile = total_magnitudes[RDM_mask]
    print(total_magnitudes.shape)
    print("magnitude_profile", magnitude_profile.shape)
    magnitude_sum = np.sum(magnitude_profile)

    each_frame_3D = [] 
    each_frame_2D = [] 

    for k in range(len(angle_profile1)):
        fb = ((cfar_ranges[k] - 1) * fs) / numADCSamples
        fd = (cfar_dopps[k] + 28 - idxProcChirp // 2 - 1) / (idxProcChirp * Tc)
        R = c * (fb - fd) / (2 * K_S)
        azimuth = angle_profile1[k] 
        elevation = elevation_angle[k] 
        

        x_2D= -R * np.sin(azimuth)
        y_2D= R * np.cos(azimuth)
        x_3D = -R * np.cos(elevation) * np.sin(azimuth )
        y_3D = R * np.cos(elevation) * np.cos(azimuth)
        z_3D = R * np.sin(elevation)
        each_frame_2D.append([x_2D, y_2D])

        each_frame_3D.append([x_3D, y_3D, z_3D])
        points_3d = np.array(each_frame_3D)
        points_2d = np.array(each_frame_2D)
