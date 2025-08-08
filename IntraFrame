import numpy as np
from scipy.fft import fft, fftshift
from scipy.signal.windows import hamming
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from scipy.spatial import distance
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler


def normalize_coordinates(points):
    scaler = MinMaxScaler()
    return scaler.fit_transform(points)


def statistical_outlier_removal_intra_frame(data, k, std_dev_multiplier):
    if k < 4:
        return data,  np.ones(len(data), dtype=bool)
    # Convert DataFrame to numpy array
    points = np.array(data)

    # Normalize coordinates
    normalized_points = normalize_coordinates(points)

    # Fit the nearest neighbors model
    nbrs = NearestNeighbors(
        n_neighbors=k, metric='manhattan').fit(normalized_points)
    distances,indices = nbrs.kneighbors(normalized_points)

    # Remove the first column (self-distance)
    distances = distances[:, 1:]  # Exclude the first column (distance to self)

    # Compute mean and standard deviation of distances
    mean_distances = distances.mean(axis=1)
    global_mean = mean_distances.mean()
    global_std_dev = mean_distances.std()

    # Determine the threshold
    threshold = global_mean + std_dev_multiplier * global_std_dev

    # Identify inliers
    inliers = mean_distances < threshold

    # Filter data
    filtered_data = points[inliers]
    return filtered_data.tolist(),inliers
