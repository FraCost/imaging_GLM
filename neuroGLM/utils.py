import numpy as np
from sklearn.model_selection import cross_val_score
from scipy.signal import butter, filtfilt

def normalize(X, norm_method='zscore'):
    if norm_method == 'zscore':
        X = (X - np.nanmean(X)) / np.nanstd(X)
    elif norm_method == 'mean centering':
        X -= np.nanmean(X)
    elif norm_method == 'min-max':
        X = (X - np.nanmin(X)) / (np.nanmax(X) - np.nanmin(X))
    elif norm_method == 'max scaling':
        X /= np.nanmax(X)
    else:
        raise ValueError("Invalid normalization method. Choose from: 'zscore', 'mean centering', 'min-max', 'max scaling'")
    return X

def map_timestamps(t1, t2): # Better to import this from utils; change to bin data
    return np.array([np.where(t2 == t2[np.abs(t2 - t).argmin()])[0][0] for t in t1])

def model_selection(model, X, y, param_grid):
    accuracies = []
    for C in param_grid:
        model.C = C
        accuracy = cross_val_score(model, X, y, cv=10)
        accuracies.append(accuracy.mean())
        best_param = param_grid[np.argmax(accuracies)]
    return accuracies, best_param

def butter_lowpass(cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y