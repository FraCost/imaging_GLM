import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_collinearity(vif, corr_mat):
    
    ticks = range(vif.shape[-1])  
    ticklabels = vif.columns  
    
    # Plot VIF
    plt.figure()
    sns.barplot(vif)
    plt.xticks(ticks, ticklabels, rotation=90, fontsize=10)  
    plt.yticks(fontsize=10)
    plt.axhline(y=1, linestyle='--', color='k')
    plt.axhline(y=5, linestyle='--', color='k')
    plt.ylabel('VIF', fontsize=12)
    
    # Plot Correlation Matrix
    plt.figure()
    sns.heatmap(corr_mat, cmap='coolwarm', center=0, xticklabels=ticklabels, yticklabels=ticklabels)
    plt.xticks(rotation=90, fontsize=10) 
    plt.yticks(rotation=0, fontsize=10) 
    cbar = plt.gcf().axes[-1]
    cbar.tick_params(labelsize=15)
    cbar.set_ylabel('Correlation', fontsize=15)
    
    plt.tight_layout()
    plt.show()


def plot_modelCV(scores, params):
    plt.figure()
    plt.plot(scores, marker='.', markersize=10, linewidth=2)
    plt.xticks(np.arange(0, len(params)), params)
    plt.xlabel('C', fontsize=15)
    plt.ylabel('Cross-validated accuracy', fontsize=15)
    

def plot_kernelsGLM(kernels, num_lags, fs, ax=None, ylabel='GLM coefficients', **kwargs):
    if ax == None: 
        fig, ax = plt.subplots()
    t = np.arange(-num_lags*fs, 0, fs)
    ax.plot(t, kernels, **kwargs)
    ax.set_xlabel('Lags (s)', fontsize=15)
    ax.set_ylabel(ylabel, fontsize=15)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.margins(0)


def plot_predictGLM(y, y_pred, fs, ax=None, **kwargs):
    if ax == None: 
        fig, ax = plt.subplots()
    t = np.arange(0, len(y)+1, fs)
    ax.plot(t, y, color='k')
    ax.plot(t, y_pred, color='darkorange')
    ax.set_xlabel('Time (s)', fontsize=15)
    ax.set_ylabel('Sp/s', fontsize=15)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.margins(0)