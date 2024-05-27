# Import modules
import warnings
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model as glm
from statsmodels.stats.outliers_influence import variance_inflation_factor

from neuroGLM.experiment.config import neural_data_path, behav_data_path, coord_data_path, paws, color_paws, trials, blocks
from neuroGLM.glm.utils import get_kernelsGLM
from neuroGLM.visualization import plot_collinearity, plot_modelCV, plot_kernelsGLM, plot_predictGLM
from neuroGLM.utils import model_selection
from neuroGLM.experiment import initExperiment as exp
from neuroGLM.design_mat import buildDesignMatrix as bdm

warnings.filterwarnings('ignore')


# Load data
neural_data = pd.read_csv(neural_data_path)
behavior_data = pd.read_csv(behav_data_path)
rois_coord = pd.read_csv(coord_data_path)


# Create experiment object
expt = exp.Experiment('split ipsi fast')
for tr in trials:
    covariates = []
    
    behav_data_tr = behavior_data[behavior_data['Trial'] == tr]
    neural_data_tr = neural_data[neural_data['trial'] == tr]

    for roi, spike_train in neural_data_tr.iloc[:, 2:].items():
        covariates.append(exp.Covariate(spike_train.values, neural_data_tr['time'].values, roi, None))
    
    for behav_label, behav in behav_data_tr.iloc[:, 2:].items():
        covariates.append(exp.Covariate(behav.values, behav_data_tr['Time'].values, behav_label, None))
    
    trial = exp.Trial(tr)
    trial.add_covariates(covariates)
    trial.register_covariates(nan_policy='drop', resampling='downsampling')
    
    expt.add_trials(trial)


# Check for multicolinearity with VIF and correlation matrix ######## Add to DM class
features = trial.aligned_covariates.iloc[:, -7:]
feature_labels = features.columns
vif = np.array([variance_inflation_factor(features, i) for i in range(features.shape[1])])
vif = pd.DataFrame(vif, index=feature_labels).T
corr_mat = features.corr()
plot_collinearity(vif, corr_mat)


# Build design matrix for one experimental block
test_block = blocks[0, 1]
sr = trial.reference_sr
num_lags = [round(0.250*sr), round(0.250*sr)]
mask = feature_labels
X = []
for tr_idx in range(test_block):
    if tr_idx == 0:
        dm = bdm.buildDesignMatrix(expt.trials[tr_idx].aligned_covariates.iloc[:, 1:], num_lags, add_offset=True, mask=mask)
        X = dm.design_mat
    else:
        dm = bdm.buildDesignMatrix(expt.trials[tr_idx].aligned_covariates.iloc[:, 1:], num_lags, add_offset=True, mask=mask)
        X = pd.concat([X, dm.design_mat[num_lags[0]-1:]])
X.shape # samples x features


# Get the variable to predict
pred_roi = 'ROI90'
for tr_idx in range(test_block):
    if tr_idx == 0:
        y = expt.trials[tr_idx].aligned_covariates[f'{pred_roi}']
    else:
        y = pd.concat([y, expt.trials[tr_idx].aligned_covariates[f'{pred_roi}']])
    print(len(expt.trials[tr_idx].aligned_covariates[f'{pred_roi}']))


# Find L1 regularization parameters via 5-fold cross-validation
GLM_model = glm.LogisticRegression(penalty="l1", solver='liblinear', max_iter=5000)
param_grid = np.logspace(-4, 4, 9)
scores, best_param = model_selection(GLM_model, X, y, param_grid)
plot_modelCV(scores, param_grid)

# Fit the model
GLM_model = glm.LogisticRegression(penalty="l1", solver='liblinear', C=best_param, max_iter=5000)
GLM_results = GLM_model.fit(X, y)

# Get kernels & plot them ########### NORMALIZE
GLM_coeff, GLM_intercept = get_kernelsGLM(GLM_results, num_lags, feature_labels)
fig, (ax_paws, ax_spikehist, ax_acc) = plt.subplots(3, 1)
for p, paw in enumerate(paws):
    plot_kernelsGLM(GLM_coeff[paw], num_lags, trial.reference_sr, ax=ax_paws, ylabel=f'{paw} coeff', color=color_paws[p], linewidth=3)
for roi in neural_data.iloc[:, 2:].columns:
    plot_kernelsGLM(GLM_coeff[roi], num_lags, trial.reference_sr, ax=ax_spikehist, ylabel='Neuronal coupling coeff')
plot_kernelsGLM(GLM_coeff['body acceleration'], num_lags, trial.reference_sr, ax=ax_acc, ylabel='Body acceleration coeff', color='green', linewidth=3)

# Compute the predicted rate & plot it ############ ADD PETH_pred
y_pred = GLM_results.predict(X)
plot_predictGLM(y, y_pred, trial.reference_sr, ax=None)

# # Performance
# logloss = round(log_loss(y, y_pred), 4)
# loglike = -1 * logloss
# mse = mean_squared_error(y, y_pred)
# r2 = r2_score(y_true, y_pred)
# cross_val_score(GLM_model, X, y, cv=5)
# R2adj = 1 - (1-r2)*(len(Y)-1)/(len(y)-X.shape[1]-1)

# print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
# print('R2 adj: %s' % round(R2adj, 2))
# print('MSE: %s' % round(mse, 2))
# print('log likelihood: %s' % loglike)
# print('log loss: %s' % logloss)
# print('R2: %s' % r2)
