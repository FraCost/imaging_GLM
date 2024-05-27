from neuroGLM.utils import map_timestamps, normalize, lowpass_filter
import pandas as pd
import numpy as np


class Covariate:
    def __init__(self, values, timestamps, var_name='covariate', norm_method=None):
        
        if len(values) < 1 or len(timestamps) < 1:
            raise ValueError("Invalid input: values or timestamps is empty")

        if norm_method:
            values = normalize(values, norm_method)
        
        self.values = values
        self.timestamps = timestamps
        self.var_name = var_name
        
        if len(self.timestamps) < 2:
            raise ValueError("At least two timestamps are required to calculate sampling rate")
        self.sr = round(1/(self.timestamps[1]-self.timestamps[0]))
    
    
class Trial:
    def __init__(self, trial_id, metadata=None):
        
        if not trial_id:
            raise ValueError("No trial_id provided")
            
        self.trial_id = trial_id
        self.metadata = metadata if metadata is not None else {}
        self.covariates = []
        self.duration = None
        self.aligned_covariates = pd.DataFrame()
        self.reference_sr = None

    def add_covariates(self, covariates):
        if not covariates:
            raise ValueError("No covariates provided")
            
        if isinstance(covariates, list):
            self.covariates.extend(covariates)
        else:
            self.covariates.append(covariates)
        self.duration = max([covariate.timestamps[-1] for covariate in self.covariates])

    def register_covariates(self, nan_policy=None): 
        if not self.covariates:
            raise ValueError("Trial is empty. Add covariates")
    
        reference_covariate = min(self.covariates, key=lambda covariate: covariate.sr)
        reference_timestamps = reference_covariate.timestamps
        self.reference_sr = reference_covariate.sr
        # cutoff_freq = self.reference_sr / 2  # Nyquist frequency of the target sampling rate
        self.aligned_covariates['time'] = reference_timestamps
        for covariate in self.covariates:
            if self.reference_sr < covariate.sr:
                # covariate.values = lowpass_filter(covariate.values, cutoff_freq, covariate.sr) # Low-pass filter the signal to remove potential aliasing components
                self.aligned_covariates[covariate.var_name] = np.interp(reference_timestamps, covariate.timestamps, covariate.values)
                # mapped_idx = map_timestamps(reference_timestamps, covariate.timestamps)
                # self.aligned_covariates[covariate.var_name] = covariate.values[mapped_idx]
            else:
                self.aligned_covariates[covariate.var_name] = covariate.values
            covariate.resampling_factor = round(covariate.sr/self.reference_sr)
        
        if nan_policy == 'interpolate':
            self.aligned_covariates.interpolate(axis=0, inplace=True)
        elif nan_policy == 'drop':
            self.aligned_covariates.dropna(inplace=True)
            

class Experiment:
    def __init__(self, exp_id=None, metadata=None):
        self.exp_id = exp_id
        self.trials = []
        self.metadata = metadata if metadata is not None else {}
        self.duration = None
        self.num_trials = None
            
    def add_trials(self, trials):
        if isinstance(trials, list):
            self.trials.extend(trials)
        else:
            self.trials.append(trials)
        trial_duration = [trial.duration for trial in self.trials]
        self.duration = sum(trial_duration)
        self.num_trials = len(trial_duration)