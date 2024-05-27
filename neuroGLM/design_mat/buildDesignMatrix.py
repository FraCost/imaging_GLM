import numpy as np
import pandas as pd
from scipy.linalg import hankel


def build_hankel(x, lags):
    
    if not isinstance(lags, (int, list, tuple)):
        raise ValueError("num_lags must be an integer, list, or tuple")
    
    if isinstance(lags, int):
        if lags <= 0:
            raise ValueError("num_lags must be a positive integer greater than zero")
        lags_before = lags
        lags_after = lags
    elif isinstance(lags, (list, tuple)):
        if not all(isinstance(l, int) for l in lags) or any(l <= 0 for l in lags):
            raise ValueError("All elements of num_lags list/tuple must be positive integers greater than zero")
        lags_before, lags_after = lags
        
    padded_x_before = np.pad(x, (lags_before, 0), mode='constant')
    hankel_mat_before = hankel(padded_x_before[:-lags_before+1], x[-lags_before-1:])
    
    x_flip = x[::-1]
    padded_x_after = np.pad(x_flip, (lags_after, 0), mode='constant')
    hankel_mat_after = hankel(padded_x_after[:-lags_after+1], x_flip[-lags_after-1:])
    hankel_mat_after = hankel_mat_after[::-1,::-1]
    hankel_mat_after = hankel_mat_after[:, 1:]
    
    hankel_mat = np.concatenate((hankel_mat_before, hankel_mat_after), axis = 1)
    
    return pd.DataFrame(hankel_mat)


class buildDesignMatrix:
    def __init__(self, features, num_lags, add_offset=True, mask=None):
        self.features = features
        self.num_lags = num_lags
        self.design_mat = None
        self.mask = mask

        if self.features.empty:
            raise ValueError("No feature found")
        if not self.num_lags:
            raise ValueError("Add num_lags")
        
        if mask is not None:
            features = features[mask]
        
        if isinstance(features, pd.Series): # If features is a Series (just one variable), convert it to a DataFrame to avoid errors
            features = features.to_frame()
        
        design_mat = pd.DataFrame(np.zeros((len(features), 0)))
        for i, col in enumerate(features.columns):
            
            if isinstance(self.num_lags, dict):
                num_lags = self.num_lags[col]
            else:
                num_lags = self.num_lags
            
            # if not isinstance(num_lags, (int, list, tuple)):
            #     raise ValueError("num_lags must be an integer, list, or tuple")
            
            # if isinstance(num_lags, int):
            #     if num_lags <= 0:
            #         raise ValueError("num_lags must be a positive integer greater than zero")
            #     lags_before = num_lags
            #     lags_after = num_lags
            # elif isinstance(num_lags, (list, tuple)):
            #     if not all(isinstance(l, int) for l in num_lags) or any(l < 0 for l in num_lags):
            #         raise ValueError("All elements of num_lags list/tuple must be positive integers greater than zero")
            #     lags_before, lags_after = num_lags
            
            # feature = features[col].values
            # headers = np.array([f'{col} lag{l}' for l in range(-lags_before, lags_after+1)])
            
            # if lags_before > 0 and lags_after > 0:
            #     padded_feature_before = np.pad(feature, (lags_before, 0), mode='constant')
            #     hankel_mat_before = hankel(padded_feature_before[:-lags_before], feature[-lags_before-1:])
                
            #     feature_flip = feature[::-1]
            #     padded_feature_after = np.pad(feature_flip, (lags_after, 0), mode='constant')
            #     hankel_mat_after = hankel(padded_feature_after[:-lags_after], feature_flip[-lags_after-1:])
            #     hankel_mat_after = hankel_mat_after[::-1,::-1]
            #     hankel_mat_after = hankel_mat_after[:, 1:]
                
            #     hankel_mat = np.concatenate((hankel_mat_before, hankel_mat_after), axis = 1) 
            #     hankel_mat = pd.DataFrame(hankel_mat, columns=headers)
                
            feature = features[col].values
            headers = np.array([f'{col} lag{l}' for l in range(-num_lags, 1)])
            padded_feature = np.pad(feature, (num_lags, 0), mode='constant')
            hankel_mat = hankel(padded_feature[:-num_lags], feature[-num_lags-1:])
            hankel_mat = pd.DataFrame(hankel_mat, columns=headers)
            
            design_mat = pd.concat([design_mat, hankel_mat], axis=1)
        self.design_mat = design_mat
            
        if add_offset == True:
            design_mat.insert(0, 'Offset', 1)
        self.design_mat = design_mat
                    
        if self.design_mat.isnull().values.any() or np.isinf(self.design_mat.values).any():
            raise ValueError("NaN or Inf found in design matrix")