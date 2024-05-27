import pandas as pd

def get_kernelsGLM(model, nlags, feature_labels):
    kernels = pd.DataFrame()
    num_features = feature_labels.shape[-1]
    i = 0
    for end in range(nlags+1, (nlags*num_features)+2, nlags):
        start = end - nlags
        kernels[feature_labels[i]] = model.coef_[0, start:end]
        i += 1    
        return kernels, model.intercept_