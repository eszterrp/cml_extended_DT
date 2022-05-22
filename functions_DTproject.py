#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 10:36:46 2022

@author: pazma
"""
import pandas as pd
import matplotlib.pyplot as plt
import math

# get dummies
def get_dummies(df,columns):
    import pandas as pd
    return pd.get_dummies(data=df, columns=columns, drop_first=True)


## the correction factor: 
def reweight(pi,q1,r1):
    r0 = 1-r1
    q0 = 1-q1
    tot = pi*(q1/r1)+(1-pi)*(q0/r0)
    w = pi*(q1/r1)
    w /= tot
    return w

# define R-squared function
def R_squared(y_pd, y_np):
    '''To be fed with numpy arrays'''
    y_pd = y_pd.reset_index()['LOS']
    return (y_pd.corr(pd.Series(y_np)))**2

# # The function plot_regression_results is used to plot the predicted and true targets for the stacked regression
def plot_regression_results(ax, y_true, y_pred, title, scores, elapsed_time):
    """Scatter plot of the predicted vs true targets."""
    ax.plot(
        [y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "--r", linewidth=2
    )
    ax.scatter(y_true, y_pred, alpha=0.2)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.spines["left"].set_position(("outward", 10))
    ax.spines["bottom"].set_position(("outward", 10))
    ax.set_xlim([y_true.min(), y_true.max()])
    ax.set_ylim([y_true.min(), y_true.max()])
    ax.set_xlabel("Measured")
    ax.set_ylabel("Predicted")
    extra = plt.Rectangle(
        (0, 0), 0, 0, fc="w", fill=False, edgecolor="none", linewidth=0
    )
    ax.legend([extra], [scores], loc="upper left")
    title = title + "\n Evaluation in {:.2f} seconds".format(elapsed_time)
    ax.set_title(title)
    


from mlens.ensemble import SuperLearner


from sklearn.feature_extraction.text import _VectorizerMixin
from sklearn.feature_selection._base import SelectorMixin
import numpy as np
from sklearn.pipeline import Pipeline

# get the feature names from the preprocessor pipeline
def get_feature_out(estimator, feature_in):
    if hasattr(estimator,'get_feature_names'):
        if isinstance(estimator, _VectorizerMixin):
            # handling all vectorizers
            return [f'vec_{f}' \
                for f in estimator.get_feature_names()]
        else:
            return estimator.get_feature_names(feature_in)
    elif isinstance(estimator, SelectorMixin):
        return np.array(feature_in)[estimator.get_support()]
    else:
        return feature_in
    
def get_ct_feature_names(ct):
    # handles all estimators, pipelines inside ColumnTransfomer
    # doesn't work when remainder =='passthrough'
    # which requires the input column names.
    output_features = []

    for name, estimator, features in ct.transformers_:
        if name!='remainder':
            if isinstance(estimator, Pipeline):
                current_features = features
                for step in estimator:
                    current_features = get_feature_out(step, current_features)
                features_out = current_features
            else:
                features_out = get_feature_out(estimator, features)
            output_features.extend(features_out)
        elif estimator=='passthrough':
            output_features.extend(ct._feature_names_in[features])
                
    return output_features

# remap icd9 codes
def remap_icd9_codes(x):
    if (x >= 1 and x<=139):
        return 'infectious and parasitic diseases'
    elif (x >= 140 and x<=239):
        return 'neoplasms'
    elif (x >= 240 and x<=279):
        return 'endocrine, nutritional and metabolic diseases, and immunity disorders'
    elif (x >= 280 and x<= 289):
        return 'diseases of the blood and blood-forming organs'
    elif (x >= 290 and x<=319):
        return 'mental disorders'
    elif (x >= 320 and x<=389):
        return 'diseases of the nervous system and sense organs'
    elif (x >= 390 and x<=459):
        return 'diseases of the circulatory system' 
    elif (x >= 460 and x<=519):
        return 'diseases of the respiratory system'
    
    elif (x >= 520 and x<=579):
        return 'diseases of the digestive system'
        
    elif (x >= 580 and x<=629):
        return 'diseases of the genitourinary system'
        
    elif (x >= 630 and x<=679):
        return 'complications of pregnancy, childbirth, and the puerperium'
        
    elif (x >= 680 and x<=709):
        return 'diseases of the skin and subcutaneous tissue'
        
    elif (x >= 710 and x<=739):
        return 'diseases of the musculoskeletal system and connective tissue'
        
    elif (x >= 740 and x<=759):
        return 'congenital anomalies'
        
    elif (x >= 710 and x<=779):
        return 'certain conditions originating in the perinatal period'
        
    elif (x >= 780 and x<=799):
        return 'symptoms, signs, and ill-defined conditions'
        
    elif (x >= 800 and x<=999):
        return "injury and poisoning"
    else:
        return 'external causes of injury and supplemental classification'
