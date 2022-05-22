# Predict Length of Stay with Decision Tree-type, Ensemble and Neural Networks models

Extended project for the Computational Machine Learning course (2nd term). 

The repository contains 2 files: 
  - final jupyter notebook of the project
  - .py file that contains the extra functions that were created and used for this project

Used dataset: the **MIMIC-III** (‘Medical Information Mart for Intensive Care’) is a large, single-center database comprising information relating to patients admitted to critical care units at a large tertiary care hospital. Data includes vital signs, medications, laboratory measurements, observations and notes charted by care providers, fluid balance, procedure codes, diagnostic codes, imaging reports, hospital length of stay, survival data, and more.
https://mimic.mit.edu/docs/gettingstarted/

**Project Summary**

1. Preprocess data
    - create new features regarding length of stay + number of comorbidities
2. Create a data preprocessing pipeline with ColumnTransformer
3. Decision Tree model (with GridSearch)
4. Other Tree-type models (RandomForest, XGBoost, GradientBoost, AdaBoost)
5. Ensembling techniques
    - Stacking predictors
    - SuperLearner
    - Feature Propagation
6. Neural Netoworks
    - with sklearn
    - with tensorflow
7. Compare results and interpret best model with SHAP
