# Stellar Classification

Multi-class classification of astronomical objects (Galaxy, Star, Quasar) from SDSS17 photometric data, built for the Kaggle Playground Series S6E6 competition.

## Overview

The goal is to classify sky objects observed by the Sloan Digital Sky Survey into three categories using photometric measurements, redshift, and positional data. The competition dataset is a synthetic expansion of the original SDSS17 dataset, evaluated on balanced accuracy.

This project combines gradient boosting models with deep tabular learning architectures in a stacked ensemble, with heavy emphasis on domain-driven feature engineering derived from astronomical color-magnitude relationships.

## Approach

**Feature Engineering**

The core signal in this dataset comes from photometric color indices rather than raw magnitudes. Features include the standard SDSS color indices (u-g, g-r, r-i, i-z and broader baselines), redshift transforms and zone thresholds, stellar locus distance, second-order color curvature, 3D spherical sky coordinates, and interaction terms between redshift and color. The original SDSS17 dataset was merged in after reconstructing the `spectral_type` and `galaxy_population` features the competition added, using the same thresholding logic found through community discussion.

**Class Imbalance**

The three classes are moderately imbalanced (roughly 65/20/14%). SMOTE was applied after feature engineering to balance the combined dataset.

**Models**

Six base models were trained with 5-fold stratified cross-validation, optimized with Optuna where applicable:

- LightGBM
- XGBoost
- ExtraTrees
- A custom residual MLP (PyTorch)
- RealMLP
- TabM

**Ensembling**

Out-of-fold predictions from all base models were combined through weighted averaging and stacking with a logistic regression meta-model, compared against rank averaging as an alternative blending strategy.

## Project Structure

```
Project_Galaxy/
├── data/
│   ├── raw/            # train.csv, test.csv, original SDSS17 dataset
│   └── processed/       # cleaned and feature-engineered data
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_baseline_models.ipynb
│   ├── 04_tuning.ipynb
│   ├── 05_deep_learning.ipynb
│   └── 06_stacking.ipynb
├── src/
│   ├── data_loader.py
│   ├── features_extraction.py
│   ├── models.py
│   └── utils.py
└── submissions/
```

## Key Insight

Redshift alone separates quasars reasonably well, but the overlap zone between stars, galaxies, and quasars at intermediate redshift is best resolved by color-color features, particularly the UV excess captured in u-g and the full optical tilt in u-r. The stellar locus distance metric, which measures how far an object falls from the empirical main sequence in color space, proved to be one of the strongest engineered features for flagging galaxies and high-redshift quasars.

## Results

Cross-validated balanced accuracy improved from ~0.953 at baseline to ~0.97+ after feature engineering, hyperparameter tuning, the addition of the original dataset, and ensembling.

## Tech Stack

Python, scikit-learn, LightGBM, XGBoost, PyTorch, pytabkit (RealMLP, TabM), Optuna, imbalanced-learn

## Acknowledgments

Built collaboratively with ([@Hozaifa-777](https://github.com/Hozaifa-777)).
