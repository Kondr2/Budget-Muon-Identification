# Prediction-on-budget-for-Muon-Identification-at-LHCb

The purpose of this project is to compare three machine learning models for muon identification at the LHCb experiment:
CatBoost library --- https://github.com/catboost;
Cost efficient gradient boosting --- http://papers.nips.cc/paper/6753-cost-efficient-gradient-boosting.pdf;
Adaptive Classification for Prediction Under a Budget --- https://papers.nips.cc/paper/7058-adaptive-classification-for-prediction-under-a-budget.pdf;

### Prerequisites

* Python ( >= 3.6.6 )

* Python modules from requirements.txt

* Catboost library --- https://github.com/catboost

* Cost efficient gradient boosting cloned repository: https://github.com/svenpeter42/LightGBM-CEGB

* GNU Make ( >= 4.1 )

* gcc ( >= 5.4.0 )

### Installing

First install CEGB library in downloaded repository.
In ./cpp/Makefile set LGBM_CEGB constant as CEGB repository path.
In ./cpp run 'make all'.
In Stand.ipynb and splitrawdata.py change constants DATA_PATH and SAVED_MODELS_PATH to preferred directories.

