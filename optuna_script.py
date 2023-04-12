import pandas as pd
import colorama

from sklearn.model_selection import cross_val_score

from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner, MedianPruner

RANDOM_STATE = 42
scorer = 'balanced_accuracy'

def objective(trial, X, y):

    # XGB
    # params = {
    #     'n_estimators': trial.suggest_int('n_estimators', 50, 1000, 50),
    #     'max_depth': trial.suggest_int('max_depth', 2, 10),
    #     'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.8, log=True),
    #     # 'gamma': trial.suggest_float('gamma', 0, 5),
    #     'subsample': trial.suggest_float('subsample', 0.1, 1.0, step=0.05),
    #     'min_child_weight': trial.suggest_float('min_child_weight', 1, 50),
    #     'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1, 1.0, step=0.05),
    #     'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 5, log=True),
    #     'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 5, log=True),
    # }
    
    # clf = xgb.XGBClassifier(**params, 
    #                         random_state=RANDOM_STATE)

    # # LGBM
    # params = {
    #     "boosting_type": "gbdt",
    #     'n_estimators': trial.suggest_int('n_estimators', 10, 600),
    #     'num_leaves': trial.suggest_int('num_leaves', 2, 500),
    #     'max_depth': trial.suggest_int('max_depth', 2, 20),
    #     'learning_rate': trial.suggest_float('learning_rate', 1e-8, 1, log=True),
    #     'min_child_samples': trial.suggest_int('min_child_samples', 1, 500),
    #     'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 100, log=True),
    #     'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 100, log=True)
    # }
    
    # clf = lgb.LGBMClassifier(**params, 
    #                          random_state=RANDOM_STATE)

    # RF
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=50),
        'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),
        'max_depth': trial.suggest_int('max_depth', 5, 30),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
        'max_features': trial.suggest_categorical('max_features', ['auto', 'sqrt', 'log2', None])
    }
    
    clf = RandomForestClassifier(**params,
                                random_state=RANDOM_STATE)


    score = cross_val_score(clf, X, y,  cv=5, scoring=scorer).mean()

    return score

if __name__ == "__main__":

    print("Importing data")
    X = pd.read_pickle("./DATA/X.pkl")
    y = pd.read_pickle("./DATA/y.pkl")

    print(f"X_train shape : {X.shape} / X_test shape : {y.shape}")

    print("Initializing study")
    study = optuna.create_study(study_name = 'rf', 
                                direction = "maximize", 
                                sampler = TPESampler(seed=RANDOM_STATE),
                                pruner = MedianPruner()
                                )

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    func = lambda trial: objective(trial, X, y)

    print("Optimization")
    study.optimize(func, n_trials = 200)

    print(colorama.Fore.GREEN+f"BEST VALUE : {study.best_trial.value}")

    print(colorama.Fore.GREEN+"BEST PARAMS")
    for key, value in study.best_trial.params.items():
        print(colorama.Fore.GREEN+"  {}: {}".format(key, value))