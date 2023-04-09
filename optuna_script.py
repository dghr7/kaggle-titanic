import pandas as pd
import colorama

from sklearn.model_selection import cross_val_score
import xgboost as xgb

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner


RANDOM_STATE = 42
scorer = 'balanced_accuracy'

def objective(trial, X, y):

    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 1000, 50),
        'max_depth': trial.suggest_int('max_depth', 2, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.8, log=True),
        # 'gamma': trial.suggest_float('gamma', 0, 5),
        'subsample': trial.suggest_float('subsample', 0.1, 1.0, step=0.05),
        'min_child_weight': trial.suggest_float('min_child_weight', 1, 50),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1, 1.0, step=0.05),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 5, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 5, log=True),
    }
    
    # V1 : cv using cross_val_score
    clf = xgb.XGBClassifier(**params, 
                            random_state=RANDOM_STATE)

    score = cross_val_score(clf, X, y,  cv=5, scoring=scorer).mean()

    return score

if __name__ == "__main__":

    print("Importing data")
    X = pd.read_pickle("./DATA/X.pkl")
    y = pd.read_pickle("./DATA/y.pkl")

    print(f"X_train shape : {X.shape} / X_test shape : {y.shape}")

    print("Initializing study")
    study = optuna.create_study(study_name = 'xgb', 
                                direction = "maximize", 
                                sampler = TPESampler(seed=RANDOM_STATE),
                                pruner = HyperbandPruner()
                                )

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    func = lambda trial: objective(trial, X, y)

    print("Optimization")
    study.optimize(func, n_trials = 100)

    print(colorama.Fore.GREEN+f"BEST VALUE : {study.best_trial.value}")

    print(colorama.Fore.GREEN+"BEST PARAMS")
    for key, value in study.best_trial.params.items():
        print(colorama.Fore.GREEN+"  {}: {}".format(key, value))