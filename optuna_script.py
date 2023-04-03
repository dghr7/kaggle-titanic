import pandas as pd
import colorama

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner, HyperbandPruner



RANDOM_STATE = 42
scorer = 'balanced_accuracy'

def objective(trial, X_train, y_train):

    params = {
        'loss': trial.suggest_categorical('loss', ['deviance', 'exponential']),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.001, 1),
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 2, 10),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 4),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
    }
    
    # V1 : cv using cross_val_score
    clf = GradientBoostingClassifier(**params, 
                                     random_state=RANDOM_STATE)

    score = cross_val_score(clf, X_train, y_train,  cv=6, scoring=scorer).mean()

    return score

if __name__ == "__main__":

    print("Importing data")
    X_train = pd.read_pickle("./DATA/X_train.pkl")
    y_train = pd.read_pickle("./DATA/y_train.pkl")
    X_test = pd.read_pickle("./DATA/X_test.pkl")
    y_test = pd.read_pickle("./DATA/y_test.pkl")

    print(f"X_train shape : {X_train.shape} / X_test shape : {X_test.shape}")

    print("Initializing study")
    study = optuna.create_study(study_name = 'gb', 
                                direction = "maximize", 
                                sampler = TPESampler(seed=RANDOM_STATE),
                                pruner = HyperbandPruner()
                                )

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    func = lambda trial: objective(trial, X_train, y_train)

    print("Optimization")
    study.optimize(func, n_trials = 100)

    print(colorama.Fore.GREEN+f"BEST VALUE : {study.best_trial.value}")

    print(colorama.Fore.GREEN+"BEST PARAMS")
    for key, value in study.best_trial.params.items():
        print(colorama.Fore.GREEN+"  {}: {}".format(key, value))