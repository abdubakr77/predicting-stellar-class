from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
import optuna
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score


def auto_tune(X,y,model_class, tuning_params:dict = None, cv=None , n_trials=50):

    if tuning_params is None:
        raise ValueError(
            "tuning_params is required. Example: CATBOOST_PARAMS"
        )


    params = {}
    def objective(trial):
        for param_name, config in tuning_params.items():
            if config[0] == 'int':
                params[param_name] = trial.suggest_int(param_name,config[1],config[2])

            elif config[0] == 'float':
                
                params[param_name] = trial.suggest_float(param_name,config[1],config[2],log=config[3])
            
            else:
                params[param_name] = config

        model = model_class(**params)

        scores = cross_val_score(
            model,
            X,
            y,
            cv=cv,
            scoring="accuracy",
            n_jobs=-1
        )
        
        return scores.mean()
        
    study = optuna.create_study(
        direction="maximize"
    )

    study.optimize(
        objective,
        n_trials=n_trials
    )

    return study
    

def train_pipline(X,y,model_class, model_params:dict = None, cv=None):

        
    if model_params is None:
        print("Warning: No model_params given! Will using default model parameters")

        models = {
            "CatBoostClassifier": CatBoostClassifier(
                random_state=42,
                verbose=0,
                allow_writing_files=False
            ),

            "LGBMClassifier": LGBMClassifier(
                random_state=42,
                verbose=0
            ),

            "XGBClassifier": XGBClassifier(
                random_state=42,
                eval_metric='mlogloss',
                verbose=0
            )
        }

        model = [models[m] for m in models if m == model_class.__name__][0]

    else:

        model = model_class(**model_params)


    name = model_class.__name__

    oof_preds = np.zeros(len(y), dtype=int)
    all_score = []

    print(f'Cross Validating with {name} Model Now...')
    for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y), start=1):

        X_train = X.iloc[train_idx]
        X_valid = X.iloc[valid_idx]

        y_train = y.iloc[train_idx]
        y_valid = y.iloc[valid_idx]

        model.fit(X_train, y_train)

        preds = model.predict(X_valid).flatten()

        oof_preds[valid_idx] = preds

        score = accuracy_score(y_valid, preds)

        all_score.append(score)

        print(f"Fold {fold}: {score:.5f}")

    ma = sum(all_score) / len(all_score)
    print(f"Mean CV Accuracy: {ma:.5f}\n")
    print('='*50)

    return model