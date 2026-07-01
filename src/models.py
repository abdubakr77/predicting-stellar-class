from pytabkit import RealMLP_TD_Classifier
from pytabkit import TabM_D_Classifier
from sklearn.ensemble import ExtraTreesClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
import optuna
import numpy as np
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression


def auto_tune(X, y, model_class, tuning_params: dict = None, cv=None, n_trials=50):
    """
    Run an Optuna hyperparameter search for a given sklearn-style model class.

    Args:
        X: training features
        y: training labels
        model_class: the model class to tune (not an instance), e.g. LGBMClassifier
        tuning_params: dict mapping param name -> config tuple.
            ('int', low, high), ('float', low, high, log_bool),
            ('categorical', [options]), or a fixed value to use as-is
        cv: cross-validation splitter used to score each trial
        n_trials: number of Optuna trials to run

    Returns:
        the completed optuna.Study, with study.best_params and
        study.best_value available
    """
    if tuning_params is None:
        raise ValueError(
            "tuning_params is required. Example: CATBOOST_PARAMS"
        )

    params = {}
    def objective(trial):
        for param_name, config in tuning_params.items():
            try:
                if config[0] == 'int':
                    params[param_name] = trial.suggest_int(param_name, config[1], config[2])

                elif config[0] == 'float':
                    params[param_name] = trial.suggest_float(param_name, config[1], config[2], log=config[3])
                elif config[0] == "categorical":
                    params[param_name] = trial.suggest_categorical(param_name, config[1])

                else:
                    params[param_name] = config
            except:
                # config wasn't a recognized tuple, treat it as a fixed value
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


def train_pipline(X, y, model_class, model_params: dict = None, cv=None, X_test=None):
    """
    Train a model either with cross-validation (to get OOF predictions) or
    as a single final fit on all the data (to predict on a held-out test set).

    Args:
        X: training features
        y: training labels
        model_class: the model class to train, e.g. LGBMClassifier
        model_params: hyperparameters to use. If None, falls back to a set
            of sane defaults for the known model classes
        cv: cross-validation splitter. If given, runs CV and returns OOF results
        X_test: test features. If cv is None and this is given, fits once
            on all of X/y and returns test set probabilities instead

    Returns:
        If cv is given: (model, oof_preds, oof_probas) from the last fold
        If X_test is given instead: test_proba array
    """
    if model_params is None:
        print("Warning: No model_params given! Will using default model parameters")

        models = {
            "RealMLP_TD_Classifier": RealMLP_TD_Classifier(
                device='cuda',
                random_state=42,
                n_cv=1,
                n_refit=0,
                verbosity=0,
                val_metric_name='cross_entropy',
                use_ls=False,
            ),
            "TabM_D_Classifier": TabM_D_Classifier(
                device='cuda',
                random_state=42,
                n_cv=1,
                n_refit=0,
                verbosity=0,

            ),
            "ExtraTreesClassifier": ExtraTreesClassifier(
                random_state=42,
                verbose=0,
            ),

            "LGBMClassifier": LGBMClassifier(
                random_state=42,
                verbose=-1
            ),

            "XGBClassifier": XGBClassifier(
                random_state=42,
                eval_metric='mlogloss',
                verbosity=0
            )
        }

        model = [models[m] for m in models if m == model_class.__name__][0]

    else:
        model = model_class(**model_params)

    name = model_class.__name__

    n_classes = len(np.unique(y))
    oof_probas = np.zeros((len(y), n_classes))
    oof_preds = np.zeros(len(y), dtype=int)

    all_score = []
    if cv:
        print(f'Cross Validating with {name} Model Now...')
        for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y), start=1):

            X_train = X.iloc[train_idx]
            X_valid = X.iloc[valid_idx]

            y_train = y.iloc[train_idx]
            y_valid = y.iloc[valid_idx]

            if name in ['RealMLP_TD_Classifier', 'TabM_D_Classifier']:
                # these two need an explicit validation split for early stopping
                model.fit(X_train, y_train, X_val=X_valid, y_val=y_valid)
            else:
                model.fit(X_train, y_train)

            preds = model.predict(X_valid).flatten()
            oof_probas[valid_idx] = model.predict_proba(X_valid)

            oof_preds[valid_idx] = preds

            score = balanced_accuracy_score(y_valid, preds)

            all_score.append(score)

            print(f"Fold {fold}: {score:.5f}")

        ma = sum(all_score) / len(all_score)
        print(f"Mean CV Accuracy: {ma:.5f}\n")
        print('=' * 50)

        return model, oof_preds, oof_probas

    if X_test is not None:
        # no CV requested, this is the final fit used to score the real test set
        model.fit(X, y)
        test_proba = model.predict_proba(X_test)
        return test_proba

    else:
        raise Exception('Please Check if you need Cross Validation or Predict the test set. all of them are None Value')


def stacking_ensemble(oof_probas_list, y,
                      test_probas_list,
                      meta_model=None):
    """
    Combine several base models' out-of-fold probabilities into a single
    stacked prediction using a meta-model.

    Args:
        oof_probas_list: list of OOF probability arrays, one per base model,
            each shaped (n_samples, n_classes)
        y: training labels matching the OOF arrays
        test_probas_list: same list of arrays but computed on the test set
        meta_model: the level-2 model to fit on the stacked probabilities.
            Defaults to a regularized LogisticRegression

    Returns:
        (final_preds, final_probas, meta_model) - the stacked predictions,
        their probabilities, and the fitted meta-model
    """
    # stack every model's probabilities side by side: (n_samples, n_models * n_classes)
    meta_train = np.hstack(oof_probas_list)
    meta_test  = np.hstack(test_probas_list)

    if meta_model is None:
        meta_model = LogisticRegression(
            max_iter=1000,
            C=1.0,
            random_state=42
        )

    meta_model.fit(meta_train, y)

    final_preds  = meta_model.predict(meta_test)
    final_probas = meta_model.predict_proba(meta_test)

    return final_preds, final_probas, meta_model