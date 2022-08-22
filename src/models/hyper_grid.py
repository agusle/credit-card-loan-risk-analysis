"""
Dictionaries with hyperparameters grid for model tuning.
"""

RANDOM_FOREST = {
    "bootstrap": [True, False],
    "max_depth": [200, None, 300, 400],
    "max_features": ["sqrt", "log2"],
    "min_samples_leaf": [2, 3, 4, 5, 6],
    "min_samples_split": [2, 5, 10, 15],
    "n_estimators": [100, 200, 300, 400],
    "criterion": ["gini", "entropy"],
}

LIGHTGBM = {
    "num_leaves": [20, 150, 300],
    "max_depth": [9, 6, 3],
    "boosting_type": ["gbdt", "dart", "goss"],
    "learning_rate": [0.1, 0.01, 0.001, 1],
    "n_estimators": [100, 200, 300],
    "min_data": [50, 70, 100],
}

XGBOOST = {
    "max_depth": [3, 4, 5, 6, 8, 10, 12, 15],
    "learning_rate": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
    "gamma": [0.0, 0.8, 0.2, 0.3, 0.4],
    "reg_lambda": [0, 1, 10],
    "scale_pos_weight": [1, 3, 5],
    "subsample": [0.3, 0.6, 0.8],
    "colsample_bytree": [0.3, 0.4, 0.5, 0.7],
}

CATBOOST = {
    "depth": [4, 5, 8, 9],
    "learning_rate": [0.01, 0.02, 0.03, 0.04],
    "iterations": [10, 20, 30, 80, 90, 100],
}

MLPC = {
    "hidden_layer_sizes": [(50, 50, 50), (50, 100, 50), (100,)],
    "activation": ["tanh", "relu"],
    "solver": ["sgd", "adam"],
    "alpha": [0.0001, 0.05],
    "learning_rate": ["constant", "adaptive"],
    "max_iter": [200, 100],
}
