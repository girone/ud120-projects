GRIDS = {
    "gradientboostingclassifier": {
        "gradientboostingclassifier__criterion": ["friedman_mse", "mae"],
        "gradientboostingclassifier__loss": ["deviance", "exponential"],
        "gradientboostingclassifier__n_estimators": [40, 50, 60, 90, 100],
        "gradientboostingclassifier__max_depth": [6, 7, 8, 9, 10],
        "gradientboostingclassifier__max_features": [None, "sqrt", "log2"],
        "gradientboostingclassifier__subsample": [0.8, 1.]
    },
    "linearsvc": {
        "linearsvc__loss": [
            "squared_hinge"
        ],  # loss "hinge" does not work with penalty "l1" when dual is False
        "linearsvc__C": [0.1, 0.5, 1.0, 10, 100, 200],
        "linearsvc__verbose": [0],
        "linearsvc__intercept_scaling": [1],
        "linearsvc__fit_intercept": [True],
        "linearsvc__max_iter": [1000],
        "linearsvc__penalty":
        ["l2"],  # penalty "l1" does not work with loss "hinge"
        "linearsvc__multi_class": ["ovr"],
        "linearsvc__random_state": [None],
        "linearsvc__dual": [True],
        "linearsvc__tol": [0.0001],
        "linearsvc__class_weight": [None]
    }
}


def get_param_grid(algorithm):
    return GRIDS[algorithm]


BEST_FOUND_PARAMETERS = {
    "gradientboostingclassifier": {
        "gradientboostingclassifier__criterion": "friedman_mse",
        "gradientboostingclassifier__max_depth": 8,
        "gradientboostingclassifier__n_estimators": 50,
        "gradientboostingclassifier__max_features": None,
        "gradientboostingclassifier__subsample": 1.0,
        "gradientboostingclassifier__loss": "deviance"
    },
    "linearsvc": {
        'linearsvc__C': 200,
        'linearsvc__fit_intercept': True,
        'linearsvc__max_iter': 1000,
        'linearsvc__penalty': 'l2',
        'linearsvc__class_weight': None,
        'linearsvc__multi_class': 'ovr',
        'linearsvc__dual': True,
        'linearsvc__verbose': 0,
        'linearsvc__tol': 0.0001,
        'linearsvc__intercept_scaling': 1,
        'linearsvc__random_state': None,
        'linearsvc__loss': 'hinge'
    }
}


def get_best_parameter_set(algorithm):
    return BEST_FOUND_PARAMETERS[algorithm]