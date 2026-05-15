import xgboost as xgb
import numpy as np


class HousePriceModel:
    def __init__(self, config):
        params = config["model"]["params"].copy()
        self.early_stopping_rounds = params.pop("early_stopping_rounds", 50)
        self.model = xgb.XGBRegressor(**params)

    def fit(self, X_train, y_train, X_val, y_val):
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=self.early_stopping_rounds,
            verbose=50
        )
        return self

    def predict(self, X):
        return self.model.predict(X)

    def get_feature_importance(self, feature_names):
        scores = self.model.feature_importances_
        return dict(zip(feature_names, scores))

    def get_booster(self):
        return self.model.get_booster()

    def best_iteration(self):
        return self.model.best_iteration
