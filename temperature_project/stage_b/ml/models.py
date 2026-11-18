"""Lightweight multi-output estimators; XGBoost is an optional lazy dependency."""
from dataclasses import dataclass
import numpy as np
from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from .optimizers import DenseRegressor


def build_model(kind,parameters=None,seed=42):
    p=dict(parameters or {})
    if kind=='ols':return LinearRegression(**p)
    if kind=='ridge':return Ridge(**p)
    if kind=='lasso':return Lasso(max_iter=10000,random_state=seed,**p)
    if kind=='elastic_net':return ElasticNet(max_iter=10000,random_state=seed,**p)
    if kind=='random_forest':return RandomForestRegressor(random_state=seed,n_jobs=1,**p)
    if kind=='xgboost':
        try:from xgboost import XGBRegressor
        except ImportError as exc:raise ImportError('XGBoost requested: install stage_b/ml/requirements-xgboost.txt; no experiment is silently skipped') from exc
        return MultiOutputRegressor(XGBRegressor(objective='reg:squarederror',tree_method='hist',random_state=seed,n_jobs=1,**p),n_jobs=1)
    if kind in ('linear_gradient','mlp'):
        hidden=p.pop('hidden_layers',() if kind=='linear_gradient' else (32,))
        if kind=='linear_gradient' and hidden:raise ValueError('Linear gradient model cannot have hidden layers')
        return DenseRegressor(hidden_layers=tuple(hidden),random_state=seed,**p)
    raise ValueError(f'Unknown model: {kind}')


@dataclass
class ModelBundle:
    estimator: object
    scaling: dict
    feature_sha256: str
    core_order: list
    simulation_contract: dict | None
    source_csv_sha256: str

    def predict(self,X):
        X=np.asarray(X,dtype=float)
        mean=np.asarray(self.scaling['x_mean']);scale=np.asarray(self.scaling['x_scale'])
        if X.ndim!=2 or X.shape[1]!=len(mean) or not np.isfinite(X).all():raise ValueError('Invalid features')
        result=self.estimator.predict((X-mean)/scale)
        result=np.asarray(result)*np.asarray(self.scaling['y_scale'])+np.asarray(self.scaling['y_mean'])
        if result.shape!=(len(X),8) or not np.isfinite(result).all():raise ValueError('Invalid model predictions')
        return result
