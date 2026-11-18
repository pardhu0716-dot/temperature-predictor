"""Metrics in Celsius, and timed inference including stored preprocessing."""
import time
import numpy as np
from .data import CORE_IDS


def metrics(y,prediction):
    y=np.asarray(y);prediction=np.asarray(prediction)
    if y.shape!=prediction.shape or y.ndim!=2 or y.shape[1]!=8 or not len(y):raise ValueError('Expected matching nonempty n×8 arrays')
    if not np.isfinite(y).all() or not np.isfinite(prediction).all():raise ValueError('Nonfinite evaluation values')
    error=prediction-y
    def score(e):return dict(mae_c=float(np.mean(np.abs(e))),rmse_c=float(np.sqrt(np.mean(e*e))),max_absolute_error_c=float(np.max(np.abs(e))))
    return dict(**score(error),samples=len(y),core_comparisons=int(y.size),per_core={name:score(error[:,i]) for i,name in enumerate(CORE_IDS)})


def latency(bundle,X,repeats=50):
    if repeats<1:raise ValueError('Repeats must be positive')
    X=np.asarray(X)
    bundle.predict(X[:1])
    samples=[]
    for _ in range(repeats):
        t=time.perf_counter_ns();bundle.predict(X[:1]);samples.append((time.perf_counter_ns()-t)/1e6)
    t=time.perf_counter_ns();bundle.predict(X);batch_ms=(time.perf_counter_ns()-t)/1e6
    return dict(single_sample_median_ms=float(np.median(samples)),single_sample_p95_ms=float(np.percentile(samples,95)),
                batch_samples=len(X),batch_ms=batch_ms,repeats=repeats,
                boundary='Engineered feature array through X scaling, regressor, inverse Y scaling; excludes CSV loading and feature engineering')
