"""Shared, deterministic geometry helpers for frozen-model feature pipelines."""
import numpy as np


def feature_inputs(geometry, powers, die):
    try:
        g=np.asarray(geometry,dtype=np.float64);p=np.asarray(powers,dtype=np.float64);d=np.asarray(die,dtype=np.float64)
    except (TypeError,ValueError,OverflowError) as exc:
        raise ValueError('Geometry, powers and die must be finite numerical arrays') from exc
    if g.shape==(8,4):g=g[None];p=p[None] if p.shape==(8,) else p;d=d[None] if d.shape==(2,) else d
    if g.ndim!=3 or g.shape[1:]!=(8,4) or not len(g) or p.shape!=(len(g),8) or d.shape!=(len(g),2):
        raise ValueError('Expected geometry (N,8,4), powers (N,8), die (N,2), or one unbatched layout')
    if not all(np.isfinite(x).all() for x in (g,p,d)) or np.any(g[:,:,:2]<=0) or np.any(d<=0):
        raise ValueError('Features require finite inputs and positive core/die dimensions')
    return g,p,d


def nearest_power(powers,distances,policy='mean_ties'):
    """Exclude self and average equidistant sources; legacy mode is audit-only.

    Absolute 1e-12 mm tolerance absorbs arithmetic roundoff at geometric symmetry;
    no relative tolerance broadens the tied set as physical distances increase.
    """
    d=np.where(np.eye(8,dtype=bool)[None],np.inf,distances)
    nearest=np.min(d,axis=2)
    if policy=='legacy_first':
        selected=np.take_along_axis(powers,np.argmin(d,axis=2),axis=1)
    elif policy=='mean_ties':
        tied=np.isclose(d,nearest[:,:,None],rtol=0,atol=1e-12)
        selected=np.sum(powers[:,None,:]*tied,axis=2)/np.sum(tied,axis=2)
    else:raise ValueError('Unsupported nearest-neighbor policy')
    return nearest,selected
