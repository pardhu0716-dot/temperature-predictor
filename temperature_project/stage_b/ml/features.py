"""Fixed geometry-aware features in canonical core-ID order; no IDs or targets as features."""
import numpy as np
from .data import CORE_IDS

VERSION=1


def build_features(dataset):
    g=dataset.geometry_mm;p=dataset.powers_w;die=dataset.die_mm
    columns=[];definitions=[]
    def add(name,values,unit):
        columns.append(np.asarray(values));definitions.append(dict(name=name,unit=unit))
    area=g[:,:,0]*g[:,:,1];centers=g[:,:,2:]+g[:,:,:2]/2
    for i,name in enumerate(CORE_IDS):
        for field,values,unit in [
            ('power',p[:,i],'W'),('width',g[:,i,0],'mm'),('height',g[:,i,1],'mm'),
            ('x',g[:,i,2],'mm'),('y',g[:,i,3],'mm'),('area',area[:,i],'mm2'),
            ('power_density',p[:,i]/area[:,i],'W/mm2'),('center_x',centers[:,i,0],'mm'),
            ('center_y',centers[:,i,1],'mm'),('is_big',np.isclose(g[:,i,0],4).astype(float),'binary')]:
            add(f'{name}.{field}',values,unit)
    add('die.width',die[:,0],'mm');add('die.height',die[:,1],'mm')
    add('die.area',die[:,0]*die[:,1],'mm2')
    add('die.whitespace_fraction',1-area.sum(axis=1)/(die[:,0]*die[:,1]),'fraction')
    for i in range(8):
        for j in range(i):
            delta=centers[:,i]-centers[:,j];dist=np.linalg.norm(delta,axis=1)
            gaps=np.maximum(np.abs(delta)-(g[:,i,:2]+g[:,j,:2])/2,0)
            for label,values in [('dx',delta[:,0]),('dy',delta[:,1]),('center_distance',dist),('edge_gap',np.linalg.norm(gaps,axis=1))]:
                add(f'{CORE_IDS[j]}--{CORE_IDS[i]}.{label}',values,'mm')
    for target in range(8):
        for source in range(8):
            if source!=target:
                distance=np.linalg.norm(centers[:,target]-centers[:,source],axis=1)
                # Fixed 1 mm softening is a feature definition, not a thermal coefficient.
                add(f'{CORE_IDS[source]}->{CORE_IDS[target]}.power_over_distance',p[:,source]/(distance+1.0),'W/mm')
    X=np.column_stack(columns)
    if not np.isfinite(X).all():raise ValueError('Nonfinite engineered features')
    return X,definitions
