"""Small named experiment sets, never an exhaustive Cartesian grid."""


def configurations(group='core'):
    core=[
        dict(name='ols',kind='ols',parameters={}),
        dict(name='ridge',kind='ridge',parameters=dict(alpha=1.)),
        dict(name='lasso',kind='lasso',parameters=dict(alpha=.001)),
        dict(name='elastic_net',kind='elastic_net',parameters=dict(alpha=.001,l1_ratio=.5)),
        dict(name='random_forest',kind='random_forest',parameters=dict(n_estimators=128,max_depth=10,min_samples_leaf=2)),
        dict(name='xgboost',kind='xgboost',parameters=dict(n_estimators=128,max_depth=3,learning_rate=.05)),
        dict(name='mlp',kind='mlp',parameters=dict(optimizer='adam',learning_rate=.003,epochs=300,l2=.0001))]
    if group=='core':return core
    if group=='optimizers':
        return [dict(name=f'{kind}_{opt}',kind=kind,parameters=dict(optimizer=opt,learning_rate=lr,epochs=300,batch_size=32))
                for kind in ('linear_gradient','mlp') for opt,lr in [('batch_gd',.01),('sgd',.001),('adam',.003)]]
    if group=='regularization':
        return [core[0],
                dict(name='ridge_weak',kind='ridge',parameters=dict(alpha=.1)),core[1],
                dict(name='lasso_weak',kind='lasso',parameters=dict(alpha=.0001)),core[2],core[3],
                dict(name='rf_shallow',kind='random_forest',parameters=dict(n_estimators=128,max_depth=5,min_samples_leaf=4)),core[4],
                dict(name='mlp_no_penalty',kind='mlp',parameters=dict(optimizer='adam',epochs=300,l2=0.)),
                dict(name='mlp_l2',kind='mlp',parameters=dict(optimizer='adam',epochs=300,l2=.001))]
    raise ValueError('Unknown experiment group')


def validate_experiments(experiments):
    import re
    names=[]
    for e in experiments:
        if not re.fullmatch(r'[A-Za-z0-9_-]+',e['name']):raise ValueError('Unsafe experiment name')
        names.append(e['name'])
        if e['kind'] in ('random_forest','xgboost') and 'optimizer' in e.get('parameters',{}):raise ValueError('SGD/Adam/GD are not tree optimizers')
    if not names or len(names)!=len(set(names)):raise ValueError('Empty/duplicate experiments')
