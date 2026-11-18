"""Small deterministic differentiable eight-output linear/MLP regressor in NumPy.

MSE is averaged over samples and outputs. L1/L2 penalties apply to weights only.
Batch GD uses all training rows, SGD uses one row, Adam uses configured minibatches.
"""
import numpy as np
from sklearn.base import BaseEstimator,RegressorMixin


class DenseRegressor(RegressorMixin,BaseEstimator):
    def __init__(self,hidden_layers=(),optimizer='adam',learning_rate=.003,epochs=300,batch_size=32,
                 tolerance=1e-7,patience=20,l1=0.,l2=0.,random_state=42,gradient_clip=10.):
        self.hidden_layers=hidden_layers;self.optimizer=optimizer;self.learning_rate=learning_rate
        self.epochs=epochs;self.batch_size=batch_size;self.tolerance=tolerance;self.patience=patience
        self.l1=l1;self.l2=l2;self.random_state=random_state;self.gradient_clip=gradient_clip

    def fit(self,X,y):
        X=np.asarray(X,dtype=float);y=np.asarray(y,dtype=float)
        if X.ndim!=2 or y.ndim!=2 or y.shape[1]!=8 or len(X)!=len(y) or not len(X):raise ValueError('Expected n×features and n×8 targets')
        if not np.isfinite(X).all() or not np.isfinite(y).all():raise ValueError('Nonfinite training inputs')
        if self.optimizer not in ('batch_gd','sgd','adam'):raise ValueError('Unknown optimizer')
        if self.epochs<1 or self.batch_size<1 or self.patience<1 or self.learning_rate<=0 or self.tolerance<0 or min(self.l1,self.l2)<0 or self.gradient_clip<=0:raise ValueError('Invalid optimizer configuration')
        if not np.isfinite([self.learning_rate,self.tolerance,self.l1,self.l2,self.gradient_clip]).all():raise ValueError('Nonfinite hyperparameter')
        if any(type(v) is not int or v<1 for v in self.hidden_layers):raise ValueError('Invalid hidden layer width')
        rng=np.random.default_rng(self.random_state);sizes=[X.shape[1],*self.hidden_layers,8]
        self.weights_=[rng.normal(0,np.sqrt(2/(a+b)),(a,b)) for a,b in zip(sizes,sizes[1:])]
        self.biases_=[np.zeros(b) for b in sizes[1:]];self.n_features_in_=X.shape[1]
        params=self.weights_+self.biases_;m=[np.zeros_like(p) for p in params];v=[np.zeros_like(p) for p in params]
        batch=len(X) if self.optimizer=='batch_gd' else 1 if self.optimizer=='sgd' else self.batch_size
        self.loss_curve_=[];best=float('inf');stale=0;step=0;best_params=None
        for epoch in range(self.epochs):
            order=np.arange(len(X)) if self.optimizer=='batch_gd' else rng.permutation(len(X))
            for start in range(0,len(X),batch):
                indices=order[start:start+batch];activations=[X[indices]]
                for i,(w,b) in enumerate(zip(self.weights_,self.biases_)):
                    a=activations[-1]@w+b;activations.append(np.tanh(a) if i<len(self.weights_)-1 else a)
                delta=2*(activations[-1]-y[indices])/(len(indices)*8)
                gw=[None]*len(self.weights_);gb=[None]*len(self.weights_)
                for i in reversed(range(len(self.weights_))):
                    gw[i]=activations[i].T@delta+self.l1*np.sign(self.weights_[i])+2*self.l2*self.weights_[i]
                    gb[i]=delta.sum(axis=0)
                    if i:delta=(delta@self.weights_[i].T)*(1-activations[i]**2)
                gradients=gw+gb;norm=np.sqrt(sum(np.sum(g*g) for g in gradients));step+=1
                if not np.isfinite(norm):raise FloatingPointError('Optimizer diverged; reduce learning rate')
                if norm>self.gradient_clip:gradients=[g*self.gradient_clip/norm for g in gradients]
                for i,(p,g) in enumerate(zip(params,gradients)):
                    if self.optimizer=='adam':
                        m[i]=.9*m[i]+.1*g;v[i]=.999*v[i]+.001*g*g
                        p-=self.learning_rate*(m[i]/(1-.9**step))/(np.sqrt(v[i]/(1-.999**step))+1e-8)
                    else:p-=self.learning_rate*g
            loss=float(np.mean((self.predict(X)-y)**2)+self.l1*sum(np.abs(w).sum() for w in self.weights_)+self.l2*sum((w*w).sum() for w in self.weights_))
            if not np.isfinite(loss):raise FloatingPointError('Nonfinite loss')
            self.loss_curve_.append(loss)
            if loss<best-self.tolerance:
                best=loss;stale=0;best_params=[p.copy() for p in params]
            else:stale+=1
            if stale>=self.patience:break
        for p,best_p in zip(params,best_params):p[:]=best_p
        self.n_iter_=epoch+1
        return self

    def predict(self,X):
        a=np.asarray(X,dtype=float)
        if a.ndim!=2 or a.shape[1]!=self.n_features_in_ or not np.isfinite(a).all():raise ValueError('Invalid prediction inputs')
        for i,(w,b) in enumerate(zip(self.weights_,self.biases_)):
            a=a@w+b
            if i<len(self.weights_)-1:a=np.tanh(a)
        return a
