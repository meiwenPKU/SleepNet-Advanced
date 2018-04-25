'''
This script implements super learner ensemble algorithms for just 1-fold cross validation with already trained models on multi-classes classification
This script is based on https://github.com/lendle/SuPyLearner
'''
from sklearn import clone, metrics
from sklearn.base import BaseEstimator, RegressorMixin
import sklearn.cross_validation as cv
import numpy as np
from scipy.optimize import fmin_l_bfgs_b, nnls, fmin_slsqp
from sklearn.metrics import log_loss

class SLError(Exception):
    """
    Base class for errors in the SupyLearner package
    """
    pass


class SuperLearner(BaseEstimator):
    """
    Loss-based super learning
    SuperLearner chooses a weighted combination of candidate estimates
    in a specified library using cross-validation.
    Parameters
    ----------
    library : list
        List of scikit-learn style estimators with fit() and predict()
        methods.
    loss : loss function, 'L2' or 'nloglik'.
    discrete : True to choose the best estimator
               from library ("discrete SuperLearner"), False to choose best
               weighted combination of esitmators in the library.
    coef_method : Method for estimating weights for weighted combination
                  of estimators in the library. 'L_BFGS_B', 'NNLS', or 'SLSQP'.
    Attributes
    ----------
    n_estimators : number of candidate estimators in the library.
    coef : Coefficients corresponding to the best weighted combination
           of candidate estimators in the libarary. 
    risk_cv : List of cross-validated risk estimates for each candidate
              estimator, and the (not cross-validated) estimated risk for
              the SuperLearner
    """
    
    def __init__(self, library, libnames=None, loss='L2', bound=0.00001, discrete=False, coef_method='SLSQP',save_pred_cv=False):
        self.library=library[:]
        self.libnames=libnames
        self.loss=loss
        self.discrete=discrete
        self.coef_method=coef_method
        self.n_estimators=len(library)
        self.save_pred_cv=save_pred_cv
        self.bound = bound
    
    def fit(self, y, y_pred_cv):
        """
        Fit SuperLearner.
        Parameters
        ----------
        y : numpy array of shape [n_samples, n_classes]
            Target values using one-hot encoding
        y_pred_cv: numpy array of shape [n_samples, n_estimators, n_classes]
        Returns
        -------
        self : returns an instance of self.
        """   
        self.coef=self._get_coefs(y, y_pred_cv)
        return self
                        
    
    def predict(self, X):
        """
        Predict using SuperLearner
        Parameters
        ----------
        X : numpy.array of shape [n_samples, n_features]
           or other object acceptable to the predict() methods
           of all candidates in the library
          
        Returns
        -------
        array, shape = [n_samples]
           Array containing the predicted class labels.
        """
        
        n_X = X.shape[0]
        y_pred_all = np.empty((n_X,self.n_estimators, 5))
        for aa in range(self.n_estimators):
            if self.libnames[aa] == 'cnn':
                batch_samples = X[:,-1,:].reshape((X.shape[0],1,-1,1))
            else:
                batch_samples = X
            y_pred_all[:,aa,:] = self.library[aa].predict(batch_samples)
        y_pred=self._get_combination(y_pred_all, self.coef)
        return y_pred


    def _get_combination(self, y_pred_mat, coef):
        """
        Calculate weighted combination of predictions
        Parameters
        ----------
        y_pred_mat: numpy.array of shape [n_samples, n_estimators, n_classes]
                    where each column is a vector of predictions from each candidate
                    estimator
        coef: numpy.array of length n_estimators, to be used to combine
              columns of y_pred_mat
        Returns
        _______
        comb: numpy.array of shape [n_samples, n_classes] of predictions. Each column is the 
              combined predicted probability of one class
        """
        #linear_comb = np.dot(np.swapaxes(y_pred_mat,1,2),coef)
        if self.loss=='L2':
            comb =  np.dot(np.swapaxes(y_pred_mat,1,2),coef)
        elif self.loss=='nloglik':
            y_pred_mat = self._trim(y_pred_mat,self.bound)         
            logit_y = np.log(y_pred_mat/(1-y_pred_mat))
            comb_logit = np.dot(np.swapaxes(logit_y,1,2), coef)
            e_comb_logit = np.exp(comb_logit)
            comb = e_comb_logit / e_comb_logit.sum(axis=1).reshape(-1,1)
        return comb

    def _trim(self, p, bound):
        """
        Trim a probabilty to be in (bound, 1-bound)
        Parameters
        ----------
        p: numpy.array of numbers (generally between 0 and 1)
        bound: small positive number <.5 to trim probabilities to
        Returns
        -------
        Trimmed p
        """
        p[p<bound]=bound
        p[p>1-bound]=1-bound
        return p

    def _get_risk(self, y, y_pred):
        """
        Calculate risk given observed y and predictions
        Parameters
        ----------
        y: numpy array of observed outcomes with shape [n_samples, n_classes]
        y_pred: numpy array of predicted outcomes with shape [n_samples, n_classes]
        Returns
        -------
        risk: estimated risk of y and predictions. 
        """
        if self.loss=='L2':
            risk=np.mean((y-y_pred)**2)
        elif self.loss=='nloglik':
            y_label = np.argmax(y,axis=1)
            risk = log_loss(y_label, self._trim(y_pred,self.bound))
        return risk
        
    def _get_coefs(self, y, y_pred_cv):
        """
        Find coefficients that minimize the estimated risk.
        Parameters
        ----------
        y: numpy.array of observed oucomes in one-hot encoding, shape [n_samples, n_classes]
        y_pred_cv: numpy.array of shape [n_samples, n_estimators, n_classes] of cross-validated
                   predictions
        Returns
        _______
        coef: numpy.array of normalized non-negative coefficents to combine
              candidate estimators
              
        
        """
        if self.coef_method is 'L_BFGS_B':
            if self.loss=='nloglik':
                raise SLError("coef_method 'L_BFGS_B' is only for 'L2' loss")            
            def ff(x):
                return self._get_risk(y, self._get_combination(y_pred_cv, x))
            x0=np.array([1./self.n_estimators]*self.n_estimators)
            bds=[(0,1)]*self.n_estimators
            coef_init,b,c=fmin_l_bfgs_b(ff, x0, bounds=bds, approx_grad=True)
            if c['warnflag'] is not 0:
                raise SLError("fmin_l_bfgs_b failed when trying to calculate coefficients")
            
        elif self.coef_method is 'NNLS':
            if self.loss=='nloglik':
                raise SLError("coef_method 'NNLS' is only for 'L2' loss")
            coef_init, b=nnls(y_pred_cv, y)

        elif self.coef_method is 'SLSQP':
            def ff(x):
                return self._get_risk(y, self._get_combination(y_pred_cv, x))
            def constr(x):
                return np.array([ np.sum(x)-1 ])
            x0=np.array([1./self.n_estimators]*self.n_estimators)
            bds=[(0,1)]*self.n_estimators
            coef_init, b, c, d, e = fmin_slsqp(ff, x0, f_eqcons=constr, bounds=bds, disp=0, full_output=1)
            if d is not 0:
                raise SLError("fmin_slsqp failed when trying to calculate coefficients")

        else: raise ValueError("method not recognized")
        coef_init = np.array(coef_init)
        #All coefficients should be non-negative or possibly a very small negative number,
        #But setting small values to zero makes them nicer to look at and doesn't really change anything
        coef_init[coef_init < np.sqrt(np.finfo(np.double).eps)] = 0
        #Coefficients should already sum to (almost) one if method is 'SLSQP', and should be really close
        #for the other methods if loss is 'L2' anyway.
        coef = coef_init/np.sum(coef_init)
        return coef


