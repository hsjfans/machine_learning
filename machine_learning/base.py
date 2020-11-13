class BaseEstimator:
    """The base class for all estimators.
    """
    def get_params(self):
        """
        return: the params of model
        """

        pass
    
    def set_params(self):
        """

        """
        pass



class ClassifierMixin:
    """Mixin class for all classifiers

    """

    _estimator_type = "classifier"


    def score(self,X, y):
        """the score of classifiers

        Parameters:
            X, input data, shape: [n_samples,dimensions]
            y, ground truth, labels, shape: [n_samples,n_outputs]

        """
        pass

