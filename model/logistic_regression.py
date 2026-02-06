import numpy as np


class linear_regression_model :

    def __init__(self, parameters, bias):
        _parameters = parameters
        _bias = bias

    def predict(self, features_instances):

        #initialize prediction with model prior bias
        prediction = self._bias
        i = 0

        #computing the product between each feature instance
        #and each corrispondent parameter, store it in prediction
        while(i < features_instances.size()):
            prediction += features_instances.get(i)*self._parameters.get(i)

        return prediction

            

class logistic_regression_model :

    #initializing the logistic regressor with a 
    #regression model
    def __init__(self, regression_model):
        _linear_regression_model = regression_model


    #logistic transformation of a regression model prediction
    def logistic_trans(regression_pred):
        return 1/(1 + np.exp(-regression_pred))
