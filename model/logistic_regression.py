import numpy as np


class linear_regression_model :

    def __init__(self, parameters, bias):
        self._parameters = parameters
        self._bias = bias

    def predict(self, features_instances):

        #initialize prediction with model prior bias
        prediction = self._bias
        i = 0

        #computing the product between each feature instance
        #and each corrispondent parameter, store it in prediction
        for i in range(len(features_instances)):
            prediction += features_instances[i]*self._parameters[i]

        return prediction

            

class logistic_regression_model :

    #initializing the logistic regressor with a 
    #regression model
    def __init__(self, regression_model):
        self._linear_regression_model = regression_model


    #logistic transformation of a regression model prediction
    @staticmethod
    def sigmoid(regression_pred):
        return 1/(1 + np.exp(regression_pred))
    
    
    #using the previous logistic transformation
    #to return the continuous prediction in interval [0,1]
    def predict(self, regression_pred):
        return self.sigmoid(regression_pred)
    

    #discretizing the continuous prediction
    def binary_classify(self, regression_pred):
        continuous_pred = self.predict(regression_pred)
        
        #creating the threshold
        if(continuous_pred < 0.5):
            return 0
        else:
            return 1
        
