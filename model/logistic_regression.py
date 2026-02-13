import numpy as np


class LinearRegressionModel :

    #parametrized init
    def __init__(self, parameters, bias):
        self._parameters = parameters
        self._bias = bias


    #non-parametrized init, 
    #init every parameters and bias with zero values
    def __init__(self):
        self._parameters[:] = 0
        self._bias = 0

    


    def predict(self, features_instances):

        #initialize prediction with model prior bias
        prediction = self._bias
        i = 0

        #computing the product between each feature instance
        #and each corrispondent parameter, store it in prediction
        for i in range(len(features_instances)):
            prediction += features_instances[i]*self._parameters[i]

        return prediction
    
    #defining getters and setters
    def get_bias(self):
        return self._bias

    #returning value of a specific parameter
    #of given index
    def get_parameter(self, param_index):
        return self._parameter[param_index]
    
    #passing parameter index and value to be set
    def set_parameter(self, param_index, value):
        self._parameters[param_index] = value

    def set_bias(self, bias):
        self._bias = bias




            

class LogisticRegressionModel :

    #initializing the logistic regressor with a 
    #regression model
    def __init__(self, regression_model):
        self._regression_model = regression_model


    #logistic transformation of any result
    @staticmethod
    def sigmoid(regression_pred):
        return 1/(1 + np.exp(-regression_pred))
    

    #using the previous logistic transformation
    #and the linear model stored 
    #to return the continuous prediction in interval [0,1]
    def predict(self, features_instances):
        return self.sigmoid(self._regression_model.predict(features_instances))
    

    #discretizing the continuous prediction
    def binary_classify(self, features_instances):
        
        #using the prediction function of
        #the logistic regression model
        continuous_pred = self.predict(features_instances)
        
        #creating the threshold
        if(continuous_pred < 0.5):
            return 0
        else:
            return 1
        
