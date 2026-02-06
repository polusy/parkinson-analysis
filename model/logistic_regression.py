import numpy as np


class lLinearRegressionModel :

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
        
        #using the yet implemented prediction function of
        #the logistic regression model
        continuous_pred = self.predict(features_instances)
        
        #creating the threshold
        if(continuous_pred < 0.5):
            return 0
        else:
            return 1
        
