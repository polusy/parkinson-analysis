from numpy import linalg,subtract
import numpy as np
import pandas as pd
from utils.constants import NUM_INSTANCE_FEATURES


class LinearRegressionModel :


    #non-parametrized init, 
    #init every parameters and bias with zero values
    def __init__(self, num_input_features):
        self._parameters = [0] * (num_input_features + 1)

    


    def predict(self, features_instances):

        prediction = 0
        i = 0

        #computing the product between each feature instance
        #and each corrispondent parameter, store it in prediction
        for i in range(len(features_instances)):
            prediction += features_instances[i]*self._parameters[i]

        #adding the bias (parameter indipendent from the features instances)
        bias = self._parameters[len(features_instances)] #we store it at the last index of the parameters list
        prediction += bias

        return  prediction
    

    def fit(self, reg_hyperparameter, batches_num, learning_rate, training_dataframe):

            """fitting the linear regressor to a specific training dataset
            given a regularization hyperparameter, the number of batches to be trained on
            and the learning rate of the gradient descent"""

            #creating batches of samples (mini dataframe), given a requested number of batches
            #dont care about same patient samples distributed over different batches 
            index_splits = np.array_split(training_dataframe.index.to_numpy(), batches_num) #list of arrays of indexes, each array denotes the list of specific indexes of a batch
            batches = [training_dataframe.loc[idx_list] for idx_list in index_splits] #using each array of indexes to store a btach in the list of batches

            if 'name' in training_dataframe.columns and 'status' in training_dataframe.columns:
                #splitting the batches in input features batches and target feature batches
                input_features_batches = [batch.drop(['name', 'status'], axis = 1) for batch in batches]
                target_feature_batches = [batch['status'] for batch in batches]
            else:
                #splitting the batches in input features batches and target feature batches
                input_features_batches = [batch.drop(columns=['status']) for batch in batches]
                target_feature_batches = [batch['status'] for batch in batches]

            #initializing the loss gradient and the gradient tollerance value
            #the dimension of the gradient is taken from the number
            #of columns of a random batch in the input_features_batches
            loss_gradient = [100 for i in range(NUM_INSTANCE_FEATURES + 1)]
            epoch_gradient = [200 for i in range(NUM_INSTANCE_FEATURES + 1)]
            

            #initialiazing the tollerance values
            parameters_subtraction_norm_tol = 10**-3
            gradient_norm_tol = 10**-3

            #initializing the first with real parameters values and the second with high values
            #such that at first, verifying the condition in the loop, their subtraction is high
            parameters_before_epoch = self._parameters
            parameters_after_epoch = [100 for i in range(len(parameters_before_epoch))]

            #iterating in epochs if 
            # - the gradient norm is still too high
            # - the norm of the subtraction between parameters after and befor epoch is too high
            while (linalg.norm(np.array(epoch_gradient)) > gradient_norm_tol) and linalg.norm(subtract(parameters_after_epoch, parameters_before_epoch)) > parameters_subtraction_norm_tol :

                #resetting epoch_gradient at the start
                epoch_gradient = [0 for i in range(NUM_INSTANCE_FEATURES + 1)]
                
                #saving the model parameters before starting the training epoch
                parameters_before_epoch = self._parameters.copy()

                #iterating over input features and target feature batches
                for input_feature_batch, target_feature_batch in zip(input_features_batches, target_feature_batches):

                    current_batch_examples_num = len(input_feature_batch)

                    #re-setting the 0 value for loss gradient, adding 1 to range to include
                    #the partial derivative of loss to respect with bias
                    loss_gradient = [0 for i in range(len(input_features_batches[0].columns) + 1)]

                    #iterating over the example in the batch
                    for (example_index,example_features), (target_index,example_target) in zip(input_feature_batch.iterrows(), target_feature_batch.items()):

                        vectorized_example_features = example_features.to_numpy()

                        #computing the linear regressor prediction and the loss between the target value
                        #and the target prediction
                        target_prediction = self.predict(vectorized_example_features)
                        example_prediction_loss = target_prediction - example_target 

                        #iterating over the example features to calculate the partial derivative
                        #of loss with respect to each parameter in the regression model
                        for i in range(len(vectorized_example_features) + 1):

                            #distinguish the partial derivative of mean squared loss with respect to bias (last parameter)
                            #from the partial derivatives of loss with respect to other parameters
                            if i == len(vectorized_example_features):
                                loss_gradient[i] += 2*example_prediction_loss #derivative of MSE with respect to bias
                            else:
                                loss_gradient[i] += 2*example_prediction_loss*vectorized_example_features[i] #derivative of MSE with respect to other parameters

                    #taking the mean partial derivative of loss with respect to each parameter
                    #to compute partial derivates of mean log loss
                    for i in range(len(vectorized_example_features) + 1):
                        loss_gradient[i] /= current_batch_examples_num
                        epoch_gradient[i] += loss_gradient[i]

                    
                    #updating linear regression model parameters
                    for i in range(len(parameters_before_epoch)):

                        #taking the previous value of the parameter
                        parameter_i = self._parameters[i]

                        #updating the parameter subtracting from it the loss gradient 
                        self._parameters[i] = parameter_i - learning_rate*loss_gradient[i]

                        #the bias term is not updated with regularization
                        if i != len(parameters_after_epoch) - 1:

                            #computing the L2 regularization term
                            L2_reg_term = learning_rate* ((reg_hyperparameter)/current_batch_examples_num)*parameter_i

                            #new update for L2 regularization, subtracting from the previously updated parameter the L2 reg. term
                            updated_parameter_i = self._parameters[i]
                            self._parameters[i] = updated_parameter_i - L2_reg_term


                #updating the epoch gradient with the mean of gradients over all the batches 
                epoch_gradient = [ddw_sum/batches_num for ddw_sum in epoch_gradient]

                #rescuing the parameters after an epoch
                parameters_after_epoch = self._parameters.copy()



    
    #defining getters and setters
    def get_bias(self):
        return self._bias

    #returning value of a specific parameter
    #of given index
    def get_parameter(self, param_index):
        return self._parameters[param_index]
    
    def get_parameters(self):
        return self._parameters
    
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
    
    
    def fit(self, reg_hyperparameter, batches_num, learning_rate, training_dataframe):

        """fitting the logistic regressor to a specific training dataset
        given a regularization hyperparameter, the number of batches to be trained on
        and the learning rate of the gradient descent"""

        #creating batches of samples (mini dataframe), given a requested number of batches
        #dont care about same patient samples distributed over different batches 
        index_splits = np.array_split(training_dataframe.index.to_numpy(), batches_num) #list of arrays of indexes, each array denotes the list of specific indexes of a batch
        batches = [training_dataframe.loc[idx_list] for idx_list in index_splits] #using each array of indexes to store a btach in the list of batches

        if 'name' in training_dataframe.columns and 'status' in training_dataframe.columns:
            #splitting the batches in input features batches and target feature batches
            input_features_batches = [batch.drop(['name', 'status'], axis = 1) for batch in batches]
            target_feature_batches = [batch['status'] for batch in batches]

        #initializing the loss gradient and the gradient tollerance value
        #the dimension of the gradient is taken from the number
        #of columns of a random batch in the input_features_batches
        loss_gradient = [100 for i in range(NUM_INSTANCE_FEATURES+ 1)]
        epoch_gradient = [200 for i in range(NUM_INSTANCE_FEATURES + 1)]
        

        #initialiazing the tollerance values
        parameters_subtraction_norm_tol = 10**-3
        gradient_norm_tol = 10**-3

        #initializing the first with real parameters value and the second with high values
        #such that at the first loop condition check, their subtraction is high
        parameters_before_epoch = self._regression_model.get_parameters()
        parameters_after_epoch = [100 for i in range(len(parameters_before_epoch))]

        #iterating in epochs if 
        # - the gradient norm is still too high
        # - the norm of the subtraction between parameters after and befor epoch is too high
        while (linalg.norm(np.array(epoch_gradient)) > gradient_norm_tol) and linalg.norm(subtract(parameters_after_epoch, parameters_before_epoch)) > parameters_subtraction_norm_tol :

            #resetting epoch_gradient at the start
            epoch_gradient = [0 for i in range(len(input_features_batches[0].columns) + 1)]
            
            #saving the model parameters before starting the training epoch
            parameters_before_epoch = self._regression_model.get_parameters().copy()

            #iterating over input features and target feature batches
            for input_feature_batch, target_feature_batch in zip(input_features_batches, target_feature_batches):

                current_batch_examples_num = len(input_feature_batch)

                #re-setting the 0 value for loss gradient, adding 1 to range to include
                #the partial derivative of loss to respect with bias
                loss_gradient = [0 for i in range(len(input_features_batches[0].columns) + 1)]

                #iterating over the example in the batch
                for (example_index,example_features), (target_index,example_target) in zip(input_feature_batch.iterrows(), target_feature_batch.items()):

                    vectorized_example_features = example_features.to_numpy()

                    #computing the logistic regressor prediction and the loss between the target value
                    #and the target prediction
                    target_prediction = self.predict(vectorized_example_features)
                    example_prediction_loss = target_prediction - example_target 

                    #iterating over the example features to calculate the partial derivative
                    #of loss with respect to each parameter in the regression model
                    for i in range(len(vectorized_example_features) + 1):

                        #distinguish the partial derivative of loss with respect to bias (first parameter)
                        #from the partial derivatives of loss with respect to other parameters
                        if i == len(vectorized_example_features):
                            loss_gradient[i] += example_prediction_loss #derivative with respect to bias
                        else:
                            loss_gradient[i] += example_prediction_loss*vectorized_example_features[i]

                #taking the mean partial derivative of loss with respect to each parameter
                #to compute partial derivatives of mean log loss
                for i in range(len(vectorized_example_features) + 1):
                    loss_gradient[i] /= current_batch_examples_num
                    epoch_gradient[i] += loss_gradient[i]

                
                #updating linear regression model parameters
                for i in range(len(parameters_before_epoch)):

                    #taking the previous value of the parameter
                    parameter_i = self._regression_model.get_parameter(i)

                    #updating the parameter by subtracting from it the loss gradient 
                    self.set_parameter(i, parameter_i - learning_rate*loss_gradient[i])

                    #do not update the bias term with regularization
                    if i != len(parameters_after_epoch) - 1:
                        
                        #computing the L2 regularization term
                        L2_reg_term = learning_rate* ((reg_hyperparameter)/current_batch_examples_num)*parameter_i

                        #new update for L2 regularization, subtracting from the previously updated parameter the L2 reg. term
                        updated_parameter_i = self._regression_model.get_parameter(i)
                        self.set_parameter(i, updated_parameter_i - L2_reg_term)


            #updating the epoch gradient with the mean of gradient (partial derivatives' vector) over all the batches 
            epoch_gradient = [ddw_sum/batches_num for ddw_sum in epoch_gradient]

            #rescuing the parameters after an epoch
            parameters_after_epoch = self._regression_model.get_parameters().copy()
    



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
        
        
    def get_regression_model(self):
        return self._regression_model
    

    def set_parameter(self, param_index, value):
        self._regression_model.set_parameter(param_index, value)
    


        
