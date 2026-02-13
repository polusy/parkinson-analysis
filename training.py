from numpy import linalg,subtract
import numpy as np
import pandas as pd



class LogisticRegressorTraining:

    #assumed logistic regressor to be already initialized, by initializing 
    #linear regressor parameter values
    def fit(logistic_regressor, reg_hyperparameter, batches_num, learning_rate, training_csv):

        training_dataframe = pd.read_csv(training_csv)

        #creating batches of samples (mini dataframe), given a requested number of batches
        #dont care about same patient samples distributed over different batches
        batches = np.array_split(training_dataframe, batches_num)


        #splitting the batches in input features batches and target feature batches
        input_features_batches = [batches[i].drop(['name', 'status']) for i in range(len(batches))]
        target_feature_batches = [batches[i]['status'] for i in range(len(batches))]

        #initializing the loss gradient and the gradient tollerance value
        #the dimension of the gradient is taken from the number
        #of columns of a random batch in the input_features_batches
        loss_gradient = [100 for i in range(len(input_features_batches[0].columns) + 1)]
        gradient_norm_tol = 10**-2

        #initialiazing the parameters subtraction norm tollerance value
        parameters_subtraction_norm_tol = 10**-2

        #initializing the first with real parameters value and the second with high values
        #such that at first, their subtraction is high
        parameters_before_epoch = logistic_regressor.get_regression_model().get_parameters()
        parameters_after_epoch = [100 for i in range(len(parameters_before_epoch))]

        #iterating in epochs if 
        # - the gradient norm is still too high
        # - the norm of the subtraction between parameters after and befor epoch is too high
        while (linalg.norm(np.array(loss_gradient), 'fro') > gradient_norm_tol) and linalg.norm(subtract(parameters_after_epoch, parameters_before_epoch)) > parameters_subtraction_norm_tol :
        
            #saving the model parameters before starting the training epoch
            parameters_before_epoch = logistic_regressor.get_regression_model().get_parameters()

            #iterating over input features and target feature batches
            for input_feature_batch, target_feature_batch in zip(input_features_batches, target_feature_batches):

                current_batch_examples_num = len(input_feature_batch)

                #re-setting the 0 value for loss gradient, adding 1 to range to include
                #the partial derivative of loss to respect with bias
                loss_gradient = [0 for i in range(len(training_dataframe.columns) + 1)]

                #iterating over the example in the batch
                for example_features, example_target in zip(input_feature_batch.iterrows(), target_feature_batch.items()):

                    vectorized_example_features = example_features.to_numpy()

                    #computing the logistic regressor prediction and the loss between the target value
                    #and the target prediction
                    target_prediction = logistic_regressor.predict(vectorized_example_features)
                    example_prediction_loss = example_target - target_prediction

                    #iterating over the example features to calculate the partial derivative
                    #of loss with respect to each parameter in the regression model
                    for i in range(len(vectorized_example_features) + 1):

                        #distinguish the partial derivative of loss with respect to bias (first parameter)
                        #from the partial derivatives of loss with respect to other parameters
                        if i == len(vectorized_example_features):
                            loss_gradient[i] = example_prediction_loss #derivative with respect to bias
                        else:
                            loss_gradient[i] += example_prediction_loss*vectorized_example_features[i]

                #taking the mean partial derivative of loss with respect to each parameter
                #to compute partial derivates of mean log loss
                for i in range(len(vectorized_example_features) + 1):
                    loss_gradient[i] /= current_batch_examples_num

                
                #updating linear regression model parameters
                for i in range(len(parameters_before_epoch) + 1):

                    #taking the previous value of the parameter
                    parameter_i = logistic_regressor.get_regression_model().get_parameter(i)

                    #updating the parameter subtracting from it the loss gradient 
                    logistic_regressor.set_parameter(i, parameter_i - learning_rate*loss_gradient[i])

                    #do not udate the bias term with regularization
                    if i != len(parameters_after_epoch):
                        #new update for L2 regularization
                        updated_parameter_i = logistic_regressor.get_regression_model().get_parameter(i)
                        L2_reg_term = learning_rate* ((reg_hyperparameter)/current_batch_examples_num)*parameter_i
                        logistic_regressor.set_parameter(i, updated_parameter_i - L2_reg_term)


            #rescuing the parameters after an epoch
            parameters_after_epoch = logistic_regressor.get_regression_model().get_parameters()
        













class LinearRegressorTraining:

    def fit(linear_regressor, reg_hyperparameter, training_csv):

        #assumed linear regressor to be already initialized, by initializing 
        #linear regressor parameter values
        training_dataframe = pd.read_csv(training_csv)

        #gradient descent using L2 regularization, given
        #reg_hyperparameter





