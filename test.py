from utils.preprocessing import DataSplitter
import numpy as np



class LogisticRegressorTest:
    
    def test(logistic_regressor, test_dataframe):

        #creating the input features dataframe and the target feature dataframe in order to test the 
        #test set of examples
        input_features_dataframe, target_feature_dataframe = DataSplitter.split_targets_from_input(test_dataframe)


        prediction_cumulative_log_loss = 0

        #iterating over the examples in the test dataframe
        for (example_index,example_features), (target_index,example_target) in zip(input_features_dataframe.iterrows(), target_feature_dataframe.items()):

            vectorized_example_features = example_features.to_numpy()

            #computing the logistic regressor prediction and the log loss between
            #the target prediction and the target real value, cumulatin it in a variable
            target_prediction = logistic_regressor.predict(vectorized_example_features)
            prediction_cumulative_log_loss += (example_target)*(-np.log(target_prediction)) + (1 - example_target)*(-np.log(1 - target_prediction))


        #to compute the mean log loss, take the cumulative log loss and divide it 
        #by the number of rows in the test_dataframe
        mean_log_loss = prediction_cumulative_log_loss/len(test_dataframe)


        return mean_log_loss
    







class LinearRegressorTest:

    def test(linear_regressor, test_dataframe):


        #creating the input features dataframe and the target feature dataframe in order to test the 
        #test set of examples
        input_features_dataframe, target_feature_dataframe = DataSplitter.split_targets_from_input(test_dataframe)


        prediction_cumulative_squared_error = 0

        #iterating over the examples in the test dataframe
        for (example_index,example_features), (target_index,example_target) in zip(input_features_dataframe.iterrows(), target_feature_dataframe.items()):

            vectorized_example_features = example_features.to_numpy()

            #computing the logistic regressor prediction and the Squared Error between
            #the target prediction and the target real value, cumulating it in a variable
            target_prediction = linear_regressor.predict(vectorized_example_features)
            prediction_cumulative_squared_error += (target_prediction - example_target)**2


        #to compute the MSE, take the cumulative Squared Error and divide it 
        #by the number of rows in the test_dataframe
        mean_squared_error = prediction_cumulative_squared_error/len(test_dataframe)


        return mean_squared_error