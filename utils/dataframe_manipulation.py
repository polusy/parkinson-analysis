from utils.preprocessing import DataSplitter
import pandas as pd
from model.logistic_regression import LogisticRegressionModel

class DataframeManipulation:

    @staticmethod
    def create_residuals_df_from_regressor(logistic_regressor, dataframe):

        """method to store in a dataframe target column, the difference
         between the real target value and the prediction, for a single logistic regressor"""
    
        #creating the input features dataframe and the target feature dataframe in order to test the 
        #test set of examples
        input_features_dataframe, target_feature_dataframe = DataSplitter.split_targets_from_input(dataframe)

        #iterating over the examples in the test dataframe
        for (example_index,example_features), (target_index,example_target) in zip(input_features_dataframe.iterrows(), target_feature_dataframe.items()):

            vectorized_example_features = example_features.to_numpy()

            #computing dhe difference between log regressor prediction and example target
            target_prediction = logistic_regressor.predict(vectorized_example_features)
            residual = example_target - target_prediction

            #storing the residuals in the target dataframe, at the right index
            target_feature_dataframe.at[target_index] = residual


        #return the complete dataframe, with target value as residuals
        return pd.concat([input_features_dataframe, target_feature_dataframe], axis = 1)
    




    @staticmethod
    def create_residuals_df_from_regressors(regressors, dataframe):

        """method to store in a dataframe target column, the difference
         between the real target value and the sigmoid of weak learners logit, for a list of weak learners"""
    
        #creating the input features dataframe and the target feature dataframe in order to test the 
        #test set of examples
        input_features_dataframe, target_feature_dataframe = DataSplitter.split_targets_from_input(dataframe)

        #iterating over the examples in the test dataframe
        for (example_index,example_features), (target_index,example_target) in zip(input_features_dataframe.iterrows(), target_feature_dataframe.items()):

            vectorized_example_features = example_features.to_numpy()
            total_logit_sum = 0

            #iterating over the regressor list to accumulate logit prediction from each model
            for i in range(len(regressors)):

                #to get the non-sigmoided prediction, for the first model, we extract the
                #regression model from the logistic one, and use the logit prediction
                if i == 0:
                    total_logit_sum += regressors[i].get_regression_model().predict(vectorized_example_features)
                else:
                    total_logit_sum += regressors[i].predict(vectorized_example_features)



            #computing the sigmoid of the total logit prediction
            #and storing it at the right index of the dataframe
            target_prediction = LogisticRegressionModel.sigmoid(total_logit_sum)
            residual = example_target - target_prediction

            #storing the residuals in the target dataframe, at the right index
            target_feature_dataframe.at[target_index] = residual


        #return the complete dataframe, with target value as residuals
        return pd.concat([input_features_dataframe, target_feature_dataframe], axis = 1)
    

