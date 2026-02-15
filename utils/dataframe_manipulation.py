from preprocessing import DataSplitter
import pandas as pd

class DataframeManipulation:

    def create_prediction_residuals_dataframe(logistic_regressor, test_dataframe):
    
        #creating the input features dataframe and the target feature dataframe in order to test the 
        #test set of examples
        input_features_dataframe, target_feature_dataframe = DataSplitter.split_targets_from_input(test_dataframe)

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