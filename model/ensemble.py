
from model.regressors import LogisticRegressionModel,LinearRegressionModel
from utils.dataframe_utils import DataframeManipulation
from utils.constants import NUM_INSTANCE_FEATURES
from utils.preprocessing import DataSplitter
import numpy as np
from statistics import mean, stdev


class GradientBoostingModel:


    def __init__(self, num_weak_learners):

        #instantiating the weak learners list
        self._weak_learners = []

        #need to have min one weak learner, otherwise
        #it is not an ensemble model
        if num_weak_learners < 2:
            raise Exception

        else:
            for i in range(num_weak_learners):
                #as first model, we instantiate the logistic regression model
                if i == 0:
                    self._weak_learners.append(LogisticRegressionModel(LinearRegressionModel(NUM_INSTANCE_FEATURES)))
                #for the other, next to the first one, we instantiate in sequence the other weak learners (lin regression models)
                else:
                    self._weak_learners.append(LinearRegressionModel(NUM_INSTANCE_FEATURES))




    def predict(self, input_features):

        total_logit_sum = 0

        #iterating over the list of weak learners
        for i in range(len(self._weak_learners)):
             
             #for the first weak learner (the logistic one), we take its linear regression model
             #and use it to predict the non-sigmoided value (logit one)
            if i == 0:
                total_logit_sum += self._weak_learners[i].get_regression_model().predict(input_features)
            #for the other weak learners (lin reg model), we predict the target value,
            #using simply its parameters and the given input features
            else:
                total_logit_sum += self._weak_learners[i].predict(input_features)

        
        #passing the sigmoid of the total logit value to compress it in [0,1]
        return LogisticRegressionModel.sigmoid(total_logit_sum)
    





    def fit(self, logistic_reg_hyperparameter, linear_reg_hyperparameter, batches_num, learning_rate, training_dataframe):

        i_residuals_training_dataframe = training_dataframe
        
        for i in range(len(self._weak_learners)):

            if i == 0:
                #training the first weak learner (logistic regression model) and producing the residuals value
                #for the next weak learner
                self._weak_learners[i].fit(logistic_reg_hyperparameter, batches_num=batches_num, learning_rate=learning_rate, training_dataframe=i_residuals_training_dataframe)
                i_residuals_training_dataframe = DataframeManipulation.create_residuals_df_from_regressor(self._weak_learners[i], training_dataframe)

            
            else:
                #training the weak learner at index i in the weak learners sequence, on the residuals
                #of the prediction of the previous ensemble
                self._weak_learners[i].fit(linear_reg_hyperparameter, batches_num=batches_num, learning_rate=learning_rate, training_dataframe=i_residuals_training_dataframe)

                #saving the list of current trained regressors
                current_regressors = [self._weak_learners[j] for j in range(i)]

                #storing in a new dataframe, at the status column, the residuals computed as: 
                # target value - sigmoid(total logit of weak learners sequence)
                i_residuals_training_dataframe = DataframeManipulation.create_residuals_df_from_regressors(current_regressors, i_residuals_training_dataframe)


    



    def test(self, test_dataframe):

        #creating the input features dataframe and the target feature dataframe in order to test the 
        #test set of examples
        input_features_dataframe, target_feature_dataframe = DataSplitter.split_targets_from_input(test_dataframe)


        prediction_log_loss_list = [0]*len(test_dataframe)

        #iterating over the examples in the test dataframe
        for (example_index,example_features), (target_index,example_target) in zip(input_features_dataframe.iterrows(), target_feature_dataframe.items()):

            vectorized_example_features = example_features.to_numpy()

            #computing the ensemble model prediction and the log loss between
            #the target prediction and the target real value, cumulatin it in a variable
            target_prediction = self.predict(vectorized_example_features)
            prediction_log_loss_list.append((example_target)*(-np.log(target_prediction)) + (1 - example_target)*(-np.log(1 - target_prediction)))


        #to compute the mean log loss, take the cumulative log loss and divide it 
        #by the number of rows in the test_dataframe
        mean_log_loss = mean(prediction_log_loss_list)
        std_dev_log_loss = stdev(prediction_log_loss_list)


        return mean_log_loss, std_dev_log_loss
                















    #training : 

    pass

