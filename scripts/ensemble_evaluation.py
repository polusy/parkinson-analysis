
from model.ensemble import GradientBoostingModel
from utils.constants import NUM_BOOSTING_ROUNDS, LOGISTIC_REG_HYPERPARAMETER, LINEAR_REG_HYPERPARAMETER, BATCHES_NUM, LEARNING_RATE
import pandas as pd


#training the final ensemble model on the initial training split with the best hyperparameters found with cross-validation
# - reg. hyperparameter of first logistic regressor
# - reg. hyperparameters of the other sequenced linear regressors weak learners
# - num. of boosting rounds
ensemble_model = GradientBoostingModel(NUM_BOOSTING_ROUNDS) 
ensemble_model.fit(LOGISTIC_REG_HYPERPARAMETER, LINEAR_REG_HYPERPARAMETER, BATCHES_NUM, LEARNING_RATE, training_dataframe=pd.read_csv("data/normalized/normalized_parkinsons_training.data"))

#and evaluating it on the test set (from the initial split), to find the mean log loss and its standard deviation on the test set
mean_log_loss, std_dev_log_loss = ensemble_model.test(test_dataframe=pd.read_csv("data/normalized/normalized_parkinsons_test.data"))

print(mean_log_loss)
print(std_dev_log_loss)