
from model.ensemble import GradientBoostingModel
from utils.constants import NUM_BOOSTING_ROUNDS, LOGISTIC_REG_HYPERPARAMETER, LINEAR_REG_HYPERPARAMETER, BATCHES_NUM, LEARNING_RATE
from utils.constants import BAL_NUM_BOOSTING_ROUNDS, BAL_LOGISTIC_REG_HYPERPARAMETER, BAL_LINEAR_REG_HYPERPARAMETER
import pandas as pd



#===============
#UNBALANCED DATASET
#===============

#training the final ensemble model on the initial training split (from unbalanced dataset) with the best hyperparameters found with cross-validation
# - reg. hyperparameter of first logistic regressor
# - reg. hyperparameters of the other sequenced linear regressors weak learners
# - num. of boosting rounds
ensemble_model = GradientBoostingModel(NUM_BOOSTING_ROUNDS) 
ensemble_model.fit(LOGISTIC_REG_HYPERPARAMETER, LINEAR_REG_HYPERPARAMETER, BATCHES_NUM, LEARNING_RATE, training_dataframe=pd.read_csv("data/unbalanced/normalized/normalized_parkinsons_training.data"))

#and evaluating it on the test set (from the initial split), to find the mean log loss and its standard deviation on the test set
mean_log_loss, std_dev_log_loss, recall, precision, f1 = ensemble_model.test(test_dataframe=pd.read_csv("data/unbalanced/normalized/normalized_parkinsons_test.data"))

print(mean_log_loss)
print(std_dev_log_loss)
print(recall)
print(precision)
print(f1)






#===============
#BALANCED DATASET
#===============

#training the final ensemble model on the initial training split (from balanced dataset) with the best hyperparameters found with cross-validation
# - reg. hyperparameter of first logistic regressor
# - reg. hyperparameters of the other sequenced linear regressors weak learners
# - num. of boosting rounds
ensemble_model = GradientBoostingModel(BAL_NUM_BOOSTING_ROUNDS) 
ensemble_model.fit(BAL_LOGISTIC_REG_HYPERPARAMETER, BAL_LINEAR_REG_HYPERPARAMETER, BATCHES_NUM, LEARNING_RATE, training_dataframe=pd.read_csv("data/balanced/normalized/normalized_balanced_parkinsons_training.data"))

#and evaluating it on the test set (from the initial split), to find the mean log loss and its standard deviation on the test set
mean_log_loss, std_dev_log_loss, recall, precision, f1 = ensemble_model.test(test_dataframe=pd.read_csv("data/balanced/normalized/normalized_balanced_parkinsons_test.data"))

print(mean_log_loss)
print(std_dev_log_loss)
print(recall)
print(precision)
print(f1)
