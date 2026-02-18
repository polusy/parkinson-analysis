
from validation.ensemble_validation import BoostingRoundsCrossValidation
from validation.weak_learners_validation import HyperparameterRegularizationCrossValidation
import pandas as pd
from utils.constants import LOGISTIC_REG_HYPERPARAMETER, BAL_LOGISTIC_REG_HYPERPARAMETER
from utils.constants import LINEAR_REG_HYPERPARAMETER, BAL_LINEAR_REG_HYPERPARAMETER

#using the same regularization hyperparameter list, because we use L2 reg. in both 
#the log. regression model and the linear regression model
#both for the balanced dataset and the unbalanced one
reg_hyperparameter_list = [0.001, 0.1, 1.0, 10.0]
boosting_rounds_num_list = [2,3,5]







#=================================================
#UNBALANCED DATASET HYPERPARAMETERS CROSS-VALIDATION
#=================================================



#finding the regularization hyperparameter that reduces mean and std. dev log loss. using 
#k-folds cross validation approach
reg_hyperparameter_stats = HyperparameterRegularizationCrossValidation.log_regressor_hyperparam_reg_cross_validate(reg_hyperparam_list=reg_hyperparameter_list, training_dataframe=pd.read_csv("data/unbalanced/normalized/normalized_parkinsons_training.data"))
print("cross-validation on logistic regressor regularization hyperparameter")
print(reg_hyperparameter_stats)


#finding the regularization hyperparameter that reduces mean squared and std. dev. of error using 
#k-folds cross validation approach
reg_hyperparameter_stats = HyperparameterRegularizationCrossValidation.lin_regressor_hyperparam_reg_cross_validate(LOGISTIC_REG_HYPERPARAMETER, reg_hyperparameter_list, training_dataframe=pd.read_csv("data/unbalanced/normalized/normalized_parkinsons_training.data"))
print("cross-validation on linear regressor regularization hyperparameter")
print(reg_hyperparameter_stats)



#finding stats (mean and std. dev. of log loss over prediction evalutation on different validation sets) for different number of boosting rounds, using k-folds cross validation approach
boosting_rounds_stats = BoostingRoundsCrossValidation.ensemble_model_boosting_rounds_cross_validate(boosting_rounds_num_list, LOGISTIC_REG_HYPERPARAMETER, LINEAR_REG_HYPERPARAMETER, training_dataframe=pd.read_csv("data/unbalanced/normalized/normalized_parkinsons_training.data"))
print("cross-validation on boosting rounds")
print(boosting_rounds_stats)










"""
#=================================================
#BALANCED DATASET HYPERPARAMETERS CROSS-VALIDATION
#=================================================




#finding the regularization hyperparameter that reduces mean and std. dev log loss. using 
#k-folds cross validation approach
reg_hyperparameter_stats = HyperparameterRegularizationCrossValidation.log_regressor_hyperparam_reg_cross_validate(reg_hyperparam_list=reg_hyperparameter_list, training_dataframe=pd.read_csv("data/balanced/normalized/normalized_balanced_parkinsons_training.data"))
print("cross-validation on logistic regressor regularization hyperparameter")
print(reg_hyperparameter_stats)


#finding the regularization hyperparameter that reduces mean squared and std. dev. of error using 
#k-folds cross validation approach
reg_hyperparameter_stats = HyperparameterRegularizationCrossValidation.lin_regressor_hyperparam_reg_cross_validate(BAL_LOGISTIC_REG_HYPERPARAMETER, reg_hyperparameter_list, training_dataframe=pd.read_csv("data/balanced/normalized/normalized_balanced_parkinsons_training.data"))
print("cross-validation on linear regressor regularization hyperparameter")
print(reg_hyperparameter_stats)


#finding stats (mean and std. dev. of log loss over prediction evalutation on different validation sets) for different number of boosting rounds, using k-folds cross validation approach
boosting_rounds_stats = BoostingRoundsCrossValidation.ensemble_model_boosting_rounds_cross_validate(boosting_rounds_num_list, BAL_LOGISTIC_REG_HYPERPARAMETER, BAL_LINEAR_REG_HYPERPARAMETER, training_dataframe=pd.read_csv("data/balanced/normalized/normalized_balanced_parkinsons_training.data"))
print("cross-validation on boosting rounds")
print(boosting_rounds_stats)"""
