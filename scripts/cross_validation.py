
from validation.ensemble_validation import BoostingRoundsCrossValidation
from validation.weak_learners_validation import HyperparameterRegularizationCrossValidation
import pandas as pd
from utils.constants import LOGISTIC_REG_HYPERPARAMETER, LINEAR_REG_HYPERPARAMETER


#using the same regularization hyperparameter list, because we use L2 reg. in both 
#the log. regression model and the linear regression model
reg_hyperparameter_list = [0.001, 0.1, 1.0, 10.0]

#finding the regularization hyperparameter that reduces mean and std. dev log loss. using 
#k-folds cross validation approach
#reg_hyperparameter_stats = HyperparameterRegularizationCrossValidation.log_regressor_hyperparam_reg_cross_validate(reg_hyperparam_list=reg_hyperparameter_list, training_dataframe=pd.read_csv("data/normalized/normalized_parkinsons_training.data"))

#print("cross-validation on logistic regressor regularization hyperparameter")
#print(reg_hyperparameter_stats)



#finding the regularization hyperparameter that reduces mean squared and std. dev. of error using 
#k-folds cross validation approach
reg_hyperparameter_stats = HyperparameterRegularizationCrossValidation.lin_regressor_hyperparam_reg_cross_validate(LOGISTIC_REG_HYPERPARAMETER, reg_hyperparam_list=reg_hyperparameter_list, training_dataframe=pd.read_csv("data/normalized/normalized_parkinsons_training.data"))

print("cross-validation on linear regressor regularization hyperparameter")
print(reg_hyperparameter_stats)