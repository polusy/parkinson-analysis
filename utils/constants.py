
#number of instance features for the current classification problem
NUM_INSTANCE_FEATURES = 22


#regularization hyperparameters, found after cross-validation, for unbalanced dataset
LOGISTIC_REG_HYPERPARAMETER = 0.001
LINEAR_REG_HYPERPARAMETER = 0.001
NUM_BOOSTING_ROUNDS = 3

#regularization hyperparameters, found after cross-validation, for balanced dataset
BAL_LOGISTIC_REG_HYPERPARAMETER = 0.001
BAL_LINEAR_REG_HYPERPARAMETER = 0.001
BAL_NUM_BOOSTING_ROUNDS = 3

#learning rate hyperparameter (fixed, not found with cross-validation)
LEARNING_RATE = 0.01

#nubers of batches for model training
BATCHES_NUM = 5
