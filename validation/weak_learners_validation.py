from sklearn.model_selection import GroupKFold
from training import LogisticRegressorTraining, LinearRegressorTraining
from model.logistic_regression import LogisticRegressionModel, LinearRegressionModel
from test import LogisticRegressorTest,LinearRegressorTest
from statistics import mean,stdev


class HyperparameterRegularizationCrossValidation:

    """this class has the purpose of finding the regularization hyperparameter of
    the first weak learner (logistic regressor) and the regularization hyperparameter 
    of the other weak learners (linear regressors)"""

    def log_regressor_hyperparam_reg_cross_validate(reg_hyperparam_list, training_dataframe):

        #creating the split using the scikit learn grouped k fold
        #in this way each group does not appear in different folders
        n_splits = 6
        gkf = GroupKFold(n_splits = n_splits) #divide the non-test dataframe in 6 folders

        #we take alle the values from the name column
        #then we split the values in two pieces
        # - the first part (patient id)
        # - the second part (sample number from the same patient)
        #then we take the first part to create groups
        groups = training_dataframe['name'].str.rsplit('_', n = 1).str[0]

        #instatiating a list in which each value will be
        #the log loss of a validation folder
        k_folds_log_loss_list = [0 for i in range(n_splits)]
        iteration_counter = 0

        #we use a list of statistics (mean , stdev) of loss 
        #for each parameter in reg_hyperparam_list
        reg_hyperparam_statistics = [(0,0,0) for i in range(len(reg_hyperparam_list))]


        #we want to test differents regularization hyperparameters, so we
        #do a k-fold on each of them, and then we compare the mean log-loss
        #on the validations set of each cross-validation
        for i in range(len(reg_hyperparam_list)):

            iteration_counter = 0

            #using the 6 split, in each iteration we take the index of the validation folder
            #and the index of the training folders, so we train the logistic regressor on the
            #training folders and we test it on the validation folder
            for train_idx, validation_idx in gkf.split(training_dataframe, groups=groups):

                current_training_folders = training_dataframe.iloc[train_idx]
                current_validation_folder = training_dataframe.iloc[validation_idx]

                current_log_regressor = LogisticRegressionModel(LinearRegressionModel())

                #training the logistic regressor on the training folders and validate it over the validation folder
                LogisticRegressorTraining.fit(current_log_regressor, reg_hyperparam_list[i], batches_num=5, learning_rate=0.01, training_data = current_training_folders)
                k_folds_log_loss_list[iteration_counter] = LogisticRegressorTest.test(current_log_regressor, current_validation_folder)

                #saving number of iteration made
                iteration_counter += 1

        
            #computing the mean log loss and the standard deviation between different validations set
            #given a fixed reg hyperparameter
            mean_log_loss = mean(k_folds_log_loss_list)
            std_deviation = stdev(k_folds_log_loss_list)

            #storing the statistics for a given lambda hyperparameter in a stats list
            reg_hyperparam_statistics[i] = (reg_hyperparam_list[i],mean_log_loss,std_deviation)

        return reg_hyperparam_statistics








        





    def lin_regressior_hyperparam_reg_cross_validate(logistic_regressor, linear_regressor, reg_hyperparam_list, training_dataframe):
        pass

