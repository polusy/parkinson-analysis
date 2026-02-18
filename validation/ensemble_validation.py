from model.ensemble import GradientBoostingModel
from utils.constants import LOGISTIC_REG_HYPERPARAMETER, LINEAR_REG_HYPERPARAMETER,BATCHES_NUM, LEARNING_RATE
from sklearn.model_selection import GroupKFold
from statistics import mean,stdev


class BoostingRoundsCrossValidation:

    @staticmethod
    def ensemble_model_boosting_rounds_cross_validate(boosting_rounds_num_list,first_learner_hyperparam, lin_reg_hyperparameter, training_dataframe):
        
        """validate the ensemble model (gradient boosting model), on multiple number of boosting
        rounds"""

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
        #the log loss of predictions on the validation folder
        k_folds_log_loss_list = []

        #we use a list of statistics (mean , stdev) of loss 
        #for each hyperparameter
        hyperparam_stats = [(0,0,0) for i in range(len(boosting_rounds_num_list))]


        for i in range(len(boosting_rounds_num_list)):

            #resetting the values of losses before starting iterating over
            #k-folders
            k_folds_log_loss_list = []

            #using the 6 split, in each iteration we take the index of the validation folder
            #and the index of the training folders, so we train the ensemble model on the
            #training folders and we test it on the validation folder
            for train_idx, validation_idx in gkf.split(training_dataframe, groups=groups):

                current_training_folders = training_dataframe.iloc[train_idx]
                current_validation_folder = training_dataframe.iloc[validation_idx]

                #instantiating a boosting model, with a given number of weak learners
                #as an hyperparameter
                current_ensemble = GradientBoostingModel(boosting_rounds_num_list[i])

                
                #training a new ensemble for each iteration over the k-folds group
                current_ensemble.fit(first_learner_hyperparam, first_learner_hyperparam, BATCHES_NUM, 
                                    LEARNING_RATE, training_dataframe=current_training_folders)
                
                #testing the ensemble on the validation folder, 
                #storing the mean log loss (derived from test on validation folder) 
                # at index correspondent to the index
                # of the the relative hyperparameter in hyperparam_list
                (mean_log_loss, log_loss_std_dev) = current_ensemble.test(test_dataframe=current_validation_folder)
                k_folds_log_loss_list.append(mean_log_loss)


            #computing and storing the mean, stdev of losses of prediction over k validation folders
            mean_log_loss = mean(k_folds_log_loss_list)
            std_dev = stdev(k_folds_log_loss_list)

            #storing the statistics for a given boosting rounds hyperparameter in a stats list
            hyperparam_stats[i] = (boosting_rounds_num_list[i],mean_log_loss,std_dev)

        return hyperparam_stats
            
