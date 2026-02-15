from model.ensemble import GradientBoostingModel
from utils.constants import LOGISTIC_REG_HYPERPARAMETER, LINEAR_REG_HYPERPARAMETER,BATCHES_NUM, LEARNING_RATE


class BoostingRoundsCrossValidation:

    def ensemble_model_boosting_rounds_cross_validate(boosting_rounds_num_list, training_dataframe):
        
        """validate the ensemble model (gradient boosting model), on multiple number of boosting
        rounds"""

        for i in range(2, len(boosting_rounds_num_list)):

            #instantiating a boosting model, with a given number of weak learners
            current_ensemble = GradientBoostingModel(boosting_rounds_num_list[i])

            #create group folding to cross validate, then fit and test
            #store the log loss for each validation set

            #at the end extract the mean and the std dev

            current_ensemble.fit(LOGISTIC_REG_HYPERPARAMETER, LINEAR_REG_HYPERPARAMETER, BATCHES_NUM, 
                                 LEARNING_RATE, training_dataframe=training_dataframe)
            current_ensemble.predict
            
