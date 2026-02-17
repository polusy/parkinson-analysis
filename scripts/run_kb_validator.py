from kb.kb_interface import KBInterface
from model.ensemble import GradientBoostingModel
from utils.constants import NUM_BOOSTING_ROUNDS, LOGISTIC_REG_HYPERPARAMETER, LINEAR_REG_HYPERPARAMETER, BATCHES_NUM, LEARNING_RATE
from model.regressors import LogisticRegressionModel
import pandas as pd
import numpy as np



#train the final ensemble model
ensemble_model = GradientBoostingModel(NUM_BOOSTING_ROUNDS) 
ensemble_model.fit(LOGISTIC_REG_HYPERPARAMETER, LINEAR_REG_HYPERPARAMETER, BATCHES_NUM, LEARNING_RATE, training_dataframe=pd.read_csv("data/normalized/normalized_parkinsons_training.data"))


kb_interface = KBInterface("kb/parkinsons_kb.pl")


normalized_test_dataframe = pd.read_csv("data/normalized/normalized_parkinsons_test.data")
non_normalized_test_dataframe = pd.read_csv("data/raw/raw_parkinsons_test.data")


warning_count = 0
conflictual_data_count = 0
critical_error_count = 0
coherent_count = 0
total_count = len(normalized_test_dataframe)



for (norm_index, normalized_row), (non_norm_index,non_normalized_row) in zip(normalized_test_dataframe.iterrows(), non_normalized_test_dataframe.iterrows()):
        
        #classify the patient, given the normalized row of his vocal values
        continuous_pred = ensemble_model.predict(normalized_row.to_numpy())
        discrete_pred = LogisticRegressionModel.binary_classify(continuous_pred)

        #rescue the result on the validation query on the kb
        result = kb_interface.query_result_kb(non_normalized_row, discrete_pred, result_only=True)

        if 'warning' in result: 
            warning_count += 1
        elif 'conflictual_data' in result:
            conflictual_data_count += 1
        elif 'critical_error' in result:
             critical_error_count += 1
        elif 'coherent' in result:
             coherent_count += 1




warning_perc = warning_count/total_count
conflictual_data_perc = conflictual_data_count/total_count
critical_error_perc = critical_error_count/total_count
coherent_perc = coherent_count/total_count