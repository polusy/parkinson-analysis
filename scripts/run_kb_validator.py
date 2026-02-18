from kb.kb_interface import KBInterface
from model.ensemble import GradientBoostingModel
from utils.constants import NUM_BOOSTING_ROUNDS, LOGISTIC_REG_HYPERPARAMETER, LINEAR_REG_HYPERPARAMETER, BATCHES_NUM, LEARNING_RATE
from utils.constants import BAL_NUM_BOOSTING_ROUNDS, BAL_LOGISTIC_REG_HYPERPARAMETER, BAL_LINEAR_REG_HYPERPARAMETER
from model.regressors import LogisticRegressionModel
from utils.preprocessing import DataNormalizer
import pandas as pd
import numpy as np

#instantiate the kb interface to consult the prolog knowledge base
kb_interface = KBInterface("kb/parkinsons_kb.pl")



#==============================================================================
#ENSEMBLE PREDICTION + KB VALIDATION ON TEST SET DERIVED FROM UNBALANCED DATASET
#===============================================================================

#train the final ensemble model on training set (derived from unbalanced dataset)
ensemble_model = GradientBoostingModel(NUM_BOOSTING_ROUNDS) 
ensemble_model.fit(LOGISTIC_REG_HYPERPARAMETER, LINEAR_REG_HYPERPARAMETER, BATCHES_NUM, LEARNING_RATE, training_dataframe=pd.read_csv("data/unbalanced/normalized/normalized_parkinsons_training.data"))



normalized_test_dataframe = DataNormalizer.drop_dataframe_columns(pd.read_csv("data/unbalanced/normalized/normalized_parkinsons_test.data"), ['name', 'status'])
non_normalized_test_dataframe = pd.read_csv("data/unbalanced/raw/raw_parkinsons_test.data")


warning_count = 0
conflictual_data_count = 0
critical_error_count = 0
severe_critical_count = 0
moderate_critical_count = 0
unreliable_evidence = 0
coherent_count = 0
total_count = len(normalized_test_dataframe)



for (norm_index, normalized_row), (non_norm_index,non_normalized_row) in zip(normalized_test_dataframe.iterrows(), non_normalized_test_dataframe.iterrows()):
        
        #classify the patient, given the normalized row of his vocal values
        continuous_pred = ensemble_model.predict(normalized_row.to_numpy())
        discrete_pred = LogisticRegressionModel.binary_classify(continuous_pred)

        #rescue the result on the validation query on the kb
        report = kb_interface.query_result_kb(non_normalized_row, discrete_pred, result_only=False)

        result_str = str(report[0]['Result'])

        if result_str  == 'warning': 
            warning_count += 1
        elif 'conflictual_data' in result_str or 'inconsistency' in result_str:
            conflictual_data_count += 1
        elif result_str == 'severe_critical_error':
            severe_critical_count += 1
        elif result_str == 'moderate_critical_error':
            moderate_critical_count += 1
        elif result_str == 'critical_error':
            critical_error_count += 1
        elif result_str == 'unreliable_evidence':
             unreliable_evidence += 1
        elif result_str == 'coherent':
             coherent_count += 1

        print(f"ANALISI KB : \n\n{report[0]['Message']}")




warning_perc = warning_count/total_count
conflictual_data_perc = conflictual_data_count/total_count
critical_error_perc = critical_error_count/total_count
coherent_perc = coherent_count/total_count

print(f"Warning proportion: {warning_perc}\n conflictual data proportion : {conflictual_data_perc}\n Critical error proportion : {critical_error_perc}\n Coherent prediction proportion: {coherent_perc}")







#==============================================================================
#ENSEMBLE PREDICTION + KB VALIDATION ON TEST SET DERIVED FROM BALANCED DATASET
#===============================================================================

#train the final ensemble model on training set (derived from balanced dataset)
ensemble_model = GradientBoostingModel(BAL_NUM_BOOSTING_ROUNDS) 
ensemble_model.fit(BAL_LOGISTIC_REG_HYPERPARAMETER, BAL_LINEAR_REG_HYPERPARAMETER, BATCHES_NUM, LEARNING_RATE, training_dataframe=pd.read_csv("data/balanced/normalized/normalized_balanced_parkinsons_training.data"))



normalized_test_dataframe = DataNormalizer.drop_dataframe_columns(pd.read_csv("data/balanced/normalized/normalized_balanced_parkinsons_test.data"), ['name', 'status'])
non_normalized_test_dataframe = pd.read_csv("data/balanced/raw/balanced_parkinsons_test.data")


warning_count = 0
conflictual_data_count = 0
critical_error_count = 0
severe_critical_count = 0
moderate_critical_count = 0
unreliable_evidence = 0
coherent_count = 0
total_count = len(normalized_test_dataframe)



for (norm_index, normalized_row), (non_norm_index,non_normalized_row) in zip(normalized_test_dataframe.iterrows(), non_normalized_test_dataframe.iterrows()):
        
        #classify the patient, given the normalized row of his vocal values
        continuous_pred = ensemble_model.predict(normalized_row.to_numpy())
        discrete_pred = LogisticRegressionModel.binary_classify(continuous_pred)

        #rescue the result on the validation query on the kb
        report = kb_interface.query_result_kb(non_normalized_row, discrete_pred, result_only=False)

        result_str = str(report[0]['Result'])

        if result_str  == 'warning': 
            warning_count += 1
        elif 'conflictual_data' in result_str or 'inconsistency' in result_str:
            conflictual_data_count += 1
        elif result_str == 'severe_critical_error':
            severe_critical_count += 1
        elif result_str == 'moderate_critical_error':
            moderate_critical_count += 1
        elif result_str == 'critical_error':
            critical_error_count += 1
        elif result_str == 'unreliable_evidence':
             unreliable_evidence += 1
        elif result_str == 'coherent':
             coherent_count += 1

        print(f"ANALISI KB : \n\n{report[0]['Message']}")




warning_perc = warning_count/total_count
conflictual_data_perc = conflictual_data_count/total_count
critical_error_perc = critical_error_count/total_count
coherent_perc = coherent_count/total_count

print(f"Warning proportion: {warning_perc}\n conflictual data proportion : {conflictual_data_perc}\n Critical error proportion : {critical_error_perc}\n Coherent prediction proportion: {coherent_perc}")