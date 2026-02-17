from pyswip import Prolog
from model.ensemble import GradientBoostingModel
from utils.constants import NUM_BOOSTING_ROUNDS, LOGISTIC_REG_HYPERPARAMETER, LINEAR_REG_HYPERPARAMETER, BATCHES_NUM, LEARNING_RATE
import pandas as pd

class KBInterface:

    def __init__(self, kb_path):
        #instantiate a prolog object, then load the kb.pl file 
        #in the prolog object, using pyswip interface
        prolog = Prolog()
        prolog.consult(kb_path)
        self._kb = prolog

    
    def query_result_kb(self, non_normalized_dataframe_row, model_prediction, result_only):
        
        selected_features_row = self.select_features_from_dataframe_row(non_normalized_dataframe_row)
        prediction_augmented_row = self.add_prediction_to_selected_features(selected_features_row, model_prediction)

        #take the selected features from the dataframe row and instante temporarily
        #in the knowledge base as facts
        self._kb.assertz(f"jitter_value(patient, {prediction_augmented_row['Jitter:DDP']})")
        self._kb.assertz(f"shimmer_value(patient, {prediction_augmented_row['Shimmer:APQ3']})")
        self._kb.assertz(f"hnr_value(patient, {prediction_augmented_row['HNR']})")
        self._kb.assertz(f"nhr_value(patient, {prediction_augmented_row['NHR']})")

        if result_only:

            #query the kb with the same patient previously instantiated as facts with his specific vocal features value
            results = list(self._kb.query(f"validation(patient, {model_prediction}, parkinson, Result)"))

            #retract previous assertions from the kb 
            self.retract_assertz_from_kb(prediction_augmented_row)

            #rescuing the value stored in the Result variable
            return results[0]['Result']

        else:
            #query the complete inference chain from the basic symptoms to other abstracts features
            #and produce a report message with the result, evidence (weighted diagnosys)
            inference_chain = list(self._kb.query(f"inference_chain(patient, {model_prediction}, parkinson, Chain)"))
            report = list(self._kb.query(f"report(patient, {model_prediction}, parkinson, Evidence, Result, Message)"))

            #retract previous assertions from the kb 
            self.retract_assertz_from_kb(prediction_augmented_row)
            
            return inference_chain, report
        


    def retract_assertz_from_kb(self, prediction_augmented_row):
        self._kb.retract(f"jitter_value(patient, {prediction_augmented_row['Jitter:DDP']})")
        self._kb.retract(f"shimmer_value(patient, {prediction_augmented_row['Shimmer:APQ3']})")
        self._kb.retract(f"hnr_value(patient, {prediction_augmented_row['HNR']})")
        self._kb.retract(f"nhr_value(patient, {prediction_augmented_row['NHR']})")





    
    def select_features_from_dataframe_row(self, non_normalized_dataframe_row):

        """choose from the non-normalized dataframe, a specific row
        and then transform it in a dataframe row of specific extracted features"""

        selected_features_row = non_normalized_dataframe_row[['Shimmer:APQ3', 'Jitter:DDP', 'NHR', 'HNR']]
        return selected_features_row








