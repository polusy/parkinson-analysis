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
        self._kb.assertz("jitter_value(patient, %p)", prediction_augmented_row['Jitter:DDP'])
        self._kb.assertz("shimmer_value(patient, %p)", prediction_augmented_row['Shimmer:APQ3'])
        self._kb.assertz("hnr_value(patient, %p)", prediction_augmented_row['HNR'])
        self._kb.assertz("nhr_value(patient, %p)", prediction_augmented_row['NHR'])

        if result_only:

            #query the kb with the same patient previously instantiated as facts with his specific vocal features value
            results = list(self._kb.query("validation(patient, %p, parkinson, Result)", model_prediction))

            #rescuing the value stored in the Result variable
            return results[0]['Result']

        else:
            #query the complete inference chain from the basic symptoms to other abstracts features
            #and produce a report message with the result, evidence (weighted diagnosys)
            inference_chain = list(self._kb.query("inference_chain(patient, %p, parkinson, Chain)", model_prediction))
            report = list(self._kb.query("report(patient, %p, parkinson, Evidence, Result, Message)", model_prediction))
            
            return inference_chain, report



    
    def select_features_from_dataframe_row(non_normalized_dataframe_row):

        """choose from the non-normalized dataframe, a specific row
        and then transform it in a dataframe row of specific extracted features"""

        selected_features_row = non_normalized_dataframe_row['Shimmer:APQ3', 'Jitter:DDP', 'NHR', 'HNR']
        return selected_features_row
    

    
    def add_prediction_to_selected_features(selected_features_row, prediction):
        """add the value prediction to a dataframe row"""

        selected_features_row['Prediction'] = prediction
        return selected_features_row








