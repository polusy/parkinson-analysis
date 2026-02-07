import pandas as pd
from sklearn.preprocessing import StandardScaler
from pathlib import Path


class DataNormalizer:

    def normalize_data(self, csv_raw_training_data, csv_raw_test_data):


        columns_name_list = ['name', 'status']


        raw_training_dataframe = pd.read_csv(csv_raw_training_data)
        raw_test_dataframe = pd.read_csv(csv_raw_test_data)

        #dropping the predefined columns list from both dataframes
        modified_training_dataframe = self.drop_dataframe_columns(raw_training_dataframe, columns_name_list)
        modified_test_dataframe = self.drop_dataframe_columns(raw_test_dataframe, columns_name_list)


        for column in modified_test_dataframe:

            #first, updating the raw test dataframe with normalized values, with standard deviation and mean derived
            #from raw training dataframe
            modified_test_dataframe[column] = (modified_training_dataframe[column] - modified_training_dataframe[column].mean())/modified_training_dataframe[column].std()

            #in order not to leak any test set informations to the training set
            #we update separately the new normalized values of the training set
            modified_training_dataframe[column] = (modified_training_dataframe[column] - modified_training_dataframe[column].mean())/modified_training_dataframe[column].std() 


        #readding previously dropped dataframe columns
        self.add_dataframe_columns(raw_test_dataframe, modified_test_dataframe, columns_name_list)
        self.add_dataframe_columns(raw_training_dataframe, modified_training_dataframe, columns_name_list)


        #todo --> convert dataframe to csv and store them in the data folder as
        #normalized_parkinsons_training and normalized_parkinsons_test










    def add_dataframe_columns(dataframe1, dataframe2, columns_name_list):

        """the method adds the columns existing in dataframe1, correspondent to the columns_name_list, 
        to dataframe2"""
 
        extracted_dataframe_columns = []

        #iterate over the columns name in the columns name list given as parameter
        #and add the correspondent column (existing in dataframe1) to the list of extracted columns
        for column in columns_name_list:
            extracted_dataframe_columns.append(dataframe1[column])


        #add each exctracted column in the new dataframe
        for extracted_column in extracted_dataframe_columns:
            dataframe2.insert(1, extracted_column.name, extracted_column)
            





    def drop_dataframe_columns(dataframe, columns_list):

        #dropping status column and name column (string)
        #cause we do not want status to be normalized
        modified_dataframe = dataframe.drop(columns = columns_list)


        #return a dataframe without columns declared in the columns list
        return modified_dataframe




    def convert_dataframe_to_csv(self, dataframe, path):

        
        #converting the new normalized dataframe to a csv
        #and store the new csv file in the data folder
        filepath = Path(path)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        dataframe.to_csv(filepath)





class DataSplitter:
    #implementare uno splitter di dati, in dati di test e in 
    #dati di training, restituendo file in formato CSV, nel
    #path /parkinson-analysis/data

    #per dividere correttamente i dati, e non fare data leakage,
    #contare le righe in multipli di 6 (numero di sample per paziente)
    #quindi prendere un multiplo di 6 righe.

    pass


