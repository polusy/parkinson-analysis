import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit


class DataNormalizer:

    @staticmethod
    def normalize_data(csv_raw_training_data, csv_raw_test_data):


        columns_name_list = ['name', 'status']


        raw_training_dataframe = pd.read_csv(csv_raw_training_data)
        raw_test_dataframe = pd.read_csv(csv_raw_test_data)

        #dropping the predefined columns list from both dataframes
        modified_training_dataframe = DataNormalizer.drop_dataframe_columns(raw_training_dataframe, columns_name_list)
        modified_test_dataframe = DataNormalizer.drop_dataframe_columns(raw_test_dataframe, columns_name_list)


        #iterating over the feature columns of the dataframe
        for column in modified_test_dataframe:

            #first, updating the raw test dataframe with normalized values, with standard deviation and mean derived
            #from raw training dataframe
            modified_test_dataframe[column] = (modified_test_dataframe[column] - modified_training_dataframe[column].mean())/modified_training_dataframe[column].std()

            #in order not to leak any test set informations to the training set
            #we update separately the new normalized values of the training set
            modified_training_dataframe[column] = (modified_training_dataframe[column] - modified_training_dataframe[column].mean())/modified_training_dataframe[column].std() 


        #readding previously dropped dataframe columns
        DataNormalizer.add_dataframe_columns(raw_test_dataframe, modified_test_dataframe, columns_name_list)
        DataNormalizer.add_dataframe_columns(raw_training_dataframe, modified_training_dataframe, columns_name_list)


       
        #convert dataframe to csv and store them in the data folder as
        #normalized_parkinsons_training and normalized_parkinsons_test
        DataNormalizer.convert_dataframe_to_csv(modified_test_dataframe, 'data/normalized_parkinsons_test.data')
        DataNormalizer.convert_dataframe_to_csv(modified_training_dataframe, 'data/normalized_parkinsons_training.data')










    def add_dataframe_columns(dataframe1, dataframe2, columns_name_list):

        """the method adds the columns existing in dataframe1, correspondent to the columns_name_list, 
        to dataframe2"""
 
        extracted_dataframe_columns = []

        #iterate over the columns name in the columns name list given as parameter
        #and add the correspondent column (existing in dataframe1) to the list of extracted columns
        for column in columns_name_list:
            extracted_dataframe_columns.append(dataframe1[column])


        #add each exctracted column (non-features columns) to the new dataframe
        #adding each extracted column in position 0, so we have non features column
        #at the beginning of the dataframe
        for extracted_column in extracted_dataframe_columns:
            dataframe2.insert(0, extracted_column.name, extracted_column)
            





    def drop_dataframe_columns(dataframe, columns_list):

        #dropping status column and name column (string)
        #cause we do not want status to be normalized
        modified_dataframe = dataframe.drop(columns = columns_list)


        #return a dataframe without columns declared in the columns list
        return modified_dataframe



    
    def convert_dataframe_to_csv(dataframe, path):

        
        #converting the new normalized dataframe to a csv
        #and store the new csv file in the data folder
        filepath = Path(path)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        dataframe.to_csv(filepath)





class DataSplitter:


    def split(raw_csv_data, test_size):

        #better implementation could have passed to split function, the ratio between
        #number of training rows and total rows number in the raw csv file

        #rescuing the complete dataframe 
        raw_dataframe = pd.read_csv(raw_csv_data)

        #creating the shuffle split using a random state (known seed)
        #the train size
        gss = GroupShuffleSplit(n_splits=1, train_size= (1-test_size), random_state=42)

        #we take alle the values from the name column
        #then we split the values in two pieces
        # - the first part (patient id)
        # - the second part (sample number from the same patient)
        #then we take the first part to create groups
        groups = raw_dataframe['name'].str.rsplit('_', n=1).str[0]

        #using the previous split and the groups generated (set of rows
        #with samples grouped by the same patient id)
        train_idx, test_idx = next(gss.split(raw_dataframe, groups=groups))

        #creating new dataframes, with the generated splits
        training_dataframe = raw_dataframe.iloc[train_idx]
        test_dataframe = raw_dataframe.iloc[test_idx]


        #converting the new exctracted dataframes to different csv file in data folder
        DataNormalizer.convert_dataframe_to_csv(training_dataframe, "data/raw_parkinsons_training.data")
        DataNormalizer.convert_dataframe_to_csv(test_dataframe, "data/raw_parkinsons_test.data")





