
from logistic_regression import LogisticRegressionModel,LinearRegressionModel

class GradientBoostingModel:


    def __init__(self, num_weak_learners):

        #instantiating the weak learners list
        self._weak_learners = []

        #need to have min one weak learner, otherwise
        #it is not an ensemble model
        if num_weak_learners < 2:
            raise Exception

        else:
            for i in range(num_weak_learners):
                #as first model, we instantiate the logistic regression model
                if i == 0:
                    self._weak_learners.append(LogisticRegressionModel(LinearRegressionModel()))
                #for the other, next to the first one, we instantiate in sequence the other weak learners (lin regression models)
                else:
                    self._weak_learners.append(LinearRegressionModel())

                    


    def predict(self, input_features):

        total_logit_sum = 0

        #iterating over the list of weak learners
        for i in range(len(self._weak_learners)):
             
             #for the first weak learner (the logistic one), we take its linear regression model
             #and use it to predict the non-sigmoided value (logit one)
            if i == 0:
                total_logit_sum += self._weak_learners[i].get_regression_model().predict(input_features)
            #for the other weak learners (lin reg model), we predict the target value,
            #using simply its parameters and the given input features
            else:
                total_logit_sum += self._weak_learners[i].predict(input_features)





    #predict : passo dal primo regressore logistico
    #uso valore predetto, e ci aggiungo tutte le predizioni
    #in sequenza degli altri regressori lineari, dopo faccio passare 
    #per funzione sigmoide


    #training : 

    pass

