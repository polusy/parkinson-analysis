/*KB FACTS*/

/*symptoms associated with parkinson, kb facts*/
symptom_of(vocal_tremor, parkinson).
symptom_of(breathy_voice, parkinson).
symptom_of(hoarse_voice, parkinson).



/*conditions on features that imply specific symptoms*/

indicates(high_jitter, vocal_tremor).
indicates(high_shimmer, vocal_tremor).
indicates(high_nhr, breathy_voice).
indicates(low_hnr, hoarse_voice).

/*weighting different symptoms with different weights as facts*/
symptom_weight(vocal_tremor, 3).
symptom_weight(hoarse_voice, 2).
symptom_weight(breathy_voice, 2).




/*KB RULES*/

/*in the validation test, we will instantiate different values as facts
jitter_value(marco, 1.50) for example, and then we will infer has_feature(marco, high_jitter)*/


/*rules on input features values that imply its position in the scale (high,low)*/
has_feature(Patient, high_jitter) :- jitter_value(Patient, Value), Value >  1.04
has_feature(Patient, high_shimmer) :- shimmer_value(Patient, Value), Value > 0.068
has_feature(Patient, high_nhr) :- nhr_value(Patient, Value), Value > 0.19
has_feature(Patient, low_hnr) :- hnr_value(Patient, Value), Value < 20


/*general symptom inference*/
has_symptom(Patient, Symptom) :- has_feature(Patient, Feature), indicates(Feature, Symptom)


/*rules to count patient symptoms for parkinson disease*/
count_symptoms(Patient, Disease, Count) :- findall(Symptom, (has_symptom(Patient, Symptom),
                                            symptom_of(Symptom,Disease)),  Symptoms), 
                                            list_to_set(Symptoms, UniqueSymptoms), 
                                            length(UniqueSymptoms, Count)


/*weighted score of disease diagnosis severity*/
weighted_diagnosis(Patient, Disease, Score) :-  findall(Symptom, (has_symptom(Patient, Symptom), symptom_of(Symptom, Disease)), Symptoms),
                                                list_to_set(Symptoms,UniqueSymptoms),
                                                findall(Weight, (member(Symptom, UniqueSymptoms), symptom_weight(Symptom, Weight)), Weight_List),
                                                sum_list(Weight_List, Score)



/*severity diagnosis by symptoms weighted count*/
diagnosis(Patient, Disease, severe) :- weighted_diagnosis(Patient, Disease, Score), Score >= 5, 
diagnosis(Patient, Disease, moderate) :- weighted_diagnosis(Patient, Disease, Score), Score >= 3, \+ Score >= 5
diagnosis(Patient, Disease, mild) :- weighted_diagnosis(Patient, Disease, Score), Score >= 1, \+ Score >= 3
diagnosis(Patient, Disease, none) :- weighted_diagnosis(Patient, Disease, Score), Score = 0



/*ensemble model validation for coherent cases*/
coherent_prediction(Patient, Prediction, Disease) :- (diagnosis(Patient, Disease, severe); diagnosis(Patient, Disease, moderate)), Prediction = 1
coherent_prediction(Patient, Prediction, Disease) :- (diagnosis(Patient, Disease, mild); diagnosis(Patient, Disease, none)), Prediction = 0


/*ensemble model validation for incoherent cases (falses positives and false negatives)*/
critical_false_negative(Patient, Prediction, Disease) :- diagnosis(Patient, Disease, severe), Prediction = 0
critical_false_positive(Patient, Prediction, Disease) :- diagnosis(Patient, Disease, none), Prediction = 1



/*conflictual features findings*/
inconsistencies_warning(Patient, Disease) :- has_feature(Patient, high_jitter),
                                            \+ has_feature(Patient, high_shimmer),
                                            \+ has_feature(Patient, high_nhr)


/*Integrating all the rules in one validator*/
validation(Patient, Prediction, Disease, Result) :- ( (critical_false_negative(Patient, Prediction, Disease); critical_false_positive(Patient, Prediction, Disease)) -> Result = critical_error;
                                        inconsistencies_warning(Patient, Disease) -> Result = conflictual_data;
                                        coherent_prediction(Patient, Prediction, Disease) -> Result = coherent;
                                        Result = warning
                                        )    





