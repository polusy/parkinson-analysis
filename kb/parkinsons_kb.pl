/*KB FACTS*/

/*symptom associated with parkinson*/
symptom_of(dysphonia, parkinson).


/*conditions on features that imply specific symptoms*/

indicates(high_jitter, vocal_tremor).
indicates(high_shimmer, vocal_tremor).
indicates(high_nhr, breathy_voice).
indicates(low_hnr, hoarse_voice).


/*weighting different symptoms with different weights as facts*/
symptom_weight(vocal_tremor, 3).
symptom_weight(hoarse_voice, 2).
symptom_weight(breathy_voice, 2).


/*hierarchy of symptoms types*/
is_subtype_of(vocal_tremor, vocal_instability).
is_subtype_of(breathy_voice, dysphonia).
is_subtype_of(vocal_instability, dysphonia).
is_subtype_of(hoarse_voice, dysphonia).

/*general subtype inference, ascending hierarchy of symptoms*/
is_subtype_of(BaseSymptom, SuperSymptom) :- is_subtype_of(BaseSymptom, IntermediateSymptom), is_subtype_of(IntermediateSymptom, SuperSymptom).








/*KB RULES*/

/*in the validation test, we will instantiate different values as facts
jitter_value(marco, 1.50) for example, and then we will infer has_feature(marco, high_jitter)*/


/*rules on input features values that imply its position in the scale (high,low)*/
has_feature(Patient, high_jitter) :- jitter_value(Patient, Value), Value >  1.04.
has_feature(Patient, high_shimmer) :- shimmer_value(Patient, Value), Value > 0.068.
has_feature(Patient, high_nhr) :- nhr_value(Patient, Value), Value > 0.19.
has_feature(Patient, low_hnr) :- hnr_value(Patient, Value), Value < 20.


/*general symptom inference*/
has_symptom(Patient, Symptom) :- has_feature(Patient, Feature), indicates(Feature, Symptom).

/*ground case*/
has_symptom_or_super(Patient, Higher_Symptom) :- has_symptom(Patient, Higher_Symptom).

/*inference of general type of symptom given a sub type*/
has_symptom_or_super(Patient, Higher_Symptom) :- has_symptom_or_super(Patient, Intermediate_Symptom), is_subtype_of(Intermediate_Symptom, Higher_Symptom).


/*understanding if a symptom is related, ascending the hierarchy, to the disease (vertex)*/
symptom_related_to_disease(Symptom, Disease) :- symptom_of(Symptom, Disease).
symptom_related_to_disease(Symptom, Disease) :- is_subtype_of(Symptom, Super_Symptom), symptom_of(Super_Symptom, Disease).





/*weighted score of disease diagnosis severity*/
weighted_diagnosis(Patient, Disease, Score) :-  findall(Symptom, (has_symptom(Patient, Symptom), symptom_related_to_disease(Symptom, Disease)), Symptoms),
                                                list_to_set(Symptoms,UniqueSymptoms),
                                                findall(Weight, (member(Symptom, UniqueSymptoms), symptom_weight(Symptom, Weight)), Weight_List),
                                                sum_list(Weight_List, Score).



/*severity diagnosis by symptoms weighted count*/
diagnosis(Patient, Disease, severe) :- weighted_diagnosis(Patient, Disease, Score), Score >= 5.
diagnosis(Patient, Disease, moderate) :- weighted_diagnosis(Patient, Disease, Score), Score >= 3, Score < 5.
diagnosis(Patient, Disease, mild) :- weighted_diagnosis(Patient, Disease, Score), Score >= 1, Score < 3.
diagnosis(Patient, Disease, none) :- weighted_diagnosis(Patient, Disease, Score), Score = 0.



/*ensemble model validation for coherent cases*/
coherent_prediction(Patient, Prediction, Disease) :- (diagnosis(Patient, Disease, severe); diagnosis(Patient, Disease, moderate)), Prediction = 1.
coherent_prediction(Patient, Prediction, Disease) :- (diagnosis(Patient, Disease, mild); diagnosis(Patient, Disease, none)), Prediction = 0.


/*ensemble model validation for incoherent cases (falses positives and false negatives)*/
critical_false_negative(Patient, Prediction, Disease) :- diagnosis(Patient, Disease, severe), Prediction = 0.
critical_false_positive(Patient, Prediction, Disease) :- diagnosis(Patient, Disease, none), Prediction = 1.



/*conflictual features findings*/
inconsistencies_warning(Patient, Disease) :- has_feature(Patient, high_jitter),
                                            \+ has_feature(Patient, high_shimmer),
                                            \+ has_feature(Patient, high_nhr).






/*Integrating all the rules in one validator*/
validation(Patient, Prediction, Disease, Result) :- ( (critical_false_negative(Patient, Prediction, Disease); critical_false_positive(Patient, Prediction, Disease)) -> Result = critical_error;
                                        inconsistencies_warning(Patient, Disease) -> Result = conflictual_data;
                                        coherent_prediction(Patient, Prediction, Disease) -> Result = coherent;
                                        Result = warning.
                                        )    



/*Explaining KB inference, showing weight assigned to each symptom*/
explain(Patient, Prediction, Disease, Evidence) :- findall(Symptom-Weight, (has_symptom(Patient, Symptom), symptom_related_to_disease(Symptom, Disease), symptom_weight(Symptom, Weight)), Evidence).

/*using a List to store each symptom, super symptom relation*/
inference_chain(Patient, Prediction, Disease, Chain) :- findall(Symptom-SuperSymptom, (has_symptom(Patient, Symptom), is_subtype_of(Symptom, SuperSymptom), symptom_related_to_disease(Symptom, Disease)), Chain).

/*Diagnostic report*/
report(Patient, Prediction, Disease, Evidence, Result, Message) :- validation(Patient, Prediction, Disease, Result),
                                                                explain(Patient, Prediction, Disease, Evidence),
                                                                format(atom(Message), "Stato: ~w. Evidenze : ~w. Predizione : ~w.", [Result, Evidence, Prediction]).











