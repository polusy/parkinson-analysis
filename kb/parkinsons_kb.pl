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
is_subtype_of(vocal_tremor, dysphonia).
is_subtype_of(breathy_voice, dysphonia).
is_subtype_of(vocal_instability, dysphonia).
is_subtype_of(hoarse_voice, dysphonia).




/*KB RULES*/


/*rules on input features values that imply its position in the scale (high,low)*/
has_feature(Patient, high_jitter) :- jitter_value(Patient, Value), Value >  0.0057.
has_feature(Patient, high_shimmer) :- shimmer_value(Patient, Value), Value > 0.030.
has_feature(Patient, high_nhr) :- nhr_value(Patient, Value), Value > 0.015.
has_feature(Patient, low_hnr) :- hnr_value(Patient, Value), Value < 20.0.


/*infer patient's vocal profile, through feature analysis*/
vocal_profile(Patient, tremor_dominant)  :- has_feature(Patient, high_jitter), has_feature(Patient, high_shimmer), \+ has_feature(Patient, low_hnr).
vocal_profile(Patient, hoarseness_dominant) :- has_feature(Patient, low_hnr), \+ has_feature(Patient, high_jitter).
vocal_profile(Patient, mixed_dysphonia)  :- has_feature(Patient, high_jitter), has_feature(Patient, low_hnr).
vocal_profile(Patient, breath_dominant)  :- has_feature(Patient, high_nhr), \+ has_feature(Patient, high_jitter).


/*count patient features and then assert the kb_confidence
based on feature count (e.g. feature confidence >= 3, then high confidence)*/
feature_count(Patient, Count) :- findall(F, has_feature(Patient, F), Fs), length(Fs, Count).
kb_confidence(Patient, high)   :- feature_count(Patient, Count), Count >= 3.
kb_confidence(Patient, medium) :- feature_count(Patient, Count), Count =:= 2.
kb_confidence(Patient, low)    :- feature_count(Patient, Count), Count =:= 1.


/*general symptom inference*/
has_symptom(Patient, Symptom) :- has_feature(Patient, Feature), indicates(Feature, Symptom).

/*ground case*/
has_symptom_or_super(Patient, Higher_Symptom) :- has_symptom(Patient, Higher_Symptom).

/*inference of general type of symptom given a sub type*/
has_symptom_or_super(Patient, Higher_Symptom) :- has_symptom_or_super(Patient, Intermediate_Symptom), is_subtype_of(Intermediate_Symptom, Higher_Symptom).


/*understanding if a symptom is related, ascending the hierarchy, to the disease (vertex)*/
symptom_related_to_disease(Symptom, Disease) :- symptom_of(Symptom, Disease).
symptom_related_to_disease(Symptom, Disease) :- is_subtype_of(Symptom, Super_Symptom), symptom_of(Super_Symptom, Disease).


/*count distinct symptoms detected*/
symptom_count(Patient, Disease, Count) :- findall(Symptom, (has_symptom(Patient, Symptom), symptom_related_to_disease(Symptom, Disease)), Symptoms),
                                          list_to_set(Symptoms, UniqueSymptoms),
                                          length(UniqueSymptoms, Count).






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



/*diagnosis reliability: a diagnosis is reliable only if supported by at least 2 distinct symptoms*/
reliable_diagnosis(Patient, Disease) :- symptom_count(Patient, Disease, Count), Count >= 2.
unreliable_diagnosis(Patient, Disease) :- symptom_count(Patient, Disease, Count), Count < 2,
                                          weighted_diagnosis(Patient, Disease, Score), Score > 0.



/*ensemble model validation for coherent cases*/
coherent_prediction(Patient, Prediction, Disease) :- (diagnosis(Patient, Disease, severe); diagnosis(Patient, Disease, moderate)), Prediction = 1.
coherent_prediction(Patient, Prediction, Disease) :- (diagnosis(Patient, Disease, mild); diagnosis(Patient, Disease, none)), Prediction = 0.


/*ensemble model validation for incoherent cases (falses positives and false negatives)*/
critical_false_negative(Patient, Prediction, Disease) :- diagnosis(Patient, Disease, severe), Prediction = 0.
critical_false_positive(Patient, Prediction, Disease) :- diagnosis(Patient, Disease, none), Prediction = 1.


/*diagnosing model error severity, based on type of error (false negative in this case),
and patient's vocal profile*/
critical_error_severity(Patient, Prediction, Disease, high) :-
    critical_false_negative(Patient, Prediction, Disease),
    vocal_profile(Patient, tremor_dominant).

critical_error_severity(Patient, Prediction, Disease, medium) :-
    critical_false_negative(Patient, Prediction, Disease),
    vocal_profile(Patient, breath_dominant).



/*INCONSISTENCY PATTERNS*/

/*Pattern 1: high jitter isolated - vocal tremor suspected but assence by shimmer or nhr*/
inconsistency_isolated_jitter(Patient) :- has_feature(Patient, high_jitter),
                                          \+ has_feature(Patient, high_shimmer),
                                          \+ has_feature(Patient, high_nhr).

/*Pattern 2: low hnr isolated - hoarseness suspected but assence of other vocal anomaly*/
inconsistency_isolated_hnr(Patient) :- has_feature(Patient, low_hnr),
                                       \+ has_feature(Patient, high_jitter),
                                       \+ has_feature(Patient, high_shimmer),
                                       \+ has_feature(Patient, high_nhr).

/*Pattern 3: high nhr isolated - breathy voice suspected but assence of other vocal anomaly*/
inconsistency_isolated_nhr(Patient) :- has_feature(Patient, high_nhr),
                                       \+ has_feature(Patient, high_jitter),
                                       \+ has_feature(Patient, high_shimmer),
                                       \+ has_feature(Patient, low_hnr).

/*aggregate inconsistency check: any of the three patterns triggers a warning*/
inconsistencies_warning(Patient) :- inconsistency_isolated_jitter(Patient).
inconsistencies_warning(Patient) :- inconsistency_isolated_hnr(Patient).
inconsistencies_warning(Patient) :- inconsistency_isolated_nhr(Patient).




/*Integrating all the rules in one validator*/
/*Priority: critical errors > unreliable diagnosis warning > inconsistent features > coherent/warning*/
validation(Patient, Prediction, Disease, Result) :-
        ( critical_false_negative(Patient, Prediction, Disease),
        critical_error_severity(Patient, Prediction, Disease, high)
    ->  Result = severe_critical_error
    ;   critical_false_negative(Patient, Prediction, Disease),
        critical_error_severity(Patient, Prediction, Disease, medium)
    ->  Result = moderate_critical_error
    ;   critical_false_negative(Patient, Prediction, Disease)
    ->  Result = critical_error
    ;   critical_false_positive(Patient, Prediction, Disease)
    ->  Result = critical_error
    ;   unreliable_diagnosis(Patient, Disease), \+ inconsistencies_warning(Patient)
    ->  Result = unreliable_evidence
    ;   inconsistencies_warning(Patient)
    ->  Result = conflictual_data
    ;   coherent_prediction(Patient, Prediction, Disease)
    ->  Result = coherent
    ;   Result = warning
    ).


/*include the kb_confidence based on features count, to weight inferred result
with the kb confidence measure*/
weighted_validation(Patient, Prediction, Disease, Result, Confidence) :-
    validation(Patient, Prediction, Disease, Result),
    kb_confidence(Patient, Confidence).



/*Explaining KB inference, showing weight assigned to each symptom*/
explain(Patient, Prediction, Disease, Evidence) :- findall(Symptom-Weight, (has_symptom(Patient, Symptom), symptom_related_to_disease(Symptom, Disease), symptom_weight(Symptom, Weight)), Evidence).

/*using a List to store each symptom, super symptom relation*/
inference_chain(Patient, Prediction, Disease, Chain) :- findall(Symptom-SuperSymptom, (has_symptom(Patient, Symptom), is_subtype_of(Symptom, SuperSymptom), symptom_related_to_disease(Symptom, Disease)), Chain).

/*Diagnostic report*/
report(Patient, Prediction, Disease, Evidence, Result, Message) :- validation(Patient, Prediction, Disease, Result),
                                                                explain(Patient, Prediction, Disease, Evidence),
                                                                format(atom(Message), "Stato: ~w. Evidenze : ~w. Predizione : ~w.", [Result, Evidence, Prediction]).











