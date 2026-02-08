
class KFoldCV:

    #per k-fold, decidere il numero di fold, su 42 pazienti,
    #togliendo 10 pazienti, avremmo 32 pazienti per il training set, su cui
    #effettuare tutte le validations.

    #faccio 7 fold da 5 pazienti, ogni fold è dato da 6*5 righe di samples
    #individuo primo folder (conservo indici con distanza (6*5) - 1 righe),
    #e faccio training sul restante, poi proseguo.
    pass