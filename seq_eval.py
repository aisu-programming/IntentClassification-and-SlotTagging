from seqeval.scheme import IOB2
classification_report(y_true, y_pred, mode='strict', scheme=IOB2)