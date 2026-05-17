import numpy as np

def compute_metrics(y_true, y_pred):
    tp, fp = np.sum((y_pred == 1) & (y_true == 1)), np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    prec, rec = tp / (tp + fp + 1e-15), tp / (tp + fn + 1e-15)
    return {"precision": prec, "recall": rec, "f1": 2 * prec * rec / (prec + rec + 1e-15)}

def compute_roc_auc(y_true, y_prob):
    desc_idx = np.argsort(y_prob)[::-1]
    y_sorted = y_true[desc_idx]
    pos_total, neg_total = np.sum(y_true == 1), np.sum(y_true == 0)
    tpr_list, fpr_list, tp, fp = [0.0], [0.0], 0, 0
    for yt in y_sorted:
        if yt == 1:
            tp += 1
        else:
            fp += 1
        tpr_list.append(tp / pos_total)
        fpr_list.append(fp / neg_total)
    tpr, fpr = np.array(tpr_list), np.array(fpr_list)
    return fpr, tpr, np.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1]) / 2)
