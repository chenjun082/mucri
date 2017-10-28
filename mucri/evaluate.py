'''
Created on 2017-2-19

@author: Jun Chen
'''

def get_acc(x_pred, x_true):
    """
    Compute the Accuracy score
    """
    assert(len(x_pred) == len(x_true) and len(x_pred) > 0)
    corrects = 0
    for i, p in enumerate(x_pred):
        q = x_true[i]
        assert((p[0] == q[0] and p[1] == q[1]) or (p[0] == q[1] and p[1] == q[0]))
        if p[0] == q[0] and p[1] == q[1]: corrects += 1
    return float(corrects) / len(x_pred)


def get_auc(x_pred, x_true):
    """
    Compute the AUC score
    """
    assert(len(x_pred) == len(x_true) and len(x_pred) > 0)
    tp, fp, p, n = 0, 0, 0, 0
    for vp, vt in zip(x_pred, x_true):
        assert((vp[0] == vt[0] and vp[1] == vt[1]) or (vp[0] == vt[1] and vp[1] == vt[0]))
        if vt[0] < vt[1]:
            p += 1
            if vp[0] < vp[1]: tp += 1
        else:
            n += 1
            if vp[0] < vp[1]: fp += 1
    if p > 0 and n > 0:
        return (1 + float(tp) / p - float(fp) / n) / 2
    elif p > 0 and n == 0:
        return (1 + float(tp) / p) / 2
    elif p == 0 and n > 0:
        return (1 - float(fp) / n) / 2
    else: return .5