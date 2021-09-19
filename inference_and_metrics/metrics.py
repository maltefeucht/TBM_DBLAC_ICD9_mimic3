import numpy as np
import torch as th
from sklearn.metrics import roc_curve, auc
import torchmetrics

#######################################################################################
# METRICS PACKAGE: This file contains metrics for multi-label classification evaluation
#######################################################################################

########################################
# PREPARE: Prepare outputs for evaluation metrics
########################################

def prepare_outputs(logits, target_labels):
    """
    Transform logits into one-hot-encoded vector for classification metric computation
    Parameters
    ----------
    logits : tensor (batch_size x target_label_classes)
        predicted logits outputed by classifier for every label class
    target_labels : tensor (batch_size x target_label_classes)
        ground truth of class labels

    Returns
    -------
    y_pred: tensor (batch_size x target_label_classes) as float
        transformed predicted logits in one-hot-encoded vector
    y_true : tensor (batch_size x target_label_classes) as int
        ground truth of class labels
    """
    #y_pred = (th.sigmoid(logits) > 0.5).float()
    y_pred = (th.sigmoid(logits)).float()
    y_true = target_labels.int()
    assert y_true.shape == y_pred.shape, "y_pred must be same shape as y_true"
    return y_pred, y_true

def prepare_outputs_caml(logits, target_labels):
    y_pred = (th.sigmoid(logits) > 0.5).int().cpu().detach().numpy()
    y_pred_raw = (th.sigmoid(logits)).float().cpu().detach().numpy()
    y_true = target_labels.int().cpu().detach().numpy()
    y_true_mic = y_true.ravel()
    y_pred_mic = y_pred.ravel()
    assert y_true.shape == y_pred.shape == y_pred_raw.shape, "y_pred must be same shape as y_true"
    assert len(y_true_mic) == len(y_pred_mic), "y_pred must be same shape as y_true"
    return y_pred, y_pred_raw, y_true, y_true_mic, y_pred_mic

def prepare_outputs_inference(logits, target_labels):
    """
    Transform logits into one-hot-encoded vector for classification metric computation
    Parameters
    ----------
    logits : tensor (batch_size x target_label_classes)
        predicted logits outputed by classifier for every label class
    target_labels : tensor (batch_size x target_label_classes)
        ground truth of class labels

    Returns
    -------
    y_pred: tensor (batch_size x target_label_classes) as float
        transformed predicted logits in one-hot-encoded vector
    y_true : tensor (batch_size x target_label_classes) as float
        ground truth of class labels
    """
    y_pred = (th.sigmoid(logits) > 0.5).int()
    y_true = target_labels.int()
    assert y_true.shape == y_pred.shape, "y_pred must be same shape as y_true"
    return y_pred, y_true

###############################################################################################
# CAML: micro- and macro metrics as implemented by caml- paper for sanity checking torchmetrics
###############################################################################################
"""
Inputs:
yhat: binary predictions matrix 
yhat_raw: prediction scores matrix (floats)
y: binary ground truth matrix
y_mic: raveled y matrix (np.ravel())
k: for @k metrics

Outputs:
dict holding relevant metrics
"""

def union_size(yhat, y, axis):
    #axis=0 for label-level union (macro). axis=1 for instance-level
    return np.logical_or(yhat, y).sum(axis=axis).astype(float)

def intersect_size(yhat, y, axis):
    #axis=0 for label-level union (macro). axis=1 for instance-level
    return np.logical_and(yhat, y).sum(axis=axis).astype(float)

# MACRO metrics
def macro_accuracy(yhat, y):
    num = intersect_size(yhat, y, 0) / (union_size(yhat, y, 0) + 1e-10)
    return np.mean(num)

def macro_precision(yhat, y):
    num = intersect_size(yhat, y, 0) / (yhat.sum(axis=0) + 1e-10)
    return np.mean(num)

def macro_recall(yhat, y):
    num = intersect_size(yhat, y, 0) / (y.sum(axis=0) + 1e-10)
    return np.mean(num)

def macro_f1(yhat, y):
    prec = macro_precision(yhat, y)
    rec = macro_recall(yhat, y)
    if prec + rec == 0:
        f1 = 0.
    else:
        f1 = 2*(prec*rec)/(prec+rec)
    return f1

# MICRO metrics
def micro_accuracy(yhatmic, ymic):
    return intersect_size(yhatmic, ymic, 0) / union_size(yhatmic, ymic, 0)

def micro_precision(yhatmic, ymic):
    return intersect_size(yhatmic, ymic, 0) / yhatmic.sum(axis=0)

def micro_recall(yhatmic, ymic):
    return intersect_size(yhatmic, ymic, 0) / ymic.sum(axis=0)

def micro_f1(yhatmic, ymic):
    prec = micro_precision(yhatmic, ymic)
    rec = micro_recall(yhatmic, ymic)
    if prec + rec == 0:
        f1 = 0.
    else:
        f1 = 2*(prec*rec)/(prec+rec)
    return f1

#AUC metrics
def auc_metrics(yhat_raw, y, ymic):
    if yhat_raw.shape[0] <= 1:
        return
    fpr = {}
    tpr = {}
    roc_auc = {}
    #get AUC for each label individually
    relevant_labels = []
    auc_labels = {}
    for i in range(y.shape[1]):
        #only if there are true positives for this label
        if y[:,i].sum() > 0:
            fpr[i], tpr[i], _ = roc_curve(y[:,i], yhat_raw[:,i], pos_label=1)
            if len(fpr[i]) > 1 and len(tpr[i]) > 1:
                auc_score = auc(fpr[i], tpr[i])
                if not np.isnan(auc_score):
                    auc_labels["auc_%d" % i] = auc_score
                    relevant_labels.append(i)
    #macro-AUC: just average the auc scores
    aucs = []
    for i in relevant_labels:
        aucs.append(auc_labels['auc_%d' % i])
    roc_auc['auc_macro'] = np.mean(aucs)

    #micro-AUC: just look at each individual prediction
    yhatmic = yhat_raw.ravel()
    fpr["micro"], tpr["micro"], _ = roc_curve(ymic, yhatmic)
    roc_auc["auc_micro"] = auc(fpr["micro"], tpr["micro"])
    return roc_auc

def precision_at_k(yhat_raw, y, k):
    '''
    This metric equals to torchmetrics.Precision(num_classes=num_classes, threshold=0.5, average='macro', top_k=5)
    '''
    #num true labels in top k predictions / k
    sortd = np.argsort(yhat_raw)[:,::-1]
    topk = sortd[:,:k]
    #get precision at k for each example
    vals = []
    for i, tk in enumerate(topk):
        if len(tk) > 0:
            num_true_in_top_k = y[i,tk].sum()
            denom = len(tk)
            vals.append(num_true_in_top_k / float(denom))
    return np.mean(vals)

#######################################################################
#COMPUTE ALL METRICS: all metrics needed for evaluation are calculated
#######################################################################

def all_metrics(y_pred, y_true, logits,  num_labels, top_k):
    print('Computing all metrics...')
    device = th.device('cuda' if th.cuda.is_available() else 'cpu')
    # move tensors if necessary
    y_pred.to(device), y_pred.to(device), logits.to(device)
    # Instantiate all metrics
    accuracy = torchmetrics.Accuracy().to(device)
    accuracy_subset = torchmetrics.Accuracy(subset_accuracy=True).to(device)
    precision_macro = torchmetrics.Precision(num_classes=num_labels, threshold=0.5, average='macro').to(device)
    precision_micro = torchmetrics.Precision(threshold=0.5, average='micro').to(device)
    precision_macro_topk = torchmetrics.Precision(num_classes=num_labels, threshold=0.5, average='macro', top_k=top_k).to(device)
    precision_micro_topk = torchmetrics.Precision(threshold=0.5, average='micro', top_k=5).to(device)
    recall_macro = torchmetrics.Recall(num_classes=num_labels, threshold=0.5, average='macro').to(device)
    recall_micro = torchmetrics.Recall(threshold=0.5, average='micro').to(device)
    f_1_macro = torchmetrics.classification.F1(num_classes=num_labels, threshold=0.5, average='macro').to(device)
    f_1_micro = torchmetrics.classification.F1(num_classes=num_labels, threshold=0.5, average='micro').to(device)

    # Compute all metrics
    accuracy = accuracy(y_pred, y_true)
    accuracy_subset = accuracy_subset(y_pred, y_true)
    precision_macro = precision_macro(y_pred, y_true)
    precision_micro = precision_micro(y_pred, y_true)
    precision_macro_topk = precision_macro_topk(y_pred, y_true)
    precision_micro_topk = precision_micro_topk(y_pred, y_true)
    recall_macro = recall_macro(y_pred, y_true)
    recall_micro = recall_micro(y_pred, y_true)
    f_1_macro = f_1_macro(y_pred, y_true)
    f_1_micro = f_1_micro(y_pred, y_true)

    # Compute auc-scroes as in CAML paper (and other caml metrics for sanity checking torchmetrics)
    y_pred_caml, y_pred_raw_caml, y_true_caml, y_true_mic_caml, y_pred_mic_caml = prepare_outputs_caml(logits, y_true)
    f_1_macro_caml = macro_f1(y_pred_caml, y_true_caml)
    f_1_micro_caml = micro_f1(y_pred_mic_caml, y_true_mic_caml)
    auc = auc_metrics(y_pred_raw_caml, y_true_caml, y_true_mic_caml)


    all_metrics = {'accuracy': accuracy, 'accuracy_subset': accuracy_subset,'precision_macro': precision_macro, 'precision_micro': precision_micro, 'precision_macro_topk': precision_macro_topk,
                   'precision_micro_topk': precision_micro_topk, 'recall_macro': recall_macro, 'recall_micro': recall_micro, 'f_1_macro': f_1_macro, 'f_1_macro_caml': f_1_macro_caml, 'f_1_micro': f_1_micro, 'f_1_micro_caml':f_1_micro_caml}
    all_metrics.update(auc)
    return all_metrics


###############
#k-fold metrics
###############
def compute_kfold_metrics(results_k_fold, k):
    accuracy = []
    accuracy_subset = []
    auc_macro = []
    auc_micro = []
    f_1_macro = []
    f_1_macro_caml = []
    f_1_micro = []
    f_1_micro_caml = []
    precision_macro = []
    precision_macro_topk = []
    precision_micro = []
    precision_micro_topk = []
    recall_macro = []
    recall_micro = []

    for i in range(k):
        accuracy.append(results_k_fold[i][0]['accuracy'])
        accuracy_subset.append(results_k_fold[i][0]['accuracy_subset'])
        auc_macro.append(results_k_fold[i][0]['auc_macro'])
        auc_micro.append(results_k_fold[i][0]['auc_micro'])
        f_1_macro.append(results_k_fold[i][0]['f_1_macro'])
        f_1_macro_caml.append(results_k_fold[i][0]['f_1_macro_caml'])
        f_1_micro.append(results_k_fold[i][0]['f_1_micro'])
        f_1_micro_caml.append(results_k_fold[i][0]['f_1_micro_caml'])
        precision_macro.append(results_k_fold[i][0]['precision_macro'])
        precision_macro_topk.append(results_k_fold[i][0]['precision_macro_topk'])
        precision_micro.append(results_k_fold[i][0]['precision_micro'])
        precision_micro_topk.append(results_k_fold[i][0]['precision_micro_topk'])
        recall_macro.append(results_k_fold[i][0]['recall_macro'])
        recall_micro.append(results_k_fold[i][0]['recall_micro'])


    accuracy_mean = (th.tensor(accuracy, dtype=th.float)).mean()
    accuracy_std = (th.tensor(accuracy, dtype=th.float)).std()

    accuracy_subset_mean = (th.tensor(accuracy_subset, dtype=th.float)).mean()
    accuracy_subset_std = (th.tensor(accuracy_subset, dtype=th.float)).std()

    auc_macro_mean = (th.tensor(auc_macro, dtype=th.float)).mean()
    auc_macro_std = (th.tensor(auc_macro, dtype=th.float)).std()

    auc_micro_mean = (th.tensor(auc_micro, dtype=th.float)).mean()
    auc_micro_std = (th.tensor(auc_micro, dtype=th.float)).std()

    f_1_macro_mean = (th.tensor(f_1_macro, dtype=th.float)).mean()
    f_1_macro_std = (th.tensor(f_1_macro, dtype=th.float)).std()

    f_1_macro_caml_mean = (th.tensor(f_1_macro_caml, dtype=th.float)).mean()
    f_1_macro_caml_std = (th.tensor(f_1_macro_caml, dtype=th.float)).std()

    f_1_micro_mean = (th.tensor(f_1_micro, dtype=th.float)).mean()
    f_1_micro_std = (th.tensor(f_1_micro, dtype=th.float)).std()

    f_1_micro_caml_mean = (th.tensor(f_1_micro_caml, dtype=th.float)).mean()
    f_1_micro_caml_std = (th.tensor(f_1_micro_caml, dtype=th.float)).std()

    precision_macro_mean = (th.tensor(precision_macro, dtype=th.float)).mean()
    precision_macro_std = (th.tensor(precision_macro, dtype=th.float)).std()

    precision_macro_topk_mean = (th.tensor(precision_macro_topk, dtype=th.float)).mean()
    precision_macro_topk_std = (th.tensor(precision_macro_topk, dtype=th.float)).std()

    precision_micro_mean = (th.tensor(precision_micro, dtype=th.float)).mean()
    precision_micro_std = (th.tensor(precision_micro, dtype=th.float)).std()

    precision_micro_topk_mean = (th.tensor(precision_micro_topk , dtype=th.float)).mean()
    precision_micro_topk_std = (th.tensor(precision_micro_topk , dtype=th.float)).std()

    recall_macro_mean = (th.tensor(recall_macro, dtype=th.float)).mean()
    recall_macro_std = (th.tensor(recall_macro, dtype=th.float)).std()

    recall_micro_mean = (th.tensor(recall_micro, dtype=th.float)).mean()
    recall_micro_std = (th.tensor(recall_micro, dtype=th.float)).std()

    return {'accuracy_mean':accuracy_mean, 'accuracy_std': accuracy_std, 'accuracy_subset_mean':accuracy_subset_mean, 'accuracy_subset_std': accuracy_subset_std,
            'auc_macro_mean':auc_macro_mean, 'auc_macro_std': auc_macro_std, 'auc_micro_mean':auc_micro_mean, 'auc_micro_std': auc_micro_std,
            'f_1_macro_mean': f_1_macro_mean, 'f_1_macro_std': f_1_macro_std, 'f_1_macro_caml_mean':f_1_macro_caml_mean, 'f_1_macro_caml_std': f_1_macro_caml_std,
            'f_1_micro_mean': f_1_micro_mean, 'f_1_micro_std': f_1_micro_std, 'f_1_micro_caml_mean': f_1_micro_caml_mean, 'f_1_micro_caml_std': f_1_micro_caml_std,
            'precision_macro_mean': precision_macro_mean, 'precision_macro_std': precision_macro_std, 'precision_macro_topk_mean': precision_macro_topk_mean, 'precision_macro_topk_std': precision_macro_topk_std,
            'precision_micro_mean': precision_micro_mean, 'precision_micro_std': precision_micro_std, 'precision_micro_topk_mean': precision_micro_topk_mean, 'precision_micro_topk_std': precision_micro_topk_std,
            'recall_macro_mean': recall_macro_mean, 'recall_macro_std': recall_macro_std, 'recall_micro_mean': recall_micro_mean, 'recall_micro_std': recall_micro_std
            }



