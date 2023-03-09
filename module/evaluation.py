import torch
import numpy as np
from sklearn.metrics import classification_report
import torch.nn as nn
import torch.nn.functional as F

emotion_label_policy = {'angry': 0, 'anger': 0,
    'disgust': 1,
    'fear': 2,
    'happy': 3, 'happines': 3, 'happiness': 3, 'excited': 3,
    'sad': 4, 'sadness': 4, 'frustrated': 4,
    'surprise': 5, 'surprised': 5, 
    'neutral': 6}

def argmax_prediction(pred_y, true_y):
    pred_argmax = torch.argmax(pred_y, dim=1).cpu()
    true_y = true_y.cpu()
    return pred_argmax, true_y


def threshold_prediction(pred_y, true_y):
    pred_y = pred_y > 0.5
    return pred_y, true_y


def metrics_report(pred_y, true_y, label, get_dict=False, multilabel=False):
    if multilabel:
        pred_y, true_y = threshold_prediction(pred_y, true_y)
        available_label = sorted(list(set((pred_y == True).nonzero()[:, -1].tolist() + (true_y == True).nonzero()[:, -1].tolist())))
    else:
        pred_y, true_y = argmax_prediction(pred_y, true_y)
        available_label = sorted(list(set(true_y.tolist() + pred_y.tolist())))

    class_name = list(label[available_label])
    if get_dict:
        return classification_report(true_y, pred_y, target_names=class_name, zero_division=0, digits=4, output_dict=True)
    else:
        return classification_report(true_y, pred_y, target_names=class_name, zero_division=0, digits=4)


def metrics_report_for_emo_binary(pred_y, true_y, get_dict=False, multilabel=False):
    if multilabel:
        pred_y, true_y = threshold_prediction(pred_y, true_y)
        available_label = sorted(list(set((pred_y == True).nonzero()[:, -1].tolist() + (true_y == True).nonzero()[:, -1].tolist())))
    else:
        pred_y, true_y = argmax_prediction(pred_y, true_y)
        available_label = sorted(list(set(true_y.tolist() + pred_y.tolist())))

    class_name = ['non-neutral', 'neutral']
    pred_y = [1 if element == 6 else 0 for element in pred_y] # element 6 means neutral.
    true_y = [1 if element == 6 else 0 for element in true_y]

    if get_dict:
        return classification_report(true_y, pred_y, target_names=class_name, zero_division=0, digits=4, output_dict=True)
    else:
        return classification_report(true_y, pred_y, target_names=class_name, zero_division=0, digits=4)

def log_metrics(logger, emo_pred_y_list, emo_true_y_list, option='train'):
    # label_ = np.array(['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'])
    label_ = np.array(['angry', 'disgust', 'fear', 'happy', 'sad', 'neutral', 'surprise'])
    logger.info('\n' + metrics_report(torch.cat(emo_pred_y_list), torch.cat(emo_true_y_list), label=label_))
    report_dict = metrics_report(torch.cat(emo_pred_y_list), torch.cat(emo_true_y_list), label=label_, get_dict=True)
    acc_emo, p_emo, r_emo, f1_emo = report_dict['accuracy'], report_dict['weighted avg']['precision'], report_dict['weighted avg']['recall'], report_dict['weighted avg']['f1-score']
    logger.info(f'\nemotion: {option}\n')

    logger.info('\n' + metrics_report_for_emo_binary(torch.cat(emo_pred_y_list), torch.cat(emo_true_y_list)))
    report_dict = metrics_report_for_emo_binary(torch.cat(emo_pred_y_list), torch.cat(emo_true_y_list), get_dict=True)
    acc_emo, p_emo, r_emo, f1_emo = report_dict['accuracy'], report_dict['weighted avg']['precision'], report_dict['weighted avg']['recall'], report_dict['weighted avg']['f1-score']
    logger.info(f'\nemotion (binary): {option}\n')