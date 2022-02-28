import torch
from sklearn.metrics import f1_score


def batch_acc(pred, label):
    """
    하나의 Batch에서의 Accuracy 값
    :param pred: 모델의 예측값
    :param label: 라벨링된 정답값
    :return: 맞으면 +1 틀리면 +0
    """
    return torch.sum(pred == label)


def batch_f1(pred, label, method):
    """
    Batch에서의 f1_score 값
    :param pred: 모델의 예측값
    :param label: 라벨링된 정답값
    :param method: f1_score의 메소드
    :return: f1_score
    """
    return f1_score(pred, label, average=method)


def epoch_mean(val, length):
    """
    Epoch당 평균값 반환
    :param val: 각 Epoch당 반환된 값
    :param length: 진행된 Epoch 수
    :return: average
    """
    return val / length