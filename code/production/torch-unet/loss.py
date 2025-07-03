import torch

def tversky(y_true, y_pred, alpha=0.7, smooth=1):
    y_pred = torch.sigmoid(y_pred)
    y_true = (y_true > 0.5).float()

    tp = torch.sum(y_pred * y_true)
    fp = torch.sum(y_pred * (1 - y_true))
    fn = torch.sum((1 - y_pred) * y_true)

    numerator = tp
    denominator = tp + alpha * fp + (1 - alpha) * fn

    score = (numerator + smooth) / (denominator + smooth)
    return 1 - score

def focal_tversky(y_true, y_pred, alpha=0.7, gamma=0.75, smooth=1):
    tversky_loss_value = tversky(y_true, y_pred, alpha, smooth)
    return torch.pow(tversky_loss_value, gamma)

def accuracy(y_true, y_pred):
    y_pred = (torch.sigmoid(y_pred) > 0.5).float()
    y_true = (y_true > 0.5).float()

    tp = torch.sum(y_pred * y_true)
    fp = torch.sum(y_pred * (1 - y_true))
    fn = torch.sum((1 - y_pred) * y_true)
    tn = torch.sum((1 - y_pred) * (1 - y_true))

    return (tp + tn) / (tp + tn + fp + fn + 1e-7)

def dice_coef(y_true, y_pred, smooth=1e-7):
    y_pred = torch.sigmoid(y_pred)
    y_true = (y_true > 0.5).float()

    tp = torch.sum(y_pred * y_true)
    fp = torch.sum(y_pred * (1 - y_true))
    fn = torch.sum((1 - y_pred) * y_true)

    two_tp = 2.0 * tp
    return (two_tp + smooth) / (two_tp + fp + fn + smooth)
