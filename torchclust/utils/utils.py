import torch


def confusion_matrix(y_true, y_pred, num_classes):
    conf_matrix = torch.zeros(
        (num_classes, num_classes), dtype=torch.int64, device=y_true.device
    )

    indices = num_classes * y_true.view(-1) + y_pred.view(-1)

    conf_matrix = torch.bincount(indices, minlength=num_classes**2).reshape(
        num_classes, num_classes
    )

    return conf_matrix
