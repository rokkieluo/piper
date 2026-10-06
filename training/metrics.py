import torch


def classification_metrics(targets, predictions, num_classes=2):
    """由真实标签和预测标签计算分类评估指标。

    输入：等长的 targets/predictions 序列和类别数（默认二分类）。
    流程：统计真实类别为行、预测类别为列的混淆矩阵；按类计算
    precision、recall、F1、support，并取类别 F1 的算术平均。
    输出：含 accuracy、macro_f1、per_class、confusion_matrix 的字典；
    空分母按 0 处理，长度不等时抛 ValueError。
    """
    targets = torch.as_tensor(targets, dtype=torch.long).view(-1)
    predictions = torch.as_tensor(predictions, dtype=torch.long).view(-1)
    if targets.numel() != predictions.numel():
        raise ValueError("targets and predictions must have the same length")

    confusion = torch.zeros((num_classes, num_classes), dtype=torch.long)
    for target, prediction in zip(targets, predictions):
        confusion[target, prediction] += 1

    per_class = []
    for class_id in range(num_classes):
        true_positive = int(confusion[class_id, class_id])
        false_positive = int(confusion[:, class_id].sum()) - true_positive
        false_negative = int(confusion[class_id, :].sum()) - true_positive
        precision_denominator = true_positive + false_positive
        recall_denominator = true_positive + false_negative
        precision = true_positive / precision_denominator if precision_denominator else 0.0
        recall = true_positive / recall_denominator if recall_denominator else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class.append(
            {"precision": precision, "recall": recall, "f1": f1, "support": recall_denominator}
        )

    accuracy = (
        float(confusion.diag().sum()) / float(confusion.sum())
        if confusion.sum()
        else 0.0
    )
    return {
        "accuracy": accuracy,
        "macro_f1": sum(item["f1"] for item in per_class) / num_classes,
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
    }
