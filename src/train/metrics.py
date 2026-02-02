import torch


def micro_f1_multilabel(
    y_true: torch.Tensor,
    y_prob: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-8,
) -> float:
    """
    Compute micro-F1 for multi-label classification.

    Args:
        y_true: Tensor [N, C] with binary labels (0/1)
        y_prob: Tensor [N, C] with probabilities in [0, 1]
        threshold: decision threshold
        eps: numerical stability

    Returns:
        micro-F1 score (float)
    """
    assert y_true.shape == y_prob.shape, "Shape mismatch"

    # binarize predictions
    y_pred = (y_prob >= threshold).to(torch.int)

    y_true = y_true.to(torch.int)

    # global counts
    tp = torch.sum((y_pred == 1) & (y_true == 1)).float()
    fp = torch.sum((y_pred == 1) & (y_true == 0)).float()
    fn = torch.sum((y_pred == 0) & (y_true == 1)).float()

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)

    f1 = 2 * precision * recall / (precision + recall + eps)
    return f1.item()