import torch

def bundle_objective(pred_loss, irm_penalty, complexity, lambda_irm=1.0, lambda_complexity=0.01):
    """
    F(B) = Prediction Error + lambda_irm * IRM + lambda_complexity * Complexity(s)
    Compatible with both standard numeric types and active PyTorch graphs.
    """
    return pred_loss + (lambda_irm * irm_penalty) + (lambda_complexity * complexity)
