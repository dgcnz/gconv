import torch

def unsqueeze_like(tensor: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """ Unsqueeze last dimensions of tensor to match another tensor's number of dimensions.

    :param tensor: tensor to unsqueeze
    :param like: tensor whose dimensions to match
    """
    n_unsqueezes = like.ndim - tensor.ndim
    if n_unsqueezes < 0:
        raise ValueError(f"tensor.ndim={tensor.ndim} > like.ndim={like.ndim}")
    elif n_unsqueezes == 0:
        return tensor
    else:
        return tensor[(...,) + (None,) * n_unsqueezes]
