import numpy as np
import numpy.typing as npt
import torch
import os
import typing

def get_batch(
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str | torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample a batch of data from the dataset.

    Args:
        dataset: 1D numpy array of token IDs.
        batch_size: Number of sequences to sample.
        context_length: Length of each sequence.
        device: Device to place the tensors on.

    Returns:
        A tuple of (inputs, targets) tensors, each of shape (batch_size, context_length).
    """
    # High index is exclusive in np.random.randint.
    # We need room for context_length tokens for inputs starting at i,
    # and the token at i + context_length for the last target.
    # So max start index is len(dataset) - context_length - 1.
    high = len(dataset) - context_length
    ix = np.random.randint(0, high, size=batch_size)
    
    x_list = [dataset[i : i + context_length] for i in ix]
    y_list = [dataset[i + 1 : i + context_length + 1] for i in ix]
    
    x = torch.from_numpy(np.stack(x_list)).to(device)
    y = torch.from_numpy(np.stack(y_list)).to(device)
    
    return x, y

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    it: int,
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]
):
    """
    Save the model and optimizer state to a checkpoint.

    Args:
        model: The model to save.
        optimizer: The optimizer to save.
        it: The current iteration.
        out: The path to save the checkpoint to.
    """
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "it": it,
    }
    torch.save(checkpoint, out)

def load_checkpoint(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
):
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint["it"]