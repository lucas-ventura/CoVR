import torch
from einops import rearrange
from lightning_utilities.core.rank_zero import rank_zero_only


@rank_zero_only
def calculate_model_params(model):
    params = {}
    params["model/params/total"] = sum(p.numel() for p in model.parameters())
    params["model/params/trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    params["model/params/non_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    print(f"Total params: {params['model/params/total']/1e6:.2f}M")
    print(f"Trainable params: {params['model/params/trainable']/1e6:.2f}M")
    print(f"Non-trainable params: {params['model/params/non_trainable']/1e6:.2f}M")

    return params


def print_dist(message):
    """
    Function to print a message only on device 0 in a distributed training setup.

    Args:
        message (str): The message to be printed.
    """
    import torch

    if torch.distributed.is_initialized():  # type: ignore
        if torch.distributed.get_rank() == 0:  # type: ignore
            print(message)
    else:
        print(message)


@torch.no_grad()
def concat_all_gather(tensor, fabric):
    if fabric.world_size > 1:
        tensor = fabric.all_gather(tensor, sync_grads=False)
        tensor = rearrange(tensor, "batch num_gpu ... -> (batch num_gpu) ...")
    return tensor


def all_gather_with_grad(tensors, fabric):
    if fabric.world_size > 1:
        tensors = fabric.all_gather(tensors, sync_grads=True)
        tensors = rearrange(tensors, "batch num_gpu ... -> (batch num_gpu) ...")
    return tensors
