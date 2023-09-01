import torch
from scipy.linalg import expm


class TrExpScipy(torch.autograd.Function):
    """
    autograd.Function to compute trace of an exponential of a matrix
    From the author's GitHub implementation: https://github.com/kurowasan/GraN-DAG.
    """

    @staticmethod
    def forward(ctx, input):
        with torch.no_grad():
            # send tensor to cpu in numpy format and compute expm using scipy
            expm_input = expm(input.detach().cpu().numpy())
            # transform back into a tensor
            expm_input = torch.as_tensor(expm_input, device=input.device)
            # save expm_input to use in backward
            ctx.save_for_backward(expm_input)

            # return the trace
            return torch.trace(expm_input)

    @staticmethod
    def backward(ctx, grad_output):
        with torch.no_grad():
            (expm_input,) = ctx.saved_tensors
            return expm_input.t() * grad_output


def is_acyclic(adjacency: torch.Tensor) -> bool:
    """Check if the given adjacency matrix is acyclic"""
    prod = torch.eye(adjacency.shape[0], dtype=adjacency.dtype, device=adjacency.device)
    for _ in range(1, adjacency.shape[0] + 1):
        prod = torch.matmul(adjacency, prod)
        if torch.trace(prod) != 0:
            return False
    return True
