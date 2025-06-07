import torch


def polyval(p, x):
    x = torch.as_tensor(x)
    result = torch.zeros_like(x)
    for ii, coeff in enumerate(p):
        result = x * result + coeff
    return result


torch.testing.assert_close(polyval([0], 1.0), torch.as_tensor(0.0))
torch.testing.assert_close(polyval([1], 1.0), torch.as_tensor(1.0))
torch.testing.assert_close(polyval([1, 1, 1], 2.0), torch.as_tensor(7.0))
