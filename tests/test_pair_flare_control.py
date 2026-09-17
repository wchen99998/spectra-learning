import torch

from scripts.pair_flare_benchmark import diagonal_control


class Readout(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(12, 5)

    def forward(self, tokens, mz, intensity, cls):
        return self.linear(tokens), cls


def test_control_keeps_capacity_and_gradients_but_ignores_off_diagonal_channels():
    torch.manual_seed(0)
    full = Readout()
    control = diagonal_control(Readout)()
    control.load_state_dict(full.state_dict())
    tokens = torch.randn(2, 3, 12, requires_grad=True)
    cls = torch.randn(2, 12)
    result, result_cls = control(tokens, None, None, cls)
    expected, _ = full(tokens[..., :4].repeat(1, 1, 3), None, None, cls)
    torch.testing.assert_close(result, expected)
    assert result_cls is cls
    perturbed = tokens.detach().clone()
    perturbed[..., 4:] += 100
    torch.testing.assert_close(control(perturbed, None, None, cls)[0], result)
    assert not torch.allclose(full(perturbed, None, None, cls)[0], full(tokens, None, None, cls)[0])
    result.sum().backward()
    assert torch.count_nonzero(tokens.grad[..., 4:]) == 0
    assert torch.count_nonzero(tokens.grad[..., :4]) > 0
    assert all(parameter.grad is not None for parameter in control.parameters())
    assert sum(p.numel() for p in full.parameters()) == sum(p.numel() for p in control.parameters())
