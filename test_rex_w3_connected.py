import unittest

import torch

from rexmodel import MLP


class RexW3ConnectionTests(unittest.TestCase):
    def test_mlp_w3_receives_gradient(self):
        torch.manual_seed(0)
        mlp = MLP(n_embd=8, dropout=0.0)
        x = torch.randn(2, 4, 8)

        loss = mlp(x).square().mean()
        loss.backward()

        self.assertIsNotNone(mlp.w3.weight.grad)
        self.assertGreater(mlp.w3.weight.grad.abs().sum().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
