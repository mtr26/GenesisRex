import unittest
from unittest import mock

import torch

import benchmark


class TinyLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(16, 8)
        self.body = torch.nn.Sequential(
            torch.nn.Linear(8, 8, bias=False),
            torch.nn.LayerNorm(8),
        )
        self.lm_head = torch.nn.Linear(8, 16, bias=False)


class TinyRexLike(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(16, 8)
        self.fc_out = torch.nn.Linear(8, 16, bias=False)
        ff = torch.nn.Module()
        ff.w3 = torch.nn.Linear(8, 32, bias=False)
        block = torch.nn.Module()
        block.ff = ff
        self.blocks = torch.nn.ModuleList([block])


class FakeFlashMuon(torch.optim.Optimizer):
    def __init__(self, params, **kwargs):
        super().__init__(params, kwargs)

    def step(self, closure=None):
        return None


class BenchmarkOptimizerMatrixTests(unittest.TestCase):
    def test_optimizer_bundle_exposes_groups_and_zero_grad_for_multiple_optimizers(self):
        p1 = torch.nn.Parameter(torch.ones(2, 2))
        p2 = torch.nn.Parameter(torch.ones(2))
        opt1 = torch.optim.SGD([p1], lr=0.1)
        opt2 = torch.optim.AdamW([p2], lr=0.01)

        bundle = benchmark.OptimizerBundle([opt1, opt2])

        self.assertEqual(len(bundle.param_groups), 2)
        (p1.sum() + p2.sum()).backward()
        bundle.zero_grad(set_to_none=True)
        self.assertIsNone(p1.grad)
        self.assertIsNone(p2.grad)

    def test_split_hidden_matrix_params_keeps_embed_and_head_on_adam_side(self):
        model = TinyLM()

        split = benchmark.split_optimizer_params(model, model_key="genesis")

        muon_ids = {id(p) for p in split.muon_params}
        adam_ids = {id(p) for p in split.adam_params}
        self.assertIn(id(model.body[0].weight), muon_ids)
        self.assertIn(id(model.embedding.weight), adam_ids)
        self.assertIn(id(model.lm_head.weight), adam_ids)
        self.assertNotIn(id(model.embedding.weight), muon_ids)
        self.assertNotIn(id(model.lm_head.weight), muon_ids)

    def test_rex_w3_is_a_muon_matrix_param(self):
        model = TinyRexLike()

        split = benchmark.split_optimizer_params(model, model_key="rex")

        self.assertIn(id(model.blocks[0].ff.w3.weight), {id(p) for p in split.muon_params})
        self.assertEqual(split.excluded_param_count, 0)

    def test_make_adamw_optimizer_is_available_for_both_models(self):
        for model_key in ("genesis", "rex"):
            model = TinyLM()
            optimizer, metadata = benchmark.make_optimizer(
                model,
                model_key=model_key,
                optimizer_key="adamw",
                muon_lr=0.02,
                adam_lr=3e-4,
                weight_decay=0.1,
                device=torch.device("cpu"),
            )
            self.assertIsInstance(optimizer, torch.optim.AdamW)
            self.assertEqual(metadata["optimizer"], "adamw")

    def test_make_flash_muon_uses_flash_optimizer_plus_aux_adam(self):
        model = TinyLM()
        with (
            mock.patch.object(benchmark, "_import_flash_muon", return_value=FakeFlashMuon),
            mock.patch.object(benchmark, "_ensure_single_process_group", return_value=None),
        ):
            optimizer, metadata = benchmark.make_optimizer(
                model,
                model_key="genesis",
                optimizer_key="flash_muon",
                muon_lr=0.02,
                adam_lr=3e-4,
                weight_decay=0.1,
                device=torch.device("cuda"),
            )

        self.assertIsInstance(optimizer, benchmark.OptimizerBundle)
        self.assertEqual(len(optimizer.optimizers), 2)
        self.assertEqual(metadata["optimizer"], "flash_muon")
        self.assertGreater(metadata["muon_param_count"], 0)
        self.assertGreater(metadata["adam_param_count"], 0)

    def test_flash_muon_requires_cuda(self):
        model = TinyLM()
        with self.assertRaisesRegex(RuntimeError, "requires CUDA"):
            benchmark.make_optimizer(
                model,
                model_key="genesis",
                optimizer_key="flash_muon",
                muon_lr=0.02,
                adam_lr=3e-4,
                weight_decay=0.1,
                device=torch.device("cpu"),
            )


if __name__ == "__main__":
    unittest.main()
