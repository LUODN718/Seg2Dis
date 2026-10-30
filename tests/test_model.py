"""CPU smoke tests for the complete model; no dataset or weights required."""

import unittest

import torch

from models import (
    SegDis,
    build_model,
    validate_checkpoint,
)
from utils.loss import structure_loss


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(42)
        cls.model = build_model(backbone_checkpoint="").eval()
        with torch.inference_mode():
            cls.output = cls.model(torch.randn(1, 3, 64, 64))

    def test_output_shape_and_probability_sum(self):
        probability = self.output["three_class_probability"]
        self.assertEqual(tuple(probability.shape), (1, 3, 64, 64))
        self.assertTrue(torch.isfinite(probability).all())
        self.assertTrue(((probability >= 0) & (probability <= 1)).all())
        torch.testing.assert_close(
            probability.sum(dim=1), torch.ones(1, 64, 64), rtol=1e-6, atol=1e-6)

    def test_foreground_type_probability_composition(self):
        foreground = self.output["final"].sigmoid()
        types = self.output["type_probability"]
        probability = self.output["three_class_probability"]
        torch.testing.assert_close(probability[:, :1], 1 - foreground)
        torch.testing.assert_close(probability[:, 1:], foreground * types)

    def test_existing_checkpoint_keys(self):
        state = self.model.state_dict()
        self.assertIn("query", state.keys())
        self.assertTrue(any(key.startswith("v3.master_decoder.") for key in state))
        result = self.model.load_state_dict(state, strict=True)
        self.assertEqual(result.missing_keys, [])
        self.assertEqual(result.unexpected_keys, [])

    def test_only_complete_pfrs_configuration_is_exposed(self):
        self.assertEqual(self.model.v3.configuration, "complete")
        self.assertTrue(self.model.v3.use_residual)
        self.assertFalse(hasattr(self.model.v3, "use_frequency"))
        self.assertFalse(self.model.v3.use_gate)
        self.assertTrue(self.model.v3.coarse_stages_are_context_only)
        self.assertNotIn("frequency_features", self.output)
        self.assertNotIn("sharpness_target", self.output)

    def test_no_historical_haar_components(self):
        from models import foreground_components

        self.assertFalse(hasattr(foreground_components, "HaarFrequencyContext"))
        self.assertFalse(hasattr(foreground_components, "RefinementBlock"))
        self.assertFalse(hasattr(self.model.v3, "refinement_blocks"))
        self.assertFalse(hasattr(self.model.v3, "residual_pyramid"))
        self.assertFalse(hasattr(self.model.v3, "missing_head"))
        self.assertFalse(any("haar" in type(module).__name__.lower()
                             for module in self.model.modules()))

    def test_removed_branches_and_fusion_channels(self):
        for name, _ in self.model.named_modules():
            self.assertNotIn("frequency", name.lower())
            self.assertNotIn("haar", name.lower())
        for key in self.model.state_dict():
            self.assertNotIn("frequency", key.lower())
            self.assertNotIn("haar", key.lower())
        for block in self.model.v3.scale_blocks:
            channels = block.original_projection[0].out_channels
            self.assertEqual(block.fusion[0][0].in_channels, 3 * channels + 2)
        block = self.model.v3.boundary_block
        channels = block.original_projection[0].out_channels
        self.assertEqual(block.smooth_body[0][0].in_channels, 3 * channels + 3)
        self.assertEqual(block.sharp_body[0][0].in_channels, 3 * channels + 3)

    def test_training_loss_backward_is_finite(self):
        from train import compute_loss

        self.model.zero_grad(set_to_none=True)
        batch = {"image": torch.randn(1, 3, 64, 64),
                 "semantic": torch.randint(0, 3, (1, 64, 64))}
        loss, _ = compute_loss(self.model, batch, torch.device("cpu"), 0.5, 0.5)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        gradients = {name: param.grad for name, param in self.model.named_parameters()
                     if param.grad is not None}
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(grad).all() for grad in gradients.values()))
        self.assertIsNotNone(self.model.v3.scale_blocks[0].fusion[0][0].weight.grad)
        self.assertIsNotNone(self.model.query.grad)
        self.model.zero_grad(set_to_none=True)

    def test_complete_configuration_metadata(self):
        metadata = {"architecture": "segdis_rgb_v2", "residual_input_mode": "rgb"}
        validate_checkpoint({**metadata, "variant": "SEGDIS_RGB_RESIDUAL"})
        validate_checkpoint({**metadata, "variant": "B4"})
        for variant, mode in (("B0", "rgb"), ("SEGDIS_RGB_RESIDUAL", "erased")):
            with self.assertRaises(ValueError):
                validate_checkpoint({**metadata, "variant": variant, "residual_input_mode": mode})
        with self.assertRaises(ValueError):
            validate_checkpoint({**metadata, "variant": "SEGDIS_RGB_RESIDUAL",
                                 "model_state": {"v3.scale_blocks.0.frequency_projection.0.weight": torch.zeros(1)}})
        with self.assertRaises(ValueError):
            validate_checkpoint({"variant": "SEGDIS_RGB_RESIDUAL", "residual_input_mode": "rgb"})
        with self.assertRaises(ValueError):
            validate_checkpoint({"variant": "B4"})
        with self.assertRaises(ValueError):
            validate_checkpoint({"variant": "SEGDIS_RGB_RESIDUAL", "residual_input_mode": "erased"})
        with self.assertRaises(ValueError):
            validate_checkpoint({"variant": "B0", "residual_input_mode": "rgb"})
        with self.assertRaises(ValueError):
            validate_checkpoint({"variant": "V3_FOREGROUND_TYPE_ALLOCATION", "residual_input_mode": "rgb"})
        with self.assertRaises(ValueError):
            validate_checkpoint({"residual_input_mode": "rgb"})

    def test_rgb_residual_keeps_image_mask_and_uncertainty(self):
        image = torch.randn(1, 3, 64, 64)
        with torch.inference_mode():
            output = self.model(image)
        torch.testing.assert_close(output['residual_input'][:, :3], image, rtol=0, atol=0)
        self.assertEqual(output['residual_input'].shape[1], 5)
        self.assertEqual(self.model.v3.residual_input_mode, 'rgb')

    def test_foreground_loss_is_finite(self):
        logit = self.output["final"].clone()
        for target in (torch.zeros_like(logit), torch.ones_like(logit)):
            self.assertTrue(torch.isfinite(structure_loss(logit, target)))


if __name__ == "__main__":
    unittest.main()
