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
        self.assertFalse(self.model.v3.use_frequency)
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

    def test_compatibility_projections_receive_only_zeros(self):
        observed = []
        projections = [block.frequency_projection for block in self.model.v3.scale_blocks]
        projections.extend([self.model.v3.boundary_block.smooth_frequency,
                            self.model.v3.boundary_block.sharp_frequency])

        def check_input(module, inputs):
            observed.append(int(torch.count_nonzero(inputs[0])))

        handles = [module.register_forward_pre_hook(check_input) for module in projections]
        try:
            with torch.inference_mode():
                self.model(torch.randn(1, 3, 64, 64))
        finally:
            for handle in handles:
                handle.remove()
        self.assertEqual(observed, [0] * 5)

    def test_zero_input_projection_can_have_learned_constant_output(self):
        from models.foreground_components import ConvGNAct

        projection = ConvGNAct(3, 16).eval()
        with torch.no_grad():
            projection[1].bias.fill_(0.5)
        with torch.inference_mode():
            result = projection(torch.zeros(1, 3, 8, 8))
        self.assertTrue((result > 0).all())

    def test_complete_configuration_metadata(self):
        validate_checkpoint({"variant": "SEGDIS_RGB_RESIDUAL", "residual_input_mode": "rgb"})
        validate_checkpoint({"variant": "B4", "residual_input_mode": "rgb"})
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
