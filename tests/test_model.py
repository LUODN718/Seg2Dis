import unittest

import torch

from models import build_model, validate_checkpoint
from train import compute_loss
from utils.checkpoint import model_metadata


class SegDisModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(42)
        cls.model = build_model(backbone_checkpoint="", variant="segdis").eval()
        with torch.inference_mode():
            cls.output = cls.model(torch.randn(1, 3, 64, 64))

    def test_three_class_probabilities_are_valid(self):
        probability = self.output["three_class_probability"]
        self.assertEqual(tuple(probability.shape), (1, 3, 64, 64))
        self.assertTrue(torch.isfinite(probability).all())
        self.assertTrue(((probability >= 0) & (probability <= 1)).all())
        torch.testing.assert_close(
            probability.sum(dim=1), torch.ones(1, 64, 64), rtol=1e-6, atol=1e-6
        )

    def test_foreground_and_type_probabilities_compose(self):
        probability = self.output["three_class_probability"]
        foreground = self.output["final"].sigmoid()
        type_probability = self.output["type_probability"]
        torch.testing.assert_close(probability[:, :1], 1 - foreground)
        torch.testing.assert_close(probability[:, 1:], foreground * type_probability)

    def test_state_dict_uses_flr_and_rta(self):
        state = self.model.state_dict()
        self.assertTrue(any(key.startswith("flr.") for key in state))
        self.assertTrue(any(key.startswith("rta.") for key in state))
        self.assertFalse(any(key.startswith("v3.") for key in state))
        self.assertFalse(any("frequency" in key.lower() or "haar" in key.lower() for key in state))
        self.model.load_state_dict(state, strict=True)

    def test_checkpoint_metadata_matches_model(self):
        checkpoint = {
            **model_metadata(self.model),
            "model_state": self.model.state_dict(),
        }
        validate_checkpoint(checkpoint, self.model)
        checkpoint["architecture"] = "other"
        with self.assertRaises(ValueError):
            validate_checkpoint(checkpoint, self.model)

    def test_training_loss_has_finite_gradients(self):
        self.model.zero_grad(set_to_none=True)
        batch = {
            "image": torch.randn(1, 3, 64, 64),
            "semantic": torch.randint(0, 3, (1, 64, 64)),
        }
        loss, _ = compute_loss(self.model, batch, torch.device("cpu"), 0.5, 0.5)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        gradients = [parameter.grad for parameter in self.model.parameters() if parameter.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))


if __name__ == "__main__":
    unittest.main()
