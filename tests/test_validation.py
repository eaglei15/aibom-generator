import json
import unittest
from unittest.mock import patch
from cyclonedx.model.bom import Bom
from cyclonedx.model.component import Component, ComponentType
from cyclonedx.output.json import JsonV1Dot6
from src.utils.validation import validate_aibom, get_validation_summary


def _valid_aibom() -> dict:
    """Generate a minimal valid CycloneDX 1.6 BOM dict via the library."""
    bom = Bom()
    bom.components.add(Component(name="test-model", type=ComponentType.MACHINE_LEARNING_MODEL))
    return json.loads(JsonV1Dot6(bom).output_as_string())


class TestValidation(unittest.TestCase):
    def test_validate_aibom_valid(self):
        is_valid, errors = validate_aibom(_valid_aibom())
        self.assertTrue(is_valid)
        self.assertEqual(errors, [])

    def test_validate_aibom_invalid(self):
        # Missing all required CycloneDX fields
        is_valid, errors = validate_aibom({"otherField": "value"})
        self.assertFalse(is_valid)
        self.assertTrue(len(errors) > 0)

    def test_validate_aibom_preserves_error_message_without_data_path(self):
        class ValidatorErrorWithoutDataPath:
            message = "useful validation detail"

        with patch(
            "src.utils.validation._validator.validate_str",
            return_value=[ValidatorErrorWithoutDataPath()],
        ):
            is_valid, errors = validate_aibom({"otherField": "value"})

        self.assertFalse(is_valid)
        self.assertIn("useful validation detail", "\n".join(errors))
        self.assertNotIn("data_path", "\n".join(errors))

    def test_validate_aibom_wrong_bom_format(self):
        aibom = _valid_aibom()
        aibom["bomFormat"] = "NotCycloneDX"
        is_valid, _ = validate_aibom(aibom)
        self.assertFalse(is_valid)

    def test_get_validation_summary_valid(self):
        summary = get_validation_summary(_valid_aibom())
        self.assertTrue(summary["valid"])
        self.assertEqual(summary["error_count"], 0)
        self.assertEqual(summary["errors"], [])

    def test_get_validation_summary_invalid(self):
        summary = get_validation_summary({"bad": "data"})
        self.assertFalse(summary["valid"])
        self.assertGreater(summary["error_count"], 0)
        self.assertGreater(len(summary["errors"]), 0)


if __name__ == "__main__":
    unittest.main()
