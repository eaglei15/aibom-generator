import json
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.models.service import AIBOMService
from src.utils.formatter import export_aibom


class FunctionalRegressionTests(unittest.TestCase):
    def _generate_local_aibom(self) -> dict:
        local_metadata = {
            "name": "local-regression-model",
            "author": "local tester",
            "commit": "abcdef1234567890",
            "pipeline_tag": "text-classification",
            "ethicalConsiderations": "May reflect training-data bias.",
        }

        with (
            patch("src.models.service.EnhancedExtractor") as extractor_cls,
            patch("src.models.service.ModelCard.load", return_value=None),
            patch("src.models.service.calculate_completeness_score") as score,
        ):
            extractor = extractor_cls.return_value
            extractor.extract_metadata.return_value = local_metadata
            extractor.extraction_results = {}
            score.return_value = {"total_score": 50}

            service = AIBOMService(
                hf_token="fake-token",
                use_inference=False,
                model_file_extractors=[],
            )
            service.hf_api = MagicMock()
            service.hf_api.model_info.return_value = MagicMock(sha=local_metadata["commit"])

            return service.generate_aibom("owner/local-regression-model")

    def test_local_service_generation_produces_parseable_core_aibom(self):
        aibom = self._generate_local_aibom()
        parsed = json.loads(json.dumps(aibom))

        for key in (
            "bomFormat",
            "specVersion",
            "serialNumber",
            "version",
            "metadata",
            "components",
        ):
            self.assertIn(key, parsed)

        self.assertEqual(parsed["bomFormat"], "CycloneDX")
        self.assertIsInstance(parsed["components"], list)
        self.assertGreaterEqual(len(parsed["components"]), 1)
        self.assertEqual(parsed["components"][0]["type"], "machine-learning-model")

    def test_ethical_considerations_do_not_emit_description_field(self):
        aibom = self._generate_local_aibom()
        ethical_considerations = (
            aibom["components"][0]["modelCard"]["considerations"]["ethicalConsiderations"]
        )

        self.assertGreaterEqual(len(ethical_considerations), 1)
        for entry in ethical_considerations:
            self.assertNotIn("description", entry)

    def test_cyclonedx_exports_pair_spec_version_and_schema_and_parse(self):
        aibom = self._generate_local_aibom()

        expected_schemas = {
            "1.6": "http://cyclonedx.org/schema/bom-1.6.schema.json",
            "1.7": "http://cyclonedx.org/schema/bom-1.7.schema.json",
        }

        for spec_version, schema in expected_schemas.items():
            with self.subTest(spec_version=spec_version):
                exported = json.loads(export_aibom(aibom, spec_version=spec_version))

                self.assertEqual(exported["bomFormat"], "CycloneDX")
                self.assertEqual(exported["specVersion"], spec_version)
                self.assertEqual(exported["$schema"], schema)
                self.assertIsInstance(exported["components"], list)

    def test_result_template_uses_neutral_export_wording(self):
        template_path = Path(__file__).resolve().parents[1] / "src" / "templates" / "result.html"
        template_text = template_path.read_text(encoding="utf-8").lower()

        self.assertNotIn("compatibility export", template_text)
        self.assertNotIn("schema-aware", template_text)
        self.assertNotIn("planned separately", template_text)


if __name__ == "__main__":
    unittest.main()
