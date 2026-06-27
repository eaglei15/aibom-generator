import json
import unittest
from typing import Any, Dict, Iterator, List, Tuple

from cyclonedx.schema import SchemaVersion
from cyclonedx.validation.json import JsonStrictValidator

from src.utils.formatter import export_aibom


CYCLONEDX_SCHEMAS = {
    "1.6": "http://cyclonedx.org/schema/bom-1.6.schema.json",
    "1.7": "http://cyclonedx.org/schema/bom-1.7.schema.json",
}

SCHEMA_VERSIONS = {
    "1.6": SchemaVersion.V1_6,
    "1.7": SchemaVersion.V1_7,
}

ALLOWED_PROPERTY_PREFIXES = ("genai:aibom:",)

# Legacy/current-output compatibility only; prefer namespaced
# genai:aibom:* properties for any new AIBOM metadata.
LEGACY_PROPERTY_NAME_ALLOWLIST = set()

KNOWN_ROOT_DEPENDENCY_REFS = {"metadata"}


def _sample_aibom() -> Dict[str, Any]:
    model_ref = "pkg:huggingface/acme/current-contract-model@12345678"

    return {
        "$schema": CYCLONEDX_SCHEMAS["1.6"],
        "bomFormat": "CycloneDX",
        "specVersion": "1.6",
        "serialNumber": "urn:uuid:12345678-1234-5678-1234-567812345678",
        "version": 1,
        "metadata": {
            "component": {
                "type": "application",
                "name": "aibom-generator",
                "version": "test",
                "bom-ref": "aibom-generator",
                "properties": [
                    {"name": "genai:aibom:fixture", "value": "true"},
                ],
            },
            "properties": [
                {"name": "genai:aibom:source", "value": "deterministic-test"},
            ],
        },
        "components": [
            {
                "type": "machine-learning-model",
                "name": "current-contract-model",
                "version": "12345678",
                "bom-ref": model_ref,
                "purl": model_ref,
                "externalReferences": [
                    {
                        "type": "website",
                        "url": "https://huggingface.co/acme/current-contract-model",
                    }
                ],
                "modelCard": {
                    "modelParameters": {
                        "task": "text-classification",
                        "modelArchitecture": "transformer",
                        "inputs": [{"format": "string"}],
                        "outputs": [{"format": "string"}],
                        "datasets": [
                            {
                                "type": "dataset",
                                "name": "contract-fixture-dataset",
                            }
                        ],
                    },
                    "considerations": {
                        "ethicalConsiderations": [
                            {"name": "May reflect training-data bias."}
                        ],
                    },
                    "properties": [
                        {"name": "genai:aibom:model-card", "value": "present"},
                    ],
                },
                "properties": [
                    {"name": "genai:aibom:model-id", "value": "acme/current-contract-model"},
                ],
            }
        ],
        "dependencies": [
            {"ref": "metadata", "dependsOn": [model_ref]},
            {"ref": model_ref, "dependsOn": []},
        ],
        "properties": [
            {"name": "genai:aibom:contract-version", "value": "current"},
        ],
    }


def _exported_documents() -> Iterator[Tuple[str, Dict[str, Any]]]:
    for spec_version in ("1.6", "1.7"):
        yield spec_version, json.loads(
            export_aibom(
                _sample_aibom(),
                bom_type="cyclonedx",
                spec_version=spec_version,
            )
        )


def _format_validation_error(error: Any) -> str:
    message = getattr(error, "message", str(error))
    path = getattr(error, "data_path", None) or getattr(error, "path", None)
    if path is None:
        return str(message)
    if isinstance(path, str):
        location = path or "root"
    else:
        location = ".".join(str(part) for part in path) or "root"
    return f"[{location}] {message}"


def _validate_cdx_schema(document: Dict[str, Any], spec_version: str) -> List[str]:
    validator = JsonStrictValidator(SCHEMA_VERSIONS[spec_version])
    errors = validator.validate_str(json.dumps(document), all_errors=True)
    if errors is None:
        return []
    return [_format_validation_error(error) for error in errors]


def _find_model_components(document: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [
        component
        for component in document.get("components", [])
        if component.get("type") == "machine-learning-model"
    ]


def _iter_properties(document: Dict[str, Any]) -> Iterator[Tuple[str, Dict[str, Any]]]:
    property_locations = [
        ("root", document),
        ("metadata", document.get("metadata", {})),
        ("metadata.component", document.get("metadata", {}).get("component", {})),
    ]

    for index, component in enumerate(document.get("components", [])):
        property_locations.append((f"components[{index}]", component))
        model_card = component.get("modelCard")
        if isinstance(model_card, dict):
            property_locations.append((f"components[{index}].modelCard", model_card))

    for location, container in property_locations:
        properties = container.get("properties") if isinstance(container, dict) else None
        if properties is None:
            continue
        for prop in properties:
            yield location, prop


def _assert_current_ai_ml_contract(test_case: unittest.TestCase, document: Dict[str, Any]) -> None:
    model_components = _find_model_components(document)
    test_case.assertGreaterEqual(len(model_components), 1)

    model = model_components[0]
    for field in ("name", "version", "purl"):
        test_case.assertIn(field, model)
    test_case.assertTrue(model.get("bom-ref") or model.get("purl"))

    model_card = model.get("modelCard")
    test_case.assertIsInstance(model_card, dict)
    model_parameters = model_card.get("modelParameters")
    test_case.assertIsInstance(model_parameters, dict)
    test_case.assertIn("task", model_parameters)
    test_case.assertIn("modelArchitecture", model_parameters)

    for field in ("inputs", "outputs", "datasets"):
        if field in model_parameters:
            test_case.assertIsInstance(model_parameters[field], list)

    if "externalReferences" in model:
        test_case.assertIsInstance(model["externalReferences"], list)

    ethical_considerations = (
        model_card.get("considerations", {}).get("ethicalConsiderations", [])
    )
    for entry in ethical_considerations:
        test_case.assertNotIn("description", entry)

    _assert_dependencies_reference_existing_refs(test_case, document)


def _assert_dependencies_reference_existing_refs(
    test_case: unittest.TestCase,
    document: Dict[str, Any],
) -> None:
    known_refs = set(KNOWN_ROOT_DEPENDENCY_REFS)

    metadata_component = document.get("metadata", {}).get("component", {})
    if metadata_component.get("bom-ref"):
        known_refs.add(metadata_component["bom-ref"])

    for component in document.get("components", []):
        if component.get("bom-ref"):
            known_refs.add(component["bom-ref"])

    for dependency in document.get("dependencies", []):
        test_case.assertIn(dependency.get("ref"), known_refs)
        for ref in dependency.get("dependsOn", []):
            test_case.assertIn(ref, known_refs)


def _assert_property_taxonomy_contract(test_case: unittest.TestCase, document: Dict[str, Any]) -> None:
    for location, prop in _iter_properties(document):
        name = prop.get("name")
        test_case.assertIsInstance(name, str, msg=location)
        test_case.assertNotEqual(name.strip(), "", msg=location)
        test_case.assertIn("value", prop, msg=location)
        test_case.assertIsInstance(prop["value"], str, msg=name)
        test_case.assertNotEqual(prop["value"], "", msg=name)

        allowed = name.startswith(ALLOWED_PROPERTY_PREFIXES)
        legacy_allowed = name in LEGACY_PROPERTY_NAME_ALLOWLIST
        test_case.assertTrue(
            allowed or legacy_allowed,
            msg=f"Unknown property namespace at {location}: {name}",
        )


class AIBOMOutputContractTests(unittest.TestCase):
    def test_exported_cdx_1_6_and_1_7_outputs_are_schema_valid(self):
        for spec_version, document in _exported_documents():
            with self.subTest(spec_version=spec_version):
                self.assertEqual(document["bomFormat"], "CycloneDX")
                self.assertEqual(document["specVersion"], spec_version)
                self.assertEqual(document["$schema"], CYCLONEDX_SCHEMAS[spec_version])
                self.assertEqual(_validate_cdx_schema(document, spec_version), [])

    def test_exported_outputs_preserve_current_ai_ml_model_contract(self):
        for spec_version, document in _exported_documents():
            with self.subTest(spec_version=spec_version):
                _assert_current_ai_ml_contract(self, document)

    def test_emitted_properties_use_allowed_taxonomy_namespaces(self):
        for spec_version, document in _exported_documents():
            with self.subTest(spec_version=spec_version):
                _assert_property_taxonomy_contract(self, document)


# Known non-enforced gaps for future implementation work:
# - model as metadata.component when the model is the BOM subject
# - first-class data components for datasets
# - dataset bom-ref relationships
# - model file hashes
# - evidence/source/confidence metadata
# - full G7/BSI completeness
# - schema-aware CycloneDX 1.7 AI/ML mapper


if __name__ == "__main__":
    unittest.main()
