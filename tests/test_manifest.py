from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from payrag.manifest import ManifestError, load_manifest, validate_manifest


MANIFEST_PATH = Path("config/source-manifest.json")


class ManifestTests(unittest.TestCase):
    def test_internal_manifest_is_valid(self) -> None:
        manifest = load_manifest(MANIFEST_PATH)
        self.assertEqual(13, len(manifest["sources"]))

    def test_duplicate_source_ids_are_rejected(self) -> None:
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        duplicate = copy.deepcopy(manifest["sources"][0])
        manifest["sources"].append(duplicate)
        manifest["source_count"] += 1

        with self.assertRaisesRegex(ManifestError, "Duplicate source_id"):
            validate_manifest(manifest)

    def test_unacquired_source_cannot_claim_a_hash(self) -> None:
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        manifest["sources"][0]["raw_sha256"] = "0" * 64

        with self.assertRaisesRegex(ManifestError, "must not have a hash"):
            validate_manifest(manifest)


if __name__ == "__main__":
    unittest.main()
