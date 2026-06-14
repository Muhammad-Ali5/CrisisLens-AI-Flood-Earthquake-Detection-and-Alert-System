import os
import pickle
import tempfile
import unittest

import test_model


class TestModelLoader(unittest.TestCase):
    def test_load_artifact_handles_empty_pickle_file(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            empty_path = os.path.join(tmp_dir, "empty.pkl")
            open(empty_path, "wb").close()

            artifact = test_model.load_artifact(empty_path, fallback="fallback")

            self.assertEqual(artifact, "fallback")


if __name__ == "__main__":
    unittest.main()
