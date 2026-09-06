import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('cloud', Path(__file__).with_name('cloud-learning.py'))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class EvidenceBoundary(unittest.TestCase):
    def test_wrong_commit_cannot_complete(self):
        row = m.jobs()[0]
        with self.assertRaises(ValueError):
            m.verify_receipt(row, dict(row, commit='0' * 40, status='complete', sourceVerified=True))

    def test_unverified_receipt_cannot_complete(self):
        row = m.jobs()[0]
        with self.assertRaises(ValueError):
            m.verify_receipt(row, dict(row, status='complete', sourceVerified=False))

    def test_rebuilt_model_must_match_inventory(self):
        row = next(r for r in m.jobs() if r['mode'] == 'stability')
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            (p/'manifest.json').write_text(json.dumps({'commit':row['commit'], 'repository':row['repository'],
                'selected':row['expectedSelected'], 'dimensions':row['expectedDimensions']}))
            (p/'files.json').write_text('[]')
            with self.assertRaises(ValueError):
                m.verify_model(row, p)

    def test_valid_exact_receipt(self):
        row = m.jobs()[0]
        m.verify_receipt(row, dict(row, status='complete', sourceVerified=True))


if __name__ == '__main__':
    unittest.main()
