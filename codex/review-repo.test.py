import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('review', Path(__file__).with_name('review-repo.py'))
review = importlib.util.module_from_spec(spec); spec.loader.exec_module(review)


class PrecisionTests(unittest.TestCase):
    def check_source(self, name, raw):
        original = review.git
        try:
            review.git = lambda *args: raw
            return review.inspect(Path('.'), ('100644', 'blob', 'fixture', str(len(raw)), name))
        finally: review.git = original

    def test_python_bom_is_valid(self):
        self.assertEqual(self.check_source('app.py', b'\xef\xbb\xbfimport json\n')['parse'], 'pass')

    def test_excerpt_is_not_a_standalone_program(self):
        self.assertEqual(self.check_source('function.excerpt.py', b'    return 1\n')['parse'], 'not-standalone-excerpt')

    def test_invalid_javascript_is_still_reported(self):
        self.assertEqual(self.check_source('app.js', 'const x = { …y };'.encode())['parse'], 'failed')


if __name__ == '__main__': unittest.main()
