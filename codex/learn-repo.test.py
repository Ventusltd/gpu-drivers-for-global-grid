import importlib.util,unittest,numpy as np
from pathlib import Path
spec=importlib.util.spec_from_file_location('learner',Path(__file__).with_name('learn-repo.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class ModelTest(unittest.TestCase):
 def test_known_terms(self):
  x,v,idf=m.train([{'capacitor':3,'voltage':2},{'capacitor':3,'voltage':2},{'pipeline':4,'news':2}]);self.assertTrue(np.allclose(x[0],x[1]));self.assertAlmostEqual(float(x[0]@x[2]),0);self.assertTrue(np.allclose(np.linalg.norm(x,axis=1),1))
 def test_reproducible_seed(self):
  x=np.eye(8,dtype=np.float32);a,_,_=m.cluster(x,5);b,_,_=m.cluster(x,5);self.assertTrue(np.array_equal(a,b))
 def test_empty(self):
  x,v,_=m.train([]);labels,_,_=m.cluster(x,0);self.assertEqual(len(labels),0)
if __name__=='__main__':unittest.main()
