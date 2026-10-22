import sys,pathlib,unittest,tempfile,json
import numpy as np,pandas as pd,joblib
from importlib.util import spec_from_file_location,module_from_spec
ROOT=pathlib.Path(__file__).resolve().parent
def load(name,rel):
 spec=spec_from_file_location(name,ROOT/rel);m=module_from_spec(spec);spec.loader.exec_module(m);return m
network=load('network_data','src/data_utils.py')
class EvaluationTests(unittest.TestCase):
 def test_held_out_values_do_not_refit_scaler(self):
  train=pd.DataFrame({'duration':[0.,10.,5.,7.],'protocol_type':['tcp','udp','tcp','udp'],'label':[0,1,0,1]})
  test=pd.DataFrame({'duration':[100.],'protocol_type':['unseen'],'label':[1]})
  x,y,p=network.preprocess_data(train); before=p.named_transformers_['num'].data_max_.copy()
  held,labels,reused=network.preprocess_data(test,p)
  self.assertIs(p,reused);np.testing.assert_array_equal(before,p.named_transformers_['num'].data_max_)
  self.assertAlmostEqual(held[0,0,0],10.);np.testing.assert_array_equal(held[0,1:,0],0.)
  self.assertEqual(held.shape,(1,x.shape[1],1))
 def test_numeric_only_records(self):
  df=network.generate_network_data(20,include_categorical=False);x,y,p=network.preprocess_data(df)
  self.assertEqual(x.shape,(20,8,1));self.assertEqual(y.shape,(20,));self.assertTrue(np.isfinite(x).all())
 def test_preprocessor_round_trip(self):
  df=network.generate_network_data(40);x,y,p=network.preprocess_data(df)
  with tempfile.TemporaryDirectory() as temp:
   dest=pathlib.Path(temp)/'p.joblib';joblib.dump(p,dest);loaded=joblib.load(dest)
   transformed,_,_=network.preprocess_data(df,loaded);np.testing.assert_allclose(x,transformed)
if __name__=='__main__':unittest.main()
