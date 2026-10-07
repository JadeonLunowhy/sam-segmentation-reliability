import unittest
import numpy as np
from src.inference import select_candidate,record_complete

class InferenceTests(unittest.TestCase):
    def test_selection_is_confidence_only(self):
        masks=np.array([[[True,False]],[[False,True]],[[True,True]]])
        mask,score,index=select_candidate(masks,np.array([0.2,0.9,0.4]))
        np.testing.assert_array_equal(mask,masks[1])
        self.assertEqual(index,1);self.assertEqual(score,0.9)

    def test_resume_integrity(self):
        self.assertFalse(record_complete({'image_id':'a'},'abc','def'))
        row={'image_id':'a','config_hash':'abc','model_hash':'def','sweep':{},'mask_path':'missing'}
        self.assertFalse(record_complete(row,'abc','def'))

    def test_resume_rejects_changed_dataset(self):
        row={'config_hash':'abc','model_hash':'def','data_manifest_hash':'old'}
        self.assertFalse(record_complete(row,'abc','def','new'))

if __name__=='__main__':unittest.main()
