import unittest
import numpy as np
from src.data import pet_trimap, split_ids, safe_extract, image_files
from src.prompts import make_prompt, perturb_points, flip_point, K_INDICES
from src.scoring import mask_iou, true_iou, threshold_stability, reliability_scores
from src.metrics import aurc, failure_metrics, selective_curve
from src.statistics import bootstrap_difference, grouped_indices
import tempfile
import tarfile
import io
from pathlib import Path


class DataTests(unittest.TestCase):
    def test_supported_image_extensions(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]/'tmp') as d:
            d=Path(d)
            for name in ['a.jpg','b.PNG','c.bmp','d.jpeg','e.txt']:(d/name).write_bytes(b'')
            self.assertEqual({p.name for p in image_files(d)},{'a.jpg','b.PNG','c.bmp','d.jpeg'})
    def test_trimap(self):
        fg, valid = pet_trimap(np.array([[1, 2, 3]]))
        np.testing.assert_array_equal(fg, [[True, False, False]])
        np.testing.assert_array_equal(valid, [[True, True, False]])

    def test_split_isolation(self):
        a, b = split_ids([str(i) for i in range(151)], 45, 20261003)
        self.assertEqual(len(a), 45)
        self.assertEqual(len(b), 106)
        self.assertFalse(set(a) & set(b))
        self.assertEqual((a,b), split_ids([str(i) for i in range(151)],45,20261003))

    def test_unsafe_archive(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]/'tmp') as d:
            path = Path(d)/'bad.tar'
            with tarfile.open(path, 'w') as t:
                info = tarfile.TarInfo('../escape.txt'); info.size = 1
                t.addfile(info, io.BytesIO(b'x'))
            with self.assertRaises(ValueError):
                safe_extract(path, Path(d)/'out')


class ScoringTests(unittest.TestCase):
    def test_iou_conventions(self):
        a = np.eye(3,dtype=bool); z = np.zeros((3,3),bool)
        self.assertEqual(mask_iou(a,a), 1)
        self.assertEqual(mask_iou(z,z), 1)
        self.assertEqual(mask_iou(a,z), 0)
        self.assertEqual(true_iou(z,a,np.ones_like(a)),0)
        with self.assertRaises(ValueError): true_iou(a,z,np.ones_like(a))

    def test_ignore_border(self):
        target=np.array([[True,False]]); valid=np.array([[True,False]])
        self.assertEqual(true_iou(np.array([[True,True]]),target,valid),1)

    def test_stability(self):
        self.assertEqual(threshold_stability(np.array([-2.,0.,2.])),0.5)
        self.assertEqual(threshold_stability(np.array([-3.,-3.])),0)

    def test_scores_without_annotations(self):
        m=np.eye(3,dtype=bool)
        s=reliability_scores(0.8,m,np.stack([m,m]),m)
        self.assertEqual(s['consistency'],1)
        self.assertAlmostEqual(s['fusion'],0.9)


class PromptTests(unittest.TestCase):
    def test_thin_mask(self):
        m=np.zeros((5,5),bool); m[2,:]=True
        for regime in ['interior','boundary']:
            p,meta=make_prompt(m,regime,7)
            self.assertTrue(m[int(p[1]),int(p[0])])
        with self.assertRaises(ValueError):make_prompt(np.zeros_like(m),'interior',7)

    def test_coordinates_and_subsets(self):
        for p in [np.array([0,0]),np.array([8,0]),np.array([0,4]),np.array([8,4])]:
            pts=perturb_points(p,(5,9),0.06)
            self.assertTrue(((pts[:,0]>=0)&(pts[:,0]<=8)).all())
            self.assertTrue(((pts[:,1]>=0)&(pts[:,1]<=4)).all())
            np.testing.assert_array_equal(flip_point(flip_point(p,9),9),p)
        self.assertEqual(K_INDICES[2],[0,4])
        self.assertEqual(K_INDICES[4],[0,2,4,6])


class MetricTests(unittest.TestCase):
    def test_order_and_ties(self):
        self.assertAlmostEqual(aurc(np.array([1.,0.]),np.array([1.,0.])),0.25)
        self.assertAlmostEqual(aurc(np.array([1.,0.]),np.array([0.,1.])),0.75)
        for q in [np.array([1.,0.]),np.array([0.,1.])]:
            self.assertAlmostEqual(aurc(q,np.ones(2)),0.5)

    def test_failure_direction(self):
        m=failure_metrics(np.array([0.1,0.9]),np.array([0.1,0.9]),0.5)
        self.assertEqual(m['auroc'],1)
        self.assertEqual(m['average_precision'],1)
        self.assertIsNone(failure_metrics(np.ones(3),np.arange(3),0.5)['auroc'])
        self.assertIsNone(failure_metrics(np.zeros(3),np.arange(3),0.5)['auroc'])


class BootstrapTests(unittest.TestCase):
    def test_image_cluster(self):
        ids=np.array(['a','a','b','b'])
        ix=grouped_indices(ids,np.random.default_rng(8))
        for name in ['a','b']:
            self.assertEqual(sum(ix==np.flatnonzero(ids==name)[0]),sum(ix==np.flatnonzero(ids==name)[1]))
        rows=[{'image_id':str(i//2),'true_iou':i/9,'a':i/9,'b':i/9} for i in range(10)]
        result=bootstrap_difference(rows,'a','b','aurc',draws=30,seed=9)
        self.assertEqual(result['estimate'],0)
        self.assertEqual(result['ci95'],[0.,0.])
        self.assertEqual(result,bootstrap_difference(rows,'a','b','aurc',30,9))

if __name__=='__main__':unittest.main()
