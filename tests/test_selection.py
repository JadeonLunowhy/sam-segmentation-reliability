import unittest
import tempfile
import json
from pathlib import Path
from src.selection import select_cell,freeze_selection
from unittest.mock import patch

class SelectionTests(unittest.TestCase):
    def test_only_validation_controls_selection(self):
        rows=[{'true_iou':1,'sweep':{'r0.01_k2':{'fusion':1},'r0.03_k4':{'fusion':0}}},
              {'true_iou':0,'sweep':{'r0.01_k2':{'fusion':0},'r0.03_k4':{'fusion':1}}}]
        chosen,table=select_cell(rows)
        self.assertEqual(chosen,'r0.01_k2')
        self.assertAlmostEqual(table[chosen],0.25)

    def test_tie_prefers_lower_budget(self):
        rows=[{'true_iou':1,'sweep':{k:{'fusion':1} for k in ['r0.03_k4','r0.03_k2','r0.01_k2']}}]
        self.assertEqual(select_cell(rows)[0],'r0.01_k2')

    def test_incomplete_validation_cannot_freeze(self):
        root=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=root/'tmp') as d:
            d=Path(d);(d/'configs').mkdir()
            rows=[{'split':'validation','image_id':'a','regime':'interior','true_iou':1,'sweep':{'r0.01_k2':{'fusion':1}}}]
            path=d/'validation.jsonl';path.write_text(json.dumps(rows[0])+'\n')
            cfg=d/'configs/study.json';cfg.write_text(json.dumps({'validation_images':45,'regimes':['interior','boundary']}))
            with patch('src.selection.ROOT',d):
                with self.assertRaises(ValueError):freeze_selection(path,cfg)

if __name__=='__main__':unittest.main()
