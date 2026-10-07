"""Independent integrity checks against stored masks and reference annotations."""
import json
import numpy as np
from src.data import ROOT,Case,load_case,sha256_file
from src.scoring import true_iou
from src.inference import record_complete

def main():
    cfg=json.loads((ROOT/'configs/study.json').read_text());frozen=json.loads((ROOT/'configs/frozen.json').read_text())
    manifest=json.loads((ROOT/'data/splits.json').read_text());cases={c['image_id']:Case(**c) for c in manifest['cases']}
    assert frozen['validation_sha256']==sha256_file(ROOT/'results/validation.jsonl')
    expected={'validation':45,'test':106,'extra':74};report={}
    for split,count in expected.items():
        rows=[json.loads(l) for l in (ROOT/'results'/(split+'.jsonl')).read_text().splitlines()]
        keys={(r['image_id'],r['regime']) for r in rows}
        assert len(rows)==len(keys)==count*2
        assert len({r['image_id'] for r in rows})==count
        metadata=json.loads((ROOT/'results'/(split+'.metadata.json')).read_text())
        assert metadata['frozen_parameters'] is True
        for row in rows:
            assert cases[row['image_id']].split==split
            assert row['regime'] in cfg['regimes']
            assert record_complete(row,sha256_file(ROOT/'configs/study.json'),frozen['model_hash'],frozen['data_manifest_hash'])
            _,target,valid=load_case(cases[row['image_id']])
            with np.load(ROOT/row['mask_path']) as masks:
                assert masks['original'].shape==target.shape
                assert masks['perturbations'].shape==(3,8,*target.shape)
                assert abs(true_iou(masks['original'],target,valid)-row['true_iou'])<1e-12
            assert 0<=row['true_iou']<=1
            for scores in row['sweep'].values():
                assert all(np.isfinite(v) for v in scores.values())
                assert all(0<=scores[k]<=1 for k in ['consistency','minimum','fusion','flip'])
        report[split]={'images':count,'cases':len(rows),'all_mask_hashes_verified':True,'all_iou_recomputed':True,'frozen_parameters':True}
    (ROOT/'results/audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':main()
