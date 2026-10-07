"""Validation-only choice; no test observations are accepted by this API."""
import json
from pathlib import Path
import numpy as np
from src.data import ROOT,sha256_file
from src.metrics import aurc

def select_cell(validation_rows):
    keys=list(validation_rows[0]['sweep']);q=np.array([r['true_iou'] for r in validation_rows])
    scores={key:aurc(q,np.array([r['sweep'][key]['fusion'] for r in validation_rows])) for key in keys}
    def rank(key):
        radius,k=key[1:].split('_k')
        return (scores[key],int(k),float(radius))
    return min(keys,key=rank),scores

def freeze_selection(validation_path=ROOT/'results/validation.jsonl',config_path=ROOT/'configs/study.json'):
    rows=[json.loads(line) for line in validation_path.read_text().splitlines()]
    if any(r['split']!='validation' for r in rows):raise ValueError('Only validation data permitted')
    config=json.loads(config_path.read_text())
    if len({r['image_id'] for r in rows})!=config['validation_images']:raise ValueError('Incomplete validation images')
    groups={r['image_id']:{q['regime'] for q in rows if q['image_id']==r['image_id']} for r in rows}
    if len(rows)!=config['validation_images']*len(config['regimes']) or any(v!=set(config['regimes']) for v in groups.values()):
        raise ValueError('Incomplete or duplicate validation prompt cases')
    selected,scores=select_cell(rows)
    frozen={'selected':selected,'validation_aurc':scores,'validation_sha256':sha256_file(validation_path),
            'data_manifest_hash':rows[0].get('data_manifest_hash'),'model_hash':rows[0].get('model_hash'),
            'study_sha256':sha256_file(config_path),'rule':'minimum validation fusion AURC; tie by lower K then radius',
            'validation_images':len({r['image_id'] for r in rows}),'validation_cases':len(rows)}
    output=config_path.parent/'frozen.json';output.write_text(json.dumps(frozen,indent=2));return output

if __name__=='__main__':print(freeze_selection().read_text())
