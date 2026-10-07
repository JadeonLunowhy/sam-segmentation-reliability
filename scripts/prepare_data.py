import json
from collections import Counter
from src.data import ROOT,prepare_data
if __name__=='__main__':
    cases=prepare_data(json.loads((ROOT/'configs/study.json').read_text()))
    print(dict(Counter(c.split for c in cases)))
