"""Record observed runtime versions and pinned model/source identifiers."""
import json,platform,subprocess
import torch
from src.data import ROOT,sha256_file
def main():
    env={'python':platform.python_version(),'platform':platform.platform(),'torch':torch.__version__,
         'cuda_available':torch.cuda.is_available(),'cuda_runtime':torch.version.cuda,
         'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
         'gpu_memory_bytes':torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None,
         'installation_scope':'project/.venv; no system Python or GPU driver changes'}
    (ROOT/'results/environment.json').write_text(json.dumps(env,indent=2))
    model={'checkpoint':'sam_vit_b_01ec64.pth','url':'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth',
           'sha256':sha256_file(ROOT/'models/sam_vit_b_01ec64.pth'),'bytes':(ROOT/'models/sam_vit_b_01ec64.pth').stat().st_size,
           'source_repository':'https://github.com/facebookresearch/segment-anything',
           'source_revision':subprocess.check_output(['git','-C',str(ROOT/'third_party/segment-anything'),'rev-parse','HEAD'],text=True).strip()}
    (ROOT/'models/provenance.json').write_text(json.dumps(model,indent=2));print(json.dumps(env,indent=2))
if __name__=='__main__':main()
