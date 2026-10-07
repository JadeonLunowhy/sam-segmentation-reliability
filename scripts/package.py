"""Small submission archive: executable code, scientific evidence and materials."""
import json,zipfile
from src.data import ROOT,sha256_file

def main():
    output=ROOT/'submission/code_and_results.zip'
    folders=['src','scripts','configs','tests','docs','figures','results']
    paths=[ROOT/'README.md',ROOT/'requirements-lock.txt',ROOT/'data/splits.json',ROOT/'data/provenance.json',ROOT/'models/provenance.json']
    for name in folders:paths.extend((ROOT/name).rglob('*'))
    paths.extend((ROOT/'third_party/segment-anything/segment_anything').rglob('*.py'))
    paths.extend([ROOT/'third_party/segment-anything/LICENSE',ROOT/'third_party/segment-anything/README.md'])
    paths.extend(p for p in (ROOT/'submission').iterdir() if p.suffix in {'.pdf','.pptx','.md','.json','.mp4'} and p.name!='package_receipt.json')
    with zipfile.ZipFile(output,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(set(paths)):
            if p.is_file() and '__pycache__' not in p.parts:z.write(p,p.relative_to(ROOT))
    with zipfile.ZipFile(output) as z:assert z.testzip() is None
    (ROOT/'submission/package_receipt.json').write_text(json.dumps({'bytes':output.stat().st_size,'sha256':sha256_file(output),
        'contains':'code, pinned upstream SAM inference source, manifests, all saved predictions, numerical results, figures and submission materials',
        'excluded':'virtual environment, raw public image archives, pretrained checkpoint and temporary installation files; download with scripts/download_data.py'},indent=2))
    print(output,output.stat().st_size,'bytes; CRC check passed')
if __name__=='__main__':main()
