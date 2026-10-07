"""Download official archives/weights and mirrored public Pet test images."""
from concurrent.futures import ThreadPoolExecutor
from src.data import ROOT,download,safe_extract
import json

URLS={
 'iseg_images':('https://robots.ox.ac.uk/~vgg/data/iseg/data/images.tgz','data/raw/iseg/images.tgz'),
 'iseg_gt':('https://robots.ox.ac.uk/~vgg/data/iseg/data/images-gt.tgz','data/raw/iseg/images-gt.tgz'),
 'pet_annotations':('https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz','data/raw/pets/annotations.tar.gz'),
 'pet_images':('https://huggingface.co/datasets/timm/oxford-iiit-pet/resolve/main/data/test-00000-of-00001.parquet','data/raw/pets/test.parquet'),
 'sam':('https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth','models/sam_vit_b_01ec64.pth')
}
def fetch(item):
    key,(url,path)=item
    print('Downloading',key,flush=True)
    if key=='pet_images':
        receipt=(ROOT/path).with_suffix('.parquet.json')
        if not (ROOT/path).exists() or not receipt.exists():
            from scripts.range_download import range_download
            range_download(url,ROOT/path,workers=8)
        meta=json.loads(receipt.read_text())
        from src.data import sha256_file
        if sha256_file(ROOT/path)!=meta['sha256']:raise IOError('Mirror hash mismatch')
    else:meta=download(url,ROOT/path)
    if key not in {'sam','pet_images'}:safe_extract(ROOT/path,(ROOT/path).parent)
    print('Verified',key,meta['bytes'],flush=True)
    return key,meta
if __name__=='__main__':
    with ThreadPoolExecutor(max_workers=3) as pool:entries=dict(pool.map(fetch,URLS.items()))
    (ROOT/'data/provenance.json').write_text(json.dumps(entries,indent=2))
