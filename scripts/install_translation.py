"""Download official Argos English-Russian model into the project, never execute it."""
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import zipfile
import requests

ROOT=Path(__file__).resolve().parents[1]
INDEX='https://raw.githubusercontent.com/argosopentech/argospm-index/main/index.json'
def main():
    response=requests.get(INDEX,timeout=60);response.raise_for_status()
    item=next(p for p in response.json() if p['from_code']=='en' and p['to_code']=='ru')
    url=next(u for u in item['links'] if u.startswith('https://'))
    destination=ROOT/'models/translation-en-ru';destination.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=ROOT) as temp:
        archive=Path(temp)/'model.zip'
        with requests.get(url,stream=True,timeout=120) as r:
            r.raise_for_status()
            with archive.open('wb') as f:
                for chunk in r.iter_content(1024*1024):f.write(chunk)
        sha=hashlib.sha256(archive.read_bytes()).hexdigest()
        with zipfile.ZipFile(archive) as z:
            for name in z.namelist():
                if '..' in Path(name).parts or Path(name).is_absolute():raise ValueError('Unsafe model archive')
            z.extractall(Path(temp)/'extracted')
        model=next((Path(temp)/'extracted').rglob('metadata.json')).parent
        shutil.copytree(model,destination,dirs_exist_ok=True)
    (destination/'provenance.json').write_text(json.dumps({'index':INDEX,'url':url,'sha256':sha,'package':item},ensure_ascii=False,indent=2),encoding='utf-8')
    print(destination,sha)
if __name__=='__main__':main()
