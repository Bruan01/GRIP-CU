"""Verify portable data/figure agreement, vector exports, fonts and local links."""
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from pypdf import PdfReader
from PIL import Image

HERE=Path(__file__).resolve().parent
PAPER=HERE.parents[1]
FIG=PAPER/'main/figures/sampling_diagnostics_seed2026'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    audit=json.loads((HERE/'independent_validation.json').read_text())
    qa=json.loads((FIG/'qa_report.json').read_text())
    assert json.loads((HERE/'validation.json').read_text())['status']=='pass'
    assert audit['status']=='pass'
    assert audit['source_data_sha256']==qa['_package']['source_data_sha256']
    for name,digest in audit['source_data_sha256'].items():
        assert sha(HERE/'source_data'/name)==digest
    previous=PAPER/'artifacts/sampling_distribution_seed2026/source_data/relation_distribution.csv'
    assert (HERE/'source_data/relation_distribution.csv').read_bytes()==previous.read_bytes()
    profiles=[]
    for p in sorted(FIG.glob('*.pdf')):
        reader=PdfReader(p)
        assert len(reader.pages)==(8 if p.stem=='sampling_diagnostics_all' else 1)
        types=set()
        for page in reader.pages:
            resources=page['/Resources'].get_object()
            for font in resources.get('/Font',{}).get_object().values():
                f=font.get_object(); types.add(str(f['/Subtype']))
                assert f['/Subtype']!='/Type3'
                for df in f.get('/DescendantFonts',[]):
                    types.add(str(df.get_object()['/Subtype']))
            assert len(page.extract_text())>100
        profiles.append({'file':p.name,'pages':len(reader.pages),'font_types':sorted(types)})
    assert len(profiles)==9
    for p in FIG.glob('*.svg'):
        root=ET.parse(p).getroot()
        assert len(root.findall('.//{http://www.w3.org/2000/svg}text'))>5
    for p in FIG.glob('*.png'):
        with Image.open(p) as im: im.verify()
    links=0
    docs=[HERE/'README.md',FIG/'README.md',PAPER/'review/sampling_plot_literature_20261110.md']
    for p in docs:
        for target in re.findall(r'\]\(([^)]+)\)',p.read_text()):
            if '://' in target: continue
            assert (p.parent/target.split('#')[0]).exists(),(p,target)
            links+=1
    assert json.loads((FIG/'render_review.json').read_text())['status']=='accepted'
    result={'status':'pass','pdf_checks':profiles,'svg_files':len(list(FIG.glob('*.svg'))),
            'png_files':len(list(FIG.glob('*.png'))),'local_links_checked':links,
            'source_data_matches_independent_and_figure_qa':True,'previous_marginals_byte_identical':True,
            'file_sha256':{str(p.relative_to(PAPER)):sha(p) for folder in [HERE,FIG] for p in sorted(folder.rglob('*'))
                           if p.is_file() and p.name!='package_check.json' and '__pycache__' not in p.parts}}
    (HERE/'package_check.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='file_sha256'},indent=2))

if __name__=='__main__': main()
