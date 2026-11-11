import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from dunhuang.alignment.dataset import prepare
source=Path(r'D:\danceRAG\outputs\radius_alignment_v3_delivery\raw\paired_alignment_v3_20261020_av311f\population.json')
print(prepare(source,ROOT/'outputs/sdmr_data',Path(r'D:\danceRAG'),Path(r'D:\danceRAG\assets\datasets')),flush=True)
