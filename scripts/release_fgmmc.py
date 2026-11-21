"""Scalar/identity publication only. Weights, private features and playback stay private."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from dunhuang.research.fgmmc_q01.common import *

def run():
 assert read(ROOT/'receipts/independent.json')['status']=='PASS'
 for phase in ['data']+CFG['models']+['controls','preview','controls_view']:verify(ROOT/phase)
 out=fresh('reports/fgmmc_q01_release');reg=read(ROOT/'data/registration.json');fields=['source_uid','dataset','split','group','genre_uid','dancer_uid','dancer_role','recording_uid','choreography_uid','raw_motion_sha256','audio_sha256','audio_offset_frames','frames','clock_scope']
 write(out/'population.json',dict(summary=reg['summary'],sources=[{k:s[k] for k in fields if k in s} for s in reg['sources']],isolation=reg['isolation'],genre_partition=reg['genre_partition'],dancer_partition=reg['dancer_partition'],initial_metadata_only_split_rejected=reg['initial_metadata_only_split_rejected'],historical_r59_exposure=True,fine_dancer_isolation=reg['fine_dancer_isolation'],new_unseen_music_certified=False))
 for name in ['windows','donors','clocks','schedule','manifest']:write(out/('data_'+name+'.json'),read(ROOT/'data'/(name+'.json')))
 for name in CFG['models']:
  for file in ['training','history']:write(out/(name+'_'+file+'.json'),read(ROOT/name/(file+'.json')))
 for name in ['results','summary','comparisons','qualification','manifest']:write(out/('controls_'+name+'.json'),read(ROOT/'controls'/(name+'.json')))
 for name in ['preview','controls_view']:write(out/(name+'_selections.json'),read(ROOT/name/'selections.json'))
 for name in ['independent','browser_acceptance']:write(out/(name+'.json'),read(ROOT/'receipts'/(name+'.json')))
 write(out/'protocol.json',CFG);launch=Path('outputs/fgmmc_launch');exits={name:int((launch/(name+'.exit')).read_text()) for name in ['main','controls','replay']};assert all(v==0 for v in exits.values());models=[read(ROOT/name/'training.json') for name in CFG['models']];assert len({r['initial_digest'] for r in models})==1 and all(r['steps']==300 for r in models)
 write(out/'runtime.json',dict(status='PASS',scientific_commit='8582331f8e15c3dde89a70000ab4dae15dd3fd22',binding=binding(),exit_status=exits,contract_tests=7,model_parameters=models[0]['parameters'],real_optimizer_updates=900,optimizer_updates_per_arm=300,checkpoint_selection='FINAL_STEP_NO_DEV_SELECTION',data_manifest_sha256=sha(ROOT/'data/manifest.json'),normalizer_sha256=sha(ROOT/'data/normalizer.npz'),weight_sha256={name:sha(ROOT/name/'weights.npz') for name in CFG['models']},parent_role_manifest_sha256=sha('outputs/paper1_role_alignment_01/preview/manifest.json'),new_geometry_solves=0,geometry_scope='EXACT_ALREADY_EXECUTED_PARENT_ARRAYS_AND_HASH_BOUND_BRIDGES; NO_NEW_MOTION_ARRAY_OR_INHERITED_QUALIFICATION_FOR_CHANGED_MOTION',independent_inference_replay='PASS',independent_training_reproduction=False,whole_song='NOT_STARTED',can_route=False,semantic_role_truth='UNKNOWN',artistic_quality='NOT_EVALUABLE',Native24='NOT_EVALUABLE'))
 write(out/'manifest.json',dict(files={p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file()}));print('FG_PUBLIC_JSON',len(list(out.iterdir())),sum(p.stat().st_size for p in out.iterdir()))
if __name__=='__main__':run()
