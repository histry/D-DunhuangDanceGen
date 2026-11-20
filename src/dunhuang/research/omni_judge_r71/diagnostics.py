"""Post-run structural diagnostics, never music-quality ground truth."""
import json,re
from pathlib import Path
from .prepare import ROOT,SOURCE,read,write

def main():
 from transformers import AutoTokenizer
 tokenizer=AutoTokenizer.from_pretrained(ROOT/'assets/model',local_files_only=True)
 rows=[]
 for stage in ['pilot','pilot_replay','controls']:
  for p in sorted((ROOT/stage).glob('*.json')):
   if not p.stem.isdigit():continue
   r=read(p);raw=r['raw_response'];complete=False
   try:json.loads(raw[raw.index('{'):raw.rindex('}')+1]);complete=True
   except ValueError:pass
   tokens=len(tokenizer.encode(raw,add_special_tokens=False))
   intervals=re.findall(r'"(\d+(?:\.\d+)?\s*-\s*\d+(?:\.\d+)?)"',raw)
   outside=[s for s in intervals if not 0<=float(s.split('-')[0])<float(s.split('-')[1])<=17.6]
   rows.append(dict(stage=stage,index=int(p.stem),condition=r['job']['condition'],complete_json=complete,
      parsed=r['parsed'] is not None,raw_reencoded_tokens=tokens,configured_max_new_tokens=512,
      consistent_with_ceiling_truncation=not complete and tokens>=512,
      raw_out_of_support_interval_strings=outside,
      token_basis='RETOKENIZED_RAW_TEXT; ORIGINAL_EOS_STOP_REASON_NOT_SAVED'))
 write(ROOT/'receipts/response_diagnostics.json',dict(rows=rows,generation_calls=0,human_reviewed=False))
 records=read(SOURCE/'records.json');by={(r['case_key'],r['method']):r for r in records};pairs=[]
 for p in read(ROOT/'registered_pairs.json'):
  if not p['eligible']:continue
  same=by[p['case_key'],'E1W']['full_R_sha256']==by[p['case_key'],p['method']]['full_R_sha256']
  pairs.append(dict(pair_id=p['pair_id'],case_index=p['case_index'],method=p['method'],pilot=p['pilot'],identical_full_rotations=same))
 write(ROOT/'receipts/pair_identity_audit.json',dict(eligible_pairs=len(pairs),distinct_full_rotation_pairs=sum(not p['identical_full_rotations'] for p in pairs),pairs=pairs))
 print('R71_POSTRUN_DIAGNOSTICS',len(rows),'ELIGIBLE_DISTINCT',sum(not p['identical_full_rotations'] for p in pairs),flush=True)

if __name__=='__main__':main()
