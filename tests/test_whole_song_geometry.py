import numpy as np
import pytest
from dunhuang.research.whole_song_geometry.study import exact_layout,coverage,path_search

@pytest.mark.parametrize('frames',[2,528,3600,3601,7189])
def test_exact_audio_coverage(frames):
 spans=exact_layout(frames);assert spans[0]['interval'][0]==0 and spans[-1]['interval'][1]==frames;assert np.all(coverage(spans,frames)==1)
 assert all(r['interval'][1]-r['interval'][0]==48 for r in spans if r['kind']=='transition')

def test_overlap_and_hole_rejected():
 with pytest.raises(ValueError):coverage([{'interval':[0,3]},{'interval':[2,4]}],4)
 with pytest.raises(ValueError):coverage([{'interval':[0,2]}],4)

def test_path_never_uses_rejected_edge():
 def n(uid,h):return {'unit':{'unit_uid':uid,'source_uid':uid},'heuristic':h}
 pools=[[n('a',1),n('b',0)],[n('c',1),n('d',0)]];good=(True,{'selected':'G0','methods':{'G0':{'gap_D3':1}}});edges={(0,0,0):(None,{}),(0,1,1):good};choice,_=path_search(pools,edges,{'D3':10});assert choice['ids']==[1,1]
 choice,layers=path_search(pools,{}, {'D3':10});assert choice is None and layers[-1]['retained']==0
