import numpy as np
from dunhuang.research.gtw_kinematic_baseline.features import ReferenceCost,METHODS,select,descriptors

def fixture():
    from scipy.spatial.transform import Rotation
    t=np.arange(90)/30;v=np.zeros((90,22,3));v[:,16,2]=.6*np.sin(t*3)
    R=Rotation.from_rotvec(v.reshape(-1,3)).as_matrix().reshape(90,22,3,3)
    parents=np.arange(22)-1;J=np.column_stack([np.arange(22)*.03,np.arange(22)*.02,np.zeros(22)])
    return R,J,parents

def test_identity_is_reference_knn_minimum_and_actual_shift_changes_cost():
    R,J,p=fixture();m=ReferenceCost(R,J,p);same=m.score(R)[0]
    changed=R.copy();changed[:,16]=R[::-1,16];different=m.score(changed)[0]
    assert all(different[k]>same[k] for k in METHODS)

def test_chest_coordinates_remove_common_global_rotation():
    from scipy.spatial.transform import Rotation
    R,J,p=fixture();other=R.copy();other[:,0]=Rotation.from_rotvec([.1,.3,-.2]).as_matrix()@R[:,0]
    a=descriptors(R,J,p);b=descriptors(other,J,p)
    assert np.allclose(a[0],b[0],atol=1e-12)

def test_coarse_queries_are_exact_full_query_subsample_and_selection_budget():
    R,J,p=fixture();m=ReferenceCost(R,J,p);full,coarse=m.score(R);only=m.score(R,coarse_only=True)[0]
    assert all(abs(only[k]-coarse[k])<1e-12 for k in METHODS)
    rows=[dict(id=str(i),eligible=True,scores={'GTWLocal':3-i},coarse={'GTWLocal':i}) for i in range(3)]
    assert select(rows,'GTWLocal')['id']=='2' and select(rows,'GTWLocal',1)['id']=='0'
    assert select([dict(eligible=False)],'GTWLocal') is None
