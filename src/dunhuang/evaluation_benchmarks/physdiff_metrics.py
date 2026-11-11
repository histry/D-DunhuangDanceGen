"""PhysDiff Appendix A reimplementation with explicit unresolved protocol choices.

Not author-code parity or a direct leaderboard score. No joint proxy is silently
substituted for the required mesh. Tolerance and contact population are separate.
"""
import numpy as np
SOURCE='https://arxiv.org/html/2212.02500v3#A1'

def metrics(vertices,joints,floor_y,contact_height=.05,tolerance=.005,contact_mask=None):
    j=np.asarray(joints)
    if j.ndim!=3 or j.shape[1:]!=(24,3) or len(j)<2 or not np.isfinite(j).all():
        raise ValueError('Finite SMPL24 joints required')
    if not np.isfinite(floor_y) or contact_height<0 or tolerance<0:raise ValueError('Explicit finite floor and nonnegative protocol thresholds required')
    result=dict(schema='physdiff_appendix_reimplementation_v1',source=SOURCE,official_code_parity=False,
        floor_y_m=float(floor_y),mesh_available=vertices is not None,contact_height_m=float(contact_height),
        geometry_tolerance_m=float(tolerance),tolerance_rule='ignore depths <=5mm; retain full depth beyond tolerance',
        skate_reduction='conditional mean over foot-joint adjacent-frame contact pairs',
        foot_joint_indices=[7,8,10,11],units='mm per frame; no FPS multiplication',direct_paper_leaderboard_comparable=False)
    feet=j[:,[7,8,10,11]]
    contacts=feet[:,:,1]-floor_y<=contact_height if contact_mask is None else np.asarray(contact_mask,bool)
    if contacts.shape!=(len(j),4):raise ValueError('Contact population [T,4] required')
    pairs=contacts[:-1]&contacts[1:]
    displacement=np.linalg.norm(feet[1:,:,[0,2]]-feet[:-1,:,[0,2]],axis=-1)*1000
    skate=float(displacement[pairs].mean()) if pairs.any() else None
    result.update(skate_mm=skate,skate_contact_pairs=int(pairs.sum()),skate_possible_pairs=int(pairs.size),
        skate_frame_normalized_mm=float(np.where(pairs,displacement,0).mean()),
        contact_population='explicit external observation mask' if contact_mask is not None else 'registered geometric height rule, NOT verified static/sliding semantics')
    if vertices is None:
        result.update(status='MISSING_MESH',penetrate_mm=None,float_mm=None,phys_err_mm=None)
        return result
    v=np.asarray(vertices)
    # Streaming callers may pass only the exact minimum mesh height per frame.
    if v.shape==(len(j),):minimum=v-floor_y
    elif v.ndim==3 and len(v)==len(j) and v.shape[2]==3:minimum=v[:,:,1].min(1)-floor_y
    else:raise ValueError('Mesh vertices or exact per-frame minima required')
    if not np.isfinite(minimum).all():raise ValueError('Nonfinite mesh')
    penetration=np.maximum(-minimum,0);floating=np.maximum(minimum,0)
    pen=float(np.where(penetration>tolerance,penetration,0).mean()*1000)
    flo=float(np.where(floating>tolerance,floating,0).mean()*1000)
    result.update(status='COMPUTED_PAPER_DEFINITION_REIMPLEMENTATION',penetrate_mm=pen,float_mm=flo,
        phys_err_mm=pen+flo+skate if skate is not None else None,
        # Appendix A does not specify whether 5mm is subtracted or is a dead zone.
        alternate_subtract_tolerance=dict(penetrate_mm=float(np.maximum(penetration-tolerance,0).mean()*1000),
            float_mm=float(np.maximum(floating-tolerance,0).mean()*1000)))
    return result
