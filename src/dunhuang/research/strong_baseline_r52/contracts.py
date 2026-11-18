"""Versioned body-only EventPlan contract shared by both research modules."""
import numpy as np

VERSION='EVENT_PLAN_R52_V1'

def validate(plan):
 if plan['schema']!=VERSION or plan['fps']!=30:raise ValueError('Explicit shared 30Hz Event Plan required')
 a,b=plan['target_interval'];cursor=a
 if b<=a:raise ValueError('Positive target duration')
 if plan['root_truth']!='UNKNOWN' or plan['contact_truth']!='UNKNOWN':raise ValueError('No inferred world Root/contact truth')
 for i,event in enumerate(plan['events']):
  s,e=event['source_interval'];u,v=event['target_interval'];w=np.asarray(event['time_warp'],float)
  if not event['event_uid'] or not event['source_uid'] or not s<e or u!=cursor or v<=u:raise ValueError('Event source identity or target clock')
  if w.ndim!=2 or w.shape[1]!=2 or not np.isfinite(w).all() or not np.all(np.diff(w,axis=0)>0):raise ValueError('Strict shared source-target warp')
  if not np.allclose(w[[0,-1]],[[s,u],[e,v]],rtol=0,atol=1e-7):raise ValueError('Full primitive endpoints must be retained')
  rate=np.diff(w[:,1])/np.diff(w[:,0])
  if rate.min()<.75-1e-7 or rate.max()>1.33+1e-7:raise ValueError('Illegal phase speed')
  if event['phase_scope']!='KINEMATIC_PROXY_NOT_SEMANTIC_TRUTH':raise ValueError('Unverified phase cannot become semantic truth')
  if event['sdmr_prior']['qualified'] or event['sdmr_prior']['consumed_by_denoiser']:raise ValueError('SDMR remains metadata-only Shadow')
  for side in ('left_context','right_context'):
   if event[side]['kind'] not in ('REFERENCE_EVENT','GENERATED_OBSERVATION','UNKNOWN'):raise ValueError('Context identity scope')
   if event[side]['kind']=='GENERATED_OBSERVATION' and not event[side].get('artifact_sha256'):raise ValueError('Generated observation needs artifact evidence')
  cursor=v+event['transition_budget']
  if i==len(plan['events'])-1 and event['transition_budget']!=0:raise ValueError('No terminal fictitious transition')
 if cursor!=b:raise ValueError('Core plus transition does not conserve target frames')
 return plan


def require_task(task,source_valid,loss_valid):
 if task not in ('FULL_REFERENCE_RECONSTRUCTION','HIDDEN_GAP_INBETWEENING'):raise ValueError('Distinct supervised task scope required')
 if task=='FULL_REFERENCE_RECONSTRUCTION' and not np.asarray(source_valid).all():raise ValueError('Identity reconstruction needs full observed source')
 if task=='HIDDEN_GAP_INBETWEENING' and np.asarray(source_valid)[104:152].any():raise ValueError('Hidden target entered source condition')
 if np.asarray(loss_valid)[...,:7].any():raise ValueError('Unknown Root/C4 cannot enter reconstruction loss')
