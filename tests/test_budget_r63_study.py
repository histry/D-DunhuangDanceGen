from dunhuang.research.budget_r63.study import selected, summarize

def case():
    return dict(key='x',role='EVAL',left=dict(source_uid='a'),right=dict(source_uid='b'))

def record(g=48,passed=True):
    ratio={'FK_path_length_mean_m_ratio':1.,'FK_motion_range_rms_m_ratio':1.}
    return dict(key='x_g'+str(g),case_key='x',gap_frames=g,status='GEOMETRY_EVALUATED',
                left=dict(retention_vs_fixed48=ratio),right=dict(retention_vs_fixed48=ratio),
                methods=dict(G1=dict(constructible=True,passes_original_screen=passed,
                    full_dynamics=dict(D1_Q95=1.,D2_Q95=2.,D3_Q95=3.))))

def test_failed_prediction_no_fallback_to_qualified_fixed48():
    c=case();cache={('x',48):record()}
    s=selected(c,'BoundaryConditioned',72,cache,[72])
    assert not s['accepted'] and s['record_key'] is None and not s['fallback_used']

def test_full_population_failure_denominator_retained():
    c=case();r=record(passed=False);cache={('x',48):r}
    s=selected(c,'Fixed48',48,cache,[48])
    summary=summarize([c],[r],[s])['EVAL']['policies']['Fixed48']
    assert summary['tasks']==1 and summary['constructible']==1
    assert summary['passes_original_screen']==0 and summary['failed_cases']==['x']

def test_more_frames_not_a_quality_certificate():
    c=case();a,b=record(),record(72);b['methods']['G1']['full_dynamics']['D3_Q95']=10
    cache={('x',48):a,('x',72):b}
    s=selected(c,'ShortestScreened',72,cache,[24,36,48,60,72])
    assert s['accepted'] and not s['retention_guard_passed']
