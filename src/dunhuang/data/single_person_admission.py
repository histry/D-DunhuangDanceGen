"""Task admission from declared provenance; a single array is not solo-capture proof."""
def admit(source):
    evidence=source.get("solo_evidence")
    count=source.get("performer_count")
    if count is not None and (type(count) is not int or count<1):raise ValueError("Invalid source performer count")
    if count is not None and count>1:
        solo="EXCLUDED_MULTI_PERFORMER"
    elif count==1 and evidence and source.get("solo_verified") is True:
        solo="VERIFIED_SINGLE_PERSON_METADATA"
    else:solo="UNKNOWN_SINGLE_PERSON_CAPTURE"
    joints=source.get("observed_joints",[])
    if any(type(j) is not int or not 0<=j<24 for j in joints) or len(joints)!=len(set(joints)):raise ValueError("Invalid observed joints")
    return dict(solo_status=solo,solo_admitted=solo=="VERIFIED_SINGLE_PERSON_METADATA",
        observed_joints=joints,unobserved_joints=[j for j in range(24) if j not in joints],
        full151_target_ready=source.get("full151_target_ready") is True and len(joints)==24 and source.get("pairing_status")=="VERIFIED",
        pairing_status=source.get("pairing_status","UNKNOWN"),contact_semantics="UNKNOWN",
        source_only_catalog_eligible=solo=="VERIFIED_SINGLE_PERSON_METADATA" and source.get("dataset_id")=="change",
        human_reviewed=False)

def require_qualified_sources(rows,*,paired=False):
    if not rows:raise ValueError("Empty task population")
    for row in rows:
        q=admit(row)
        if not q["solo_admitted"]:raise ValueError("Single-person provenance is incomplete")
        if paired and (row.get("pairing_status")!="VERIFIED" or not q["full151_target_ready"]):
            raise ValueError("Real paired full151 support required")
    return rows
