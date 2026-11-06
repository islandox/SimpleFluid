"""Run the finite 48,900-cell convergence campaign; never qualify physical limits."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import collect_evidence as e
import common_sampling as sampling
import reference_field_gate as gate
import target_study


def authenticated(directory):
    index=e.read_json(directory/'checksums.json')
    for name,digest in index.items():
        if e.sha256(e.inside(directory,name))!=digest:
            raise ValueError(f'changed prior evidence: {directory}/{name}')
    return index



def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--previous',type=Path,required=True)
    p.add_argument('--resume-starts',action='store_true',help='wait for the two already launched starting runs')
    p.add_argument('--delivery',type=Path)
    a=p.parse_args();root=a.root.resolve();previous=a.previous.resolve()
    if not a.resume_starts:root.mkdir(parents=True,exist_ok=False)
    if not root.is_dir():raise ValueError('missing campaign directory')
    if (root/'campaign-plan.json').exists():raise ValueError('campaign already has a plan')
    settings=previous/'native-fine-002/settings.json'
    selection=previous/'reviewed-reference-sources.json'
    shutil.copy2(selection,root/'reference-selection.json')
    reviewed=e.read_json(selection)
    reference=previous/reviewed['references']['primary']['case']
    plan={'schema_version':1,'started_utc':datetime.now(timezone.utc).isoformat(),
          'reference_selection_sha256':e.sha256(selection),'base_settings_sha256':e.sha256(settings),
          'source_revision':e.output(['git','rev-parse','HEAD']),
          'roles':{'baseline-001':{'steps':6000,'min_steps':20,'dt':1e-4},
                   'stationarity-001':{'steps':8000,'min_steps':6100,'dt':1e-4},
                   'half-step-001':{'steps':12000,'min_steps':20,'dt':5e-5},
                   'quarter-step-001':{'steps':26000,'min_steps':20,'dt':2.5e-5}},
          'qualification_policy':'measure only; pending_external_qualification; no physical tolerances'}
    e.write_json(root/'campaign-plan.json',plan)
    shutil.copy2(__file__,root/'campaign-script.py')
    deadline=time.monotonic()+48*3600

    def wait_for_run(name):
        while not (root/name/'checksums.json').exists():
            if time.monotonic()>deadline:raise TimeoutError(name)
            time.sleep(15)
        target_study.verify(root/name)

    def launch(name,cpus):
        role=plan['roles'][name]
        cmd=[sys.executable,str(e.HERE/'run_target_study.py'),'--settings',str(settings),
             '--output',str(root/name),'--steps',str(role['steps']),
             '--min-steps',str(role['min_steps']),'--dt',str(role['dt'])]
        env=dict(os.environ,PITZ_CPUS=cpus)
        with (root/(name+'-launcher.log')).open('w') as stream:
            result=subprocess.run(cmd,cwd=e.REPO,env=env,stdout=stream,stderr=subprocess.STDOUT)
        if result.returncode:raise ValueError(f'{name} failed; see retained launcher/solver logs')
        wait_for_run(name)

    def lane(first,second,cpus):
        if a.resume_starts:wait_for_run(first)
        else:launch(first,cpus)
        launch(second,cpus)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(lane,'baseline-001','half-step-001','0,2,4'),
                 pool.submit(lane,'stationarity-001','quarter-step-001','6,8,10')]
        while not all(f.done() for f in futures):
            target_study.capture_runtime(root)
            status={}
            for name in plan['roles']:
                path=root/name/'solver.log'
                lines=path.read_text(errors='replace').splitlines() if path.exists() else []
                status[name]=next((line for line in reversed(lines) if line.startswith(('steady_step=','steady_state_search: reached='))),'queued or building')
            e.write_json(root/'campaign-progress.json',{'observed_utc':datetime.now(timezone.utc).isoformat(),'runs':status})
            print(json.dumps(status),flush=True)
            time.sleep(60)
        for f in futures:f.result()

    # The reference was selected before these errors were measured. Recheck all
    # raw-field hashes and the independent field history before using it.
    if e.sha256(selection)!=plan['reference_selection_sha256']:
        raise ValueError('reference selection changed during campaign')
    primary=reviewed['references']['primary']
    authenticated(reference.parent)
    history=previous/primary['stationarity_case']
    authenticated(history.parent)
    field_proof=gate.window_summary(history,primary['mpi_ranks'],
                                   int(primary['retained_profile_time']),
                                   reviewed['numerical_field_controls'])
    if not field_proof['passed'] or field_proof['cells']!=48900:
        raise ValueError('reviewed physical reference no longer passes its field test')
    old_target=previous/'native-fine-002'
    old_index=authenticated(old_target)
    fields=sorted((old_target/'result').glob('simplefluid_cells_rank*.csv'))
    def required(path):
        if path.relative_to(old_target).as_posix() not in old_index:
            raise ValueError('unauthenticated previous target input')
        return path
    sampling.verify_sampling(e.HERE/'openfoam',fields,1,old_target/'common-sampling',required,mesh_multiplier=2)
    summary=target_study.summarize(root,reference,old_target)
    summary['reference_field_proof']=field_proof
    e.write_json(root/'target-measurements.json',summary)
    text=['# 48,900-cell target study','',
          'All four native studies passed the retained convergence and evidence audits.',
          'Physical qualification remains pending. No acceptance tolerances were assigned.',
          'The wall-function regime changes and noncontracting spatial trend still require review.','',
          '| Study | Steps | Time (s) | Final dt (s) |','|---|---:|---:|---:|']
    for role,record in summary['audited_runs'].items():
        d=record['diagnostics'];text.append(f"| {role} | {d['summary']['steps']} | {d['summary']['time_s']} | {d['endpoint_values']['final_dt']} |")
    text+=['','Candidate errors and all profile changes are retained in target-measurements.json.',
           'These are measurements; the workflow deliberately does not convert them into physical limits.']
    (root/'TARGET-REVIEW.md').write_text('\n'.join(text)+'\n')
    e.write_json(root/'campaign-completion.json',{'completed_utc':datetime.now(timezone.utc).isoformat(),
                 'status':'studies_completed_physical_qualification_pending','stations':{}})
    if a.delivery:
        destination=a.delivery.resolve();destination.mkdir(parents=True,exist_ok=True)
        for name in ('target-measurements.json','TARGET-REVIEW.md','campaign-completion.json','campaign-plan.json'):
            shutil.copy2(root/name,destination/name)
        archive=shutil.make_archive(str(destination/'target-evidence'),'gztar',root.parent,root.name)
        (destination/'target-evidence.sha256').write_text(e.sha256(Path(archive))+'  '+Path(archive).name+'\n')
    print('Completed numerical studies; physical qualification remains pending.',flush=True)


if __name__=='__main__':
    try:main()
    except Exception as error:
        print(f'CAMPAIGN FAILED CLOSED: {error}',file=sys.stderr,flush=True)
        raise
