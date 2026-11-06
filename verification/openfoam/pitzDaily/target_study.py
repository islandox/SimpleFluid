"""Audit individual retained pitzDaily target studies; never qualify physical limits."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import math
from pathlib import Path
import re
import collect_evidence as e
import common_sampling as s
import evidence_checks


def endpoint(text, settings):
    """Require declared algorithms, full linear convergence and a complete steady window."""
    d=e.diagnostics(text,'simplefluid')
    transport=settings['transport_backend']
    pressure={'cg-symmetric-muelu':('cg','MueLuSymmetric'),
              'gmres-muelu':('gmres','MueLu')}[settings['pressure_mode']]
    if re.findall(r'^transport_linear_solver: backend=(\w+) preconditioner=(\w+)$',text,re.M)!=[(transport,settings['transport_preconditioner'])]:
        raise ValueError('transport solver declaration mismatch')
    declared=re.findall(r'^pressure_linear_solver: backend=(\w+) preconditioner=(\w+) tolerance=([^\s]+)$',text,re.M)
    if len(declared)!=1 or declared[0][:2]!=pressure or float(declared[0][2])!=1e-9:
        raise ValueError('pressure solver declaration mismatch')
    if not d['steady_reported'] or d['reported_unconverged_steps'] or re.search(r'linear_converged=no',text):
        raise ValueError('missing steady endpoint or failed linear solve')
    if d['pressure_convention']!='modified' or d['velocity_neumann']!='zeroGradient_stress' or d['mesh_multiplier']!=2:
        raise ValueError('target pressure/outlet/mesh mismatch')
    if not d['summary'] or int(d['summary']['mesh_divisor'])!=1 or int(d['summary']['mpi_ranks'])!=3:
        raise ValueError('target endpoint mismatch')
    steps=int(d['summary']['steps'])
    if not settings['steady_min_steps']<=steps<=settings['steps']:
        raise ValueError('endpoint outside requested step budget')
    matches=re.findall(r'^steady_step=(\d+)/(\d+) .*? dt=([^\s]+).*?steady_window=(\d+)/(\d+) steady=yes linear_converged=yes.*?linear_tolerance=([^\s]+)',text,re.M)
    if not matches or tuple(map(int,matches[-1][3:5]))!=(settings['steady_consecutive_steps'],)*2 or int(matches[-1][0])!=steps:
        raise ValueError('incomplete final steady window')
    if int(matches[-1][1])!=settings['steps']:
        raise ValueError('logged budget differs from settings')
    values={k:float(v) for k,v in re.findall(r'(final_update_rate|final_dt|last_transport_linear_tolerance)=([^,\s]+)',d['steady_diagnostics'])}
    if set(values)!={'final_update_rate','final_dt','last_transport_linear_tolerance'} or not all(math.isfinite(v) and v>=0 for v in values.values()):
        raise ValueError('missing or invalid final diagnostics')
    if values['final_update_rate']>settings['steady_tolerance'] or values['last_transport_linear_tolerance']!=settings['linear_tolerance']:
        raise ValueError('endpoint exceeds declared convergence tolerance')
    if not settings['steady_min_dt_s']<=values['final_dt']<=settings['steady_max_dt_s']*(1+1e-6):
        raise ValueError('final time step outside bounds')
    if not math.isfinite(float(matches[-1][5])) or float(matches[-1][5])>max(1e-9,settings['linear_tolerance'])*(1+1e-6):
        raise ValueError('final linear residual above tolerance')
    return dict(d,endpoint_values=values)


def verify(directory):
    directory=Path(directory)
    index=e.read_json(directory/'checksums.json')
    def required(name):
        path=e.inside(directory,name)
        if name not in index or not path.is_file() or e.sha256(path)!=index[name]:
            raise ValueError(f'missing or changed evidence: {name}')
        return path
    for name in index:required(name)
    settings=e.read_json(required('settings.json'))
    environment=e.read_json(required('provenance.json'))['environment']
    for key,value in settings.items():
        name={'transport_backend':'TRANSPORT_BACKEND','pressure_mode':'PRESSURE_MODE'}.get(key,e.ENV_NAMES.get(key))
        if name is None:raise ValueError(f'unknown target setting: {key}')
        expected=str(int(value)) if isinstance(value,bool) else str(value)
        if environment.get('SIMPLEFLUID_PITZ_'+name)!=expected:
            raise ValueError(f'launch environment does not match settings: {key}')

    receipt=e.read_json(required('command.json'))
    if receipt['returncode']!=0:raise ValueError('solver command failed')
    provenance=e.read_json(required('provenance.json'))
    if provenance['git_status'] or provenance['source_diff'] or not re.fullmatch('[0-9a-f]{40}',provenance['revision']):
        raise ValueError('source is not the clean frozen revision')
    required('run-script.py');required('binary.json');required('CMakeCache.txt')
    runtime=e.read_json(required('runtime-library-provenance.json'))
    binary=e.read_json(required('binary.json'))
    if len(runtime['processes'])!=3 or any(record['mapped_file_sha256'].get(record['exe'])!=binary['sha256'] for record in runtime['processes']):
        raise ValueError('actual loaded executable provenance mismatch')
    if not datetime.fromisoformat(receipt['started_utc'])<=datetime.fromisoformat(runtime['captured_utc'])<=datetime.fromisoformat(receipt['finished_utc']):
        raise ValueError('runtime provenance was captured outside the solver run')

    d=endpoint(required('solver.log').read_text(),settings)
    paths=sorted((directory/'result').glob('simplefluid_cells_rank*.csv'))
    if len(paths)!=3:raise ValueError('missing rank fields')
    for path in paths:required(path.relative_to(directory).as_posix())
    quality=evidence_checks.cell_quality(paths,require_physical_fields=True)
    if sum(1 for p in paths for _ in p.open())-len(paths)!=48900:raise ValueError('wrong target cell count')
    s.verify_sampling(e.HERE/'openfoam',paths,1,directory/'common-sampling',lambda path:required(path.relative_to(directory).as_posix()),mesh_multiplier=2)
    return {'status':'pending_external_qualification','stations':{},'diagnostics':d,'field_quality':quality,'revision':provenance['revision'],'binary':e.read_json(required('binary.json'))}



def matching_settings(base, other, role):
    """Keep physics and numerical controls fixed except declared study axes."""
    allowed={'steps','steady_min_steps'}
    if role in ('half','quarter'):
        allowed |= {'dt_s','steady_max_dt_s'}
    if set(base)!=set(other) or any(base[k]!=other[k] for k in base.keys()-allowed):
        raise ValueError(f'{role}: unrelated controls changed')
    if role=='stationarity':
        if other['steady_min_steps']<=base['steps']:
            raise ValueError('stationarity must exceed the baseline maximum budget')
    elif role in ('half','quarter'):
        factor={'half':2,'quarter':4}[role]
        if any(not math.isclose(other[k]*factor,base[k],rel_tol=1e-12) for k in ('dt_s','steady_max_dt_s')):
            raise ValueError('incorrect temporal refinement ratio')
    else:
        raise ValueError('unknown study role')


def summarize(root, reference_case, previous_target):
    """Return measured changes; never create acceptance tolerances."""
    root=Path(root)
    names={'baseline':'baseline-001','stationarity':'stationarity-001',
           'half':'half-step-001','quarter':'quarter-step-001'}
    audited={role:verify(root/name) for role,name in names.items()}
    settings={role:e.read_json(root/name/'settings.json') for role,name in names.items()}
    for role in ('stationarity','half','quarter'):
        matching_settings(settings['baseline'],settings[role],role)
        if audited[role]['binary']['sha256']!=audited['baseline']['binary']['sha256']:
            raise ValueError('campaign executable changed')
    dt={role:audited[role]['diagnostics']['endpoint_values']['final_dt'] for role in names}
    if not math.isclose(dt['baseline'],dt['stationarity'],rel_tol=1e-6):
        raise ValueError('later-time study changed actual time step')
    if not math.isclose(dt['half']*2,dt['baseline'],rel_tol=1e-6) or not math.isclose(dt['quarter']*4,dt['baseline'],rel_tol=1e-6):
        raise ValueError('actual endpoint time steps do not realize refinement')
    cases={role:root/name/'common-sampling' for role,name in names.items()}
    changes={'later_time':s.profile_metrics(cases['baseline'],cases['stationarity']),
             'full_to_half_step':s.profile_metrics(cases['baseline'],cases['half']),
             'half_to_quarter_step':s.profile_metrics(cases['half'],cases['quarter']),
             'linear_algorithm_change':s.profile_metrics(Path(previous_target)/'common-sampling',cases['baseline'])}
    errors={role:s.profile_metrics(reference_case,case) for role,case in cases.items()}
    return {'status':'pending_external_qualification','stations':{},
            'tolerances_assigned':False,'actual_final_time_steps_s':dt,
            'audited_runs':audited,'measured_profile_changes':changes,
            'candidate_reference_errors':errors,
            'limitation':'Observed differences are measurements, not guaranteed uncertainty bounds. The noncontracting spatial trend still requires review.'}

def capture_runtime(root):
    """Bind each study to the actual executable and shared objects in /proc."""
    groups={}
    executable=(e.REPO/'build/gcc/bin/RelWithDebInfo/pitz_daily').resolve()
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():continue
        try:
            if (proc/'exe').resolve()!=executable:continue
            cwd=(proc/'cwd').resolve()
            if not cwd.is_relative_to(root) or cwd.name!='result':continue
            destination=cwd.parent/'runtime-library-provenance.json'
            if destination.exists():continue
            files={line.split(maxsplit=5)[5] for line in (proc/'maps').read_text().splitlines()
                   if len(line.split(maxsplit=5))==6 and line.split(maxsplit=5)[5].startswith('/')}
            record={'pid':int(proc.name),'exe':str(executable),'cwd':str(cwd),
                    'affinity':(proc/'status').read_text().split('Cpus_allowed_list:')[1].splitlines()[0].strip(),
                    'mapped_file_sha256':{name:e.sha256(Path(name)) for name in sorted(files) if Path(name).is_file()}}
            groups.setdefault(destination,[]).append(record)
        except (FileNotFoundError,PermissionError):continue
    for destination,records in groups.items():
        if len(records)==3:
            e.write_json(destination,{'captured_utc':datetime.now(timezone.utc).isoformat(),'processes':records})

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path);a=p.parse_args()
    print(__import__('json').dumps(verify(a.directory),indent=2))
if __name__=='__main__':main()
