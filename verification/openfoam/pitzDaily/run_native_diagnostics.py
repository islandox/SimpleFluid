#!/usr/bin/env python3
"""Authenticate and replay retained pitzDaily endpoints; never qualify tolerances.

Run in the usual verification/environments.sh build environment. Optional profile
sampling uses the existing evidence branch's common_sampling module supplied by
--sampling-tools; it must run with OpenFOAM.com v2606 loaded.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

STUDIES=('baseline-001','stationarity-001','half-step-001','quarter-step-001')
COEFFICIENTS=(1e-4,5e-5,2.5e-5,0.0)
REPO=Path(__file__).resolve().parents[3]

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda:f.read(1024*1024),b''):h.update(data)
    return h.hexdigest()

def save(path,data):
    path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')

def authenticate(directory):
    index=directory/'checksums.json'
    manifest=json.loads(index.read_text())
    scope_path=directory/'input_scope.json'
    original_index_sha256=None
    if scope_path.exists():
        scope=json.loads(scope_path.read_text())
        if scope.get('kind')!='native-replay-input-subset-v1':raise ValueError('Unknown input subset scope')
        original_path=directory/'original-checksums.json'
        original_index_sha256=digest(original_path)
        if original_index_sha256!=scope.get('original_checksum_manifest_sha256'):
            raise ValueError('Original checksum manifest changed')
        original=json.loads(original_path.read_text())
        if any(original.get(name)!=value for name,value in manifest.items()):
            raise ValueError('Subset is not authenticated by original checksum manifest')
    root=directory.resolve()
    for name, expected in manifest.items():
        path=(root/name).resolve()
        if not path.is_relative_to(root):raise ValueError('Checksum path escapes study')
        if not path.is_file() or digest(path)!=expected:raise ValueError(f'Checksum mismatch: {path}')
    required=['settings.json','diagnostics.json','provenance.json','binary.json','command.json']
    required += [f'result/simplefluid_cells_rank{rank}.csv' for rank in range(3)]
    if any(name not in manifest for name in required):raise ValueError('Required replay evidence is not authenticated')
    settings=json.loads((root/'settings.json').read_text())
    endpoint=json.loads((root/'diagnostics.json').read_text())
    if not endpoint['steady_reported'] or endpoint['reported_unconverged_steps']:
        raise ValueError('Input endpoint did not pass its historical solver checks')
    if (settings['mesh_divisor'],settings['mesh_multiplier'],settings['modified_pressure'],settings['openfoam_outlet_stress'])!=(1,2,True,True):
        raise ValueError('Campaign geometry/pressure/outlet mismatch')
    dt=float(re.search(r'final_dt=([^,\s]+)',endpoint['steady_diagnostics'])[1])
    return dict(index_sha256=digest(index),original_index_sha256=original_index_sha256,verified_files=len(manifest),settings=settings,endpoint=endpoint,dt=dt,
                original_source=json.loads((root/'provenance.json').read_text()),original_binary=json.loads((root/'binary.json').read_text()))

def executable_mapping_paths(content):
    """Only code mappings are provenance: MPI shared-memory buffers are mutable data."""
    paths=set()
    for line in content.splitlines():
        fields=line.split(maxsplit=5)
        if len(fields)==6 and 'x' in fields[1] and fields[5].startswith('/') and Path(fields[5]).is_file():
            paths.add(fields[5])
    return paths

def runtime_maps(executable, run_directory, captured, digest_cache):
    """Capture mappings of actual rank executables, not only ldd predictions."""
    for item in Path('/proc').iterdir():
        if not item.name.isdigit():continue
        try:
            if (item/'exe').resolve()!=executable or (item/'cwd').resolve()!=run_directory:continue
            content=(item/'maps').read_text()
            paths=executable_mapping_paths(content)
            observed=captured.setdefault(item.name,{})
            for name in sorted(paths):
                info=Path(name).stat();key=(name,info.st_size,info.st_mtime_ns,info.st_ino)
                if key not in digest_cache:digest_cache[key]=digest(name)
                value=digest_cache[key]
                if name in observed and observed[name]!=value:
                    raise RuntimeError('Executable/library changed during diagnostic run: '+name)
                observed[name]=value
            (run_directory/f'process-{item.name}-maps.txt').write_text(content)
        except (OSError,PermissionError):continue

def execute(args,study,verified,run_name,coefficient,steps):
    directory=(args.output/study/run_name).resolve()
    directory.mkdir(parents=True,exist_ok=False)
    env=dict(os.environ)
    # Explicit clean case settings: no inherited pitzDaily override can change this study.
    env={k:v for k,v in env.items() if not k.startswith('SIMPLEFLUID_PITZ_')}
    settings=verified['settings']
    values=dict(MESH_DIVISOR=1,MESH_MULTIPLIER=2,DT=verified['dt'],STEPS=max(steps,1),STEADY_STATE=0,
        LINEAR_TOLERANCE=settings['linear_tolerance'],TRANSPORT_BACKEND=settings['transport_backend'],
        TRANSPORT_PRECONDITIONER=settings['transport_preconditioner'],PRESSURE_MODE=settings['pressure_mode'],
        MODIFIED_PRESSURE=1,OPENFOAM_OUTLET_STRESS=1,REPLAY_PREFIX=str(args.campaign/study/'result/simplefluid_cells'),
        REPLAY_RANKS=3,DIAGNOSTIC_ONLY=int(steps==0),DIAGNOSTICS_PREFIX=str(directory/'diagnostics'),
        DIAGNOSTICS_INTERVAL=10,DIAGNOSTIC_FIELDS=int(steps==0),OUTPUT_PREFIX=str(directory/'simplefluid_cells'))
    if coefficient is not None:values['DIAGNOSTIC_COEFFICIENT_DT']=coefficient
    env.update({'SIMPLEFLUID_PITZ_'+k:str(v) for k,v in values.items()})
    command=['mpiexec','-n',str(args.ranks),str(args.executable)]
    if args.cpus:command=['taskset','-c',args.cpus]+command
    receipt=dict(argv=command,cwd=str(directory),environment={k:v for k,v in env.items() if k.startswith('SIMPLEFLUID_PITZ_')},
                 started_utc=datetime.now(timezone.utc).isoformat(),executable_sha256=digest(args.executable),
                 original_time_s=verified['endpoint']['summary']['time_s'],original_dt=verified['dt'],
                 note='Continuation is bounded diagnostic evidence, not a renewed stationarity qualification.')
    save(directory/'command.json',receipt)
    captured={};digest_cache={}
    with (directory/'solver.log').open('w') as log:
        process=subprocess.Popen(command,cwd=directory,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            while process.poll() is None:
                runtime_maps(args.executable,directory,captured,digest_cache)
                time.sleep(.2)
        except BaseException as error:
            if process.poll() is None:
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGKILL);process.wait()
            receipt.update(returncode=process.returncode,error=str(error),finished_utc=datetime.now(timezone.utc).isoformat())
            save(directory/'command.json',receipt);save(directory/'actual-runtime-mappings.json',captured)
            raise
    receipt.update(returncode=process.returncode,finished_utc=datetime.now(timezone.utc).isoformat())
    save(directory/'command.json',receipt);save(directory/'actual-runtime-mappings.json',captured)
    save(directory/'checksums.json',{str(p.relative_to(directory)):digest(p) for p in directory.rglob('*') if p.is_file() and p.name!='checksums.json'})
    if process.returncode:raise RuntimeError(f'{study}/{run_name} failed; retained logs and receipt')
    if len(captured)<args.ranks:raise RuntimeError(f'{study}/{run_name}: actual rank executable mappings incomplete')
    final=directory/f'diagnostics_step{steps}.json'
    if not final.exists():raise RuntimeError(f'{study}/{run_name}: final diagnostic summary missing')
    result=json.loads(final.read_text(),parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite diagnostic JSON')))
    if result['step_index']!=steps:raise RuntimeError('Diagnostic endpoint step mismatch')
    if 'linear_converged=no' in (directory/'solver.log').read_text():raise RuntimeError('Continuation contains unconverged solves')
    if steps:
        with (directory/'diagnostics_step_history.csv').open() as f:history=list(csv.DictReader(f))
        if len(history)!=steps or [int(row['step']) for row in history]!=list(range(1,steps+1)):
            raise RuntimeError('Clipping history is incomplete')
        if any(float(row['dt_s'])!=verified['dt'] for row in history):raise RuntimeError('Continuation timestep changed')
    print(f'Completed {study}/{run_name}',flush=True)

def summarize(args):
    rows=[]
    for p in sorted(args.output.glob('*/frozen-*/diagnostics_step0.json')):
        d=json.loads(p.read_text());row=dict(study=p.parent.parent.name,coefficient_dt=d['pressure_coefficient_dt_s'])
        for name,r in [('ux',d['momentum'][0]),('uy',d['momentum'][1]),('uz',d['momentum'][2]),('k',d['k']),('epsilon',d['secondary']),('continuity',d['reconstructed_continuity'])]:
            for key in ('volume_rms','max_density','scale','dimensionless'):row[name+'_'+key]=r[key]
        squares={'velocity':0.,'pressure':0.,'reconstructed':0.};maxima=dict.fromkeys(squares,0.)
        count=0
        for f in sorted(p.parent.glob('diagnostics_step0_faces_rank*.csv')):
            with f.open() as stream:
                for face in csv.DictReader(stream):
                    count+=1
                    for name in squares:
                        value=float(face[name+'_flux_m3_s']);squares[name]+=value*value;maxima[name]=max(maxima[name],abs(value))
        row['owned_faces']=count
        for name in squares:row[name+'_flux_l2']=squares[name]**.5;row[name+'_flux_max']=maxima[name]
        rows.append(row)
    save(args.output/'frozen-measurements.json',rows)
    if rows:
        with (args.output/'frozen-measurements.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    if args.sampling_tools:
        sys.path.insert(0,str(args.sampling_tools));sampler=importlib.import_module('common_sampling')
        comparisons={}
        for study in STUDIES:
            directory=args.output/study/'continuation'
            if not directory.exists():continue
            profile=directory/'common-sampling'
            if not profile.exists():sampler.sample(args.sampling_tools/'openfoam',sorted(directory.glob('simplefluid_cells_rank*.csv')),1,profile,mesh_multiplier=2)
            comparisons[study]={'continued_vs_original':sampler.profile_metrics(args.campaign/study/'common-sampling',profile),
                'continued_vs_reference':sampler.profile_metrics(args.reference,profile)}
        save(args.output/'profile-comparisons.json',comparisons)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--executable',type=Path,required=True)
    parser.add_argument('--mode',choices=('verify','replay','continue','report','all'),default='all')
    parser.add_argument('--ranks',type=int,default=3);parser.add_argument('--steps',type=int,default=200)
    parser.add_argument('--cpus');parser.add_argument('--sampling-tools',type=Path);parser.add_argument('--reference',type=Path)
    args=parser.parse_args()
    for key in ('campaign','output','executable','sampling_tools','reference'):
        value=getattr(args,key)
        if value is not None:setattr(args,key,value.resolve())
    if args.steps<1 or args.steps>200 or args.ranks<1:parser.error('Use 1..200 continuation steps and positive ranks')
    if args.sampling_tools and not args.reference:parser.error('--sampling-tools requires --reference')
    args.output.mkdir(parents=True,exist_ok=True)
    lock=args.output/'campaign.lock'
    with lock.open('x') as f:f.write(str(os.getpid()))
    try:
        if args.mode!='report':
            verified={name:authenticate(args.campaign/name) for name in STUDIES}
            save(args.output/'authenticated-inputs.json',verified)
            provenance=dict(head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
                branch=subprocess.check_output(['git','branch','--show-current'],cwd=REPO,text=True).strip(),
                executable=str(args.executable),executable_sha256=digest(args.executable),
                source_files={str(p.relative_to(REPO)):digest(p) for p in (REPO/'src').rglob('*') if p.is_file()},
                collector_sha256=digest(Path(__file__)),qualification='pending_external_qualification')
            cache=args.executable.parents[2]/'CMakeCache.txt'
            if cache.exists():
                provenance['cmake_cache_sha256']=digest(cache)
                (args.output/'CMakeCache.txt').write_bytes(cache.read_bytes())
            save(args.output/'new-source-provenance.json',provenance)
            if args.mode in ('replay','all'):
                for study in STUDIES:
                    for coefficient in COEFFICIENTS:execute(args,study,verified[study],f'frozen-{coefficient:.8g}',coefficient,0)
            if args.mode in ('continue','all'):
                for study in STUDIES:execute(args,study,verified[study],'continuation',None,args.steps)
        if args.mode in ('report','all'):summarize(args)
    finally:lock.unlink()

if __name__=='__main__':main()
