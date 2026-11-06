#!/usr/bin/env python3
"""Independently verify pitzDaily reference field stationarity.

The controls are numerical stopping criteria, never physical error limits.
This module is self-contained so copied OpenFOAM cases remain executable.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess

FIELDS=('U','p','k','epsilon','nut')


def write(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def digest(path):
    result=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            result.update(block)
    return result.hexdigest()


def validate_controls(value):
    expected={'schema_version','probe_steps','required_consecutive_steps','scaled_update_tolerance','burnin_steps','maximum_burnin_cycles'}
    if set(value)!=expected or value['schema_version']!=1:
        raise ValueError('unsupported reference field-control schema')
    for key in ('probe_steps','required_consecutive_steps','burnin_steps','maximum_burnin_cycles'):
        if type(value[key]) is not int or value[key]<(0 if key=='maximum_burnin_cycles' else 1):
            raise ValueError(f'invalid reference {key}')
    if value['probe_steps']<value['required_consecutive_steps']:
        raise ValueError('reference probe is shorter than its convergence window')
    tolerance=value['scaled_update_tolerance']
    if type(tolerance) not in (float,int) or not math.isfinite(tolerance) or tolerance<=0:
        raise ValueError('invalid reference field-change tolerance')
    return value


def field_values(path):
    text=path.read_text()
    match=re.search(r'internalField\s+nonuniform\s+List<(scalar|vector)>\s+(\d+)\s*\((.*?)\)\s*;',text,re.S)
    if not match:
        raise ValueError(f'expected an ASCII nonuniform reference field: {path}')
    values=[float(x) for x in re.findall(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?',match[3])]
    width=3 if match[1]=='vector' else 1
    if (path.name=='U') != (width==3):
        raise ValueError(f'incorrect reference field type: {path}')
    if len(values)!=int(match[2])*width or not values or not all(math.isfinite(v) for v in values):
        raise ValueError(f'invalid reference field: {path}')
    if path.name in ('k','epsilon') and min(values)<=0:
        raise ValueError(f'nonpositive reference turbulence field: {path}')
    if path.name=='nut' and min(values)<0:
        raise ValueError(f'negative reference eddy viscosity: {path}')
    return values


def field_paths(case,ranks,time,field):
    if type(ranks) is not int or not 1<=ranks<=6 or type(time) is not int or time<0 or field not in FIELDS:
        raise ValueError('invalid reference field role')
    return [case/(f'processor{i}' if ranks>1 else '')/str(time)/field for i in range(ranks)]


def window_summary(case,ranks,start,controls,required=lambda path:path):
    controls=validate_controls(controls)
    result={'start_time':start,'end_time':start+controls['probe_steps'],'ranks':ranks,'fields':{}}
    cells=None
    for field in FIELDS:
        def read(time):
            rows=[]
            for path in field_paths(case,ranks,time,field):
                rows.extend(field_values(required(path)))
            return rows
        old=read(start)
        if field=='U':
            if len(old)%3:
                raise ValueError('invalid reference velocity dimensions')
            cells=len(old)//3
        elif len(old)!=cells:
            raise ValueError('inconsistent reference scalar cell count')
        changes=[]
        for time in range(start+1,start+controls['probe_steps']+1):
            new=read(time)
            if len(new)!=len(old):
                raise ValueError('reference field topology changed during stationarity probe')
            absolute=max(abs(b-a) for a,b in zip(old,new))
            scale=max(1.,max(map(abs,old)),max(map(abs,new)))
            changes.append({'time':time,'max_absolute_change':absolute,'global_field_scale':scale,'scaled_change':absolute/scale})
            old=new
        last=changes[-controls['required_consecutive_steps']:]
        result['fields'][field]={'changes':changes,'maximum_final_window_change':max(row['scaled_change'] for row in last)}
    result['cells']=cells
    result['passed']=all(v['maximum_final_window_change']<=controls['scaled_update_tolerance'] for v in result['fields'].values())
    return result


def latest_time(case,ranks):
    sets=[]
    for rank in range(ranks):
        folder=case/(f'processor{rank}' if ranks>1 else '')
        sets.append({int(path.name) for path in folder.iterdir() if path.is_dir() and re.fullmatch(r'\d+',path.name)})
    common=set.intersection(*sets)
    if not common:
        raise ValueError('no common retained reference checkpoint')
    return max(common)


def run(case,ranks,controls_path):
    if os.environ.get('WM_PROJECT_VERSION')!='v2606':
        raise ValueError('load OpenFOAM.com v2606 before checking reference fields')
    case=case.resolve()
    controls=validate_controls(json.loads(controls_path.read_text()))
    destination=case/'reference-diagnostics'
    destination.mkdir(exist_ok=False)
    control=case/'system/controlDict'
    solution=case/'system/fvSolution'
    originals={control:control.read_text(),solution:solution.read_text()}
    for path,text in originals.items():
        (destination/(path.name+'.input')).write_text(text)
    receipts=[]
    summary={'status':'failed','controls':controls,'ranks':ranks,'windows':[],'commands':receipts}
    summary['openfoam_version']=os.environ['WM_PROJECT_VERSION']
    summary['executables']={}
    for name in ('foamDictionary','simpleFoam','postProcess',*(['mpirun','reconstructPar'] if ranks>1 else [])):
        executable=shutil.which(name)
        if executable is None:
            raise ValueError(f'missing reference tool: {name}')
        path=Path(executable).resolve()
        summary['executables'][name]={'path':str(path),'sha256':digest(path)}
    env=dict(os.environ,PWD=str(case))
    def command(name,args):
        log=destination/(name+'.log')
        started=datetime.now(timezone.utc).isoformat()
        with log.open('w') as stream:
            result=subprocess.run(args,cwd=case,env=env,stdout=stream,stderr=subprocess.STDOUT)
        receipts.append({'name':name,'argv':args,'started_utc':started,'finished_utc':datetime.now(timezone.utc).isoformat(),'returncode':result.returncode,'log':log.relative_to(case).as_posix()})
        write(destination/'summary.json',summary)
        if result.returncode:
            raise ValueError(f'reference command failed: {name}')
        return log
    def configure(prefix,end,interval,residuals):
        text=originals[solution]
        if not residuals:
            text,count=re.subn(r'residualControl\s*\{[^{}]*\}','residualControl {}',text)
            if count!=1:
                raise ValueError('expected one reference residual-control dictionary')
        solution.write_text(text)
        for entry,value in [('startFrom','latestTime'),('endTime',str(end)),('writeInterval',str(interval))]:
            command(f'{prefix}-set-{entry}',['foamDictionary',str(control),'-entry',entry,'-set',value])
        (destination/(prefix+'.controlDict')).write_text(control.read_text())
        (destination/(prefix+'.fvSolution')).write_text(solution.read_text())
    def solve(name):
        args=(['mpirun','-np',str(ranks)] if ranks>1 else [])+['simpleFoam']+(['-parallel'] if ranks>1 else [])
        log=command(name,args)
        with (case/'log.solver').open('a') as stream:
            stream.write(f'\n# reference field protocol: {name}\n')
            stream.write(log.read_text())
        return log
    try:
        if 'SIMPLE solution converged' not in (case/'log.solver').read_text():
            raise ValueError('initial reference solve did not meet residual controls')
        for cycle in range(controls['maximum_burnin_cycles']+1):
            start=latest_time(case,ranks)
            configure(f'probe-{cycle}',start+controls['probe_steps'],1,False)
            solve(f'probe-{cycle}')
            probe=window_summary(case,ranks,start,controls)
            summary['windows'].append(probe)
            write(destination/'summary.json',summary)
            if probe['passed']:
                final_start=latest_time(case,ranks)
                configure('final',final_start+1000,100,True)
                log=solve('final')
                if 'SIMPLE solution converged' not in log.read_text():
                    raise ValueError('final reference residual controls were not met')
                if ranks>1:
                    command('reconstruct',['reconstructPar','-latestTime'])
                command('profiles',['postProcess','-latestTime','-fields','(U p k)'])
                summary.update(status='passed_numerical_field_stationarity',final_time=latest_time(case,ranks),accepted_window_index=cycle)
                break
            if cycle==controls['maximum_burnin_cycles']:
                raise ValueError('reference fields did not become stationary within the retained continuation budget')
            start=latest_time(case,ranks)
            configure(f'burnin-{cycle}',start+controls['burnin_steps']+100,100,False)
            solve(f'burnin-{cycle}')
        write(destination/'summary.json',summary)
    except Exception as error:
        summary['error']=str(error)
        write(destination/'summary.json',summary)
        raise
    finally:
        for path,text in originals.items():
            path.write_text(text)
    return summary


def verify(case,ranks,controls,required=lambda path:path):
    summary=json.loads(required(case/'reference-diagnostics/summary.json').read_text())
    controls=validate_controls(controls)
    if summary['status']!='passed_numerical_field_stationarity' or summary['controls']!=controls or summary['ranks']!=ranks:
        raise ValueError('missing or mismatched reference field-stationarity proof')
    index=summary['accepted_window_index']
    if type(index) is not int or index!=len(summary['windows'])-1 or not 0<=index<=controls['maximum_burnin_cycles']:
        raise ValueError('invalid accepted reference field window')
    expected=summary['windows'][index]
    actual=window_summary(case,ranks,expected['start_time'],controls,required)
    if actual!=expected or not actual['passed']:
        raise ValueError('reference field-stationarity summary does not reproduce')
    for receipt in summary['commands']:
        if receipt['returncode']!=0:
            raise ValueError('failed reference field-protocol command')
        path=(case/receipt['log']).resolve()
        if not path.is_relative_to(case.resolve()):
            raise ValueError('reference command log escapes its case')
        required(path)
    final_log=required(case/'reference-diagnostics/final.log').read_text()
    if 'SIMPLE solution converged' not in final_log:
        raise ValueError('missing final reference residual-control convergence')
    times=re.findall(r'^Time = (\d+)\s*$',final_log,re.M)
    if (type(summary['final_time']) is not int or not times
            or int(times[-1])!=summary['final_time']
            or summary['final_time']<=actual['end_time']):
        raise ValueError('mismatched final reference endpoint')
    for field in FIELDS:
        values=[]
        for path in field_paths(case,ranks,summary['final_time'],field):
            values.extend(field_values(required(path)))
        if len(values)!=actual['cells']*(3 if field=='U' else 1):
            raise ValueError('incomplete final reference field')
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',type=Path,required=True)
    parser.add_argument('--ranks',type=int,required=True)
    parser.add_argument('--controls',type=Path,required=True)
    args=parser.parse_args()
    if not 1<=args.ranks<=6:
        parser.error('reference ranks must be1..6')
    try:
        run(args.case,args.ranks,args.controls)
    except (OSError,ValueError,KeyError,TypeError) as error:
        parser.exit(1,f'reference field verification failed: {error}\n')


if __name__=='__main__':
    main()
