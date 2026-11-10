#!/usr/bin/env python3
"""Exercise real graded-mesh replay; retain generated evidence outside source."""
from pathlib import Path
import json,os,subprocess,sys,argparse,tempfile
from replay_comparison import compare_replays, format_comparison
parser=argparse.ArgumentParser(description="Native pitzDaily replay, MPI reproducibility, and rejection checks")
parser.add_argument('--executable',type=Path,required=True)
parser.add_argument('--output',type=Path)
parser.add_argument('--mpiexec',default='mpiexec')
parser.add_argument('--parallel-procs',type=int,default=2)
parser.add_argument('--numproc-flag',default='-n')
parser.add_argument('--preflag',action='append',default=[])
parser.add_argument('--postflag',action='append',default=[])
args=parser.parse_args()
if args.parallel_procs < 2:
    parser.error('--parallel-procs must be at least 2')
producer_count=args.parallel_procs
exe=args.executable.resolve()
if args.output:
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
else:
    out=Path(tempfile.mkdtemp(prefix='simplefluid-native-replay-'))
print('Retained integration evidence:',out,flush=True)
base={k:v for k,v in os.environ.items() if not k.startswith('SIMPLEFLUID_PITZ_')}
settings=dict(MESH_DIVISOR=4,MESH_MULTIPLIER=1,DT=1e-6,STEPS=2,STEADY_STATE=0,
              LINEAR_TOLERANCE=1e-9,TRANSPORT_BACKEND='bicgstab',TRANSPORT_PRECONDITIONER='ilut',
              PRESSURE_MODE='cg-symmetric-muelu',MODIFIED_PRESSURE=1,OPENFOAM_OUTLET_STRESS=1,
              DIAGNOSTIC_FIELDS=1,DIAGNOSTICS_INTERVAL=1)
def run(name,ranks,extra,success=True):
    d=out/name;d.mkdir()
    env=base|{'SIMPLEFLUID_PITZ_'+k:str(v) for k,v in (settings|extra|dict(DIAGNOSTICS_PREFIX=d/'diagnostics',OUTPUT_PREFIX=d/'simplefluid_cells')).items()}
    command=[args.mpiexec,args.numproc_flag,str(ranks),*args.preflag,str(exe),*args.postflag]
    with (d/'solver.log').open('w') as log:
        result=subprocess.run(command,cwd=d,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=90)
    (d/'command.json').write_text(json.dumps(dict(argv=command,returncode=result.returncode,environment={k:v for k,v in env.items() if k.startswith('SIMPLEFLUID_PITZ_')}),indent=2))
    if (result.returncode==0)!=success:
        tail='\n'.join((d/'solver.log').read_text(errors='replace').splitlines()[-40:])
        raise RuntimeError(f'{name}: unexpected return code {result.returncode}; solver log: {d/"solver.log"}\n{tail}')
    return d
generated=run('generated',producer_count,{})
initial=json.loads((generated/'diagnostics_step0.json').read_text())
final=json.loads((generated/'diagnostics_step2.json').read_text())
assert initial['stored_flux_available'] is False and initial['stored_continuity'] is None
assert final['stored_flux_available'] is True and len(final['pressure_update_l2_history'])==2
prefix=generated/'simplefluid_cells'
serial=run('serial-replay',1,dict(REPLAY_PREFIX=prefix,REPLAY_RANKS=producer_count,DIAGNOSTIC_ONLY=1))
parallel=run('parallel-replay',producer_count,dict(REPLAY_PREFIX=prefix,REPLAY_RANKS=producer_count,DIAGNOSTIC_ONLY=1))
comparison=compare_replays(serial,parallel)
comparison_path=out/'comparison.json'
comparison_path.write_text(json.dumps(comparison,indent=2)+'\n')
integrated=comparison['integrated_residual_check']
counts=integrated['counts']
print(f'Integrated accumulation-only diagnostic status={integrated["status"]} '
      f'checked={counts["checked"]} within={counts["within"]} exceeds={counts["exceeds"]} '
      f'unavailable={counts["unavailable"]} tau=0 upstream=unassessed',flush=True)
if not comparison['passed']:
    print(format_comparison(comparison),file=sys.stderr)
    print(f'Retained replay comparison: {comparison_path}',file=sys.stderr)
    raise AssertionError(f'{len(comparison["failures"])} serial/parallel replay summaries failed. Evidence written to {comparison_path}')
run('missing-input-partition',producer_count,dict(REPLAY_PREFIX=prefix,REPLAY_RANKS=producer_count-1,DIAGNOSTIC_ONLY=1),False)
run('wrong-mesh',producer_count,dict(REPLAY_PREFIX=prefix,REPLAY_RANKS=producer_count,DIAGNOSTIC_ONLY=1,MESH_DIVISOR=3),False)
run('bad-coefficient',producer_count,dict(REPLAY_PREFIX=prefix,REPLAY_RANKS=producer_count,DIAGNOSTIC_ONLY=1,DIAGNOSTIC_COEFFICIENT_DT='nan'),False)
(out/'result.json').write_text(json.dumps(dict(passed=True,serial_parallel_relative_tolerance=1e-10,
    meaning='Numerical reproducibility check, not a physical acceptance tolerance',cases=6),indent=2))
print('Native replay integration smoke: 6 scenarios passed')

