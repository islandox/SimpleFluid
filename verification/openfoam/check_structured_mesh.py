#!/usr/bin/env python3
"""Check every OpenFOAM mesh point against the shared Cartesian coordinates."""
import argparse
import bisect
import json
import math
from pathlib import Path
from structured_mesh import read_mesh,mesh_statistics


def check_points(mesh_file,case):
    axes=read_mesh(mesh_file);sizes=[len(axes[k]) for k in 'xyz']
    expected=sizes[0]*sizes[1]*sizes[2];seen=bytearray(expected)
    cache={axis:{} for axis in 'xyz'};maximum={axis:0.0 for axis in 'xyz'};count=0
    def index(axis,value):
        if not math.isfinite(value):raise ValueError('Nonfinite mesh point')
        if value in cache[axis]:return cache[axis][value]
        values=axes[axis];candidate=bisect.bisect_left(values,value)
        choices=[i for i in [candidate-1,candidate] if 0<=i<len(values)]
        best=min(choices,key=lambda i:abs(values[i]-value));error=abs(values[best]-value)
        if error>1e-10:raise ValueError(f'OpenFOAM {axis} coordinate differs by {error} m')
        maximum[axis]=max(maximum[axis],error);cache[axis][value]=best
        return best
    with (Path(case)/'constant/polyMesh/points').open() as stream:
        for line in stream:
            line=line.strip()
            if not (line.startswith('(') and line.endswith(')')):continue
            parts=line[1:-1].split()
            if len(parts)!=3:continue
            i,j,k=[index(axis,float(value)) for axis,value in zip('xyz',parts)]
            key=(k*sizes[1]+j)*sizes[0]+i
            if seen[key]:raise ValueError('Duplicate or mislocated mesh point')
            seen[key]=1;count+=1
    if count!=expected:raise ValueError(f'Expected {expected} mesh points, read {count}')
    return {'passed':True,'points':count,'maximum_coordinate_error_m':maximum,'mesh':mesh_statistics(axes)}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mesh',required=True);p.add_argument('--case',required=True);p.add_argument('--report',required=True,type=Path)
    args=p.parse_args();result=check_points(args.mesh,args.case)
    args.report.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
