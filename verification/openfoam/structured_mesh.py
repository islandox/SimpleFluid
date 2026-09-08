#!/usr/bin/env python3
"""Consume shared native mesh edges; write exact one-cell OpenFOAM blocks."""
import argparse
import json
import math
from pathlib import Path


def read_mesh(path):
    axes={}
    for raw in Path(path).read_text().splitlines():
        tokens=raw.split('#',1)[0].split()
        if not tokens: continue
        name=tokens[0]
        if name not in 'xyz' or len(name)!=1 or name in axes: raise ValueError('Invalid or duplicate mesh axis')
        values=[float(value) for value in tokens[1:]]
        if len(values)<2 or not all(map(math.isfinite,values)) or any(b<=a for a,b in zip(values,values[1:])):
            raise ValueError('Invalid mesh edges')
        axes[name]=values
    if set(axes)!=set('xyz') or len(axes['y'])!=2: raise ValueError('Expected x/y/z with one extrusion cell')
    return axes


def write_openfoam(mesh_file,output):
    edges=read_mesh(mesh_file);out=Path(output)
    x,y,z=(edges[k] for k in 'xyz');nx,nz=len(x)-1,len(z)-1
    def vertex(i,j,k):return (k*2+j)*(nx+1)+i
    points=[f'({xx:.17g} {yy:.17g} {zz:.17g})' for zz in z for yy in y for xx in x]
    blocks=[];patches={name:[] for name in ['xmin','xmax','ymin','ymax','zmin','zmax']}
    for k in range(nz):
        for i in range(nx):
            v=[vertex(i,0,k),vertex(i+1,0,k),vertex(i+1,1,k),vertex(i,1,k),vertex(i,0,k+1),vertex(i+1,0,k+1),vertex(i+1,1,k+1),vertex(i,1,k+1)]
            blocks.append('hex ('+' '.join(map(str,v))+') (1 1 1) simpleGrading (1 1 1)')
            for name,active,indices in [('xmin',i==0,[0,4,7,3]),('xmax',i==nx-1,[1,2,6,5]),('ymin',True,[0,1,5,4]),('ymax',True,[3,7,6,2]),('zmin',k==0,[0,3,2,1]),('zmax',k==nz-1,[4,5,6,7])]:
                if active:patches[name].append('('+' '.join(str(v[q]) for q in indices)+')')
    header=lambda name:f'FoamFile {{version 2.0; format ascii; class dictionary; object {name};}}\n'
    text=header('blockMeshDict')+'scale 1;\nvertices (\n'+'\n'.join(points)+'\n);\nblocks (\n'+'\n'.join(blocks)+'\n);\nedges ();\nboundary (\n'
    for name,faces in patches.items():
        kind='wall' if name in ['xmin','xmax','zmin'] else 'patch'
        text+=f'{name} {{type {kind}; faces (\n'+ '\n'.join(faces)+'\n);}\n'
    text+=');\nmergePatchPairs ();\n'
    (out/'system').mkdir(parents=True,exist_ok=True);(out/'constant').mkdir(parents=True,exist_ok=True)
    (out/'system/blockMeshDict').write_text(text)
    dictionary=header('referenceMesh')+''.join(f'{axis}Edges ('+' '.join(f'{v:.17g}' for v in edges[axis])+');\n' for axis in 'xyz')
    (out/'constant/referenceMesh').write_text(dictionary)
    (out/'mesh.dat').write_text(Path(mesh_file).read_text())
    return edges


def with_mesh_manifest(manifest_path,mesh_file,output):
    data=json.loads(Path(manifest_path).read_text());edges=read_mesh(mesh_file)
    count=(len(edges['x'])-1)*(len(edges['z'])-1)
    samples=[str(i) for i in range(count)]
    data['spatial_output']['cell_samples']=samples
    if data['case']=='dispersedBubbleFlow':data['expected_samples']=samples
    data['mesh_edges_m']=edges
    data['mesh_statistics']=mesh_statistics(edges)
    Path(output).write_text(json.dumps(data,indent=2)+'\n')


def mesh_statistics(edges):
    result={'cells':math.prod(len(edges[k])-1 for k in 'xyz'),
            'volume_m3':math.prod(edges[k][-1]-edges[k][0] for k in 'xyz')}
    for axis,values in edges.items():
        widths=[b-a for a,b in zip(values,values[1:])]
        result[axis]={'cells':len(widths),'first_cell_m':widths[0],'last_cell_m':widths[-1],
                      'minimum_cell_m':min(widths),'maximum_cell_m':max(widths),
                      'max_adjacent_ratio':max([1,*[max(a/b,b/a) for a,b in zip(widths,widths[1:])]])}
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mesh',required=True);p.add_argument('--openfoam-case');p.add_argument('--manifest');p.add_argument('--output-manifest')
    a=p.parse_args()
    if a.openfoam_case:write_openfoam(a.mesh,a.openfoam_case)
    if a.manifest:with_mesh_manifest(a.manifest,a.mesh,a.output_manifest)
