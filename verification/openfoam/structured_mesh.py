#!/usr/bin/env python3
"""Consume shared native mesh edges; write matching OpenFOAM multi-grading."""
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
    if set(axes)!=set('xyz'): raise ValueError('Expected x/y/z cell edges')
    return axes


def write_openfoam(mesh_file,output):
    edges=read_mesh(mesh_file);out=Path(output)
    x,y,z=(edges[k] for k in 'xyz');nx,ny,nz=len(x)-1,len(y)-1,len(z)-1
    header=lambda name:f'FoamFile {{version 2.0; format ascii; class dictionary; object {name};}}\n'
    vertices=[(x[0],y[0],z[0]),(x[-1],y[0],z[0]),(x[-1],y[-1],z[0]),(x[0],y[-1],z[0]),
              (x[0],y[0],z[-1]),(x[-1],y[0],z[-1]),(x[-1],y[-1],z[-1]),(x[0],y[-1],z[-1])]
    # One multi-grading section per cell: its length and one cell, expansion 1.
    # Avoid a million one-cell blocks while retaining every canonical edge.
    grading=['('+' '.join(f'({b-a:.17g} 1 1)' for a,b in zip(values,values[1:]))+')' for values in (x,y,z)]
    text=header('blockMeshDict')+'scale 1;\nvertices (\n'+'\n'.join('('+' '.join(f'{v:.17g}' for v in point)+')' for point in vertices)+'\n);\n'
    text+=f'blocks (hex (0 1 2 3 4 5 6 7) ({nx} {ny} {nz}) simpleGrading ('+' '.join(grading)+'));\nedges ();\nboundary (\n'
    faces={'xmin':'0 4 7 3','xmax':'1 2 6 5','ymin':'0 1 5 4','ymax':'3 7 6 2','zmin':'0 3 2 1','zmax':'4 5 6 7'}
    for name,face in faces.items():
        kind='wall' if name in ['xmin','xmax','zmin'] else 'patch'
        text+=f'{name} {{type {kind}; faces (({face}));}}\n'
    text+=');\nmergePatchPairs ();\n'
    (out/'system').mkdir(parents=True,exist_ok=True);(out/'constant').mkdir(parents=True,exist_ok=True)
    (out/'system/blockMeshDict').write_text(text)
    dictionary=header('referenceMesh')+''.join(f'{axis}Edges ('+' '.join(f'{v:.17g}' for v in edges[axis])+');\n' for axis in 'xyz')
    (out/'constant/referenceMesh').write_text(dictionary)
    (out/'mesh.dat').write_text(Path(mesh_file).read_text())
    return edges


def with_mesh_manifest(manifest_path,mesh_file,output):
    data=json.loads(Path(manifest_path).read_text());edges=read_mesh(mesh_file)
    old_edges=data.get('mesh_edges_m',edges)
    old_volume=data.get('tolerance_reference_volume_m3',mesh_statistics(old_edges)['volume_m3'])
    new_volume=mesh_statistics(edges)['volume_m3'];volume_factor=new_volume/old_volume
    old_height=data.get('tolerance_reference_height_m',old_edges['z'][-1]-old_edges['z'][0])
    new_height=edges['z'][-1]-edges['z'][0];length_factor=new_height/old_height
    volume_quantities={'volume_m3','liquid_mass_kg','energy_J','cumulative_heat_J','hydrogen_mol','produced_mol','escaped_mol','heat_power_W','wall_heat_loss_W'}
    volume_residuals={'relative_flux_max_m3_s','hydrogen_balance_mol','thermal_step_residual_J','mass_residual_kg','energy_balance_residual_J'}
    for name,entry in data['quantities'].items():
        if name in volume_quantities:entry['absolute_tolerance']*=volume_factor
        elif data['case']=='planarALE' and name=='level_m':entry['absolute_tolerance']*=length_factor
    for name,entry in data.get('conservation',{}).items():
        if name in volume_residuals:entry['absolute_tolerance']*=volume_factor
        elif name=='analytic_level_error_m':entry['absolute_tolerance']*=length_factor
    data['tolerance_reference_volume_m3']=new_volume
    data['tolerance_reference_height_m']=new_height
    source=data['spatial_output'].get('source_region')
    if source:
        for axis in ['x','z']:
            factor=(edges[axis][-1]-edges[axis][0])/(old_edges[axis][-1]-old_edges[axis][0])
            for suffix in ['lower_m','upper_m']:source[f'{axis}_{suffix}']*=factor
    count=(len(edges['x'])-1)*(len(edges['z'])-1)
    samples=[str(i) for i in range(count)]
    data['spatial_output']['cell_samples']=samples
    data['spatial_output']['layout']='xz'
    data['spatial_output']['width_m']=edges['x'][-1]-edges['x'][0]
    middle=(len(edges['y'])-1)//2
    data['spatial_output']['y_slice_index']=middle
    data['spatial_output']['y_slice_bounds_m']=edges['y'][middle:middle+2]
    if data['case']=='dispersedBubbleFlow':data['expected_samples']=[str(i) for i in range(len(edges['z'])-1)]
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
