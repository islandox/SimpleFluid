#!/usr/bin/env python3
"""Summarize controlled bottom-convection mesh sensitivity and wall heat loss."""
import argparse
import csv
import html
import json
import math
from pathlib import Path
import shutil
import subprocess
from plot_water_fields import text,rectangle


def final_fields(path):
    with path.open() as stream:rows=list(csv.DictReader(stream))
    end=max(float(row['time_s']) for row in rows)
    return end,[{k:(v if k=='sample' else float(v)) for k,v in row.items()}
                for row in rows if abs(float(row['time_s'])-end)<1e-10]


def measure(run):
    report=json.loads((run/'comparison.json').read_text())
    if not report['passed']:raise ValueError(f'Comparison failed: {run}')
    manifest=json.loads((run/'manifest.json').read_text())
    if manifest['case']!='bottomHeatedBubblyConvection':raise ValueError('Expected convection case')
    edges=manifest['mesh_edges_m'];width=edges['x'][-1];height=edges['z'][-1];depth=edges['y'][-1]
    result={'cells':manifest['mesh_statistics']['cells'],'slice_cells':len(manifest['spatial_output']['cell_samples']),'run':str(run),'solvers':{}}
    props={}
    for line in (run/'openfoam/constant/verificationProperties').read_text().splitlines():
        words=line.rstrip(';').split()
        if len(words)==2:
            try:props[words[0]]=float(words[1])
            except ValueError:pass
    for solver in ['simplefluid','openfoam']:
        end,rows=final_fields(run/solver/'fields.csv')
        if len(rows)!=result['slice_cells']:raise ValueError('Incomplete final spatial output')
        wall_heat=mean_temperature=0
        for r in rows:
            dx=r['x_upper_m']-r['x_lower_m'];dz=r['z_upper_m']-r['z_lower_m']
            mean_temperature+=r['temperature_K']*dx*dz/(width*height)
            delta=r['temperature_K']-props['temperature_K']
            if abs(r['x_lower_m'])<1e-12 or abs(r['x_upper_m']-width)<1e-12:
                wall_heat+=props['thermal_conductivity_W_m_K']*depth*dz*delta/(dx/2)
            if abs(r['z_upper_m']-height)<1e-12:
                wall_heat+=props['thermal_conductivity_W_m_K']*depth*dx*delta/(dz/2)
        with (run/solver/'history.csv').open() as stream:history=list(csv.DictReader(stream))[-1]
        mean_temperature=float(history['temperature_mean_K'])
        if 'wall_heat_loss_W' in history:wall_heat=float(history['wall_heat_loss_W'])
        elif len(edges['y'])>2:raise ValueError('3D wall heat loss requires the global wall_heat_loss_W history column')
        result['time_s']=end
        result['solvers'][solver]={'temperature_rise_K':max(r['temperature_K'] for r in rows)-props['temperature_K'],
            'mean_temperature_rise_K':mean_temperature-props['temperature_K'],
            'speed_mm_s':1000*max(math.hypot(r['ux_m_s'],r['uy_m_s'],r['uz_m_s']) for r in rows),
            'gas_percent':100*max(r['alpha_g'] for r in rows),'wall_heat_loss_W':wall_heat}
    return result,props


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',action='append',required=True,help='LABEL=run-directory');p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();data=[];reference=None
    for item in a.run:
        label,path=item.split('=',1);result,props=measure(Path(path))
        if reference is not None and props!=reference:raise ValueError('Physics/time inputs differ between grids')
        reference=props;data.append({'label':label,**result})
    data.sort(key=lambda d:d['cells'])
    if len(data)<2 or len({d['cells'] for d in data})!=len(data):raise ValueError('Need distinct mesh sizes')
    change={solver:{key:100*(data[-1]['solvers'][solver][key]-data[-2]['solvers'][solver][key])/abs(data[-2]['solvers'][solver][key])
                     for key in data[-1]['solvers'][solver] if data[-2]['solvers'][solver][key]!=0}
            for solver in ['simplefluid','openfoam']}
    a.output.mkdir(parents=True,exist_ok=True)
    (a.output/'mesh_sensitivity.json').write_text(json.dumps({'runs':data,'last_refinement_percent_change':change,'scope':'fixed-dt spatial sensitivity, not established mesh independence'},indent=2)+'\n')
    metrics=[('temperature_rise_K','Maximum temperature rise','K'),('speed_mm_s','Peak liquid speed','mm/s'),('gas_percent','Peak gas fraction','%'),('wall_heat_loss_W','Cold-wall heat loss','W')]
    elements=['<svg xmlns="http://www.w3.org/2000/svg" width="1640" height="620">','<style>text{font-family:DejaVu Sans,sans-serif;fill:#17212b}</style>',rectangle(0,0,1640,620,'#f7f9fb'),text(35,40,'Bottom convection: spatial mesh sensitivity at t = 20 s',25),text(35,70,'Same heat, gas, water, boundaries and dt = 0.02 s on every grid. Lines connect measured values.',16)]
    colors={'simplefluid':'#d95f02','openfoam':'#1f77b4'}
    for j,(metric,label,unit) in enumerate(metrics):
        left,top,w,h=90+j*400,135,280,325
        values=[d['solvers'][solver][metric] for d in data for solver in colors]
        lo,hi=min(values),max(values);pad=max(hi-lo,abs(hi)*.05)*.15;lo-=pad;hi+=pad
        elements.extend([text(left,top-30,label,18),text(left,top-8,unit,15),rectangle(left,top,w,h,'#fff')])
        for i in range(5):
            y=top+h*(1-i/4);v=lo+(hi-lo)*i/4
            elements.append(f'<line x1="{left}" y1="{y}" x2="{left+w}" y2="{y}" stroke="#d3d9df"/>')
            elements.append(text(left-8,y+5,f'{v:.4g}',13,'end'))
        def xpos(cells):return left+w*math.log(cells/data[0]['cells'])/math.log(data[-1]['cells']/data[0]['cells'])
        for solver,color in colors.items():
            points=[(xpos(d['cells']),top+h*(hi-d['solvers'][solver][metric])/(hi-lo)) for d in data]
            elements.append('<polyline points="'+' '.join(f'{x},{y}' for x,y in points)+f'" fill="none" stroke="{color}" stroke-width="2"/>')
            for x,y in points:elements.append(f'<circle cx="{x}" cy="{y}" r="4" fill="{color}"/>')
        for d in data:elements.append(text(xpos(d['cells']),top+h+25,str(d['cells']),13,'middle'))
        elements.append(text(left+w/2,top+h+52,'Cells (log scale)',15,'middle'))
    elements.extend([text(70,570,'Orange: SimpleFluid     Blue: OpenFOAM',17),text(70,599,'Remaining differences between the graded meshes are reported explicitly; no mesh-independence claim.',16),'</svg>'])
    svg=a.output/'mesh_sensitivity.svg';svg.write_text('\n'.join(elements))
    converter=shutil.which('rsvg-convert')
    if converter:
        for extension in ['png','pdf']:subprocess.run([converter,'--format',extension,'--output',str(a.output/f'mesh_sensitivity.{extension}'),str(svg)],check=True)
    lines=['# Bottom-convection mesh sensitivity','', 'All runs use the same physics, dt=0.02 s and t=20 s.','', '| Grid | Solver | Cells | Peak rise (K) | Peak speed (mm/s) | Peak gas (%) | Wall heat loss (W) |','| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for d in data:
        for solver,r in d['solvers'].items():lines.append(f"| {d['label']} | {solver} | {d['cells']} | {r['temperature_rise_K']:.6f} | {r['speed_mm_s']:.6f} | {r['gas_percent']:.6f} | {r['wall_heat_loss_W']:.6f} |")
    lines.extend(['','Last refinement percentage changes:','', '```json',json.dumps(change,indent=2),'```','','This is spatial sensitivity at fixed time step, not proof of mesh independence.'])
    (a.output/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(change,indent=2))


if __name__=='__main__':main()
