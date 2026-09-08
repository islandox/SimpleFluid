#!/usr/bin/env python3
"""Create an independent OpenFOAM buoyant-circulation case from shared SI inputs."""
import argparse
import math
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from reference_water import read_reference_water
from structured_mesh import write_openfoam


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--mesh',type=Path,default=Path(__file__).with_name('mesh.dat'))
    args=parser.parse_args()
    case=Path(__file__).resolve().parent
    props={}
    for raw in (case/'reference.properties').read_text().splitlines():
        words=raw.split('#',1)[0].split()
        if not words: continue
        if len(words)!=2 or words[0] in props: raise ValueError('Malformed or duplicate case parameter')
        value=float(words[1])
        if not math.isfinite(value): raise ValueError('Nonfinite case parameter')
        props[words[0]]=value
    water=read_reference_water(case.parent/'reference_water.properties')
    if props.keys() & water.keys(): raise ValueError('Case parameters cannot override the water snapshot')
    props.update(water)
    out=args.output
    if out.exists() and any(out.iterdir()): raise ValueError('OpenFOAM case directory must be empty')
    for name in ['0','constant','system']: (out/name).mkdir(parents=True,exist_ok=True)
    def header(name,kind='dictionary'):
        return f'FoamFile {{version 2.0; format ascii; class {kind}; object {name};}}\n'
    (out/'constant/verificationProperties').write_text(header('verificationProperties')+''.join(f'{k} {v:.17g};\n' for k,v in props.items()))
    write_openfoam(args.mesh,out)
    (out/'system/controlDict').write_text(header('controlDict')+f'''
application bottomBubblyConvectionFoam;
startFrom startTime; startTime 0; stopAt endTime;
endTime {props['end_time']}; deltaT {props['dt']};
writeControl timeStep; writeInterval 100000;
writeFormat ascii; writePrecision 17; timePrecision 12;
runTimeModifiable false;
''')
    (out/'system/fvSchemes').write_text(header('fvSchemes')+'''
ddtSchemes {default Euler;}
gradSchemes {default Gauss linear;}
divSchemes {default Gauss upwind;}
laplacianSchemes {default Gauss linear orthogonal;}
interpolationSchemes {default linear;}
snGradSchemes {default orthogonal;}
fluxRequired {default no; p;}
''')
    (out/'system/fvSolution').write_text(header('fvSolution')+'''
solvers
{
 p {solver PCG; preconditioner DIC; tolerance 1e-12; relTol 0; maxIter 2000;}
 "(U|T)" {solver smoothSolver; smoother symGaussSeidel; tolerance 1e-12; relTol 0; maxIter 2000;}
 "(moles|number)" {solver smoothSolver; smoother symGaussSeidel; tolerance GAS_TRANSPORT_TOLERANCE; relTol 0; maxIter 2000;}
}
'''.replace('GAS_TRANSPORT_TOLERANCE',str(props['gas_transport_tolerance'])))
    for name,kind,dims,value in [('U','volVectorField','0 1 -1 0 0 0 0','(0 0 0)'),('p','volScalarField','0 2 -2 0 0 0 0','0'),('T','volScalarField','0 0 0 1 0 0 0',str(props['temperature_K']))]:
        body=header(name,kind)+f'dimensions [{dims}];\ninternalField uniform {value};\nboundaryField\n{{\n'
        for patch in ['xmin','xmax','ymin','ymax','zmin','zmax']:
            condition='type zeroGradient;'
            if name=='U': condition='type slip;' if patch in ['ymin','ymax','zmax'] else 'type fixedValue; value uniform (0 0 0);'
            if name=='T' and patch in ['xmin','xmax','zmax']: condition=f'type fixedValue; value uniform {value};'
            body+=f'{patch} {{{condition}}}\n'
        (out/'0'/name).write_text(body+'}\n')


if __name__=='__main__': main()
