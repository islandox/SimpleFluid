#!/usr/bin/env python3
"""SST distributions and errors on the same validated cell planes as water fields."""
import argparse
from pathlib import Path
from compare_verification import ComparisonError, load_manifest, read_samples, validate_manifest
from plot_water_fields import load_fields, generate

SST_FIELDS = {
    'k_m2_s2': ('Turbulent kinetic energy', 'm²/s²', 1e-12),
    'omega_1_s': ('Specific dissipation rate', '1/s', 1e-12),
    'nut_m2_s': ('Eddy viscosity', 'm²/s', 1e-14),
    'wall_yplus': ('Wall-adjacent y+', '1', 1e-12),
}

def load_sst(manifest, openfoam, simplefluid, not_before=None):
    data = load_fields(manifest, openfoam/'fields.csv', simplefluid/'fields.csv', not_before)
    columns = [*SST_FIELDS, 'wall_distance_m']
    contract = {**manifest, 'expected_samples': manifest['spatial_output']['cell_samples'],
                'quantities': {key: {'units': 'SI', 'absolute_tolerance': 0, 'relative_tolerance': 0} for key in columns},
                'conservation': {}}
    contract = validate_manifest(contract)
    for directory, fields in zip((openfoam, simplefluid), data):
        values = read_samples(directory/'turbulence.csv', contract, not_before)
        if values.keys() != fields.keys():
            raise ComparisonError('SST and physical fields have different time/cell coverage')
        for key, row in values.items():
            if (row['k_m2_s2'] <= 0 or row['omega_1_s'] <= 0 or row['nut_m2_s'] < 0
                    or row['wall_yplus'] < 0 or row['wall_distance_m'] <= 0):
                raise ComparisonError('SST fields violate positivity or wall-distance bounds')
            fields[key].update(row)
    return data

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',required=True,type=Path)
    p.add_argument('--openfoam-directory',required=True,type=Path)
    p.add_argument('--simplefluid-directory',required=True,type=Path)
    p.add_argument('--output-directory',required=True,type=Path)
    p.add_argument('--not-before',type=float)
    p.add_argument('--formats',nargs='+',default=['svg'],choices=['svg','png','pdf'])
    a=p.parse_args();manifest=load_manifest(a.manifest)
    manifest['field_context']='Menter-1994 SST, resolved viscous sublayer. y+ is a wall-adjacent diagnostic; interior cells report zero.'
    reference,actual=load_sst(manifest,a.openfoam_directory,a.simplefluid_directory,a.not_before)
    generate(manifest,reference,actual,a.output_directory,a.formats,SST_FIELDS)
    print(f'SST figures: {a.output_directory}/index.html')

if __name__=='__main__':main()
