"""Recompute official sampled profiles in fresh cases; never modify evidence."""
from pathlib import Path
import shutil
import tempfile

import collect_evidence as evidence
import common_sampling as sampling
import compare_profiles as comparator


def replay_profiles(source_case, paths, divisor, stored, mesh_multiplier=1):
    metadata=sampling.verify_sampling(source_case,paths,divisor,stored,lambda p:p,mesh_multiplier)
    executable=shutil.which('postProcess')
    if executable is None or evidence.sha256(Path(executable).resolve())!=metadata['postProcess']['sha256']:
        raise ValueError('sampling replay requires the recorded OpenFOAM postProcess executable')
    with tempfile.TemporaryDirectory(prefix='pitz-sampling-replay-') as folder:
        fresh=Path(folder)/'case'
        sampling.sample(source_case,paths,divisor,fresh,mesh_multiplier)
        expected=comparator.latest_common_profiles(fresh)
        retained=comparator.latest_common_profiles(stored)
        for station in comparator.STATIONS:
            if comparator.read_openfoam(expected[station])!=comparator.read_openfoam(retained[station]):
                raise ValueError(f'{station}: retained profiles do not reproduce from raw cells')
    return True
