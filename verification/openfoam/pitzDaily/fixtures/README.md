# pitzDaily comparator-contract fixture

These files are synthetic and test only the deterministic profile-selection,
interpolation, norm, station-offset, and per-component tolerance behavior in
`compare_profiles.py`. They are not CFD results and must never be used as
physical validation evidence.

Each OpenFOAM-format profile has five samples. The SimpleFluid-format CSV has
three samples per station with analytically injected component errors. The
acceptance manifest records the exact expected norms and rounds each limit
upward with roughly 10 percent headroom. `test_compare_profiles.py` verifies
the recorded norms, the passing limits, a deliberately tightened failing
limit, input checksums, and rejection of the pending physical manifest.

The fixture uses nearest SimpleFluid layers at `x=0.049`, `0.101`, and
`0.201 m`. Their `0.001 m` offsets are accepted by a fixture-only
`0.0011 m` sampling tolerance.
