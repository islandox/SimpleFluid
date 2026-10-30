# Ordinary general VoF pressure projection

The Ordinary native solver projects physical cell momentum and canonical face
rates together. Independent gas and carrier inventories determine the cell
masses; represented mixture density is not a replacement for supplied physical
mass. The transport backend remains independently selected, with
`SweptPartition` as the default. BoundedCertification and Exact retain their
separate projection implementations and arithmetic contracts; the energy
identity below is not a qualification claim for those paths.

## Common discrete operators

Let `q` contain integrated canonical interior-face volume rates, with exterior
rates fixed to zero. `B` is signed cell incidence, positive at the canonical
owner. The native face-centroid moment operator is

```text
(R q)_c = sum_f sign(c,f) * (x_f - x_c) * q_f / V_c.
```

Every covering coarse/fine subface occurs once. Actual native displacement and
volume queries define `R`; no Cartesian reconstruction or donor approximation
is substituted. With positive physical cell masses `M`, the projector minimizes

```text
0.5 * ||R q - u_predict||_M^2, subject to B q = b.
```

The source-free target `b` is zero except for the existing closed-compatible
signed-occupancy restoration target, which remains within the original physical
continuity and transport-readiness limits. It never edits either phase inventory
or resets arithmetic history.

Writing `A = R^T M R` and `pi = dt*p`, the coupled equations are

```text
[A  -B^T] [q ] = [R^T M u_predict + dt*s]
[-B    0] [pi]   [-b                    ].
```

The capillary driving value is
`s_f = sigma*kappa_f*(fraction_neighbor-fraction_owner)`. Pressure and capillarity
use the same cell-potential jump. Physical geometry is already present in `R`;
adding a tangential correction to only one potential would break the constant
curvature balance `p=sigma*kappa*fraction+constant`. Curvature acquisition,
coherence, sample-count and conditioning requirements are unchanged.

The accepted velocity is `R*q`. On an admitted native shell,
`sum_f A_f*n_cf*(x_f-x_c)^T = V_c*I`, so the corresponding pressure and capillary
tractions reproduce the physical cell impulse. Wall reactions include resolved
and unsampled normal components; an unsampled axis still requires the existing
straight closed-slab support declaration. Missing-axis pressure offsets remain
reported separately from cell-center pressure.

## Energy and solver admission

The common operators give the physical work identity

```text
E_new - E_predict
  = dt*p^T*b + dt*s^T*q - 0.5*||u_new-u_predict||_M^2.
```

The implementation checks this identity using the published binary64 velocity,
independent physical masses, measured original-equation defects and a derived
arithmetic allowance. The allowance does not replace physical divergence,
occupancy or transport gates. There is no energy clipping, face-flux damping or
mass regularization. An already compatible circulation is preserved, including
when the timestep or physical mass scale changes.

The face mass matrix can be singular while the constrained problem is well
posed. The implementation solves the symmetric coupled system, with symmetric
positive diagonal scaling formed in wide arithmetic and one pressure gauge.
Fresh zero-start, unpreconditioned GMRES supplies a minimum weighted norm choice
for invisible canonical-rate modes when the system is consistent. Inconsistent
forcing, unresolved original-equation defects or exhausted work reject the
candidate. No diagonal perturbation changes the physical minimization.

Corrections accumulate in wide arithmetic and re-evaluate the original native
mass operator, rather than accepting the narrowed sparse-matrix residual alone.
All solves and correction rounds share the original pressure iteration budget.
Physical tolerances, timestep and request controls are agreed collectively before
rank-dependent work, including on empty owners. Resource capacities remain rank-local;
exhaustion on any rank rejects collectively. Caller storage, matrix payload
and conservative Krylov workspace estimates share the configured byte cap;
opaque backend allocator overhead is distinct from this accounting and from RSS.

## Predictor and publication contracts

For Ordinary, physical cell momentum is the predictor authority. Face inputs
provide the initial physical residual scale and are not an independent force.
When initialization supplies faces alone, cell velocities are first reconstructed
from their native moments. Tests of prescribed circulation supply matching cell
momentum. An arbitrary nonzero circulation paired with zero physical momentum
cannot simultaneously be preserved and satisfy the closed, unforced energy law.

Each Ordinary advance forms the face estimate from the current momentum
predictor. It does not accumulate old face/cell pressure discrepancies. The
coupled solve then publishes one canonical rate used by phase, mass and momentum
transport. Fields, clock, preparation and inventories still publish only after
the surrounding accepted-state transaction succeeds.

Diagnostic `SIMPLEFLUID_VOF_FLOW_TRACE` records actual input cell masses and face
geometry, correction residuals, original physical limits, pressure work, energy
identity and peak admitted workspace. Diagnostics do not change admission.

Component energy and replay tests establish focused properties. The original
Q0/Q3/Q5 and capillary endpoints, accuracy gates and per-rank results remain
separate requirements; consult the maintained verification contracts and dated
build artifacts for the source-pinned qualification scope.
