# Conservative low-Mach reacting flow

`Reacting::LowMachMixtureSolver` owns native mapped flow fields and conservative
physical phase inventories. `ReactingMixtureCoupler` retains shared ownership of
an initialized solver and holds its exclusive advancement lease. The controller
owns optional chemistry; the flow solver owns transport, momentum, thermodynamic
pressure, mechanical pressure, and the accepted clock.

## Physical state and pressure

Each owned cell stores species moles and internal energy for every declared
physical phase. Liquid and dissolved chemical compartments share one physical
solution phase. Gas phases use supplied mixture EOS providers. Density is the
sum of actual phase masses divided by the control volume, rather than a
Boussinesq reference density or a void correction to a fixed liquid mass.

The thermodynamic pressure closes total material volume at the supplied energy
and composition. The mechanical pressure correction enforces local volume
continuity using the variable-density face coefficient. Accepted phase volumes
come from the EOS; neither a separately advected volume label nor a numerical
pressure compatibility correction may change physical inventories.

Species and phase energy use the same conservative upwind face transport.
Momentum uses the resulting physical mass flux in kg/s. Multiplying this flux by
density again is incorrect. Dynamic viscosity belongs in the stress operator;
non-Laplacian deviatoric stress terms require a supplied contribution when used.
Pressure work uses the phase-volume trial and the same transported volume packet
as the continuity budget.

## Reaction and heat integration

`MixturePhaseBinding` explicitly maps catalog species compartments to physical
energy phases. The controller consumes supplied reaction rates or integrated
extents through the shared finite-donor reaction kernel. A source receives the
accepted chemical state and a thermodynamic trial; it must be deterministic and
must not mutate external history during a preview.

Each outer corrector starts from the same accepted state. Convergence includes
species, phase energy and physical momentum inputs. A successful preview is an
immutable candidate. Other successful or rejected previews from the same
accepted generation do not invalidate it. Commit, initialization, restore, or
controller ownership changes invalidate incompatible candidates.

Independent external heating or cooling is signed energy, separate from material
transfer energy and chemical heat. With formation energy in the caloric EOS,
chemistry does not also supply reaction heat. Separate-temperature reactions that
cross physical phases require explicit energy partitioning. The partition may
redistribute phase energy with zero net heat; it cannot introduce an undeclared
net source. The controller accumulates heat and material-transfer energy
together with compensation before narrowing each phase packet. Its
`net_thermal_energy` diagnostic retains the resulting interval energy when large
opposite reservoir channels cancel; adding the individually rounded channel
totals is not an equivalent energy budget.

## Acceptance and replay

Mass, species, internal energy, momentum, EOS occupancy, and volume-continuity
checks are independent of Krylov convergence. Energy tolerance uses a caloric
scale that does not grow with an arbitrary additive mass-energy reference.
Compensated sums retain source/work/transport contributions, and a separately
reported operand-roundoff bound must remain a small part of the physical
accuracy budget. An ill-conditioned caloric reference is rejected rather than
used to weaken that budget.

The equilibrium mode retains a single recovered temperature. Separate-phase
mode recovers each occupied phase from its own internal energy. Exact empty
phases keep zero temperature and energy. Exact incompressible liquids are
supported when the supplied derivatives define that limit; an incompatible
closed-volume change with zero susceptibility rejects.

The initial solver stage supports closed, fixed, orthogonal native mapped meshes.
Focused geometry coverage includes Cartesian, cylindrical, and orthogonal
`SemiStructuredXY_Z` meshes with contiguous algebraic maps. The geometry check
rejects nonorthogonal incidences; construction captures a fixed geometry epoch.
Prescribed phase drift is an explicit closure. ALE, population
number ownership, diffusion, and interphase thermal integration require their
complete transport and acceptance adapters before their support gates open.
VoF solvers are disabled by default and excluded from the supported reacting-flow
scope while their qualification remains incomplete.

Thermodynamic pressure uses a safeguarded solve within supplied EOS bounds.
Caloric range failures may backtrack from a previously feasible state; arbitrary
constitutive-provider errors remain hard failures. The initial source trial must
be calorically feasible at the initial pressure guess. Discovery of a different
feasible pressure branch is not implemented in this stage.

MPI configuration and source failures are validated collectively. Tests include
actual empty owners, rejected-source replay, independent heat, two gas products,
nonuniform heating, and caloric-reference shifts. These are numerical acceptance
checks using manufactured closures, not physical validation of a chemistry law.
