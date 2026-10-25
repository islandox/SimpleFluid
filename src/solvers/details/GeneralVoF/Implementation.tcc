/** @file Implementation.tcc
 * @brief Compiled mode-selected candidate preparation and numerical operations.
 */
namespace SimpleFluid
{
template<ArithmeticMode Mode>
std::shared_ptr<GeneralVoFSolver::Impl::State> GeneralVoFSolver::Impl::construct_mode(
    std::shared_ptr<const VoF::GeneralVoF> phases, VoFMaterial m, GeneralVoFSolverOptions o)
{
    using A = VoF::finite_detail::Arithmetic<Mode>;
    bool valid = options.byte_budget > 0 && options.maximum_momentum_iterations > 0 &&
                 std::isfinite(options.momentum_relative_tolerance) && options.momentum_relative_tolerance > 0;
    for (double value : {m.gas_density, m.liquid_density})
        valid = valid && std::isfinite(value) && value > 0;
    for (double value : {m.gas_viscosity, m.liquid_viscosity, m.surface_tension})
        valid = valid && std::isfinite(value) && value >= 0;
    for (double value : o.flow.gravity)
        valid = valid && std::isfinite(value);
    for (double value : {o.flow.max_courant, o.flow.capillary_safety, o.flow.viscous_safety,
             o.flow.pressure_absolute_tolerance, o.flow.divergence_tolerance})
        valid = valid && std::isfinite(value) && value > 0;
    valid = valid && std::isfinite(o.flow.pressure_relative_tolerance) && o.flow.pressure_relative_tolerance >= 0 &&
            o.flow.max_pressure_iterations > 0;
    require(valid, "General VoF flow has invalid material or numerical controls.");
    std::array<std::uint64_t, 18> controls{};
    const std::array<double, 15> real_controls{m.gas_density, m.liquid_density, m.gas_viscosity, m.liquid_viscosity,
        m.surface_tension, o.flow.gravity[0], o.flow.gravity[1], o.flow.gravity[2], o.flow.max_courant,
        o.flow.capillary_safety, o.flow.viscous_safety, o.flow.pressure_relative_tolerance,
        o.flow.pressure_absolute_tolerance, o.flow.divergence_tolerance, o.momentum_relative_tolerance};
    for (std::size_t i = 0; i < real_controls.size(); ++i)
        controls[i] = std::bit_cast<std::uint64_t>(real_controls[i]);
    controls[15] = std::uint64_t(o.maximum_momentum_iterations);
    controls[16] = std::uint64_t(o.flow.max_pressure_iterations);
    controls[17] = std::uint64_t(arithmetic_mode_);
    auto agreed = controls;
    Teuchos::broadcast(
        *mesh->owned_cell_map()->getComm(), 0, int(sizeof(agreed)), reinterpret_cast<char*>(agreed.data()));
    require(controls == agreed, "General VoF material/numerical controls differ across ranks.");
    require(phases->hierarchy()->geometry_unchanged(), "General VoF source geometry is stale.");
    const auto geometry = phases->owned_geometry();
    const auto bounded_geometry = arithmetic_mode_ != ArithmeticMode::Exact
                                      ? phases->owned_finite_geometry<Mode>()
                                      : std::vector<PLIC::ConvexGeometryPtr<Mode>>{};
    volume.resize(mesh->num_owned_cells());
    thickness.resize(volume.size());
    collective(
        [&]
        {
            if (geometry.size() != volume.size())
                throw std::runtime_error("General VoF phase/native geometry order differs.");
            for (std::size_t c = 0; c < volume.size(); ++c)
            {
                if (!bounded_geometry.empty())
                {
                    const auto& p = bounded_geometry[c]->points();
                    long double area = 0;
                    for (const auto& face : bounded_geometry[c]->faces())
                    {
                        auto normal = A::point({});
                        for (std::size_t i = 1; i + 1 < face.size(); ++i)
                            normal = A::add(normal, A::cross(A::subtract(p[face[i]].dynamic(), p[face[0]].dynamic()),
                                                        A::subtract(p[face[i + 1]].dynamic(), p[face[0]].dynamic())));
                        const auto square = A::dot(normal, normal);
                        area += std::sqrt(square.value) / 2;
                    }
                    const auto v = VoF::finite_detail::runtime_value<Mode>(bounded_geometry[c]->volume());
                    volume[c] = A::as_double(v);
                    thickness[c] = static_cast<double>(2 * v.value / area);
                    if (!std::isfinite(thickness[c]) || !(thickness[c] > 0))
                        throw std::runtime_error("Finite VoF cell has no physical thickness.");
                    continue;
                }
                volume[c] = geometry[c]->moments().volume.as_double();
                long double area = 0;
                for (std::size_t face = 0; face < geometry[c]->num_faces(); ++face)
                    area += geometry[c]->face_geometry(face)->area().value;
                thickness[c] = static_cast<double>(2 * geometry[c]->moments().volume.value / area);
                if (!std::isfinite(thickness[c]) || thickness[c] <= 0)
                    throw std::runtime_error("General VoF cell has no positive physical thickness.");
            }
        });
    auto state = std::make_shared<State>();
    state->phases = std::move(phases);
    return state;
}

template<ArithmeticMode Mode> void GeneralVoFSolver::Impl::surface_curvature_mode(State& state) const
{
    state.curvature = detail::general_vof::Curvature<Mode>::assemble(
        *mesh, *state.phases, material.surface_tension, options.byte_budget);
}

template<ArithmeticMode Mode>
double GeneralVoFSolver::Impl::stable_timestep_mode(
    const State& state, const std::vector<VoF::GeneralVoFNodeVelocity>& nodes) const
{
    using A = VoF::finite_detail::Arithmetic<Mode>;
    constexpr bool bounded_mode = Mode == ArithmeticMode::BoundedCertification;
    const auto bound =
        flow.stable_timestep(state.face_velocity, std::min(material.gas_density, material.liquid_density),
            std::max(material.gas_viscosity, material.liquid_viscosity), material.surface_tension, options.flow);
    double local_thickness = std::numeric_limits<double>::infinity(), global_thickness = 0;
    for (double h : thickness)
        local_thickness = std::min(local_thickness, h);
    Teuchos::reduceAll(*mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_MIN, 1, &local_thickness, &global_thickness);
    double speed = 0;
    for (const auto& node : nodes)
        speed = std::max(speed, std::hypot(node.velocity[0], node.velocity[1], node.velocity[2]));
    if (speed > 0 && std::isfinite(speed))
        speed = std::nextafter(speed, std::numeric_limits<double>::infinity());
    speed = maximum(speed);
    if (!finite(state))
        return Exact::GeneralVoFSolver(*this).stable_timestep(state, bound, speed, global_thickness);
    namespace fp = VoF::finite_detail;
    const auto rates = finite_rates_mode<Mode>(state);
    const auto geometry = state.phases->owned_finite_geometry<Mode>();
    double local_rate = std::numeric_limits<double>::infinity();
    collective(
        [&]
        {
            const auto lease = mesh->acquire_execution_view();
            for (std::size_t c = 0; c < volume.size(); ++c)
            {
                VoF::finite_detail::SumFor<Mode> sum(arithmetic_mode_);
                for (const auto face : mesh->faces(LO(c)))
                    sum.add(A::make_value(std::abs(rates[face].value), rates[face].absolute_error));
                const auto outgoing = sum.value();
                const auto upper_rate = outgoing.value + (bounded_mode ? outgoing.absolute_error : 0);
                if (upper_rate > 0)
                {
                    const auto lower_volume =
                        VoF::finite_detail::runtime_value<Mode>(geometry[c]->volume()).value -
                        (bounded_mode ? VoF::finite_detail::runtime_value<Mode>(geometry[c]->volume()).absolute_error
                                      : 0);
                    const auto time = lower_volume * options.flow.max_courant / upper_rate;
                    double value = static_cast<double>(time);
                    if (static_cast<long double>(value) > time)
                        value = std::nextafter(value, 0.);
                    if (!(value > 0) || std::isnan(value))
                        throw std::runtime_error("Finite VoF canonical rates have no positive timestep.");
                    local_rate = std::min(local_rate, value);
                }
            }
        });
    double rate = 0;
    Teuchos::reduceAll(*mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_MIN, 1, &local_rate, &rate);
    return std::min(rate, speed > 0 ? std::min(bound, options.flow.max_courant * global_thickness / speed) : bound);
}

template<ArithmeticMode Mode>
std::shared_ptr<GeneralVoFSolver::Impl::State> GeneralVoFSolver::Impl::prepare_initialize_mode(
    const std::vector<Vector>& owned_velocity, const std::vector<double>& face_velocity)
{
    require(
        accepted->phases->arithmetic_mode() == arithmetic_mode_, "General VoF initialization changed arithmetic mode.");
    // Empty owners still participate in the same scalar/face imports. Local
    // vector emptiness cannot select whether a field was supplied globally.
    const int local_fields[2] = {!owned_velocity.empty(), !face_velocity.empty()};
    int supplied_fields[2] = {};
    Teuchos::reduceAll(*mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_MAX, 2, local_fields, supplied_fields);
    const bool supplied_velocity = supplied_fields[0], supplied_faces = supplied_fields[1];
    require((!supplied_velocity || owned_velocity.size() == volume.size()) &&
                (!supplied_faces || face_velocity.size() == mesh->num_faces()),
        "General VoF initialization supplied field shape mismatch.");
    auto next = std::make_shared<State>();
    next->phases = accepted->phases;
    next->velocity = supplied_velocity ? owned_velocity : std::vector<Vector>(volume.size());
    if (!supplied_velocity && supplied_faces)
    {
        const auto canonical = flow.faces(face_velocity);
        collective(
            [&]
            {
                const auto lease = mesh->acquire_execution_view();
                for (std::size_t c = 0; c < volume.size(); ++c)
                    for (auto face : mesh->faces(LO(c)))
                    {
                        const auto offset = mesh->face_center_vector(face, LO(c));
                        const double amount = (mesh->owner_cell(face) == LO(c) ? 1 : -1) * mesh->face_area(face) *
                                              canonical[face] / volume[c];
                        next->velocity[c][0] += amount * offset.x;
                        next->velocity[c][1] += amount * offset.y;
                        next->velocity[c][2] += amount * offset.z;
                    }
            });
    }
    bool valid = next->velocity.size() == volume.size();
    for (const auto& velocity : next->velocity)
        for (double value : velocity)
            valid = valid && std::isfinite(value);
    require(valid, "General VoF initialization requires finite owned cell velocities.");
    phase_storage_mode<Mode>(*next);
    if (finite(*next))
        surface_curvature_mode<Mode>(*next);
    else
        surface_curvature_mode<ArithmeticMode::Exact>(*next);
    next->face_velocity = supplied_faces ? flow.faces(face_velocity) : flow.interpolate(next->velocity);
    next->projection_arithmetic.byte_budget = options.byte_budget;
    if (finite(*next))
    {
        const auto closure_topology =
            projection_topology<detail::GeneralVoFFiniteProjectionClosureTopology>(*next->phases);
        next->finite_projection.byte_budget = options.byte_budget;
        flow.project(next->face_velocity, next->velocity, next->pressure, next->density, next->fraction,
            next->curvature, 1, 0, options.flow, next->diagnostics, &next->finite_projection,
            std::span<const long double>(next->finite_mass), &closure_topology,
            std::span<const long double>(next->finite_mass_error));
    }
    else
    {
        const auto closure_topology = projection_topology(*next->phases);
        detail::Exact::GeneralVoFFlowWorkspace::project(flow, next->face_velocity, next->velocity, next->pressure,
            next->density, next->fraction, next->curvature, 1, 0, options.flow, next->diagnostics,
            &next->projection_arithmetic,
            std::span<const SimpleFluid::PLIC::GeometryMeasure<SimpleFluid::ArithmeticMode::Exact>>(
                next->physical_mass),
            &closure_topology);
    }
    budget_mode<Mode>(*next);
    return next;
}

template<class Topology> Topology GeneralVoFSolver::Impl::projection_topology(const VoF::GeneralVoF& phases) const
{
    return detail::general_vof::projection_topology<Topology>(*mesh, phases);
}

template<ArithmeticMode Mode> void GeneralVoFSolver::Impl::phase_storage_mode(State& state) const
{
    if (!finite(state))
    {
        Exact::GeneralVoFSolver(*this).phase_storage(state);
        return;
    }
    auto result = detail::general_vof::PhaseStorage<Mode>::evaluate(*mesh, *state.phases, material);
    state.fraction = std::move(result.fraction);
    state.density = std::move(result.density);
    state.finite_mass = std::move(result.mass);
    state.finite_mass_error = std::move(result.mass_error);
    state.physical_mass.clear();
}

template<ArithmeticMode Mode>
std::vector<PLIC::DynamicGeometryValue> GeneralVoFSolver::Impl::finite_rates_mode(const State& state) const
{
    return detail::general_vof::face_rates<Mode>(*mesh, *state.phases, state.finite_projection, flow);
}

template<ArithmeticMode Mode>
std::vector<VoF::GeneralVoFFaceVolume> GeneralVoFSolver::Impl::finite_face_volumes_mode(
    const State& state, double dt) const
{
    return detail::general_vof::face_volumes<Mode>(*mesh, *state.phases, state.finite_projection, flow, dt);
}

template<ArithmeticMode Mode>
void GeneralVoFSolver::Impl::verify_phase_publication_mode(
    const VoF::GeneralVoF::Trial& trial, const VoF::GeneralVoF& candidate) const
{
    if (candidate.arithmetic_mode() == ArithmeticMode::Exact)
        return Exact::GeneralVoFSolver(*this).verify_phase_publication(trial, candidate);
    detail::general_vof::PhaseStorage<Mode>::validate(*mesh, *accepted->phases, trial, candidate);
}

template<ArithmeticMode Mode> void GeneralVoFSolver::Impl::budget_mode(State& state) const
{
    if (!finite(state))
    {
        Exact::GeneralVoFSolver(*this).budget(state);
        return;
    }
    state.diagnostics = detail::general_vof::Diagnostics<Mode>::evaluate(
        *mesh, *state.phases, material, state.finite_mass, state.velocity, state.diagnostics);
}

template<ArithmeticMode Mode>
std::shared_ptr<GeneralVoFSolver::Impl::State> GeneralVoFSolver::Impl::prepare_step_mode(double dt)
{
    using A = VoF::finite_detail::Arithmetic<Mode>;
    constexpr bool bounded_mode = Mode == ArithmeticMode::BoundedCertification;
    require(
        accepted->phases->arithmetic_mode() == arithmetic_mode_, "General VoF accepted state changed arithmetic mode.");
    if (!finite(*accepted))
        return Exact::GeneralVoFSolver(*this).prepare_step(dt);
    namespace fp = VoF::finite_detail;
    require(std::isfinite(dt) && dt > 0, "Finite VoF timestep must be positive and finite.");
    double minimum = 0, maximum_dt = 0;
    Teuchos::reduceAll(*mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_MIN, 1, &dt, &minimum);
    Teuchos::reduceAll(*mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_MAX, 1, &dt, &maximum_dt);
    require(minimum == maximum_dt, "Finite VoF timestep differs across ranks.");
    const auto& old = *accepted;
    const auto nodes = node_velocity(old);
    require(
        dt <= stable_timestep_mode<Mode>(old, nodes), "Finite VoF timestep exceeds physical geometry/trajectory CFL.");
    const auto trial = old.phases->plan_transport(nodes, finite_face_volumes_mode<Mode>(old, dt), dt);
    double local_h = std::numeric_limits<double>::infinity(), global_h = 0;
    for (auto h : thickness)
        local_h = std::min(local_h, h);
    Teuchos::reduceAll(*mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_MIN, 1, &local_h, &global_h);
    collective(
        [&]
        {
            if (trial.finite_node_trajectories().size() != nodes.size())
                throw std::runtime_error("Finite VoF trajectory certificate is incomplete.");
            const auto limit = A::multiply(A::make_value(global_h, 0), A::make_value(options.flow.max_courant, 0));
            const auto square_limit = A::multiply(limit, limit);
            for (const auto& node : trial.finite_node_trajectories())
            {
                const auto displacement = A::subtract(node.point, node.departure);
                const auto square = A::dot(displacement, displacement);
                if (square.value + (bounded_mode ? square.absolute_error : 0) >
                    square_limit.value - (bounded_mode ? square_limit.absolute_error : 0))
                    throw std::runtime_error("Finite VoF bounded trajectory exceeds physical displacement CFL.");
            }
        });
    auto next = std::make_shared<State>();
    next->phases = trial.candidate();
    require(next->phases->mesh().get() == mesh.get(), "Finite VoF transport changed its immutable native mesh.");
    verify_phase_publication_mode<Mode>(trial, *next->phases);
    phase_storage_mode<Mode>(*next);
    std::vector<long double> outgoing(volume.size()), outgoing_error(volume.size()), source(volume.size());
    std::vector<detail::ConservativeVoFTransport::FiniteTransfer> incoming;
    std::vector<detail::ConservativeVoFTransport::FiniteFaceContribution> face_contributions;
    const auto mass = [&](const PLIC::DynamicPhaseMeasures& p)
    {
        return A::add(A::multiply(p.gas, A::make_value(material.gas_density, 0)),
            A::multiply(p.carrier, A::make_value(material.liquid_density, 0)));
    };
    collective(
        [&]
        {
            std::set<GO> seen;
            for (const auto& donor : trial.finite_donor_budgets())
            {
                const auto row = mesh->owned_cell_map()->getLocalElement(GO(donor.donor));
                if (row < 0 || !seen.insert(GO(donor.donor)).second)
                    throw std::runtime_error("Finite VoF donor budget is duplicate or foreign.");
                const auto exported_mass = mass(donor.exported);
                outgoing[row] = exported_mass.value;
                outgoing_error[row] = exported_mass.absolute_error;
            }
            if (seen.size() != volume.size())
                throw std::runtime_error("Finite VoF donor budgets are incomplete.");
            for (const auto& route : trial.finite_transfers())
                if (route.donor != route.recipient)
                {
                    const auto value = mass(route.amount);
                    incoming.push_back({GO(route.donor), GO(route.recipient), value.value, value.absolute_error});
                }
            for (const auto& face : trial.finite_face_contributions())
            {
                const auto row = mesh->owned_face_map()->getLocalElement(GO(face.face));
                if (row < 0 || (face.direction != 1 && face.direction != -1))
                    throw std::runtime_error("Finite VoF raw stream is not a canonical owned face.");
                const auto value = mass(face.amount);
                face_contributions.push_back({row, GO(face.donor), face.direction * value.value, value.absolute_error});
            }
        });
    next->velocity.resize(volume.size());
    for (unsigned a = 0; a < 3; ++a)
    {
        std::vector<long double> momentum(volume.size()), momentum_error(volume.size());
        for (std::size_t c = 0; c < volume.size(); ++c)
        {
            const auto value = A::multiply(
                A::make_value(old.finite_mass[c], old.finite_mass_error[c]), A::make_value(old.velocity[c][a], 0));
            momentum[c] = value.value;
            momentum_error[c] = value.absolute_error;
        }
        const auto carried = transport.solve(momentum, source, next->finite_mass, outgoing, incoming,
            face_contributions, false, options.maximum_momentum_iterations, options.momentum_relative_tolerance,
            options.byte_budget, momentum_error, {}, next->finite_mass_error, outgoing_error, arithmetic_mode_);
        next->momentum_transport[a] = {
            carried.physical_relative_residual, carried.iterations, carried.physical_refinements};
        for (std::size_t c = 0; c < volume.size(); ++c)
            next->velocity[c][a] = carried.cell_trace[c];
    }
    std::vector<double> viscosity(volume.size());
    collective(
        [&]
        {
            const auto& phases = old.phases->phase_volumes();
            const auto geometry = old.phases->owned_finite_geometry<Mode>();
            for (std::size_t c = 0; c < volume.size(); ++c)
            {
                const auto weighted =
                    A::add(A::multiply(A::make_value(phases[c].gas, 0), A::make_value(material.gas_viscosity, 0)),
                        A::multiply(A::make_value(phases[c].carrier, 0), A::make_value(material.liquid_viscosity, 0)));
                viscosity[c] =
                    A::as_double(A::divide(weighted, VoF::finite_detail::runtime_value<Mode>(geometry[c]->volume())));
            }
        });
    const auto viscous = flow.viscous_force(old.velocity, viscosity);
    std::vector<Vector> change(volume.size());
    collective(
        [&]
        {
            for (std::size_t c = 0; c < volume.size(); ++c)
                for (unsigned a = 0; a < 3; ++a)
                {
                    const auto force = A::divide(A::multiply(A::make_value(viscous[c][a], 0), A::make_value(dt, 0)),
                        A::make_value(next->finite_mass[c], next->finite_mass_error[c]));
                    const auto updated =
                        A::add(A::add(A::make_value(next->velocity[c][a], 0),
                                   A::multiply(A::make_value(options.flow.gravity[a], 0), A::make_value(dt, 0))),
                            force);
                    next->velocity[c][a] = A::as_double(updated);
                    change[c][a] = next->velocity[c][a] - old.velocity[c][a];
                }
        });
    next->face_velocity = old.face_velocity;
    const auto face_change = flow.interpolate(change);
    for (std::size_t f = 0; f < next->face_velocity.size(); ++f)
        next->face_velocity[f] += face_change[f];
    surface_curvature_mode<Mode>(*next);
    next->finite_projection.byte_budget = options.byte_budget;
    const auto closure_topology = projection_topology<detail::GeneralVoFFiniteProjectionClosureTopology>(*next->phases);
    flow.project(next->face_velocity, next->velocity, next->pressure, next->density, next->fraction, next->curvature,
        dt, material.surface_tension, options.flow, next->diagnostics, &next->finite_projection,
        std::span<const long double>(next->finite_mass), &closure_topology,
        std::span<const long double>(next->finite_mass_error));
    next->diagnostics.time = old.diagnostics.time + dt;
    require(std::isfinite(next->diagnostics.time) && next->diagnostics.time > old.diagnostics.time,
        "Finite VoF timestep did not advance the accepted clock.");
    budget_mode<Mode>(*next);
    next->phases = old.phases->apply(trial);
    return next;
}

template<ArithmeticMode Mode>
double GeneralVoFSolver::Impl::handoff_finite_adapted_faces_mode(
    const State& old, Impl& target, State& state, const VoF::GeneralVoF::FiniteRemapTrial& trial) const
{
    auto result = detail::general_vof::transfer_adapted_faces<Mode>(
        {*mesh, *old.phases, old.velocity, old.finite_projection, flow, arithmetic_mode_},
        {*target.mesh, *state.phases, target.volume, target.flow, target.arithmetic_mode_},
        {options.byte_budget, options.flow.divergence_tolerance}, trial);
    state.face_velocity = std::move(result.face_velocity);
    state.finite_projection = std::move(result.projection);
    return result.divergence;
}

template<ArithmeticMode Mode>
double GeneralVoFSolver::Impl::handoff_finite_owned_faces_mode(const State& old, Impl& target, State& state) const
{
    auto result = detail::general_vof::transfer_owned_faces<Mode>(
        {*mesh, *old.phases, old.velocity, old.finite_projection, flow, arithmetic_mode_},
        {*target.mesh, *state.phases, target.volume, target.flow, target.arithmetic_mode_},
        {options.byte_budget, options.flow.divergence_tolerance});
    state.face_velocity = std::move(result.face_velocity);
    state.finite_projection = std::move(result.projection);
    return result.divergence;
}

template<ArithmeticMode Mode>
GeneralVoFSolver::Impl::PreparedRemap GeneralVoFSolver::Impl::prepare_remap_mode(bool migration,
    std::span<const int> destinations, std::span<const Meshes::AMR::AdaptiveMark> refine,
    std::span<const Meshes::AMR::AdaptiveBytes> coarsen) const
{
    using A = VoF::finite_detail::Arithmetic<Mode>;
    constexpr bool bounded_mode = Mode == ArithmeticMode::BoundedCertification;
    if (!finite(*accepted))
        return Exact::GeneralVoFSolver(*this).prepare_remap(migration, destinations, refine, coarsen);
    namespace fp = VoF::finite_detail;
    const auto& old = *accepted;
    const auto hierarchy = old.phases->hierarchy();
    std::vector<int> ordered;
    if (migration)
    {
        require(
            destinations.size() == old.phases->owned_order().size(), "Finite migration destination shape mismatch.");
        collective(
            [&]
            {
                std::map<VoF::GeneralVoF::Key, int> requested;
                for (std::size_t i = 0; i < destinations.size(); ++i)
                    requested.emplace(old.phases->owned_order()[i], destinations[i]);
                for (const auto& [key, leaf] : hierarchy->owned())
                    ordered.push_back(requested.at(key));
            });
    }
    const auto plan = migration ? hierarchy->plan_migration(ordered) : hierarchy->plan_adaptation(refine, coarsen);
    const auto raw_target = hierarchy->apply(plan);
    const auto native = Meshes::AMR::assemble_adaptive_native(*raw_target, raw_target->configuration_descriptor());
    const auto bound = raw_target->with_partition(native.partition, native.mesh->owned_cell_map());
    const auto target = hierarchy->apply(plan, bound);
    VoF::GeneralVoFFiniteRemapOptions remap_options;
    remap_options.byte_budget = options.byte_budget;
    remap_options.phase_density = {material.gas_density, material.liquid_density};
    remap_options.channels = {{"momentum_x", VoF::GeneralVoFFiniteRemapCarrier::MixtureMass, false},
        {"momentum_y", VoF::GeneralVoFFiniteRemapCarrier::MixtureMass, false},
        {"momentum_z", VoF::GeneralVoFFiniteRemapCarrier::MixtureMass, false},
        {"pressure_volume", VoF::GeneralVoFFiniteRemapCarrier::BulkVolume, false}};
    std::vector<VoF::GeneralVoFFiniteOwnedChannels> channels;
    collective(
        [&]
        {
            const auto geometry = old.phases->owned_finite_geometry<Mode>();
            for (std::size_t c = 0; c < old.velocity.size(); ++c)
            {
                VoF::GeneralVoFFiniteOwnedChannels item;
                item.key = old.phases->owned_order()[c];
                for (unsigned a = 0; a < 3; ++a)
                    item.amounts.push_back(A::multiply(A::make_value(old.finite_mass[c], old.finite_mass_error[c]),
                        A::make_value(old.velocity[c][a], 0)));
                item.amounts.push_back(A::multiply(
                    VoF::finite_detail::runtime_value<Mode>(geometry[c]->volume()), A::make_value(old.pressure[c], 0)));
                channels.push_back(std::move(item));
            }
        });
    const auto trial = old.phases->plan_remap(plan, target, channels, remap_options);
    trial.validate_observations();
    const auto phases = trial.candidate();
    require(phases->arithmetic_mode() == arithmetic_mode_, "General VoF remap changed arithmetic mode.");
    auto candidate = std::make_unique<Impl>(phases, material, options);
    auto state = std::make_shared<State>();
    state->phases = phases;
    candidate->phase_storage_mode<Mode>(*state);
    state->velocity.resize(candidate->volume.size());
    state->pressure.resize(candidate->volume.size());
    std::map<VoF::GeneralVoF::Key, const VoF::GeneralVoFFiniteOwnedChannels*> result;
    for (const auto& item : trial.owned_channels())
        result.emplace(item.key, &item);
    candidate->collective(
        [&]
        {
            const auto geometry = phases->owned_finite_geometry<Mode>();
            for (std::size_t c = 0; c < candidate->volume.size(); ++c)
            {
                const auto& item = *result.at(phases->owned_order()[c]);
                if (item.amounts.size() != 4)
                    throw std::runtime_error("Finite remap channel schema differs.");
                for (unsigned a = 0; a < 3; ++a)
                    state->velocity[c][a] = A::as_double(
                        A::divide(item.amounts[a], A::make_value(state->finite_mass[c], state->finite_mass_error[c])));
                state->pressure[c] = A::as_double(
                    A::divide(item.amounts[3], VoF::finite_detail::runtime_value<Mode>(geometry[c]->volume())));
            }
        });
    candidate->surface_curvature_mode<Mode>(*state);
    const auto handoff_divergence = trial.ownership_only()
                                        ? handoff_finite_owned_faces_mode<Mode>(old, *candidate, *state)
                                        : handoff_finite_adapted_faces_mode<Mode>(old, *candidate, *state, trial);
    state->diagnostics = old.diagnostics;
    state->diagnostics.time = old.diagnostics.time;
    state->momentum_transport = old.momentum_transport;
    state->diagnostics.maximum_divergence = handoff_divergence;
    candidate->budget_mode<Mode>(*state);
    // Recovering a finite velocity must preserve the original physical
    // phase-mass-weighted momentum gate independently of channel transfer.
    for (unsigned a = 0; a < 3; ++a)
    {
        VoF::finite_detail::SumFor<Mode> change(arithmetic_mode_), reference(arithmetic_mode_);
        for (std::size_t c = 0; c < old.velocity.size(); ++c)
        {
            const auto amount = A::multiply(
                A::make_value(old.finite_mass[c], old.finite_mass_error[c]), A::make_value(old.velocity[c][a], 0));
            change.add(A::negate(amount));
            reference.add(A::make_value(std::abs(amount.value), amount.absolute_error));
        }
        for (std::size_t c = 0; c < state->velocity.size(); ++c)
        {
            const auto amount = A::multiply(A::make_value(state->finite_mass[c], state->finite_mass_error[c]),
                A::make_value(state->velocity[c][a], 0));
            change.add(amount);
            reference.add(A::make_value(std::abs(amount.value), amount.absolute_error));
        }
        const auto residual = change.value(), l1 = reference.value();
        int local_exponent = std::numeric_limits<int>::min(), exponent = 0;
        if (l1.value != 0)
            (void) std::frexp(l1.value + (bounded_mode ? l1.absolute_error : 0), &local_exponent);
        Teuchos::reduceAll(*mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_MAX, 1, &local_exponent, &exponent);
        const auto scale = exponent == std::numeric_limits<int>::min() ? 1.L : std::scalbn(1.L, exponent - 1);
        const auto center = residual.value / scale;
        const auto narrowed_center = static_cast<double>(center);
        const auto reference_difference =
            !bounded_mode || l1.absolute_error == 0
                ? l1.value
                : std::nextafter(l1.value - l1.absolute_error, -std::numeric_limits<long double>::infinity());
        const auto scaled_reference =
            A::divide(A::make_value(std::max(0.L, reference_difference), 0), A::make_value(scale, 0));
        const auto lower_reference = std::max(0.L, A::lower(scaled_reference));
        auto narrowed_reference = static_cast<double>(lower_reference);
        if (bounded_mode && static_cast<long double>(narrowed_reference) > lower_reference)
            narrowed_reference = std::nextafter(narrowed_reference, 0.);
        double local[4] = {narrowed_center,
            bounded_mode
                ? fp::upper_double(fp::bound_add(residual.absolute_error / scale, std::abs(center - narrowed_center)))
                : 0,
            narrowed_reference, bounded_mode ? fp::upper_double(std::abs(center)) : double(std::abs(center))},
               global[4]{};
        Teuchos::reduceAll(*mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_SUM, 4, local, global);
        const auto u = bounded_mode ? (8.L * mesh->owned_cell_map()->getComm()->getSize() + 16) *
                                          std::numeric_limits<double>::epsilon()
                                    : 0;
        const auto error = bounded_mode ? (global[1] + u * (global[3] + global[1])) / (1 - u) : 0;
        require(std::abs(global[0]) + error <= options.momentum_relative_tolerance * global[2] * (1 - u),
            "Finite fixed-clock remap changed physical phase-mass-weighted momentum.");
    }
    trial.validate_observations();
    return {std::move(candidate), std::move(state)};
}
} // namespace SimpleFluid
