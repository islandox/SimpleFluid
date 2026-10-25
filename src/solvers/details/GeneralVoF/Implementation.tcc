/** @file Implementation.tcc
 * @brief Compiled mode-selected candidate preparation and numerical operations.
 */
namespace SimpleFluid
{
template<ArithmeticMode Mode>
std::shared_ptr<GeneralVoFSolver::Impl::State> GeneralVoFSolver::Impl::construct_mode(std::shared_ptr<const VoF::GeneralVoF> phases, VoFMaterial m, GeneralVoFSolverOptions o)
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
                            normal =
                                A::add(normal, A::cross(A::subtract(p[face[i]].dynamic(), p[face[0]].dynamic()),
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
    state.curvature.assign(volume.size(), 0);
    if (material.surface_tension == 0)
        return;
    const auto patches = state.phases->reconstruction_patches();
    std::map<GO, int> phase_geometry;
    for (const auto& patch : patches)
        phase_geometry.emplace(GO(patch.cell), patch.interface.pure_gas       ? 1
                                               : patch.interface.pure_carrier ? -1
                                                                              : 0);
    collective(
        [&]
        {
            const auto lease = mesh->acquire_execution_view();
            for (std::size_t c = 0; c < volume.size(); ++c)
                for (auto face : mesh->faces(LO(c)))
                {
                    if (mesh->is_geometry_exterior_face(face))
                        continue;
                    const auto a = phase_geometry.at(mesh->cell_global_id(LO(c)));
                    const auto b = phase_geometry.at(mesh->cell_global_id(mesh->opposite_cell(face, LO(c))));
                    // A binary interface still has capillary physics. Until a
                    // canonical pure-interface face fit is implemented, reject
                    // that unresolved surface rather than return zero curvature.
                    if (a * b == -1)
                        throw std::runtime_error("General VoF capillarity requires an admissible reconstructed "
                                                 "surface stencil at a pure-cell interface.");
                }
        });
    std::vector<detail::GeneralVoFCurvaturePatch> surface;
    collective(
        [&]
        {
            for (const auto& patch : patches)
            {
                const auto sample = [&]
                {
                    if constexpr (Mode == ArithmeticMode::Exact)
                        return detail::GeneralVoFCurvature::patch<ArithmeticMode::Exact>(patch.cell,
                            *PLIC::ConvexCellGeometry<ArithmeticMode::Exact>::from_geometry(*patch.geometry),
                            patch.interface);
                    else
                        return detail::GeneralVoFCurvature::patch<Mode>(patch.cell,
                            *PLIC::ConvexCellGeometry<Mode>::from_dynamic(patch.finite_geometry), patch.interface);
                }();
                if (sample)
                    surface.push_back(*sample);
            }
            constexpr auto index_entry_bytes =
                sizeof(std::pair<const std::uint64_t, std::size_t>) + 4 * sizeof(void*);
            if (surface.size() > options.byte_budget / index_entry_bytes)
                throw std::runtime_error("General VoF curvature index exceeds its byte budget.");
            const auto index_bytes = surface.size() * index_entry_bytes;
            std::map<std::uint64_t, std::size_t> surface_index;
            for (std::size_t index = 0; index < surface.size(); ++index)
                surface_index.emplace(surface[index].cell, index);
            for (const auto& patch : surface)
            {
                const auto local = mesh->owned_cell_map()->getLocalElement(GO(patch.cell));
                if (local >= 0)
                {
                    const auto support = state.phases->cell_support(patch.cell);
                    if (support.size() > (options.byte_budget - index_bytes) / sizeof(std::size_t))
                        throw std::runtime_error("General VoF curvature support index exceeds its byte budget.");
                    std::vector<std::size_t> selected;
                    selected.reserve(support.size());
                    for (const auto cell : support)
                        if (const auto found = surface_index.find(cell); found != surface_index.end())
                            selected.push_back(found->second);
                    // Retain the original surface order: curvature fits sum
                    // floating-point stencil contributions in this order.
                    std::sort(selected.begin(), selected.end());
                    std::vector<detail::GeneralVoFCurvaturePatch> stencil;
                    stencil.reserve(selected.size());
                    for (const auto index : selected)
                        stencil.push_back(surface[index]);
                    state.curvature[local] = detail::GeneralVoFCurvature::fit(patch, stencil).curvature;
                }
            }
        });
    Teuchos::Array<GO> ids;
    for (const auto& patch : surface)
        ids.push_back(GO(patch.cell));
    auto patch_map = Teuchos::rcp(new Pack::map_type(
        Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid(), ids(), GO{}, mesh->owned_cell_map()->getComm()));
    const auto k = import(state.curvature, patch_map);
    std::set<GO> mixed_cells;
    for (const auto& patch : surface)
        mixed_cells.insert(GO(patch.cell));
    // Curvature is fitted only at actual mixed PLIC sections. Extend that
    // fitted value to neighboring pure cells for the balanced face force;
    // no manufactured normal or pure-cell curvature fit is substituted.
    collective(
        [&]
        {
            for (std::size_t c = 0; c < volume.size(); ++c)
            {
                const auto gid = mesh->cell_global_id(LO(c));
                if (mixed_cells.contains(gid))
                    continue;
                const auto support = state.phases->cell_support(std::uint64_t(gid));
                const std::set<std::uint64_t> admitted(support.begin(), support.end());
                const auto center = mesh->cell_centroid(LO(c));
                double nearest = std::numeric_limits<double>::infinity();
                for (std::size_t p = 0; p < surface.size(); ++p)
                {
                    if (!admitted.contains(surface[p].cell))
                        continue;
                    const auto& point = surface[p].centroid;
                    const double dx = point[0] - center.x, dy = point[1] - center.y, dz = point[2] - center.z;
                    const double distance = dx * dx + dy * dy + dz * dz;
                    if (distance < nearest)
                    {
                        nearest = distance;
                        state.curvature[c] = k[p];
                    }
                }
            }
        });
}

template<ArithmeticMode Mode>
double GeneralVoFSolver::Impl::stable_timestep_mode(const State& state, const std::vector<VoF::GeneralVoFNodeVelocity>& nodes) const
{
    using A = VoF::finite_detail::Arithmetic<Mode>;
    constexpr bool bounded_mode = Mode == ArithmeticMode::BoundedCertification;
    const auto bound =
        flow.stable_timestep(state.face_velocity, std::min(material.gas_density, material.liquid_density),
            std::max(material.gas_viscosity, material.liquid_viscosity), material.surface_tension, options.flow);
    double local_thickness = std::numeric_limits<double>::infinity(), global_thickness = 0;
    for (double h : thickness)
        local_thickness = std::min(local_thickness, h);
    Teuchos::reduceAll(
        *mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_MIN, 1, &local_thickness, &global_thickness);
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
                        (bounded_mode
                                ? VoF::finite_detail::runtime_value<Mode>(geometry[c]->volume()).absolute_error
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
std::shared_ptr<GeneralVoFSolver::Impl::State> GeneralVoFSolver::Impl::prepare_initialize_mode(const std::vector<Vector>& owned_velocity, const std::vector<double>& face_velocity)
{
    require(accepted->phases->arithmetic_mode() == arithmetic_mode_,
        "General VoF initialization changed arithmetic mode.");
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

template<class Topology>
Topology GeneralVoFSolver::Impl::projection_topology(const VoF::GeneralVoF& phases) const
{
    Topology result;
    collective(
        [&]
        {
            std::map<GO, const Meshes::AMR::AdaptiveTile*> tiles;
            for (const auto& tile : phases.tiles())
                tiles.emplace(GO(tile.face_gid), &tile);
            const auto lease = mesh->acquire_execution_view();
            for (std::size_t c = 0; c < mesh->num_owned_cells(); ++c)
                result.cells.push_back(
                    {mesh->cell_global_id(LO(c)), Meshes::AMR::encode_adaptive_key(phases.owned_order()[c]), {}});
            for (std::size_t f = 0; f < mesh->num_owned_faces(); ++f)
            {
                const auto face = LO(f);
                if (mesh->is_geometry_exterior_face(face))
                    continue;
                const auto& tile = *tiles.at(mesh->face_global_id(face));
                std::vector<char> identity;
                for (const auto* bytes : {&tile.domain, &tile.surface, &tile.identity})
                {
                    append_word(identity, bytes->size());
                    identity.insert(identity.end(), bytes->begin(), bytes->end());
                }
                result.faces.push_back({mesh->face_global_id(face), mesh->cell_global_id(mesh->owner_cell(face)),
                    mesh->cell_global_id(mesh->neighbor_cell(face)), std::move(identity), {}, {}});
            }
        });
    return result;
}

template<ArithmeticMode Mode> void GeneralVoFSolver::Impl::phase_storage_mode(State& state) const
{
    using A = VoF::finite_detail::Arithmetic<Mode>;
    constexpr bool bounded_mode = Mode == ArithmeticMode::BoundedCertification;
    if (!finite(state))
    {
        Exact::GeneralVoFSolver(*this).phase_storage(state);
        return;
    }
    namespace fp = VoF::finite_detail;
    const auto& phases = state.phases->phase_volumes();
    const auto geometry = state.phases->owned_finite_geometry<Mode>();
    const auto fraction = state.phases->geometric_gas_fractions();
    collective(
        [&]
        {
            if (phases.size() != volume.size() || geometry.size() != volume.size())
                throw std::runtime_error("Finite VoF phase/native dimensions differ.");
            state.fraction = fraction;
            state.density.resize(volume.size());
            state.finite_mass.resize(volume.size());
            state.finite_mass_error.resize(volume.size());
            state.physical_mass.clear();
            for (std::size_t c = 0; c < volume.size(); ++c)
            {
                const auto mass = A::add(
                    A::multiply(A::make_value(phases[c].gas, 0), A::make_value(material.gas_density, 0)),
                    A::multiply(A::make_value(phases[c].carrier, 0), A::make_value(material.liquid_density, 0)));
                const auto density =
                    A::divide(mass, VoF::finite_detail::runtime_value<Mode>(geometry[c]->volume()));
                state.finite_mass[c] = mass.value;
                state.finite_mass_error[c] = mass.absolute_error;
                state.density[c] = A::as_double(density);
                if (!(mass.value > (bounded_mode ? mass.absolute_error : 0)) || !std::isfinite(state.density[c]) ||
                    !(state.density[c] > 0) || state.fraction[c] < 0 || state.fraction[c] > 1)
                    throw std::runtime_error("Finite VoF physical mixture storage is unresolved.");
            }
        });
}

template<ArithmeticMode Mode> std::vector<PLIC::DynamicGeometryValue> GeneralVoFSolver::Impl::finite_rates_mode(const State& state) const
{
    using A = VoF::finite_detail::Arithmetic<Mode>;
    require(state.phases->arithmetic_mode() == arithmetic_mode_ &&
                state.finite_projection.arithmetic_mode == arithmetic_mode_ &&
                state.finite_projection.face_rate.size() == mesh->num_owned_faces() &&
                state.finite_projection.face_radius.size() == mesh->num_owned_faces(),
        "Finite VoF projection rate authority is incomplete.");
    std::array<std::vector<double>, 6> wire;
    collective(
        [&]
        {
            for (auto& v : wire)
                v.resize(mesh->num_faces());
            for (std::size_t f = 0; f < mesh->num_owned_faces(); ++f)
                for (unsigned k = 0; k < 2; ++k)
                {
                    const auto value =
                        k == 0 ? state.finite_projection.face_rate[f] : state.finite_projection.face_radius[f];
                    int exponent = 0;
                    const auto mantissa = std::frexp(value, &exponent);
                    const auto high = static_cast<double>(mantissa), low = static_cast<double>(mantissa - high);
                    if (!std::isfinite(value) || (k && value < 0) ||
                        static_cast<long double>(high) + low != mantissa)
                        throw std::runtime_error("Finite VoF rate exceeds fixed wire precision.");
                    wire[3 * k][f] = high;
                    wire[3 * k + 1][f] = low;
                    wire[3 * k + 2][f] = exponent;
                }
        });
    for (auto& value : wire)
        value = flow.faces(value);
    std::vector<PLIC::DynamicGeometryValue> result(mesh->num_faces());
    collective(
        [&]
        {
            for (std::size_t f = 0; f < result.size(); ++f)
            {
                long double values[2]{};
                for (unsigned k = 0; k < 2; ++k)
                {
                    const auto exponent = wire[3 * k + 2][f];
                    if (!std::isfinite(exponent) || exponent < std::numeric_limits<int>::min() ||
                        exponent > std::numeric_limits<int>::max() || exponent != std::trunc(exponent))
                        throw std::runtime_error("Finite VoF canonical wire exponent is malformed.");
                    values[k] =
                        std::scalbn(static_cast<long double>(wire[3 * k][f]) + wire[3 * k + 1][f], int(exponent));
                    if (!std::isfinite(values[k]) || (k && values[k] < 0))
                        throw std::runtime_error("Finite VoF canonical wire rate is nonfinite.");
                }
                result[f] = A::make_value(values[0], values[1]);
            }
        });
    return result;
}

template<ArithmeticMode Mode>
std::vector<VoF::GeneralVoFFaceVolume> GeneralVoFSolver::Impl::finite_face_volumes_mode(const State& state, double dt) const
{
    using A = VoF::finite_detail::Arithmetic<Mode>;
    const auto rates = finite_rates_mode<Mode>(state);
    std::vector<VoF::GeneralVoFFaceVolume> result;
    collective(
        [&]
        {
            for (const auto& tile : state.phases->tiles())
            {
                const auto local = mesh->overlap_face_map()->getLocalElement(GO(tile.face_gid));
                if (local < 0)
                    throw std::runtime_error("Finite VoF missing canonical face rate.");
                const auto integrated = A::multiply(rates[local], A::make_value(dt, 0));
                const auto diagnostic = static_cast<double>(integrated.value);
                if (!std::isfinite(diagnostic))
                    throw std::runtime_error("Finite VoF face diagnostic overflows.");
                result.push_back({tile.face_gid, diagnostic, std::nullopt, integrated});
            }
        });
    return result;
}

template<ArithmeticMode Mode>
void GeneralVoFSolver::Impl::verify_phase_publication_mode(const VoF::GeneralVoF::Trial& trial, const VoF::GeneralVoF& candidate) const
{
    using A = VoF::finite_detail::Arithmetic<Mode>;
    constexpr bool bounded_mode = Mode == ArithmeticMode::BoundedCertification;
    if (candidate.arithmetic_mode() == ArithmeticMode::Exact)
    {
        Exact::GeneralVoFSolver(*this).verify_phase_publication(trial, candidate);
        return;
    }
    namespace fp = VoF::finite_detail;
    collective(
        [&]
        {
            if (candidate.arithmetic_mode() != arithmetic_mode_ ||
                candidate.arithmetic_mode() != accepted->phases->arithmetic_mode())
                throw std::runtime_error("Finite VoF candidate changed inventory policy.");
            if (candidate.accepted_identity() == accepted->phases->accepted_identity())
                return;
            std::vector<VoF::finite_detail::SumFor<Mode>> gas(
                volume.size(), VoF::finite_detail::SumFor<Mode>(arithmetic_mode_)),
                carrier(volume.size(), VoF::finite_detail::SumFor<Mode>(arithmetic_mode_));
            std::vector<detail::CompensatedSum<>> gas_magnitude(volume.size()), carrier_magnitude(volume.size());
            std::vector<std::size_t> publication_terms(volume.size());
            for (const auto& route : trial.finite_transfers())
            {
                const auto row = mesh->owned_cell_map()->getLocalElement(GO(route.recipient));
                if (row < 0)
                    throw std::runtime_error("Finite phase publication has foreign recipient.");
                gas[row].add(route.amount.gas);
                carrier[row].add(route.amount.carrier);
                gas_magnitude[row] += std::abs(route.amount.gas.value);
                carrier_magnitude[row] += std::abs(route.amount.carrier.value);
                ++publication_terms[row];
            }
            const auto& phases = candidate.phase_volumes();
            const auto& bounds = trial.phase_roundoff_bounds();
            if (phases.size() != volume.size() || bounds.size() != volume.size())
                throw std::runtime_error("Finite phase publication has incompatible shape.");
            for (std::size_t c = 0; c < volume.size(); ++c)
                for (const auto& item :
                    {std::tuple{phases[c].gas, bounds[c].gas, gas[c].value(), gas_magnitude[c].value()},
                        std::tuple{phases[c].carrier, bounds[c].carrier, carrier[c].value(),
                            carrier_magnitude[c].value()}})
                {
                    const auto expected = std::get<2>(item);
                    const auto difference = std::abs(static_cast<long double>(std::get<0>(item)) - expected.value);
                    // Two independently ordered sums of the same n represented
                    // terms differ by at most twice the standard gamma bound.
                    // This allowance uses count/magnitude/precision, not defect.
                    const auto u = (publication_terms[c] + 4.L) * std::numeric_limits<long double>::epsilon();
                    if (!(u < .25L))
                        throw std::runtime_error("Finite phase publication reduction bound exhausted.");
                    const auto sum_roundoff = bounded_mode ? 0.L : 2 * u / (1 - u) * std::get<3>(item);
                    if (!std::isfinite(std::get<1>(item)) || std::get<1>(item) < 0 ||
                        difference + (bounded_mode ? expected.absolute_error : 0) >
                            std::get<1>(item) + sum_roundoff)
                        throw std::runtime_error("Finite phase publication exceeds its own bounded ledger.");
                }
        });
}

template<ArithmeticMode Mode> void GeneralVoFSolver::Impl::budget_mode(State& state) const
{
    using A = VoF::finite_detail::Arithmetic<Mode>;
    if (!finite(state))
    {
        Exact::GeneralVoFSolver(*this).budget(state);
        return;
    }
    namespace fp = VoF::finite_detail;
    std::array<VoF::finite_detail::SumFor<Mode>, 7> sums;
    for (auto& sum : sums)
        sum = VoF::finite_detail::SumFor<Mode>(arithmetic_mode_);
    collective(
        [&]
        {
            const auto& phases = state.phases->phase_volumes();
            for (std::size_t c = 0; c < volume.size(); ++c)
            {
                sums[0].add(A::make_value(phases[c].gas, 0));
                sums[1].add(A::multiply(A::make_value(phases[c].gas, 0), A::make_value(material.gas_density, 0)));
                sums[2].add(
                    A::multiply(A::make_value(phases[c].carrier, 0), A::make_value(material.liquid_density, 0)));
                for (unsigned a = 0; a < 3; ++a)
                    sums[3 + a].add(A::multiply(
                        A::make_value(state.finite_mass[c], 0), A::make_value(state.velocity[c][a], 0)));
                const auto square = static_cast<long double>(state.velocity[c][0]) * state.velocity[c][0] +
                                    static_cast<long double>(state.velocity[c][1]) * state.velocity[c][1] +
                                    static_cast<long double>(state.velocity[c][2]) * state.velocity[c][2];
                sums[6].add(A::multiply(A::make_value(state.finite_mass[c], 0), A::make_value(square / 2, 0)));
            }
        });
    std::array<double, 7> global{};
    for (unsigned a = 0; a < 7; ++a)
    {
        const auto local = sums[a].value();
        int local_exponent = std::numeric_limits<int>::min(), exponent = 0;
        if (local.value != 0)
            (void) std::frexp(std::abs(local.value), &local_exponent);
        Teuchos::reduceAll(*mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_MAX, 1, &local_exponent, &exponent);
        const auto scale = exponent == std::numeric_limits<int>::min() ? 1.L : std::scalbn(1.L, exponent - 1);
        const auto normalized = static_cast<double>(local.value / scale);
        double sum = 0;
        Teuchos::reduceAll(*mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_SUM, 1, &normalized, &sum);
        collective(
            [&]
            {
                const auto value = static_cast<long double>(sum) * scale;
                global[a] = static_cast<double>(value);
                if (!std::isfinite(global[a]) || (value != 0 && global[a] == 0))
                    throw std::runtime_error(
                        "Finite VoF global physical diagnostic is outside representable range.");
            });
    }
    state.diagnostics.gas_volume = global[0];
    state.diagnostics.gas_mass = global[1];
    state.diagnostics.liquid_mass = global[2];
    for (unsigned a = 0; a < 3; ++a)
        state.diagnostics.momentum[a] = global[3 + a];
    state.diagnostics.kinetic_energy = global[6];
    double speed = 0;
    for (const auto& velocity : state.velocity)
        speed = std::max(speed, std::sqrt(dot(velocity, velocity)));
    state.diagnostics.maximum_velocity = maximum(speed);
}

template<ArithmeticMode Mode> std::shared_ptr<GeneralVoFSolver::Impl::State> GeneralVoFSolver::Impl::prepare_step_mode(double dt)
{
    using A = VoF::finite_detail::Arithmetic<Mode>;
    constexpr bool bounded_mode = Mode == ArithmeticMode::BoundedCertification;
    require(accepted->phases->arithmetic_mode() == arithmetic_mode_,
        "General VoF accepted state changed arithmetic mode.");
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
    require(dt <= stable_timestep_mode<Mode>(old, nodes),
        "Finite VoF timestep exceeds physical geometry/trajectory CFL.");
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
                face_contributions.push_back(
                    {row, GO(face.donor), face.direction * value.value, value.absolute_error});
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
                const auto weighted = A::add(
                    A::multiply(A::make_value(phases[c].gas, 0), A::make_value(material.gas_viscosity, 0)),
                    A::multiply(A::make_value(phases[c].carrier, 0), A::make_value(material.liquid_viscosity, 0)));
                viscosity[c] = A::as_double(
                    A::divide(weighted, VoF::finite_detail::runtime_value<Mode>(geometry[c]->volume())));
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
    const auto closure_topology =
        projection_topology<detail::GeneralVoFFiniteProjectionClosureTopology>(*next->phases);
    flow.project(next->face_velocity, next->velocity, next->pressure, next->density, next->fraction,
        next->curvature, dt, material.surface_tension, options.flow, next->diagnostics, &next->finite_projection,
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
    using A = VoF::finite_detail::Arithmetic<Mode>;
    constexpr bool bounded_mode = Mode == ArithmeticMode::BoundedCertification;
    require(target.arithmetic_mode_ == arithmetic_mode_ && state.phases->arithmetic_mode() == arithmetic_mode_,
        "General VoF adapted handoff changed arithmetic mode.");
    namespace fp = VoF::finite_detail;
    using Point = PLIC::Point<Mode>;
    using PA = fp::PointArithmetic<Mode>;
    using Value = PLIC::DynamicGeometryValue;
    using Key = VoF::GeneralVoF::Key;
    using Loop = std::vector<Point>;
    using Support = PLIC::PlaneAuthority;
    const auto comm = mesh->owned_cell_map()->getComm();
    const auto& common = trial.source_common_cells();
    std::vector<PLIC::ConvexGeometryPtr<Mode>> common_geometry;
    std::size_t common_geometry_bytes = 0;
    collective(
        [&]
        {
            constexpr auto per_cell =
                sizeof(PLIC::ConvexGeometryPtr<Mode>) + sizeof(PLIC::ConvexCellGeometry<Mode>);
            if (common.size() > options.byte_budget / per_cell)
                throw std::runtime_error("Finite common-cell typed geometry storage cap exhausted.");
            common_geometry_bytes = common.size() * per_cell;
            common_geometry.reserve(common.size());
            for (const auto& cell : common)
                common_geometry.push_back(PLIC::ConvexCellGeometry<Mode>::from_dynamic(cell.geometry));
        });
    std::vector<Key> requested;
    std::map<Key, std::vector<std::size_t>> columns;
    for (std::size_t i = 0; i < common.size(); ++i)
    {
        columns[common[i].source_key].push_back(i);
        requested.push_back(common[i].target_key);
    }
    const auto draft = Meshes::AMR::fetch_adaptive_native_draft(*state.phases->hierarchy(), requested);
    std::map<Key, SimpleFluid::PLIC::ConvexGeometryPtr<SimpleFluid::ArithmeticMode::Ordinary>> native_sources;
    std::map<std::uint64_t, std::set<Key>> native_fans;
    collective(
        [&]
        {
            for (const auto* leaves : {&draft.owned, &draft.ghosts})
                for (const auto& leaf : *leaves)
                    native_sources.emplace(leaf.key, state.phases->hierarchy()->geometry(leaf));
            for (const auto& tile : draft.tiles)
                for (auto node : tile.node_gids)
                    for (const auto* key : {&tile.first, tile.second ? &*tile.second : nullptr})
                        if (key && native_sources.contains(*key))
                            native_fans[node].insert(*key);
        });
    std::map<std::uint64_t, Point> native_points;
    collective(
        [&]
        {
            for (const auto& node : draft.nodes)
            {
                const auto fan = native_fans.find(node.gid);
                if (fan == native_fans.end() || fan->second.empty())
                    continue;
                std::vector<SimpleFluid::PLIC::ConvexGeometryPtr<SimpleFluid::ArithmeticMode::Ordinary>> cells;
                for (const auto& key : fan->second)
                    cells.push_back(native_sources.at(key));
                native_points.emplace(node.gid, PLIC::source_point<Mode>(cells, node.coordinate_identity));
            }
        });
    std::map<std::uint64_t, Loop> target_loops;
    std::map<Key, std::vector<const Meshes::AMR::AdaptiveTile*>> target_faces;
    collective(
        [&]
        {
            std::set<Key> required(requested.begin(), requested.end());
            for (const auto& tile : draft.tiles)
                if (required.contains(tile.first))
                {
                    Loop loop;
                    for (auto node : tile.node_gids)
                        loop.push_back(native_points.at(node));
                    target_loops.emplace(tile.face_gid, std::move(loop));
                    target_faces[tile.first].push_back(&tile);
                }
        });
    const auto rates = finite_rates_mode<Mode>(old);
    std::map<Key, std::vector<const Meshes::AMR::AdaptiveTile*>> source_faces;
    std::map<std::uint64_t, Loop> source_loops;
    collective(
        [&]
        {
            for (const auto& tile : old.phases->tiles())
            {
                if (columns.contains(tile.first))
                    source_faces[tile.first].push_back(&tile);
                if (tile.second && columns.contains(*tile.second))
                    source_faces[*tile.second].push_back(&tile);
                source_loops.emplace(tile.face_gid, old.phases->finite_face_points<Mode>(tile.face_gid));
            }
        });
    PLIC::GeometryLimits geometry_limits;
    geometry_limits.byte_budget = options.byte_budget / 2;
    geometry_limits.arithmetic_mode = arithmetic_mode_;
    std::size_t pairs = 0, patches = 0, retained_bytes = common_geometry_bytes;
    const auto charge = [&](std::size_t bytes)
    {
        if (bytes > options.byte_budget - std::min(retained_bytes, options.byte_budget))
            throw std::runtime_error("Finite common-face handoff storage cap exhausted.");
        retained_bytes += bytes;
    };
    const auto shared_support = [](const Loop& first, const Loop& second) -> std::optional<Support>
    {
        const auto a = PLIC::common_supports<Mode>(first), b = PLIC::common_supports<Mode>(second);
        for (const auto& key : a)
            if (std::find(b.begin(), b.end(), key) != b.end())
                return key;
        return {};
    };
    const auto overlap = [&](const Loop& first, const Loop& second)
    {
        if (++pairs > VoF::ExactFaceTransferOptions{}.maximum_pair_tests)
            throw std::runtime_error("Finite common-face pair cap exhausted.");
        const auto support = shared_support(first, second);
        if (!support)
            return PLIC::PlanarFaceOverlap<Mode>{};
        auto result = PLIC::planar_face_overlap<Mode>(first, second, *support, geometry_limits);
        if (!result.overlap_vertices.empty())
        {
            if (++patches > VoF::ExactFaceTransferOptions{}.maximum_patches)
                throw std::runtime_error("Finite common-face patch cap exhausted.");
            charge(sizeof(result) + result.overlap_vertices.capacity() * sizeof(Point));
            for (const auto& point : result.overlap_vertices)
                charge(PLIC::point_storage_bytes<Mode>(point) - sizeof(Point));
        }
        return result;
    };
    const auto nonempty = [](const PLIC::PlanarFaceOverlap<Mode>& value)
    {
        if (value.overlap_projected_area.value == 0 &&
            fp::runtime_value<Mode>(value.overlap_projected_area).absolute_error == 0)
            return false;
        if (!(value.overlap_projected_area.value >
                fp::runtime_value<Mode>(value.overlap_projected_area).absolute_error))
            throw std::runtime_error("Finite common-face overlap positivity is unresolved.");
        return true;
    };
    const auto certify_cover = [&](const Loop& whole, const std::vector<Loop>& pieces)
    {
        VoF::finite_detail::SumFor<Mode> fraction(arithmetic_mode_);
        for (std::size_t i = 0; i < pieces.size(); ++i)
        {
            const auto part = overlap(whole, pieces[i]);
            if (!nonempty(part))
                throw std::runtime_error("Finite common-face cover contains empty piece.");
            fraction.add(part.orientation > 0 ? fp::runtime_value<Mode>(part.overlap_over_first)
                                              : A::negate(fp::runtime_value<Mode>(part.overlap_over_first)));
            for (std::size_t j = 0; j < i; ++j)
            {
                const auto collision = overlap(pieces[i], pieces[j]);
                if (collision.overlap_projected_area.value != 0 ||
                    fp::runtime_value<Mode>(collision.overlap_projected_area).absolute_error != 0)
                    throw std::runtime_error("Finite common-face cover has overlapping or unresolved interiors.");
            }
        }
        const auto difference = A::subtract(fraction.value(), A::make_value(1, 0));
        if (bounded_mode
                ? (std::abs(difference.value) > difference.absolute_error ||
                      difference.absolute_error >
                          VoF::GeneralVoFFiniteRemapOptions{}.maximum_relative_target_defect)
                : std::abs(difference.value) > VoF::GeneralVoFFiniteRemapOptions{}.maximum_relative_target_defect)
            throw std::runtime_error("Finite common-face cover has a hole or exhausted arithmetic budget.");
    };
    struct Part
    {
        Loop loop;
        Value rate;
        std::optional<std::size_t> edge;
        int sign = 1;
    };
    struct Facet
    {
        std::size_t cell;
        Loop loop;
        std::vector<Part> parts;
        bool inherited = false;
    };
    struct Internal
    {
        std::size_t first, second;
        Value rate;
    };
    struct Output
    {
        Key source, target;
        std::uint64_t source_revision, face, proof;
        int target_owner;
        Value rate, fraction;
    };
    std::vector<Output> outputs;
    collective(
        [&]
        {
            for (const auto& [source_key, unordered_indices] : columns)
            {
                auto indices = unordered_indices;
                std::sort(indices.begin(), indices.end(),
                    [&](auto a, auto b) { return common[a].target_key < common[b].target_key; });
                const auto source_row =
                    mesh->owned_cell_map()->getLocalElement(GO(common[indices.front()].source_gid));
                if (source_row < 0)
                    throw std::runtime_error("Finite common-face source is not locally owned.");
                const auto velocity = old.velocity[source_row];
                std::vector<Facet> facets;
                std::vector<std::vector<std::size_t>> cell_facets(indices.size());
                std::map<std::uint64_t, std::vector<Loop>> old_cover;
                for (std::size_t c = 0; c < indices.size(); ++c)
                {
                    const auto& geometry = *common_geometry[indices[c]];
                    const auto points = geometry.points();
                    for (const auto& face : geometry.faces())
                    {
                        Facet facet{c, {}, {}};
                        for (auto vertex : face)
                            facet.loop.push_back(points[vertex]);
                        for (const auto* tile : source_faces.at(source_key))
                        {
                            const auto part = overlap(facet.loop, source_loops.at(tile->face_gid));
                            if (!nonempty(part))
                                continue;
                            const auto local = mesh->overlap_face_map()->getLocalElement(GO(tile->face_gid));
                            if (local < 0)
                                throw std::runtime_error("Finite source face rate is absent.");
                            const auto rate =
                                A::multiply(rates[local], fp::runtime_value<Mode>(part.overlap_over_second));
                            facet.parts.push_back({part.overlap_vertices, rate, {}, 1});
                            old_cover[tile->face_gid].push_back(part.overlap_vertices);
                        }
                        if (!facet.parts.empty())
                        {
                            std::vector<Loop> cover;
                            for (const auto& part : facet.parts)
                                cover.push_back(part.loop);
                            certify_cover(facet.loop, cover);
                            facet.inherited = true;
                        }
                        charge(sizeof(Facet) + facet.loop.size() * sizeof(Point));
                        if (++patches > VoF::ExactFaceTransferOptions{}.maximum_patches)
                            throw std::runtime_error("Finite common-face patch cap exhausted.");
                        cell_facets[c].push_back(facets.size());
                        facets.push_back(std::move(facet));
                    }
                }
                for (const auto* tile : source_faces.at(source_key))
                {
                    const auto cover = old_cover.find(tile->face_gid);
                    if (cover == old_cover.end())
                        throw std::runtime_error("Finite old source face is uncovered.");
                    certify_cover(source_loops.at(tile->face_gid), cover->second);
                }
                std::vector<Internal> internal;
                for (std::size_t f = 0; f < facets.size(); ++f)
                {
                    if (facets[f].inherited)
                        continue;
                    for (std::size_t g = f + 1; g < facets.size(); ++g)
                    {
                        if (facets[g].inherited || facets[g].cell == facets[f].cell)
                            continue;
                        const auto piece = overlap(facets[f].loop, facets[g].loop);
                        if (!nonempty(piece))
                            continue;
                        if (piece.orientation != -1)
                            throw std::runtime_error("Finite interior common faces have equal orientation.");
                        Point area;
                        const auto& loop = piece.overlap_vertices;
                        for (std::size_t i = 1; i + 1 < loop.size(); ++i)
                            area = PA::add(area,
                                PA::cross(PA::subtract(loop[i], loop[0]), PA::subtract(loop[i + 1], loop[0])));
                        const auto rate = A::divide(PA::dot(area, Point(A::point(velocity))), A::make_value(2, 0));
                        const auto index = internal.size();
                        internal.push_back({facets[f].cell, facets[g].cell, rate});
                        facets[f].parts.push_back({loop, A::make_value(0), index, 1});
                        auto reversed = loop;
                        std::reverse(reversed.begin(), reversed.end());
                        facets[g].parts.push_back({std::move(reversed), A::make_value(0), index, -1});
                    }
                }
                for (const auto& face : facets)
                {
                    std::vector<Loop> pieces;
                    for (const auto& part : face.parts)
                        pieces.push_back(part.loop);
                    certify_cover(face.loop, pieces);
                }
                const auto divergences = [&]
                {
                    std::vector<VoF::finite_detail::SumFor<Mode>> sums(
                        indices.size(), VoF::finite_detail::SumFor<Mode>(arithmetic_mode_));
                    for (const auto& face : facets)
                        for (const auto& part : face.parts)
                            sums[face.cell].add(part.edge ? (part.sign > 0 ? internal[*part.edge].rate
                                                                           : A::negate(internal[*part.edge].rate))
                                                          : part.rate);
                    std::vector<Value> result;
                    for (const auto& sum : sums)
                        result.push_back(sum.value());
                    return result;
                };
                std::vector<std::vector<std::size_t>> adjacent(indices.size());
                for (std::size_t e = 0; e < internal.size(); ++e)
                {
                    adjacent[internal[e].first].push_back(e);
                    adjacent[internal[e].second].push_back(e);
                }
                const auto none = std::numeric_limits<std::size_t>::max();
                std::vector<std::size_t> parent(indices.size(), none), parent_edge(indices.size(), none), order{0};
                parent[0] = 0;
                for (std::size_t i = 0; i < order.size(); ++i)
                    for (auto edge : adjacent[order[i]])
                    {
                        const auto other =
                            internal[edge].first == order[i] ? internal[edge].second : internal[edge].first;
                        if (parent[other] == none)
                        {
                            parent[other] = order[i];
                            parent_edge[other] = edge;
                            order.push_back(other);
                        }
                    }
                if (order.size() != indices.size())
                    throw std::runtime_error("Finite common refinement is not face connected.");
                auto defect = divergences();
                for (std::size_t i = order.size(); i-- > 1;)
                {
                    const auto child = order[i], edge = parent_edge[child];
                    internal[edge].rate = A::add(internal[edge].rate,
                        internal[edge].first == child ? A::negate(defect[child]) : defect[child]);
                    defect[parent[child]] = A::add(defect[parent[child]], defect[child]);
                }
                const auto remaining = divergences();
                for (std::size_t cell = 0; cell < remaining.size(); ++cell)
                {
                    const auto& residual = remaining[cell];
                    const auto allowed = bounded_mode
                                             ? residual.absolute_error
                                             : static_cast<long double>(options.flow.divergence_tolerance) *
                                                   common_geometry[indices[cell]]->volume().value;
                    if (std::abs(residual.value) > allowed)
                        throw std::runtime_error("Finite new common-face extension fails source-free closure.");
                }
                std::uint64_t proof = 0;
                for (const auto& face : facets)
                    for (const auto& part : face.parts)
                    {
                        const auto& cell = common[indices[face.cell]];
                        const auto rate = part.edge ? (part.sign > 0 ? internal[*part.edge].rate
                                                                     : A::negate(internal[*part.edge].rate))
                                                    : part.rate;
                        for (const auto* tile : target_faces.at(cell.target_key))
                        {
                            const auto piece = overlap(part.loop, target_loops.at(tile->face_gid));
                            if (!nonempty(piece))
                                continue;
                            if (piece.orientation != 1)
                                throw std::runtime_error("Finite target boundary has inconsistent orientation.");
                            outputs.push_back({source_key, cell.target_key,
                                old.phases->hierarchy()->owned().at(source_key).revision, tile->face_gid, proof++,
                                tile->first_owner,
                                A::multiply(rate, fp::runtime_value<Mode>(piece.overlap_over_first)),
                                fp::runtime_value<Mode>(piece.overlap_over_second)});
                        }
                    }
            }
        });
    // The producer has independently proved every common-cell facet cover,
    // inherited source-face cover and opposite internal incidence. Together
    // with the opaque phase trial's donor/recipient cover, these immutable
    // proofs compose into a target boundary partition. The receiver checks
    // the bounded area closure and authenticates each proof exactly once.
    const auto put_scalar = [](std::vector<char>& out, long double value)
    {
        int exponent = 0;
        const auto m = std::frexp(value, &exponent);
        const auto hi = static_cast<double>(m), lo = static_cast<double>(m - hi);
        if (!std::isfinite(value) || static_cast<long double>(hi) + lo != m)
            throw std::runtime_error("Finite adapted face scalar exceeds wire precision.");
        append_word(out, std::bit_cast<std::uint64_t>(hi));
        append_word(out, std::bit_cast<std::uint64_t>(lo));
        append_word(out, std::bit_cast<std::uint64_t>(static_cast<std::int64_t>(exponent)));
    };
    const auto put_bytes = [](std::vector<char>& out, const std::vector<char>& key)
    {
        append_word(out, key.size());
        out.insert(out.end(), key.begin(), key.end());
    };
    std::vector<OwnerPacket> packets;
    collective(
        [&]
        {
            std::size_t used = 0;
            for (const auto& item : outputs)
            {
                OwnerPacket packet;
                packet.peer = item.target_owner;
                append_word(packet.bytes, old.phases->accepted_control());
                append_word(packet.bytes, state.phases->accepted_control());
                put_bytes(packet.bytes, Meshes::AMR::encode_adaptive_key(item.source));
                append_word(packet.bytes, item.source_revision);
                put_bytes(packet.bytes, Meshes::AMR::encode_adaptive_key(item.target));
                append_word(packet.bytes, item.face);
                append_word(packet.bytes, item.proof);
                for (const auto value : {item.rate, item.fraction})
                {
                    put_scalar(packet.bytes, value.value);
                    put_scalar(packet.bytes, value.absolute_error);
                }
                used += sizeof(OwnerPacket) + packet.bytes.size();
                if (used > options.byte_budget / 2)
                    throw std::runtime_error("Finite adapted face packet cap exhausted.");
                packets.push_back(std::move(packet));
            }
        });
    const auto received = exchange_owner_packets(packets, comm, options.byte_budget / 2);
    struct Incoming
    {
        Key source, target;
        std::uint64_t revision, face, proof;
        int owner;
        Value rate, fraction;
    };
    std::vector<Incoming> incoming;
    std::vector<Key> donor_keys;
    target.collective(
        [&]
        {
            for (const auto& packet : received.packets)
            {
                std::size_t offset = 0;
                const auto word = [&]()
                {
                    const auto value = read_word(packet.bytes, offset);
                    offset += 8;
                    return value;
                };
                const auto blob = [&]()
                {
                    const auto count = word();
                    if (count > packet.bytes.size() - std::min(offset, packet.bytes.size()))
                        throw std::runtime_error("Finite adapted face identity is malformed.");
                    std::vector<char> value(packet.bytes.begin() + offset, packet.bytes.begin() + offset + count);
                    offset += count;
                    return value;
                };
                const auto scalar = [&]()
                {
                    const auto hi = std::bit_cast<double>(word()), lo = std::bit_cast<double>(word());
                    const auto exponent = std::bit_cast<std::int64_t>(word());
                    if (exponent < std::numeric_limits<int>::min() || exponent > std::numeric_limits<int>::max())
                        throw std::runtime_error("Finite adapted face exponent is malformed.");
                    const auto value = std::scalbn(static_cast<long double>(hi) + lo, int(exponent));
                    if (!std::isfinite(value))
                        throw std::runtime_error("Finite adapted face scalar is nonfinite.");
                    return value;
                };
                if (word() != old.phases->accepted_control() || word() != state.phases->accepted_control())
                    throw std::runtime_error("Finite adapted face proof publication is stale.");
                Incoming item;
                item.source = Meshes::AMR::decode_adaptive_key(blob());
                item.revision = word();
                item.target = Meshes::AMR::decode_adaptive_key(blob());
                item.face = word();
                item.proof = word();
                const auto rate_value = scalar(), rate_error = scalar();
                const auto fraction_value = scalar(), fraction_error = scalar();
                item.rate = A::make_value(rate_value, rate_error);
                item.fraction = A::make_value(fraction_value, fraction_error);
                item.owner = packet.peer;
                if (offset != packet.bytes.size() || item.rate.absolute_error < 0 ||
                    item.fraction.absolute_error < 0 ||
                    item.fraction.value < (bounded_mode ? item.fraction.absolute_error : 0))
                    throw std::runtime_error("Finite adapted face proof has malformed bounded values.");
                donor_keys.push_back(item.source);
                incoming.push_back(std::move(item));
            }
        });
    const auto owners = old.phases->hierarchy()->lookup_cells(donor_keys);
    target.collective(
        [&]
        {
            std::map<GO, const Meshes::AMR::AdaptiveTile*> actual;
            for (const auto& tile : state.phases->tiles())
                if (target.mesh->owned_face_map()->getLocalElement(GO(tile.face_gid)) >= 0)
                    actual.emplace(GO(tile.face_gid), &tile);
            std::vector<VoF::finite_detail::SumFor<Mode>> flux(
                target.mesh->num_owned_faces(), VoF::finite_detail::SumFor<Mode>(arithmetic_mode_)),
                coverage(flux.size(), VoF::finite_detail::SumFor<Mode>(arithmetic_mode_));
            std::set<std::tuple<Key, std::uint64_t, std::uint64_t>> seen;
            for (std::size_t i = 0; i < incoming.size(); ++i)
            {
                const auto& item = incoming[i];
                const auto local = target.mesh->owned_face_map()->getLocalElement(GO(item.face));
                if (owners[i].owner != item.owner || owners[i].revision != item.revision || local < 0 ||
                    actual.at(GO(item.face))->first != item.target ||
                    !seen.emplace(item.source, item.face, item.proof).second)
                    throw std::runtime_error(
                        "Finite adapted face proof has foreign or duplicate source/target authority.");
                flux[local].add(item.rate);
                coverage[local].add(item.fraction);
            }
            state.finite_projection.byte_budget = options.byte_budget;
            state.finite_projection.arithmetic_mode = arithmetic_mode_;
            state.finite_projection.face_rate.resize(flux.size());
            state.finite_projection.face_radius.resize(flux.size());
            state.finite_projection.cell_radius.resize(target.volume.size());
            state.face_velocity.resize(target.mesh->num_faces());
            for (std::size_t f = 0; f < flux.size(); ++f)
            {
                const auto cover = A::subtract(coverage[f].value(), A::make_value(1, 0));
                if (bounded_mode ? (std::abs(cover.value) > cover.absolute_error ||
                                       cover.absolute_error >
                                           VoF::GeneralVoFFiniteRemapOptions{}.maximum_relative_target_defect)
                                 : std::abs(cover.value) >
                                       VoF::GeneralVoFFiniteRemapOptions{}.maximum_relative_target_defect)
                    throw std::runtime_error("Finite adapted target face has incomplete bounded coverage.");
                const auto rate = flux[f].value();
                if (target.mesh->is_geometry_exterior_face(LO(f)) && (rate.value != 0 || rate.absolute_error != 0))
                    throw std::runtime_error("Finite adapted handoff changes a physical impermeable wall.");
                state.finite_projection.face_rate[f] = rate.value;
                state.finite_projection.face_radius[f] = rate.absolute_error;
                const auto diagnostic = A::divide(rate, A::make_value(target.mesh->face_area(LO(f)), 0));
                state.face_velocity[f] = static_cast<double>(diagnostic.value);
                if (!std::isfinite(state.face_velocity[f]))
                    throw std::runtime_error("Finite adapted face velocity diagnostic overflows.");
            }
        });
    const auto final_rates = target.finite_rates_mode<Mode>(state);
    double local_divergence = 0;
    target.collective(
        [&]
        {
            const auto lease = target.mesh->acquire_execution_view();
            for (std::size_t c = 0; c < target.volume.size(); ++c)
            {
                VoF::finite_detail::SumFor<Mode> net(arithmetic_mode_);
                for (auto face : target.mesh->faces(LO(c)))
                    net.add(
                        target.mesh->owner_cell(face) == LO(c) ? final_rates[face] : A::negate(final_rates[face]));
                const auto residual = net.value();
                if (bounded_mode && std::abs(residual.value) > residual.absolute_error)
                    throw std::runtime_error("Finite adapted handoff fails canonical target closure.");
                state.finite_projection.cell_radius[c] = residual.absolute_error;
                const auto divergence_bound = A::divide(
                    A::make_value(bounded_mode ? fp::bound_add(std::abs(residual.value), residual.absolute_error)
                                               : std::abs(residual.value),
                        0),
                    A::make_value(target.volume[c], 0));
                local_divergence = std::max(local_divergence, fp::upper_double(A::upper(divergence_bound)));
            }
        });
    const auto divergence = target.maximum(local_divergence);
    target.require(divergence <= options.flow.divergence_tolerance,
        "Finite adapted handoff exceeds the unchanged physical divergence target.");
    state.face_velocity = target.flow.faces(state.face_velocity);
    trial.validate_observations();
    return divergence;
}

template<ArithmeticMode Mode>
double GeneralVoFSolver::Impl::handoff_finite_owned_faces_mode(const State& old, Impl& target, State& state) const
{
    using A = VoF::finite_detail::Arithmetic<Mode>;
    constexpr bool bounded_mode = Mode == ArithmeticMode::BoundedCertification;
    require(target.arithmetic_mode_ == arithmetic_mode_ && state.phases->arithmetic_mode() == arithmetic_mode_,
        "General VoF ownership handoff changed arithmetic mode.");
    namespace fp = VoF::finite_detail;
    const auto comm = mesh->owned_cell_map()->getComm();
    const auto identity = [](const Meshes::AMR::AdaptiveTile& tile)
    {
        std::vector<char> result;
        for (const auto* key : {&tile.domain, &tile.surface, &tile.identity})
        {
            append_word(result, key->size());
            result.insert(result.end(), key->begin(), key->end());
        }
        return result;
    };
    const auto authority = [&](const std::vector<char>& key)
    {
        std::uint64_t hash = 1469598103934665603ULL;
        for (unsigned char c : key)
            hash = (hash ^ c) * 1099511628211ULL;
        return int(hash % comm->getSize()); // Routing only; full identity is compared.
    };
    const auto bytes = [](std::vector<char>& out, const std::vector<char>& value)
    {
        append_word(out, value.size());
        out.insert(out.end(), value.begin(), value.end());
    };
    const auto scalar = [](std::vector<char>& out, long double value)
    {
        int exponent = 0;
        const auto mantissa = std::frexp(value, &exponent);
        const auto high = static_cast<double>(mantissa), low = static_cast<double>(mantissa - high);
        if (!std::isfinite(value) || static_cast<long double>(high) + low != mantissa)
            throw std::runtime_error("Finite face handoff scalar exceeds fixed precision.");
        append_word(out, std::bit_cast<std::uint64_t>(high));
        append_word(out, std::bit_cast<std::uint64_t>(low));
        append_word(out, std::bit_cast<std::uint64_t>(static_cast<std::int64_t>(exponent)));
    };
    const auto word = [](const std::vector<char>& in, std::size_t& offset)
    {
        const auto value = read_word(in, offset);
        offset += 8;
        return value;
    };
    const auto read_bytes = [&](const std::vector<char>& in, std::size_t& offset)
    {
        const auto count = word(in, offset);
        if (count > in.size() - std::min(offset, in.size()))
            throw std::runtime_error("Finite face handoff identity length is malformed.");
        std::vector<char> result(in.begin() + offset, in.begin() + offset + count);
        offset += count;
        return result;
    };
    const auto read_scalar = [&](const std::vector<char>& in, std::size_t& offset)
    {
        const auto high = std::bit_cast<double>(word(in, offset)), low = std::bit_cast<double>(word(in, offset));
        const auto exponent = std::bit_cast<std::int64_t>(word(in, offset));
        if (exponent < std::numeric_limits<int>::min() || exponent > std::numeric_limits<int>::max())
            throw std::runtime_error("Finite face handoff exponent is malformed.");
        const auto value = std::scalbn(static_cast<long double>(high) + low, int(exponent));
        if (!std::isfinite(value))
            throw std::runtime_error("Finite face handoff scalar is nonfinite.");
        return value;
    };
    std::map<GO, const Meshes::AMR::AdaptiveTile*> target_tiles;
    std::vector<OwnerPacket> packets;
    collective(
        [&]
        {
            if (old.finite_projection.face_rate.size() != mesh->num_owned_faces() ||
                old.finite_projection.face_radius.size() != mesh->num_owned_faces())
                throw std::runtime_error("Finite ownership handoff lacks source rate authority.");
            std::size_t used = 0;
            for (const auto& tile : old.phases->tiles())
            {
                const auto face = mesh->owned_face_map()->getLocalElement(GO(tile.face_gid));
                if (face < 0)
                    continue;
                auto key = identity(tile);
                OwnerPacket packet;
                packet.peer = authority(key);
                append_word(packet.bytes, 0);
                append_word(packet.bytes, old.phases->accepted_control());
                bytes(packet.bytes, key);
                bytes(packet.bytes, Meshes::AMR::encode_adaptive_key(tile.first));
                bytes(packet.bytes,
                    tile.second ? Meshes::AMR::encode_adaptive_key(*tile.second) : std::vector<char>{});
                scalar(packet.bytes, old.finite_projection.face_rate[face]);
                scalar(packet.bytes, old.finite_projection.face_radius[face]);
                used += packet.bytes.size() + sizeof(OwnerPacket);
                if (used > options.byte_budget / 2)
                    throw std::runtime_error("Finite face source directory exceeds budget.");
                packets.push_back(std::move(packet));
            }
            for (const auto& tile : state.phases->tiles())
            {
                const auto face = target.mesh->owned_face_map()->getLocalElement(GO(tile.face_gid));
                if (face < 0)
                    continue;
                target_tiles.emplace(GO(tile.face_gid), &tile);
                auto key = identity(tile);
                OwnerPacket packet;
                packet.peer = authority(key);
                append_word(packet.bytes, 1);
                append_word(packet.bytes, old.phases->accepted_control());
                bytes(packet.bytes, key);
                append_word(packet.bytes, tile.face_gid);
                bytes(packet.bytes, Meshes::AMR::encode_adaptive_key(tile.first));
                used += packet.bytes.size() + sizeof(OwnerPacket);
                if (used > options.byte_budget / 2)
                    throw std::runtime_error("Finite face target directory exceeds budget.");
                packets.push_back(std::move(packet));
            }
        });
    const auto directory = exchange_owner_packets(packets, comm, options.byte_budget / 2);
    struct SourceFace
    {
        std::vector<char> first, second;
        long double rate, radius;
    };
    struct Request
    {
        std::vector<char> identity, first;
        std::uint64_t face;
        int owner;
    };
    std::map<std::vector<char>, SourceFace> sources;
    std::vector<Request> requests;
    packets.clear();
    collective(
        [&]
        {
            for (const auto& packet : directory.packets)
            {
                std::size_t offset = 0;
                const auto type = word(packet.bytes, offset), control = word(packet.bytes, offset);
                auto key = read_bytes(packet.bytes, offset);
                if (control != old.phases->accepted_control() || authority(key) != comm->getRank())
                    throw std::runtime_error("Finite face directory has stale or foreign source authority.");
                if (type == 0)
                {
                    auto first = read_bytes(packet.bytes, offset), second = read_bytes(packet.bytes, offset);
                    const auto rate = read_scalar(packet.bytes, offset), radius = read_scalar(packet.bytes, offset);
                    if (radius < 0 || !sources.emplace(key, SourceFace{first, second, rate, radius}).second)
                        throw std::runtime_error("Finite canonical source face is duplicated.");
                }
                else if (type == 1)
                {
                    const auto face = word(packet.bytes, offset);
                    requests.push_back({key, read_bytes(packet.bytes, offset), face, packet.peer});
                }
                else
                    throw std::runtime_error("Finite face directory type is malformed.");
                if (offset != packet.bytes.size())
                    throw std::runtime_error("Finite face directory has trailing payload.");
            }
            for (const auto& request : requests)
            {
                const auto found = sources.find(request.identity);
                if (found == sources.end())
                    throw std::runtime_error(
                        "Finite adaptive face handoff requires a bounded surface-overlap certificate.");
                const auto& source = found->second;
                const auto sign = request.first == source.first ? 1 : request.first == source.second ? -1 : 0;
                if (!sign)
                    throw std::runtime_error("Finite face handoff incident-cell orientation is foreign.");
                OwnerPacket reply;
                reply.peer = request.owner;
                append_word(reply.bytes, request.face);
                bytes(reply.bytes, request.identity);
                scalar(reply.bytes, sign * source.rate);
                scalar(reply.bytes, source.radius);
                packets.push_back(std::move(reply));
            }
        });
    const auto result = exchange_owner_packets(packets, comm, options.byte_budget / 2);
    target.collective(
        [&]
        {
            state.finite_projection.byte_budget = options.byte_budget;
            state.finite_projection.arithmetic_mode = arithmetic_mode_;
            state.finite_projection.face_rate.resize(target.mesh->num_owned_faces());
            state.finite_projection.face_radius.resize(target.mesh->num_owned_faces());
            state.finite_projection.cell_radius.resize(target.volume.size());
            state.face_velocity.resize(target.mesh->num_faces());
            std::set<GO> seen;
            for (const auto& packet : result.packets)
            {
                std::size_t offset = 0;
                const auto gid = GO(word(packet.bytes, offset));
                const auto key = read_bytes(packet.bytes, offset);
                const auto rate = read_scalar(packet.bytes, offset), radius = read_scalar(packet.bytes, offset);
                const auto local = target.mesh->owned_face_map()->getLocalElement(gid);
                if (local < 0 || !seen.insert(gid).second || !target_tiles.contains(gid) ||
                    identity(*target_tiles.at(gid)) != key || packet.peer != authority(key) ||
                    offset != packet.bytes.size() || radius < 0)
                    throw std::runtime_error("Finite ownership face handoff has foreign or duplicate target.");
                if (target.mesh->is_geometry_exterior_face(local) && rate != 0)
                    throw std::runtime_error("Finite ownership handoff changes a closed wall.");
                const auto velocity =
                    A::divide(A::make_value(rate, radius), A::make_value(target.mesh->face_area(local), 0));
                state.face_velocity[local] = static_cast<double>(velocity.value);
                if (!std::isfinite(state.face_velocity[local]))
                    throw std::runtime_error("Finite ownership face velocity diagnostic overflows.");
                state.finite_projection.face_rate[local] = rate;
                state.finite_projection.face_radius[local] = radius;
            }
            if (seen.size() != target.mesh->num_owned_faces())
                throw std::runtime_error("Finite ownership handoff lacks target face coverage.");
        });
    const auto rates = target.finite_rates_mode<Mode>(state);
    double local_divergence = 0;
    target.collective(
        [&]
        {
            const auto lease = target.mesh->acquire_execution_view();
            for (std::size_t c = 0; c < target.volume.size(); ++c)
            {
                VoF::finite_detail::SumFor<Mode> net(arithmetic_mode_);
                for (const auto face : target.mesh->faces(LO(c)))
                    net.add(target.mesh->owner_cell(face) == LO(c) ? rates[face] : A::negate(rates[face]));
                const auto residual = net.value();
                if (bounded_mode && std::abs(residual.value) > residual.absolute_error)
                    throw std::runtime_error("Finite ownership handoff loses canonical source-free closure.");
                state.finite_projection.cell_radius[c] = residual.absolute_error;
                const auto divergence_bound = A::divide(
                    A::make_value(bounded_mode ? fp::bound_add(std::abs(residual.value), residual.absolute_error)
                                               : std::abs(residual.value),
                        0),
                    A::make_value(target.volume[c], 0));
                local_divergence = std::max(local_divergence, fp::upper_double(A::upper(divergence_bound)));
            }
        });
    state.face_velocity = target.flow.faces(state.face_velocity);
    const auto divergence = target.maximum(local_divergence);
    target.require(divergence <= options.flow.divergence_tolerance,
        "Finite fixed-clock face handoff exceeds the unchanged physical divergence target.");
    return divergence;
}

template<ArithmeticMode Mode>
GeneralVoFSolver::Impl::PreparedRemap GeneralVoFSolver::Impl::prepare_remap_mode(bool migration, std::span<const int> destinations,
    std::span<const Meshes::AMR::AdaptiveMark> refine, std::span<const Meshes::AMR::AdaptiveBytes> coarsen) const
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
        require(destinations.size() == old.phases->owned_order().size(),
            "Finite migration destination shape mismatch.");
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
                item.amounts.push_back(A::multiply(VoF::finite_detail::runtime_value<Mode>(geometry[c]->volume()),
                    A::make_value(old.pressure[c], 0)));
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
                    state->velocity[c][a] = A::as_double(A::divide(
                        item.amounts[a], A::make_value(state->finite_mass[c], state->finite_mass_error[c])));
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
            bounded_mode ? fp::upper_double(
                               fp::bound_add(residual.absolute_error / scale, std::abs(center - narrowed_center)))
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
