/** @file Operations.tcc
 * @brief Optional rational backend in the Exact namespace.
 * Included only by SolverImpl.cc after its shared support provider.
 */
namespace SimpleFluid::Exact
{
template<class Context> struct GeneralVoFSolver
{
    using Impl = std::remove_const_t<Context>;
    using State = typename Impl::State;
    Context& self;
    explicit GeneralVoFSolver(Context& context) : self(context) {}
    using FaceMeasures = std::map<GO, std::pair<SimpleFluid::PLIC::GeometryMeasure<SimpleFluid::ArithmeticMode::Exact>,
                                          SimpleFluid::PLIC::GeometryMeasure<SimpleFluid::ArithmeticMode::Exact>>>;
    void phase_storage(State& state) const
    {
        const auto& gas = state.phases->gas_volumes();
        const auto& carrier = state.phases->carrier_volumes();
        const auto geometric_fraction = state.phases->geometric_gas_fractions();
        std::vector<SimpleFluid::PLIC::PhaseMeasures<SimpleFluid::ArithmeticMode::Exact>> authority;
        self.require(gas.size() == self.volume.size() && carrier.size() == self.volume.size(),
            "General VoF phase/native storage dimensions differ.");
        std::vector<SimpleFluid::PLIC::ConvexGeometryPtr<SimpleFluid::ArithmeticMode::Ordinary>> geometry;
        self.collective(
            [&]
            {
                authority = current_inventory(*state.phases);
                geometry = state.phases->owned_geometry();
                state.fraction.resize(self.volume.size());
                state.density.resize(self.volume.size());
                state.physical_mass.resize(self.volume.size());
            });
        self.collective(
            [&]
            {
                for (std::size_t c = 0; c < self.volume.size(); ++c)
                {
                    state.fraction[c] = geometric_fraction[c];
                    state.physical_mass[c] = phase_mass_measure(authority[c]);
                    const auto& mass = state.physical_mass[c];
                    const auto density = mass.divided_by(
                        SimpleFluid::PLIC::GeometryMeasure<SimpleFluid::ArithmeticMode::Exact>::volume(*geometry[c]),
                        self.options.byte_budget);
                    state.density[c] = density.sign() ? density.certified().as_double() : 0;
                    if (mass.sign() <= 0 || !std::isfinite(state.density[c]) || !(state.density[c] > 0) ||
                        !std::isfinite(state.fraction[c]) || state.fraction[c] < 0 || state.fraction[c] > 1)
                        throw std::runtime_error("General VoF final extensive mixture mass/fraction is invalid.");
                }
            });
    }

    void budget(State& state) const
    {
        std::array<SimpleFluid::PLIC::GeometryMeasure<SimpleFluid::ArithmeticMode::Exact>, 7> local;
        std::vector<SimpleFluid::PLIC::PhaseMeasures<SimpleFluid::ArithmeticMode::Exact>> authority;
        bool valid = true;
        self.collective(
            [&]
            {
                authority = current_inventory(*state.phases);
                for (std::size_t c = 0; c < self.volume.size(); ++c)
                {
                    const auto& gas = authority[c].gas;
                    const auto& carrier = authority[c].carrier;
                    const auto& mass = state.physical_mass[c];
                    const auto& velocity = state.velocity[c];
                    local[0] = detail::vof_exact::add(local[0], gas, self.options.byte_budget);
                    local[1] = detail::vof_exact::add(
                        local[1], gas.scaled(self.material.gas_density), self.options.byte_budget);
                    local[2] = detail::vof_exact::add(
                        local[2], carrier.scaled(self.material.liquid_density), self.options.byte_budget);
                    for (unsigned a = 0; a < 3; ++a)
                    {
                        const auto momentum = mass.scaled(velocity[a]);
                        local[3 + a] = detail::vof_exact::add(local[3 + a], momentum, self.options.byte_budget);
                        valid = valid && std::isfinite(velocity[a]);
                    }
                    local[6] = detail::vof_exact::add(
                        local[6], mass.scaled(.5 * dot(velocity, velocity)), self.options.byte_budget);
                    valid = valid && std::isfinite(state.density[c]) && state.density[c] > 0;
                }
            });
        self.require(valid, "General VoF accepted candidate has nonfinite velocity/storage.");
        std::array<double, 7> global{};
        for (unsigned a = 0; a < 7; ++a)
        {
            const auto exact = global_sum({local[a]});
            self.collective([&] { global[a] = exact.sign() ? exact.certified().as_double() : 0; });
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
        state.diagnostics.maximum_velocity = self.maximum(speed);
        self.require(std::all_of(global.begin(), global.end(), [](double value) { return std::isfinite(value); }),
            "General VoF global accepted budget is nonfinite.");
    }

    std::shared_ptr<State> prepare_step(double dt)
    {
        self.require(std::isfinite(dt) && dt > 0, "General VoF timestep must be positive and finite.");
        double minimum = 0, maximum_dt = 0;
        Teuchos::reduceAll(*self.mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_MIN, 1, &dt, &minimum);
        Teuchos::reduceAll(*self.mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_MAX, 1, &dt, &maximum_dt);
        self.require(minimum == maximum_dt, "General VoF timestep differs across ranks.");
        const auto& old = *self.accepted;
        const auto nodes = self.node_velocity(old);
        self.require(dt <= self.stable_timestep(old, nodes),
            "General VoF timestep exceeds its physical geometry/trajectory limit.");
        const auto controls = closed_face_volumes(old, dt);
        const auto trial = old.phases->plan_transport(nodes, controls, dt);
        double local_h = std::numeric_limits<double>::infinity(), global_h = 0;
        for (double h : self.thickness)
            local_h = std::min(local_h, h);
        Teuchos::reduceAll(*self.mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_MIN, 1, &local_h, &global_h);
        self.collective(
            [&]
            {
                const auto scalar_budget = self.options.byte_budget / 32;
                SimpleFluid::PLIC::TrajectoryLimits limits;
                limits.byte_budget = scalar_budget;
                const auto displacement_limit =
                    SimpleFluid::PLIC::GeometryMeasure<SimpleFluid::ArithmeticMode::Exact>::from_value(global_h)
                        .scaled(self.options.flow.max_courant)
                        .squared(scalar_budget);
                if (trial.node_trajectories().size() != nodes.size())
                    throw std::runtime_error("Incomplete exact node trajectory CFL certificate.");
                for (const auto& node : trial.node_trajectories())
                    if (node.trajectory.displacement_squared(limits).compare(displacement_limit) > 0)
                        throw std::runtime_error("Exact node trajectory exceeds physical displacement CFL.");
            });
        auto next = std::make_shared<State>();
        next->phases = trial.candidate();
        self.require(
            next->phases->mesh().get() == self.mesh.get(), "General VoF transport changed its immutable native mesh.");
        verify_phase_publication(trial, *next->phases);
        phase_storage(*next);
        std::vector<SimpleFluid::PLIC::GeometryMeasure<SimpleFluid::ArithmeticMode::Exact>> outgoing(
            self.volume.size()),
            source(self.volume.size()), capacity(self.volume.size());
        std::vector<detail::ConservativeVoFTransport::ExactTransfer> incoming;
        std::vector<detail::ConservativeVoFTransport::ExactFaceContribution> face_contributions;
        self.collective(
            [&]
            {
                capacity = next->physical_mass;
                std::set<GO> seen;
                for (const auto& donor : trial.donor_budgets())
                {
                    const auto local = self.mesh->owned_cell_map()->getLocalElement(GO(donor.donor));
                    if (local < 0 || !seen.insert(GO(donor.donor)).second)
                        throw std::runtime_error("General VoF donor budget is duplicate or foreign.");
                    outgoing[local] = phase_mass_measure(donor.exact_exported);
                }
                if (seen.size() != self.volume.size())
                    throw std::runtime_error("General VoF donor budgets do not cover native owned cells.");
                for (const auto& route : trial.transfers())
                    if (route.donor != route.recipient)
                        incoming.push_back({GO(route.donor), GO(route.recipient), phase_mass_measure(route.exact)});
                for (const auto& face : trial.face_contributions())
                {
                    const auto local = self.mesh->owned_face_map()->getLocalElement(GO(face.face));
                    if (local < 0 || (face.direction != 1 && face.direction != -1))
                        throw std::runtime_error("General VoF raw donor stream is not a canonical owned face.");
                    const auto mass = phase_mass_measure(face.exact);
                    face_contributions.push_back({local, GO(face.donor), mass.scaled(face.direction)});
                }
            });
        next->velocity.resize(self.volume.size());
        for (unsigned a = 0; a < 3; ++a)
        {
            std::vector<SimpleFluid::PLIC::GeometryMeasure<SimpleFluid::ArithmeticMode::Exact>> momentum(
                self.volume.size());
            self.collective(
                [&]
                {
                    for (std::size_t c = 0; c < self.volume.size(); ++c)
                    {
                        momentum[c] = old.physical_mass[c].scaled(old.velocity[c][a]);
                    }
                });
            const auto carried = detail::Exact::ConservativeVoFTransport::solve(self.transport, momentum, source,
                capacity, outgoing, incoming, face_contributions, false, self.options.maximum_momentum_iterations,
                self.options.momentum_relative_tolerance, self.options.byte_budget);
            next->momentum_transport[a] = {
                carried.physical_relative_residual, carried.iterations, carried.physical_refinements};
            for (std::size_t c = 0; c < self.volume.size(); ++c)
                next->velocity[c][a] = carried.cell_trace[c];
        }
        std::vector<double> viscosity(self.volume.size());
        const auto physical_geometry = old.phases->owned_geometry();
        std::vector<SimpleFluid::PLIC::PhaseMeasures<SimpleFluid::ArithmeticMode::Exact>> old_inventory;
        self.collective(
            [&]
            {
                old_inventory = current_inventory(*old.phases);
                for (std::size_t c = 0; c < self.volume.size(); ++c)
                {
                    const auto weighted = old_inventory[c].gas.scaled(self.material.gas_viscosity) +
                                          old_inventory[c].carrier.scaled(self.material.liquid_viscosity);
                    const auto coefficient = weighted.divided_by(
                        SimpleFluid::PLIC::GeometryMeasure<SimpleFluid::ArithmeticMode::Exact>::volume(
                            *physical_geometry[c]),
                        self.options.byte_budget);
                    viscosity[c] = coefficient.sign() ? coefficient.certified().as_double() : 0;
                }
            });
        const auto viscous = self.flow.viscous_force(old.velocity, viscosity);
        std::vector<Vector> change(self.volume.size());
        self.collective(
            [&]
            {
                for (std::size_t c = 0; c < self.volume.size(); ++c)
                    for (unsigned a = 0; a < 3; ++a)
                    {
                        const auto mass = next->physical_mass[c];
                        const auto viscous_change =
                            SimpleFluid::PLIC::GeometryMeasure<SimpleFluid::ArithmeticMode::Exact>::from_value(
                                viscous[c][a])
                                .scaled(dt)
                                .divided_by(mass, self.options.byte_budget);
                        const auto updated =
                            SimpleFluid::PLIC::GeometryMeasure<SimpleFluid::ArithmeticMode::Exact>::from_value(
                                next->velocity[c][a]) +
                            SimpleFluid::PLIC::GeometryMeasure<SimpleFluid::ArithmeticMode::Exact>::from_value(
                                self.options.flow.gravity[a])
                                .scaled(dt) +
                            viscous_change;
                        next->velocity[c][a] = updated.sign() ? updated.certified().as_double() : 0;
                        change[c][a] = next->velocity[c][a] - old.velocity[c][a];
                    }
            });
        next->face_velocity = old.face_velocity;
        const auto face_change = self.flow.interpolate(change);
        for (std::size_t f = 0; f < next->face_velocity.size(); ++f)
            next->face_velocity[f] += face_change[f];
        self.surface_curvature(*next);
        next->projection_arithmetic.byte_budget = self.options.byte_budget;
        const auto closure_topology = self.projection_topology(*next->phases);
        detail::Exact::GeneralVoFFlowWorkspace::project(self.flow, next->face_velocity, next->velocity, next->pressure,
            next->density, next->fraction, next->curvature, dt, self.material.surface_tension, self.options.flow,
            next->diagnostics, &next->projection_arithmetic,
            std::span<const SimpleFluid::PLIC::GeometryMeasure<SimpleFluid::ArithmeticMode::Exact>>(
                next->physical_mass),
            &closure_topology);
        next->diagnostics.time = old.diagnostics.time + dt;
        self.require(std::isfinite(next->diagnostics.time) && next->diagnostics.time > old.diagnostics.time,
            "General VoF timestep did not advance its finite clock.");
        budget(*next);
        // The authority verifies source identity again only after every flow
        // gate passes. The coordinator publishes the returned immutable snapshot.
        next->phases = old.phases->apply(trial);
        return next;
    }

    typename Impl::PreparedRemap prepare_remap(bool migration, std::span<const int> destinations,
        std::span<const Meshes::AMR::AdaptiveMark> refine, std::span<const Meshes::AMR::AdaptiveBytes> coarsen) const
    {
        using M = SimpleFluid::PLIC::GeometryMeasure<SimpleFluid::ArithmeticMode::Exact>;
        const auto& old = *self.accepted;
        const auto hierarchy = old.phases->hierarchy();
        auto source = old.phases->export_transfer_source(!migration);
        std::vector<VoF::ExactTransferChannel> schema = {{"momentum_x", VoF::ExactTransferCarrier::MixtureMass, false},
            {"momentum_y", VoF::ExactTransferCarrier::MixtureMass, false},
            {"momentum_z", VoF::ExactTransferCarrier::MixtureMass, false},
            {"pressure_volume", VoF::ExactTransferCarrier::BulkVolume, false}};
        std::vector<VoF::ExactTransferOwnedChannels> channels;
        const auto source_geometry = old.phases->owned_geometry();
        std::vector<SimpleFluid::PLIC::PhaseMeasures<SimpleFluid::ArithmeticMode::Exact>> source_inventory;
        self.collective(
            [&]
            {
                source_inventory = current_inventory(*old.phases);
                for (std::size_t c = 0; c < old.phases->owned_order().size(); ++c)
                {
                    const auto& gas = source_inventory[c].gas;
                    const auto& carrier = source_inventory[c].carrier;
                    VoF::ExactTransferOwnedChannels item;
                    item.key = old.phases->owned_order()[c];
                    item.phase_mass = VoF::ExactTransferPhaseMass{
                        gas.scaled(self.material.gas_density), carrier.scaled(self.material.liquid_density)};
                    const auto mass = item.phase_mass->gas + item.phase_mass->carrier;
                    for (unsigned a = 0; a < 3; ++a)
                        item.amounts.push_back(mass.scaled(old.velocity[c][a]));
                    item.amounts.push_back(M::volume(*source_geometry[c]).scaled(old.pressure[c]));
                    channels.push_back(std::move(item));
                }
            });
        VoF::ExactPhaseTransferOptions transfer_options;
        transfer_options.byte_budget = self.options.byte_budget;
        source = source.with_channels(std::move(schema), std::move(channels), transfer_options);
        std::vector<int> ordered_destinations;
        if (migration)
        {
            self.require(destinations.size() == old.phases->owned_order().size(),
                "General VoF repartition destination count differs.");
            self.collective(
                [&]
                {
                    std::map<VoF::GeneralVoF::Key, int> requested;
                    for (std::size_t i = 0; i < destinations.size(); ++i)
                        requested.emplace(old.phases->owned_order()[i], destinations[i]);
                    for (const auto& [key, leaf] : hierarchy->owned())
                        ordered_destinations.push_back(requested.at(key));
                });
        }
        const auto plan =
            migration ? hierarchy->plan_migration(ordered_destinations) : hierarchy->plan_adaptation(refine, coarsen);
        const auto raw_target = hierarchy->apply(plan);
        const auto native_target =
            Meshes::AMR::assemble_adaptive_native(*raw_target, raw_target->configuration_descriptor());
        const auto bound_target =
            raw_target->with_partition(native_target.partition, native_target.mesh->owned_cell_map());
        const auto target = hierarchy->apply(plan, bound_target);
        const auto transferred = VoF::ExactPhaseTransfer::prepare(source, plan, target, transfer_options);
        const auto candidate_phase = old.phases->adopt_transfer(transferred);
        self.require(candidate_phase->inventory_policy() == old.phases->inventory_policy(),
            "General VoF remap changed inventory authority policy.");
        auto candidate = std::make_unique<Impl>(candidate_phase, self.material, self.options);
        auto state = std::make_shared<State>();
        state->phases = candidate_phase;
        candidate->phase_storage(*state);
        state->velocity.resize(candidate->volume.size());
        std::vector<double> mapped_pressure(candidate->volume.size());
        std::map<VoF::GeneralVoF::Key, const VoF::ExactPhaseTransferCapsule*> result;
        for (const auto& item : transferred.owned())
            result.emplace(item.key, &item);
        const auto target_geometry = candidate_phase->owned_geometry();
        candidate->collective(
            [&]
            {
                for (std::size_t c = 0; c < candidate_phase->owned_order().size(); ++c)
                {
                    const auto& item = *result.at(candidate_phase->owned_order()[c]);
                    if (item.amounts.size() != 4 || !item.phase_mass)
                        throw std::runtime_error("Exact general flow transfer schema differs.");
                    const auto mass = state->physical_mass[c];
                    if (!std::isfinite(item.roundoff.gas) || !std::isfinite(item.roundoff.carrier) ||
                        item.roundoff.gas < 0 || item.roundoff.carrier < 0)
                        throw std::runtime_error("Remapped phase publication bound is invalid.");
                    const auto mass_difference = item.phase_mass->gas + item.phase_mass->carrier - mass;
                    const auto mass_error = mass_difference.sign() < 0 ? mass_difference.scaled(-1) : mass_difference;
                    // Ownership copies perform no new publication. Their
                    // historical phase metadata cannot admit a mass mismatch.
                    const bool exact_policy =
                        candidate_phase->inventory_policy() == VoF::PhaseInventoryPolicy::ExperimentalExact;
                    const auto mass_bound =
                        migration || exact_policy
                            ? M{}
                            : M::from_value(item.roundoff.gas).scaled(self.material.gas_density) +
                                  M::from_value(item.roundoff.carrier).scaled(self.material.liquid_density);
                    if (mass_error.compare(mass_bound) > 0)
                        throw std::runtime_error("Remapped phase mass exceeds its finite publication bound.");
                    for (unsigned a = 0; a < 3; ++a)
                    {
                        const auto value = item.amounts[a].divided_by(mass, self.options.byte_budget);
                        state->velocity[c][a] = value.sign() ? value.certified().as_double() : 0;
                    }
                    const auto pressure =
                        item.amounts[3].divided_by(M::volume(*target_geometry[c]), self.options.byte_budget);
                    mapped_pressure[c] = pressure.sign() ? pressure.certified().as_double() : 0;
                }
            });
        candidate->surface_curvature(*state);
        // Cell momentum/pV and face q are independent accepted degrees of
        // freedom. A fixed-clock handoff must not apply a physical dt=1 impulse.
        using Domain = detail::GeneralVoFStorageDomain;
        detail::GeneralVoFExactStorage handoff_storage(self.options.byte_budget);
        std::size_t face_budget = 0, certificate_budget = 0, scalar_budget = 0;
        std::vector<VoF::ExactOwnedFaceRate> old_rates;
        std::vector<VoF::ExactOwnedCellVelocity> seeds;
        candidate->collective(
            [&]
            {
                const auto source_faces = self.mesh->num_owned_faces(),
                           target_faces = candidate->mesh->num_owned_faces();
                if (old.projection_arithmetic.face_rate.size() != source_faces)
                    throw std::runtime_error("Missing authoritative source face rates for fixed-clock handoff.");
                handoff_storage.capacity(Domain::Input, old.projection_arithmetic.face_rate.capacity(), sizeof(M));
                for (const auto& q : old.projection_arithmetic.face_rate)
                    handoff_storage.measure(Domain::Input, q);
                handoff_storage.capacity(Domain::Scratch, source_faces, sizeof(VoF::ExactOwnedFaceRate));
                handoff_storage.capacity(
                    Domain::Scratch, migration ? 0 : old.velocity.size(), sizeof(VoF::ExactOwnedCellVelocity));
                if (!migration)
                    for (const auto& key : old.phases->owned_order())
                        handoff_storage.capacity(Domain::Scratch, Meshes::AMR::encode_adaptive_key(key).size(), 1);
                handoff_storage.capacity(
                    Domain::Result, target_faces, sizeof(M) + sizeof(detail::GeneralVoFRoundedValue));
                handoff_storage.capacity(Domain::Scratch, target_faces, sizeof(M) + sizeof(unsigned char));
                handoff_storage.capacity(Domain::Result, candidate->mesh->num_faces(), sizeof(double));
                // Preserve one admitted half for the face-transfer result throughout
                // validation; the other covers caller views and exact closure proof.
                face_budget = handoff_storage.remaining() / 2;
                handoff_storage.capacity(Domain::Scratch, face_budget, 1);
                certificate_budget = handoff_storage.remaining() / 2;
                handoff_storage.capacity(Domain::Scratch, certificate_budget, 1);
                if (target_faces > (std::numeric_limits<std::size_t>::max() - 32) / 4)
                    throw std::runtime_error("Fixed-clock face handoff scalar count overflow.");
                const auto slots = 4 * target_faces + 32;
                scalar_budget = handoff_storage.remaining() / slots;
                if (face_budget < 1024 || certificate_budget < 1024 || scalar_budget < 1024)
                    throw std::runtime_error("Fixed-clock face handoff cumulative reserve exhausted.");
                handoff_storage.capacity(Domain::Scratch, slots, scalar_budget);
                old_rates.reserve(source_faces);
                const auto lease = self.mesh->acquire_execution_view();
                for (std::size_t f = 0; f < source_faces; ++f)
                    old_rates.push_back(
                        {std::uint64_t(self.mesh->face_global_id(LO(f))), old.projection_arithmetic.face_rate[f]});
                if (!migration)
                {
                    seeds.reserve(old.velocity.size());
                    for (std::size_t c = 0; c < old.velocity.size(); ++c)
                        seeds.push_back({old.phases->owned_order()[c], old.velocity[c]});
                }
            });
        VoF::ExactFaceTransferOptions face_options;
        face_options.byte_budget = face_budget;
        const auto transferred_faces =
            VoF::ExactFaceTransfer::prepare(*old.phases, *candidate_phase, transferred, old_rates, seeds, face_options);
        transferred_faces.validate_observations(*old.phases, *candidate_phase);
        std::vector<M> zero_radius;
        candidate->collective(
            [&]
            {
                const auto count = candidate->mesh->num_owned_faces();
                state->projection_arithmetic.byte_budget = self.options.byte_budget;
                state->projection_arithmetic.face_rate.resize(count);
                state->projection_arithmetic.face_velocity.resize(count);
                state->face_velocity.resize(candidate->mesh->num_faces());
                zero_radius.resize(count);
                std::vector<unsigned char> seen(count);
                detail::GeneralVoFProjectionArithmetic arithmetic(scalar_budget);
                if (transferred_faces.owned_rates().size() != count)
                    throw std::runtime_error("Fixed-clock face handoff has incomplete actual target coverage.");
                const auto lease = candidate->mesh->acquire_execution_view();
                for (const auto& item : transferred_faces.owned_rates())
                {
                    const auto f = candidate->mesh->owned_face_map()->getLocalElement(GO(item.face));
                    if (f < 0 || seen[f]++)
                        throw std::runtime_error("Foreign/duplicate target face in fixed-clock handoff.");
                    if (candidate->mesh->is_geometry_exterior_face(f) && item.rate.sign())
                        throw std::runtime_error("Fixed-clock handoff changes a closed physical wall rate.");
                    const auto view = detail::vof_exact::leased_divide(
                        item.rate, M::from_value(candidate->mesh->face_area(f)), scalar_budget)
                                          .enclosure();
                    const double velocity = static_cast<double>(view.value);
                    if (!std::isfinite(velocity))
                        throw std::runtime_error("Fixed-clock face diagnostic overflows.");
                    state->projection_arithmetic.face_rate[f] = item.rate;
                    // This leaf describes the diagnostic view only. Radius zero
                    // belongs to retained EXACT q, not to a rounded q/area product.
                    state->projection_arithmetic.face_velocity[f] = arithmetic.leaf(velocity);
                    state->face_velocity[f] = velocity;
                }
                state->pressure = std::move(mapped_pressure);
            });
        const auto proof = detail::Exact::certify_general_vof_projection_rate(candidate->mesh,
            state->projection_arithmetic.face_velocity, certificate_budget, &state->projection_arithmetic.face_rate,
            &zero_radius);
        candidate->require(
            proof.arithmetic_admitted, "Transferred exact face rates do not close the actual target topology.");
        candidate->collective(
            [&]
            {
                for (const auto& q : proof.owned_net_rate)
                    if (q.sign())
                        throw std::runtime_error("Fixed-clock face handoff has nonzero exact recipient divergence.");
            });
        state->face_velocity = candidate->flow.faces(state->face_velocity);
        state->diagnostics = old.diagnostics;
        state->momentum_transport = old.momentum_transport;
        state->diagnostics.time = old.diagnostics.time;
        state->diagnostics.maximum_divergence = 0; // Proven on actual target faces.
        candidate->budget(*state);
        // The opaque remap proves exact channel columns. Final represented
        // velocity recovery retains its independent physical momentum gate.
        for (unsigned a = 0; a < 3; ++a)
        {
            std::vector<M> before, after, reference;
            self.collective(
                [&]
                {
                    for (std::size_t c = 0; c < old.velocity.size(); ++c)
                    {
                        const auto value = old.physical_mass[c].scaled(old.velocity[c][a]);
                        before.push_back(value);
                        reference.push_back(value.sign() < 0 ? value.scaled(-1) : value);
                    }
                    for (std::size_t c = 0; c < state->velocity.size(); ++c)
                    {
                        const auto value = state->physical_mass[c].scaled(state->velocity[c][a]);
                        after.push_back(value);
                        reference.push_back(value.sign() < 0 ? value.scaled(-1) : value);
                    }
                });
            const auto difference = global_sum(after) - global_sum(before);
            const auto magnitude = difference.sign() < 0 ? difference.scaled(-1) : difference;
            const auto limit = global_sum(reference).scaled(self.options.momentum_relative_tolerance);
            // This is a residual-vs-L1 bound, not a value-vs-reference relative
            // error. An exactly zero L1 reference requires an exact zero residual.
            self.require(magnitude.compare(limit) <= 0,
                "General VoF fixed-clock remap changed physical phase-mass-weighted momentum.");
        }
        transferred_faces.validate_observations(*old.phases, *candidate_phase);
        return {std::move(candidate), std::move(state)};
    }

    double stable_timestep(const State& state, double bound, double speed, double global_thickness) const
    {
        // Supplement rounded face CFL with the authoritative exact rate, including
        // nonzero streams whose individual binary64 product is zero.
        FaceMeasures owned;
        self.collective(
            [&]
            {
                if (state.projection_arithmetic.face_rate.size() != self.mesh->num_owned_faces())
                    throw std::runtime_error("Missing exact rate for timestep admission.");
                for (std::size_t f = 0; f < self.mesh->num_owned_faces(); ++f)
                    owned.emplace(self.mesh->face_global_id(LO(f)),
                        std::pair{state.projection_arithmetic.face_rate[f],
                            SimpleFluid::PLIC::GeometryMeasure<SimpleFluid::ArithmeticMode::Exact>{}});
            });
        const auto rates = exchange_face_measures(state, owned, self.options.byte_budget / 2);
        double local_rate_bound = std::numeric_limits<double>::infinity();
        self.collective(
            [&]
            {
                using M = SimpleFluid::PLIC::GeometryMeasure<SimpleFluid::ArithmeticMode::Exact>;
                const auto geometry = state.phases->owned_geometry();
                const auto slots = geometry.size() + 32;
                const auto scalar_budget = self.options.byte_budget / 2 / slots;
                if (scalar_budget < 1024)
                    throw std::runtime_error("Exact rate CFL cumulative reserve exhausted.");
                std::vector<M> sums(geometry.size());
                const auto lease = self.mesh->acquire_execution_view();
                for (const auto& tile : state.phases->tiles())
                {
                    const auto face = self.mesh->overlap_face_map()->getLocalElement(GO(tile.face_gid));
                    if (face < 0)
                        throw std::runtime_error("Missing local exact CFL face.");
                    const auto rate = rates.at(GO(tile.face_gid)).first;
                    const auto magnitude = rate.sign() < 0 ? rate.scaled(-1) : rate;
                    for (const auto cell : {self.mesh->owner_cell(face), self.mesh->neighbor_cell(face)})
                        if (cell >= 0 && std::size_t(cell) < sums.size())
                            sums[cell] = detail::vof_exact::add(sums[cell], magnitude, scalar_budget);
                }
                for (std::size_t c = 0; c < sums.size(); ++c)
                    if (sums[c].sign())
                    {
                        const auto exact = M::volume(*geometry[c])
                                               .scaled(self.options.flow.max_courant)
                                               .divided_by(sums[c], scalar_budget);
                        if (exact.compare(M::from_value(std::numeric_limits<double>::max())) > 0)
                            continue;
                        double value = exact.certified().as_double();
                        if (M::from_value(value).compare(exact) > 0)
                            value = std::nextafter(value, 0.0);
                        if (!(value > 0))
                            throw std::runtime_error("Exact rate has no positive representable timestep.");
                        local_rate_bound = std::min(local_rate_bound, value);
                    }
            });
        double rate_bound = 0;
        Teuchos::reduceAll(
            *self.mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_MIN, 1, &local_rate_bound, &rate_bound);
        // A trajectory displacement bound supplements net-face CFL: opposing
        // swept streams can exist on a face whose projected net Phi is zero.
        return std::min(
            rate_bound, speed > 0 ? std::min(bound, self.options.flow.max_courant * global_thickness / speed) : bound);
    }

    SimpleFluid::PLIC::GeometryMeasure<SimpleFluid::ArithmeticMode::Exact> phase_mass_measure(
        const SimpleFluid::PLIC::PhaseMeasures<SimpleFluid::ArithmeticMode::Exact>& phase) const
    {
        return phase.gas.scaled(self.material.gas_density) + phase.carrier.scaled(self.material.liquid_density);
    }

    std::vector<SimpleFluid::PLIC::PhaseMeasures<SimpleFluid::ArithmeticMode::Exact>> current_inventory(
        const VoF::GeneralVoF& phases) const
    {
        // The immutable FVM source selects the policy. Finite is the default;
        // experimental exact authority is explicit and remains unqualified.
        return phases.inventory_measures();
    }

    void verify_phase_publication(const VoF::GeneralVoF::Trial& trial, const VoF::GeneralVoF& candidate) const
    {
        self.collective(
            [&]
            {
                using M = SimpleFluid::PLIC::GeometryMeasure<SimpleFluid::ArithmeticMode::Exact>;
                using Domain = detail::GeneralVoFStorageDomain;
                detail::GeneralVoFExactStorage storage(self.options.byte_budget);
                // Only finite centers/bounds enter these small interval sums.
                // Keep their pending values and one operation disjoint from
                // the variable-size recipient aggregation below.
                storage.capacity(Domain::Scratch, 16 * 1024, 1);
                const auto n = self.volume.size();
                storage.capacity(Domain::Scratch, n, 4 * sizeof(M));
                M zero;
                storage.measure(Domain::Input, zero);
                for (const auto& route : trial.transfers())
                {
                    storage.measure(Domain::Input, route.exact.gas);
                    storage.measure(Domain::Input, route.exact.carrier);
                }
                std::vector<M> gas(n, zero), carrier(n, zero);
                if (gas.capacity() > 2 * n || carrier.capacity() > 2 * n)
                    throw std::runtime_error("General VoF publication vector capacity exceeds admission.");
                for (std::size_t c = 0; c < n; ++c)
                {
                    storage.measure(Domain::Scratch, gas[c]);
                    storage.measure(Domain::Scratch, carrier[c]);
                }
                const auto add = [&](M& value, const M& part)
                {
                    const auto remaining = storage.remaining();
                    if (remaining <= 2048)
                        throw std::runtime_error("General VoF finite publication proof exhausted its operation lease.");
                    auto sum = detail::vof_exact::add(value, part, remaining - 1024);
                    storage.measure(Domain::Scratch, sum);
                    storage.release(Domain::Scratch, value);
                    value = std::move(sum);
                };
                for (const auto& route : trial.transfers())
                {
                    const auto local = self.mesh->owned_cell_map()->getLocalElement(GO(route.recipient));
                    if (local < 0)
                        throw std::runtime_error("General VoF publication proof has a foreign recipient.");
                    add(gas[local], route.exact.gas);
                    add(carrier[local], route.exact.carrier);
                }
                if (candidate.inventory_policy() != self.accepted->phases->inventory_policy())
                    throw std::runtime_error("General VoF trial changed inventory authority policy.");
                if (candidate.inventory_policy() == VoF::PhaseInventoryPolicy::ExperimentalExact)
                {
                    const auto& exact = candidate.experimental_phases();
                    if (exact.size() != n)
                        throw std::runtime_error("Experimental exact phase publication shape differs.");
                    for (std::size_t c = 0; c < n; ++c)
                        if (gas[c].compare(exact[c].gas) || carrier[c].compare(exact[c].carrier))
                            throw std::runtime_error("Experimental exact phase publication differs from its routes.");
                    return;
                }
                const auto& phases = candidate.phase_volumes();
                // Identity trials retain the accepted object and its historical
                // publication metadata; only THIS trial's bounds may admit error.
                const auto& bounds = trial.phase_roundoff_bounds();
                if (phases.size() != n || bounds.size() != n)
                    throw std::runtime_error("General VoF finite publication bound shape differs.");
                const auto check = [&](double value, double bound, const M& expected)
                {
                    if (!std::isfinite(value) || !std::isfinite(bound) || bound < 0)
                        throw std::runtime_error("General VoF finite publication bound is invalid.");
                    const auto center = M::from_value(value);
                    const auto lower = detail::vof_exact::add(center, M::from_value(-bound), 8192);
                    const auto upper = detail::vof_exact::add(center, M::from_value(bound), 8192);
                    if (expected.compare(lower) < 0 || expected.compare(upper) > 0)
                        throw std::runtime_error("General VoF phase publication exceeds its own conversion bound.");
                };
                for (std::size_t c = 0; c < n; ++c)
                {
                    check(phases[c].gas, bounds[c].gas, gas[c]);
                    check(phases[c].carrier, bounds[c].carrier, carrier[c]);
                }
                // Positive densities give |Mnew-Mrouted| <= rhoG*boundG+
                // rhoC*boundC. This is arithmetic provenance only: momentum
                // still solves its original equation with represented Mnew.
            });
    }

    FaceMeasures exchange_face_measures(
        const State& state, const FaceMeasures& owned, std::size_t resource_budget) const
    {
        using Domain = detail::GeneralVoFStorageDomain;
        detail::GeneralVoFExactStorage storage(resource_budget);
        std::size_t scalar_budget = 0, packet_budget = 0;
        self.collective(
            [&]
            {
                storage.capacity(Domain::Input, owned.size(), sizeof(FaceMeasures::value_type) + 4 * sizeof(void*));
                for (const auto& [gid, value] : owned)
                {
                    storage.measure(Domain::Input, value.first);
                    storage.measure(Domain::Input, value.second);
                }
                const auto count = state.phases->tiles().size();
                storage.capacity(Domain::Scratch, count, 3 * sizeof(GO) + 2 * sizeof(int) + 12 * sizeof(void*));
                storage.capacity(Domain::Result, count, sizeof(FaceMeasures::value_type) + 4 * sizeof(void*));
                const auto pool = storage.remaining() / 2;
                storage.capacity(Domain::Scratch, pool, 1);
                packet_budget = pool / 8;
                if (count > (std::numeric_limits<std::size_t>::max() - 32) / 2)
                    throw std::runtime_error("Phi publication scalar count overflow.");
                scalar_budget = storage.remaining() / (2 * count + 32);
                if (scalar_budget < 1024 || packet_budget < 1024)
                    throw std::runtime_error("Phi publication cumulative reserve exhausted.");
                storage.capacity(Domain::Result, 2 * count, scalar_budget);
                storage.capacity(Domain::Scratch, 32, scalar_budget);
            });
        std::set<GO> required;
        for (const auto& tile : state.phases->tiles())
            required.insert(GO(tile.face_gid));
        Teuchos::Array<GO> ids(required.begin(), required.end());
        Teuchos::Array<int> owners(ids.size(), -1);
        (void) self.mesh->owned_face_map()->getRemoteIndexList(ids(), owners());
        std::map<GO, int> authority;
        std::vector<OwnerPacket> packets;
        self.collective(
            [&]
            {
                for (std::size_t i = 0; i < std::size_t(ids.size()); ++i)
                {
                    if (owners[i] < 0)
                        throw std::runtime_error("Missing canonical Phi authority.");
                    authority.emplace(ids[i], owners[i]);
                    OwnerPacket p;
                    p.peer = owners[i];
                    append_word(p.bytes, std::uint64_t(ids[i]));
                    packets.push_back(std::move(p));
                }
            });
        const auto requests = exchange_owner_packets(packets, self.mesh->owned_cell_map()->getComm(), packet_budget);
        packets.clear();
        self.collective(
            [&]
            {
                std::size_t used = 0;
                for (const auto& p : requests.packets)
                {
                    if (p.bytes.size() != 8)
                        throw std::runtime_error("Bad Phi certificate request.");
                    const GO face = GO(read_word(p.bytes, 0));
                    const auto& values = owned.at(face);
                    if (self.mesh->owned_face_map()->getLocalElement(face) < 0)
                        throw std::runtime_error("Foreign Phi certificate request.");
                    OwnerPacket answer;
                    answer.peer = p.peer;
                    append_word(answer.bytes, std::uint64_t(face));
                    for (const auto* value : {&values.first, &values.second})
                    {
                        const auto encoded = value->encode(scalar_budget);
                        if (answer.bytes.size() > packet_budget || packet_budget - answer.bytes.size() < 8 ||
                            encoded.size() > packet_budget - answer.bytes.size() - 8)
                            throw std::runtime_error("Phi certificate exceeds budget.");
                        append_word(answer.bytes, encoded.size());
                        answer.bytes.insert(answer.bytes.end(), encoded.begin(), encoded.end());
                    }
                    if (answer.bytes.size() > packet_budget || used > packet_budget - answer.bytes.size() ||
                        sizeof(OwnerPacket) > packet_budget - used - answer.bytes.size())
                        throw std::runtime_error("Phi certificate packets exceed budget.");
                    used += sizeof(OwnerPacket) + answer.bytes.size();
                    packets.push_back(std::move(answer));
                }
            });
        const auto answers = exchange_owner_packets(packets, self.mesh->owned_cell_map()->getComm(), packet_budget);
        FaceMeasures result;
        self.collective(
            [&]
            {
                for (const auto& p : answers.packets)
                {
                    if (p.bytes.size() < 24)
                        throw std::runtime_error("Truncated Phi certificate.");
                    const GO face = GO(read_word(p.bytes, 0));
                    if (!authority.contains(face) || authority.at(face) != p.peer)
                        throw std::runtime_error("Wrong Phi certificate authority.");
                    std::size_t offset = 8;
                    std::array<SimpleFluid::PLIC::GeometryMeasure<SimpleFluid::ArithmeticMode::Exact>, 2> value;
                    for (auto& item : value)
                    {
                        if (offset > p.bytes.size() || p.bytes.size() - offset < 8)
                            throw std::runtime_error("Truncated Phi certificate length.");
                        const auto count = read_word(p.bytes, offset);
                        offset += 8;
                        if (count > p.bytes.size() - offset)
                            throw std::runtime_error("Invalid Phi certificate length.");
                        item = SimpleFluid::PLIC::GeometryMeasure<SimpleFluid::ArithmeticMode::Exact>::decode(
                            std::span<const char>(p.bytes.data() + offset, std::size_t(count)), scalar_budget);
                        offset += count;
                    }
                    if (offset != p.bytes.size() || !result.emplace(face, std::pair{value[0], value[1]}).second)
                        throw std::runtime_error("Duplicate Phi certificate.");
                }
                if (result.size() != required.size())
                    throw std::runtime_error("Incomplete Phi certificate publication.");
            });
        return result;
    }

    std::vector<VoF::GeneralVoFFaceVolume> closed_face_volumes(const State& state, double dt) const
    {
        using Domain = detail::GeneralVoFStorageDomain;
        detail::GeneralVoFExactStorage storage(self.options.byte_budget);
        std::size_t scalar_budget = 0, publication_budget = 0;
        self.collective(
            [&]
            {
                const auto nf = self.mesh->num_owned_faces(), nt = state.phases->tiles().size();
                const auto& rate = state.projection_arithmetic.face_rate;
                if (rate.size() != nf)
                    throw std::runtime_error("Missing accepted divergence-free face-rate ledger.");
                storage.capacity(Domain::Input, rate.capacity(),
                    sizeof(SimpleFluid::PLIC::GeometryMeasure<SimpleFluid::ArithmeticMode::Exact>));
                for (const auto& value : rate)
                    storage.measure(Domain::Input, value);
                storage.capacity(Domain::Scratch, nf, sizeof(FaceMeasures::value_type) + 4 * sizeof(void*));
                storage.capacity(Domain::Result, nt, sizeof(VoF::GeneralVoFFaceVolume));
                const auto pool = storage.remaining() / 2;
                storage.capacity(Domain::Scratch, pool, 1);
                publication_budget = pool;
                const auto slots = nf + 2 * nt + 16;
                if (slots < nf)
                    throw std::runtime_error("Integrated Phi scalar capacity overflow.");
                scalar_budget = storage.remaining() / slots;
                if (scalar_budget < 1024 || publication_budget < 1024)
                    throw std::runtime_error("Integrated Phi cumulative reserve exhausted.");
                storage.capacity(Domain::Scratch, nf + 16, scalar_budget);
                storage.capacity(Domain::Result, 2 * nt, scalar_budget);
            });
        FaceMeasures owned;
        self.collective(
            [&]
            {
                for (std::size_t f = 0; f < self.mesh->num_owned_faces(); ++f)
                    owned.emplace(self.mesh->face_global_id(LO(f)),
                        std::pair{detail::vof_exact::scale(state.projection_arithmetic.face_rate[f], dt, scalar_budget),
                            SimpleFluid::PLIC::GeometryMeasure<SimpleFluid::ArithmeticMode::Exact>{}});
            });
        const auto controls = exchange_face_measures(state, owned, publication_budget);
        std::vector<VoF::GeneralVoFFaceVolume> result;
        self.collective(
            [&]
            {
                result.reserve(state.phases->tiles().size());
                for (const auto& tile : state.phases->tiles())
                {
                    const auto& exact = controls.at(GO(tile.face_gid)).first;
                    const double diagnostic = static_cast<double>(exact.enclosure().value);
                    if (!std::isfinite(diagnostic))
                        throw std::runtime_error("Integrated Phi diagnostic overflows.");
                    result.push_back({tile.face_gid, diagnostic, exact});
                }
            });
        return result;
    }

    SimpleFluid::PLIC::GeometryMeasure<SimpleFluid::ArithmeticMode::Exact> global_sum(
        const std::vector<SimpleFluid::PLIC::GeometryMeasure<SimpleFluid::ArithmeticMode::Exact>>& values) const
    {
        SimpleFluid::PLIC::GeometryMeasure<SimpleFluid::ArithmeticMode::Exact> local;
        std::vector<OwnerPacket> packets;
        self.collective(
            [&]
            {
                for (const auto& value : values)
                    local = detail::vof_exact::add(local, value, self.options.byte_budget);
                packets.push_back({0, local.encode(self.options.byte_budget)});
            });
        const auto received =
            exchange_owner_packets(packets, self.mesh->owned_cell_map()->getComm(), self.options.byte_budget);
        SimpleFluid::PLIC::GeometryMeasure<SimpleFluid::ArithmeticMode::Exact> sum;
        self.collective(
            [&]
            {
                if (self.mesh->owned_cell_map()->getComm()->getRank() == 0)
                {
                    std::set<int> senders;
                    for (const auto& p : received.packets)
                    {
                        if (!senders.insert(p.peer).second)
                            throw std::runtime_error("Duplicate exact global inventory sender.");
                        sum = detail::vof_exact::add(sum,
                            SimpleFluid::PLIC::GeometryMeasure<SimpleFluid::ArithmeticMode::Exact>::decode(
                                p.bytes, self.options.byte_budget),
                            self.options.byte_budget);
                    }
                    if (senders.size() != std::size_t(self.mesh->owned_cell_map()->getComm()->getSize()))
                        throw std::runtime_error("Incomplete exact global inventory reduction.");
                }
            });
        std::vector<char> encoded;
        self.collective(
            [&]
            {
                if (self.mesh->owned_cell_map()->getComm()->getRank() == 0)
                    encoded = sum.encode(self.options.byte_budget);
            });
        std::uint64_t count = encoded.size();
        Teuchos::broadcast(*self.mesh->owned_cell_map()->getComm(), 0, 1, &count);
        self.collective(
            [&]
            {
                if (count > self.options.byte_budget || count > std::uint64_t(std::numeric_limits<int>::max()))
                    throw std::runtime_error("Exact global inventory broadcast exceeds budget.");
                encoded.resize(std::size_t(count));
            });
        Teuchos::broadcast(*self.mesh->owned_cell_map()->getComm(), 0, int(count), encoded.data());
        self.collective(
            [&]
            {
                sum = SimpleFluid::PLIC::GeometryMeasure<SimpleFluid::ArithmeticMode::Exact>::decode(
                    encoded, self.options.byte_budget);
            });
        return sum;
    }
};
} // namespace SimpleFluid::Exact
