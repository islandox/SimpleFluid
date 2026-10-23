/** @file GeneralVoFFiniteSolver.inc
 * @brief Finite numerical phase, mass, momentum and projection composition.
 * Included only inside GeneralVoFSolver::Impl.
 */
    void phase_storage_finite(State& state) const
    {
        namespace fp = VoF::finite_detail;
        const auto& phases = state.phases->phase_volumes();
        const auto geometry = state.phases->owned_finite_geometry();
        const auto fraction = state.phases->geometric_gas_fractions();
        collective([&]
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
                const auto mass = fp::add(fp::multiply({phases[c].gas, 0}, {material.gas_density, 0}),
                    fp::multiply({phases[c].carrier, 0}, {material.liquid_density, 0}));
                const auto density = fp::divide(mass, geometry[c]->volume());
                state.finite_mass[c] = mass.value;
                state.finite_mass_error[c] = mass.absolute_error;
                state.density[c] = density.as_double();
                if (!(mass.value > mass.absolute_error) || !std::isfinite(state.density[c]) ||
                    !(state.density[c] > 0) || state.fraction[c] < 0 || state.fraction[c] > 1)
                    throw std::runtime_error("Finite VoF physical mixture storage is unresolved.");
            }
        });
    }
    std::vector<Meshes::BoundedGeometryValue> finite_rates(const State& state) const
    {
        require(state.finite_projection.face_rate.size() == mesh->num_owned_faces() &&
            state.finite_projection.face_radius.size() == mesh->num_owned_faces(),
            "Finite VoF projection rate authority is incomplete.");
        std::array<std::vector<double>, 6> wire;
        collective([&]
        {
            for (auto& v : wire) v.resize(mesh->num_faces());
            for (std::size_t f = 0; f < mesh->num_owned_faces(); ++f)
                for (unsigned k = 0; k < 2; ++k)
                {
                    const auto value = k == 0 ? state.finite_projection.face_rate[f] : state.finite_projection.face_radius[f];
                    int exponent = 0;
                    const auto mantissa = std::frexp(value, &exponent);
                    const auto high = static_cast<double>(mantissa), low = static_cast<double>(mantissa - high);
                    if (!std::isfinite(value) || (k && value < 0) || static_cast<long double>(high) + low != mantissa)
                        throw std::runtime_error("Finite VoF rate exceeds fixed wire precision.");
                    wire[3 * k][f] = high;
                    wire[3 * k + 1][f] = low;
                    wire[3 * k + 2][f] = exponent;
                }
        });
        for (auto& value : wire) value = flow.faces(value);
        std::vector<Meshes::BoundedGeometryValue> result(mesh->num_faces());
        collective([&]
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
                    values[k] = std::scalbn(static_cast<long double>(wire[3 * k][f]) + wire[3 * k + 1][f], int(exponent));
                    if (!std::isfinite(values[k]) || (k && values[k] < 0))
                        throw std::runtime_error("Finite VoF canonical wire rate is nonfinite.");
                }
                result[f] = {values[0], values[1]};
            }
        });
        return result;
    }
    double stable_timestep_finite(const State& state, double bound, double speed, double global_thickness) const
    {
        namespace fp = VoF::finite_detail;
        const auto rates = finite_rates(state);
        const auto geometry = state.phases->owned_finite_geometry();
        double local_rate = std::numeric_limits<double>::infinity();
        collective([&]
        {
            const auto lease = mesh->acquire_execution_view();
            for (std::size_t c = 0; c < volume.size(); ++c)
            {
                fp::Sum sum;
                for (const auto face : mesh->faces(LO(c)))
                    sum.add({std::abs(rates[face].value), rates[face].absolute_error});
                const auto outgoing = sum.value();
                if (outgoing.value + outgoing.absolute_error > 0)
                {
                    const auto lower_volume = geometry[c]->volume().value - geometry[c]->volume().absolute_error;
                    const auto time = lower_volume * options.flow.max_courant / (outgoing.value + outgoing.absolute_error);
                    double value = static_cast<double>(time);
                    if (static_cast<long double>(value) > time) value = std::nextafter(value, 0.);
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
    std::vector<VoF::GeneralVoFFaceVolume> finite_face_volumes(const State& state, double dt) const
    {
        const auto rates = finite_rates(state);
        std::vector<VoF::GeneralVoFFaceVolume> result;
        collective([&]
        {
            for (const auto& tile : state.phases->tiles())
            {
                const auto local = mesh->overlap_face_map()->getLocalElement(GO(tile.face_gid));
                if (local < 0) throw std::runtime_error("Finite VoF missing canonical face rate.");
                const auto integrated = VoF::finite_detail::multiply(rates[local], {dt, 0});
                const auto diagnostic = static_cast<double>(integrated.value);
                if (!std::isfinite(diagnostic)) throw std::runtime_error("Finite VoF face diagnostic overflows.");
                result.push_back({tile.face_gid, diagnostic, std::nullopt, integrated});
            }
        });
        return result;
    }
    void verify_phase_publication_finite(const VoF::GeneralVoF::Trial& trial, const VoF::GeneralVoF& candidate) const
    {
        namespace fp = VoF::finite_detail;
        collective([&]
        {
            if (candidate.inventory_policy() != accepted->phases->inventory_policy())
                throw std::runtime_error("Finite VoF candidate changed inventory policy.");
            if (candidate.accepted_identity() == accepted->phases->accepted_identity())
                return;
            std::vector<fp::Sum> gas(volume.size()), carrier(volume.size());
            for (const auto& route : trial.finite_transfers())
            {
                const auto row = mesh->owned_cell_map()->getLocalElement(GO(route.recipient));
                if (row < 0) throw std::runtime_error("Finite phase publication has foreign recipient.");
                gas[row].add(route.amount.gas);
                carrier[row].add(route.amount.carrier);
            }
            const auto& phases = candidate.phase_volumes();
            const auto& bounds = trial.phase_roundoff_bounds();
            if (phases.size() != volume.size() || bounds.size() != volume.size())
                throw std::runtime_error("Finite phase publication has incompatible shape.");
            for (std::size_t c = 0; c < volume.size(); ++c)
                for (const auto& item : {std::tuple{phases[c].gas, bounds[c].gas, gas[c].value()},
                         std::tuple{phases[c].carrier, bounds[c].carrier, carrier[c].value()}})
                {
                    const auto expected = std::get<2>(item);
                    const auto difference = std::abs(static_cast<long double>(std::get<0>(item)) - expected.value);
                    if (!std::isfinite(std::get<1>(item)) || std::get<1>(item) < 0 ||
                        difference + expected.absolute_error > std::get<1>(item))
                        throw std::runtime_error("Finite phase publication exceeds its own bounded ledger.");
                }
        });
    }
    void budget_finite(State& state) const
    {
        namespace fp = VoF::finite_detail;
        std::array<fp::Sum, 7> sums;
        collective([&]
        {
            const auto& phases = state.phases->phase_volumes();
            for (std::size_t c = 0; c < volume.size(); ++c)
            {
                sums[0].add({phases[c].gas, 0});
                sums[1].add(fp::multiply({phases[c].gas, 0}, {material.gas_density, 0}));
                sums[2].add(fp::multiply({phases[c].carrier, 0}, {material.liquid_density, 0}));
                for (unsigned a = 0; a < 3; ++a)
                    sums[3 + a].add(fp::multiply({state.finite_mass[c], 0}, {state.velocity[c][a], 0}));
                const auto square = static_cast<long double>(state.velocity[c][0]) * state.velocity[c][0] +
                    static_cast<long double>(state.velocity[c][1]) * state.velocity[c][1] +
                    static_cast<long double>(state.velocity[c][2]) * state.velocity[c][2];
                sums[6].add(fp::multiply({state.finite_mass[c], 0}, {square / 2, 0}));
            }
        });
        std::array<double, 7> global{};
        for (unsigned a = 0; a < 7; ++a)
        {
            const auto local = sums[a].value();
            int local_exponent = std::numeric_limits<int>::min(), exponent = 0;
            if (local.value != 0) (void) std::frexp(std::abs(local.value), &local_exponent);
            Teuchos::reduceAll(*mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_MAX, 1, &local_exponent, &exponent);
            const auto scale = exponent == std::numeric_limits<int>::min() ? 1.L : std::scalbn(1.L, exponent - 1);
            const auto normalized = static_cast<double>(local.value / scale);
            double sum = 0;
            Teuchos::reduceAll(*mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_SUM, 1, &normalized, &sum);
            collective([&]
            {
                const auto value = static_cast<long double>(sum) * scale;
                global[a] = static_cast<double>(value);
                if (!std::isfinite(global[a]) || (value != 0 && global[a] == 0))
                    throw std::runtime_error("Finite VoF global physical diagnostic is outside representable range.");
            });
        }
        state.diagnostics.gas_volume = global[0];
        state.diagnostics.gas_mass = global[1];
        state.diagnostics.liquid_mass = global[2];
        for (unsigned a = 0; a < 3; ++a) state.diagnostics.momentum[a] = global[3 + a];
        state.diagnostics.kinetic_energy = global[6];
        double speed = 0;
        for (const auto& velocity : state.velocity) speed = std::max(speed, std::sqrt(dot(velocity, velocity)));
        state.diagnostics.maximum_velocity = maximum(speed);
    }
    VoFStepDiagnostics step_finite(double dt)
    {
        namespace fp = VoF::finite_detail;
        require(std::isfinite(dt) && dt > 0, "Finite VoF timestep must be positive and finite.");
        double minimum = 0, maximum_dt = 0;
        Teuchos::reduceAll(*mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_MIN, 1, &dt, &minimum);
        Teuchos::reduceAll(*mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_MAX, 1, &dt, &maximum_dt);
        require(minimum == maximum_dt, "Finite VoF timestep differs across ranks.");
        const auto& old = *accepted;
        const auto nodes = node_velocity(old);
        require(dt <= stable_timestep(old, nodes), "Finite VoF timestep exceeds physical geometry/trajectory CFL.");
        const auto trial = old.phases->plan_transport(nodes, finite_face_volumes(old, dt), dt);
        double local_h = std::numeric_limits<double>::infinity(), global_h = 0;
        for (auto h : thickness) local_h = std::min(local_h, h);
        Teuchos::reduceAll(*mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_MIN, 1, &local_h, &global_h);
        collective([&]
        {
            if (trial.finite_node_trajectories().size() != nodes.size())
                throw std::runtime_error("Finite VoF trajectory certificate is incomplete.");
            const auto limit = fp::multiply({global_h, 0}, {options.flow.max_courant, 0});
            const auto square_limit = fp::multiply(limit, limit);
            for (const auto& node : trial.finite_node_trajectories())
            {
                const auto displacement = fp::subtract(node.point, node.departure);
                const auto square = fp::dot(displacement, displacement);
                if (square.value + square.absolute_error > square_limit.value - square_limit.absolute_error)
                    throw std::runtime_error("Finite VoF bounded trajectory exceeds physical displacement CFL.");
            }
        });
        auto next = std::make_shared<State>();
        next->phases = trial.candidate();
        require(next->phases->mesh().get() == mesh.get(), "Finite VoF transport changed its immutable native mesh.");
        verify_phase_publication_finite(trial, *next->phases);
        phase_storage_finite(*next);
        std::vector<long double> outgoing(volume.size()), outgoing_error(volume.size()), source(volume.size());
        std::vector<detail::ConservativeVoFTransport::FiniteTransfer> incoming;
        std::vector<detail::ConservativeVoFTransport::FiniteFaceContribution> face_contributions;
        const auto mass = [&](const Meshes::BoundedPhaseMeasures& p)
        {
            return fp::add(fp::multiply(p.gas, {material.gas_density, 0}),
                fp::multiply(p.carrier, {material.liquid_density, 0}));
        };
        collective([&]
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
            if (seen.size() != volume.size()) throw std::runtime_error("Finite VoF donor budgets are incomplete.");
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
                const auto value = fp::multiply({old.finite_mass[c], old.finite_mass_error[c]}, {old.velocity[c][a], 0});
                momentum[c] = value.value;
                momentum_error[c] = value.absolute_error;
            }
            const auto carried = transport.solve_finite(momentum, source, next->finite_mass, outgoing, incoming,
                face_contributions, false, options.maximum_momentum_iterations,
                options.momentum_relative_tolerance, options.byte_budget, momentum_error, {}, next->finite_mass_error,
                outgoing_error);
            next->momentum_transport[a] = {carried.physical_relative_residual, carried.iterations, carried.physical_refinements};
            for (std::size_t c = 0; c < volume.size(); ++c) next->velocity[c][a] = carried.cell_trace[c];
        }
        std::vector<double> viscosity(volume.size());
        collective([&]
        {
            const auto& phases = old.phases->phase_volumes();
            const auto geometry = old.phases->owned_finite_geometry();
            for (std::size_t c = 0; c < volume.size(); ++c)
            {
                const auto weighted = fp::add(fp::multiply({phases[c].gas, 0}, {material.gas_viscosity, 0}),
                    fp::multiply({phases[c].carrier, 0}, {material.liquid_viscosity, 0}));
                viscosity[c] = fp::divide(weighted, geometry[c]->volume()).as_double();
            }
        });
        const auto viscous = flow.viscous_force(old.velocity, viscosity);
        std::vector<Vector> change(volume.size());
        collective([&]
        {
            for (std::size_t c = 0; c < volume.size(); ++c)
                for (unsigned a = 0; a < 3; ++a)
                {
                    const auto force = fp::divide(fp::multiply({viscous[c][a], 0}, {dt, 0}), {next->finite_mass[c], next->finite_mass_error[c]});
                    const auto updated = fp::add(fp::add({next->velocity[c][a], 0},
                        fp::multiply({options.flow.gravity[a], 0}, {dt, 0})), force);
                    next->velocity[c][a] = updated.as_double();
                    change[c][a] = next->velocity[c][a] - old.velocity[c][a];
                }
        });
        next->face_velocity = old.face_velocity;
        const auto face_change = flow.interpolate(change);
        for (std::size_t f = 0; f < next->face_velocity.size(); ++f) next->face_velocity[f] += face_change[f];
        surface_curvature(*next);
        next->finite_projection.byte_budget = options.byte_budget;
        const auto closure_topology = projection_topology<detail::GeneralVoFFiniteProjectionClosureTopology>(*next->phases);
        flow.project_finite(next->face_velocity, next->velocity, next->pressure, next->density, next->fraction,
            next->curvature, dt, material.surface_tension, options.flow, next->diagnostics, &next->finite_projection,
            std::span<const long double>(next->finite_mass), &closure_topology,
            std::span<const long double>(next->finite_mass_error));
        next->diagnostics.time = old.diagnostics.time + dt;
        require(std::isfinite(next->diagnostics.time) && next->diagnostics.time > old.diagnostics.time,
            "Finite VoF timestep did not advance the accepted clock.");
        budget_finite(*next);
        next->phases = old.phases->apply(trial);
        accepted = std::move(next);
        return accepted->diagnostics;
    }
