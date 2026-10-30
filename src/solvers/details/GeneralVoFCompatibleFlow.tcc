/** @file GeneralVoFCompatibleFlow.tcc
 * @brief Ordinary coupled projection in the physical cell-mass metric.
 */
void GeneralVoFFlowWorkspace::project_compatible(std::vector<double>& face_velocity, std::vector<Vector>& velocity,
    std::vector<double>& pressure, const std::vector<double>& density, const std::vector<double>& fraction,
    const std::vector<double>& kappa, double dt, double sigma, const VoFSolverOptions& options,
    VoFStepDiagnostics& diagnostics, GeneralVoFFiniteProjectionProvenance* provenance,
    std::optional<std::span<const long double>> physical_mass,
    const GeneralVoFFiniteProjectionClosureTopology* closure_topology,
    std::optional<std::span<const long double>> physical_mass_error)
{
    constexpr auto Mode = ArithmeticMode::Ordinary;
    const auto comm = mesh_->owned_cell_map()->getComm();
    const auto n = volumes_.size(), nf = mesh_->num_owned_faces();
    const auto budget = provenance ? provenance->byte_budget : 64 * 1024 * 1024;
    constexpr bool bounded = Mode == ArithmeticMode::BoundedCertification;
    const std::optional<GeneralVoFFiniteProjectionProvenance::TransportReadiness> no_readiness;
    const auto& readiness = provenance ? provenance->transport_readiness : no_readiness;
    bool valid = arithmetic_mode_ != ArithmeticMode::Exact && (!bounded || PLIC::certification_environment()) &&
                 (!provenance || !provenance->arithmetic_mode || *provenance->arithmetic_mode == arithmetic_mode_) &&
                 current_geometry() && velocity.size() == n && density.size() == n && fraction.size() == n &&
                 kappa.size() == n && face_velocity.size() == geometry_.size() &&
                 (!physical_mass || physical_mass->size() == n) &&
                 (!physical_mass_error || (physical_mass && physical_mass_error->size() == n)) && std::isfinite(dt) &&
                 dt > 0 && std::isfinite(sigma) && sigma >= 0 && std::numeric_limits<long double>::is_iec559 &&
                 std::numeric_limits<long double>::digits >= std::numeric_limits<double>::digits &&
                 std::numeric_limits<long double>::digits <= 106 && (!bounded || std::fegetround() == FE_TONEAREST);
    valid = valid && std::isfinite(options.pressure_absolute_tolerance) && options.pressure_absolute_tolerance >= 0 &&
            std::isfinite(options.pressure_relative_tolerance) && options.pressure_relative_tolerance >= 0 &&
            std::isfinite(options.divergence_tolerance) && options.divergence_tolerance > 0 &&
            options.max_pressure_iterations > 0;
    valid = valid &&
            (!readiness || (!bounded && std::isfinite(readiness->volume_tolerance) && readiness->volume_tolerance > 0 &&
                               std::isfinite(readiness->max_courant) && readiness->max_courant > 0 &&
                               (!readiness->cumulative ||
                                   (readiness->signed_discrepancy.size() == n && readiness->volume.size() == n))));
    for (std::size_t c = 0; c < n; ++c)
        valid = valid && std::isfinite(density[c]) && density[c] > 0 && std::isfinite(fraction[c]) &&
                fraction[c] >= 0 && fraction[c] <= 1 && std::isfinite(kappa[c]) &&
                (!physical_mass || (std::isfinite((*physical_mass)[c]) && (*physical_mass)[c] > 0)) &&
                (!physical_mass_error || (std::isfinite((*physical_mass_error)[c]) && (*physical_mass_error)[c] >= 0 &&
                                             (!bounded || (*physical_mass)[c] > (*physical_mass_error)[c])));
    for (const auto& u : velocity)
        for (double value : u)
            valid = valid && std::isfinite(value);
    if (readiness && readiness->cumulative && valid)
        for (std::size_t c = 0; c < n; ++c)
            valid = valid && std::isfinite(readiness->signed_discrepancy[c]) && std::isfinite(readiness->volume[c]) &&
                    readiness->volume[c] > 0 &&
                    std::abs(readiness->signed_discrepancy[c]) <=
                        VoF::algebraic_physical_occupancy_relative_limit * readiness->volume[c];
    for (std::size_t f = 0; f < nf; ++f)
        valid = valid && std::isfinite(face_velocity[f]) && (!geometry_[f].boundary || face_velocity[f] == 0);
    // Fixed-width vector/halo/gradient/force buffers occupy one half; the
    // authenticated sparse closure uses the other half. Backend allocations
    // remain outside this application-payload accounting, as in the exact API.
    const auto cells_count = mesh_->num_cells(), faces_count = mesh_->num_faces();
    const bool missing_direction =
        std::any_of(absent_gradient_axis_.begin(), absent_gradient_axis_.end(), [](bool absent) { return absent; });
    const auto payload_units =
        256 * (static_cast<long double>(cells_count) + faces_count + n + nf + 1) + boundary_gradient_storage_bytes() +
        static_cast<long double>(nf) * sizeof(long double) +
        2 * static_cast<long double>(n) * sizeof(std::array<PLIC::DynamicGeometryValue, 3>) +
        (provenance && missing_direction ? sizeof(long double) * static_cast<long double>(nf) : 0) +
        (readiness && readiness->cumulative
                ? sizeof(long double) * static_cast<long double>(
                                            readiness->signed_discrepancy.capacity() + readiness->volume.capacity()) +
                      2 * static_cast<long double>(n) * sizeof(long double) +
                      6 * static_cast<long double>(comm->getSize()) * sizeof(double)
                : 0);
    valid =
        valid && budget > 0 && payload_units <= budget / 2 && comm->getSize() <= std::numeric_limits<int>::max() / 6;
    require(valid,
        "Finite General VoF projection has invalid inputs, unsupported FP environment or exhausted payload cap.");
    const double pressure_controls[] = {dt, sigma, options.pressure_absolute_tolerance,
        options.pressure_relative_tolerance, options.divergence_tolerance, double(options.max_pressure_iterations)};
    double minimum_pressure_controls[6]{}, maximum_pressure_controls[6]{};
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MIN, 6, pressure_controls, minimum_pressure_controls);
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 6, pressure_controls, maximum_pressure_controls);
    require(std::equal(std::begin(minimum_pressure_controls), std::end(minimum_pressure_controls),
                std::begin(maximum_pressure_controls)),
        "Compatible projection physical controls differ across ranks.");
    int local_requests[4] = {bool(physical_mass), bool(closure_topology), bool(physical_mass_error), bool(readiness)},
        minimum_requests[4]{}, maximum_requests[4]{};
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MIN, 4, local_requests, minimum_requests);
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 4, local_requests, maximum_requests);
    require(minimum_requests[0] == maximum_requests[0] && minimum_requests[1] == maximum_requests[1] &&
                minimum_requests[2] == maximum_requests[2] && minimum_requests[3] == maximum_requests[3],
        "Finite projection requests differ across ranks.");
    long double readiness_limit = 0;
    if (readiness)
    {
        double controls[3] = {readiness->volume_tolerance, readiness->max_courant, double(readiness->cumulative)},
               minimum[3]{}, maximum[3]{};
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_MIN, 3, controls, minimum);
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 3, controls, maximum);
        require(minimum[0] == maximum[0] && minimum[1] == maximum[1] && minimum[2] == maximum[2],
            "Finite projection transport readiness controls differ across ranks.");
        readiness_limit = static_cast<long double>(readiness->volume_tolerance) / (2.L * readiness->max_courant);
        require(std::isfinite(readiness_limit) && readiness_limit > 0,
            "Finite projection transport readiness has an unrepresentable ratio threshold.");
    }
    const auto wide_maximum = [&](long double local)
    {
        int exponent = std::numeric_limits<int>::min(), global_exponent = 0;
        double mantissa = 0, global_mantissa = 0;
        if (local > 0)
        {
            const auto wide = std::frexp(local, &exponent);
            mantissa = static_cast<double>(wide);
            if (static_cast<long double>(mantissa) < wide)
                mantissa = std::nextafter(mantissa, std::numeric_limits<double>::infinity());
        }
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &exponent, &global_exponent);
        if (exponent != global_exponent)
            mantissa = 0;
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &mantissa, &global_mantissa);
        return global_mantissa > 0 ? std::ldexp(static_cast<long double>(global_mantissa), global_exponent) : 0.L;
    };
    const auto halo_rates = [&](const std::vector<long double>& owned)
    {
        std::vector<double> high(geometry_.size()), low(geometry_.size()), exponent(geometry_.size());
        for (std::size_t f = 0; f < nf; ++f)
        {
            int e = 0;
            const auto m = std::frexp(owned[f], &e);
            high[f] = static_cast<double>(m);
            low[f] = static_cast<double>(m - high[f]);
            exponent[f] = e;
        }
        high = faces(high);
        low = faces(low);
        exponent = faces(exponent);
        std::vector<long double> result(geometry_.size());
        for (std::size_t f = 0; f < result.size(); ++f)
            result[f] = std::ldexp(static_cast<long double>(high[f]) + low[f], int(exponent[f]));
        return result;
    };
    namespace fp = VoF::finite_detail;
    using A = fp::Arithmetic<Mode>;
    const auto arithmetic_value = [](long double value, long double error = 0) { return A::make_value(value, error); };
    const auto arithmetic_guard = [&](auto&& work, const char* message)
    {
        bool accepted = true;
        try
        {
            work();
        }
        catch (...)
        {
            accepted = false;
        }
        require(accepted, message);
    };
    std::vector<long double> mass(n), force(nf), rate(nf), radius(nf);
    const auto fraction_halo = cells(fraction), curvature_halo = cells(kappa);
    arithmetic_guard(
        [&]
        {
            for (std::size_t c = 0; c < n; ++c)
            {
                mass[c] = physical_mass ? (*physical_mass)[c] : static_cast<long double>(density[c]) * volumes_[c];
                for (unsigned a = 0; a < 3; ++a)
                    if (absent_gradient_axis_[a] && !wall_reaction_axis_[a] && velocity[c][a] != 0)
                        throw std::invalid_argument(
                            "Unsampled normal motion requires a supported closed-slab pressure mode.");
            }
            for (std::size_t f = 0; f < nf; ++f)
            {
                const auto& g = geometry_[f];
                if (g.boundary)
                    continue;
                rate[f] = static_cast<long double>(face_velocity[f]) * g.area;
                const long double curvature = (static_cast<long double>(g.right) * curvature_halo[g.owner] +
                                                  static_cast<long double>(g.left) * curvature_halo[g.neighbor]) /
                                              g.distance;
                // B and R already use the physical native geometry. Pressure and
                // capillarity must use the same cell-potential jump, including skew.
                force[f] = static_cast<long double>(sigma) * curvature *
                           (static_cast<long double>(fraction_halo[g.neighbor]) - fraction_halo[g.owner]);
            }
        },
        "Compatible projection mass, wall support or capillary force is invalid.");
    general_vof::flow_trace(comm->getRank(), "compatible_projection_input",
        [&](auto& out)
        {
            const auto lease = mesh_->acquire_execution_view();
            out << ",\"dt\":" << dt << ",\"sigma\":" << sigma
                << ",\"cell_columns\":[\"gid\",\"volume\",\"mass\",\"ux\",\"uy\",\"uz\"]"
                << ",\"cells\":[";
            for (std::size_t c = 0; c < n; ++c)
                out << (c ? ",[" : "[") << mesh_->cell_global_id(LO(c)) << ',' << volumes_[c] << ',' << mass[c] << ','
                    << velocity[c][0] << ',' << velocity[c][1] << ',' << velocity[c][2] << ']';
            out << "],\"face_columns\":[\"gid\",\"owner_gid\",\"neighbor_gid\",\"area\","
                   "\"owner_rx\",\"owner_ry\",\"owner_rz\",\"neighbor_rx\",\"neighbor_ry\",\"neighbor_rz\","
                   "\"capillary_force\",\"nx\",\"ny\",\"nz\"],\"faces\":[";
            for (std::size_t f = 0; f < nf; ++f)
            {
                const auto& g = geometry_[f];
                const auto left = convert(mesh_->face_center_vector(LO(f), g.owner));
                const auto right = g.boundary ? Vector{} : convert(mesh_->face_center_vector(LO(f), g.neighbor));
                out << (f ? ",[" : "[") << mesh_->face_global_id(LO(f)) << ',' << mesh_->cell_global_id(g.owner) << ','
                    << (g.boundary ? -1 : mesh_->cell_global_id(g.neighbor)) << ',' << g.area << ',' << left[0] << ','
                    << left[1] << ',' << left[2] << ',' << right[0] << ',' << right[1] << ',' << right[2] << ','
                    << force[f] << ',' << g.normal[0] << ',' << g.normal[1] << ',' << g.normal[2] << ']';
            }
            out << ']';
        });
    std::vector<long double> net(n), cell_radius(n), overlap_rate, overlap_radius;
    const bool cumulative = readiness && readiness->cumulative;
    std::vector<long double> occupancy_target(n), incident_rate;
    arithmetic_guard(
        [&]
        {
            if (cumulative)
            {
                occupancy_target.resize(n);
                incident_rate.resize(n);
            }
        },
        "Occupancy pressure target scratch allocation failed.");
    bool occupancy_target_frozen = false;
    long double occupancy_target_defect = 0, occupancy_target_limit = 0, occupancy_target_maximum = 0;
    long double occupancy_compatibility_residual = 0, occupancy_compatibility_bound = 0;
    long double readiness_ratio = 0, readiness_step_defect = 0, cumulative_step_defect = 0;
    const auto refresh_rates = [&]
    {
        overlap_rate = halo_rates(rate);
        overlap_radius = halo_rates(radius);
        std::fill(net.begin(), net.end(), 0);
        std::fill(cell_radius.begin(), cell_radius.end(), 0);
        long double local_divergence = 0;
        long double local_readiness_ratio = 0;
        long double local_step_defect = 0;
        long double local_cumulative_defect = 0;
        long double local_target_defect = 0;
        arithmetic_guard(
            [&]
            {
                const auto lease = mesh_->acquire_execution_view();
                for (std::size_t c = 0; c < n; ++c)
                {
                    fp::SumFor<Mode> row(arithmetic_mode_);
                    fp::SumFor<Mode> incident(arithmetic_mode_);
                    for (auto fid : mesh_->faces(LO(c)))
                    {
                        row.add(arithmetic_value(
                            (geometry_[fid].owner == LO(c) ? 1 : -1) * overlap_rate[fid], overlap_radius[fid]));
                        if (readiness)
                            incident.add(arithmetic_value(std::abs(overlap_rate[fid]), 0));
                    }
                    const auto sum = row.value();
                    net[c] = sum.value;
                    cell_radius[c] = sum.absolute_error;
                    local_divergence = std::max(local_divergence, std::abs(net[c]) / volumes_[c]);
                    if (readiness)
                    {
                        const auto total = incident.value().value;
                        const auto ratio = total > 0 ? std::abs(net[c]) / total : 0.L;
                        local_readiness_ratio = std::max(local_readiness_ratio, ratio);
                        local_step_defect = std::max(local_step_defect, std::abs(net[c]) * dt / volumes_[c]);
                        if (readiness->cumulative)
                        {
                            incident_rate[c] = total;
                            if (occupancy_target_frozen)
                                local_target_defect =
                                    std::max(local_target_defect, std::abs(net[c] - occupancy_target[c]) / volumes_[c]);
                            const auto volume = readiness->volume[c];
                            const auto delta = readiness->signed_discrepancy[c];
                            const auto u = (32.L + 12.L * mesh_->faces(LO(c)).size()) *
                                           std::numeric_limits<long double>::epsilon();
                            const auto uncertainty = total == 0 ? 0.L
                                                                : (std::numeric_limits<double>::epsilon() + u) *
                                                                      (volume + std::abs(delta) + dt * total);
                            local_cumulative_defect = std::max(
                                local_cumulative_defect, (std::abs(delta - dt * net[c]) + uncertainty) / volume);
                        }
                    }
                }
            },
            "Finite projection row accumulation exceeds its supported arithmetic range.");
        if (readiness)
        {
            readiness_ratio = wide_maximum(local_readiness_ratio);
            readiness_step_defect = wide_maximum(local_step_defect);
            cumulative_step_defect = wide_maximum(local_cumulative_defect);
            if (cumulative)
                occupancy_target_defect = wide_maximum(local_target_defect);
        }
        return wide_maximum(local_divergence);
    };
    auto residual = refresh_rates();
    const auto pressure_limit =
        A::add(A::multiply(arithmetic_value(dt, 0), arithmetic_value(options.pressure_absolute_tolerance, 0)),
            A::multiply(arithmetic_value(residual, 0), arithmetic_value(options.pressure_relative_tolerance, 0)));
    const long double limit =
        std::min(static_cast<long double>(options.divergence_tolerance), A::lower(pressure_limit));
    require(std::isfinite(limit) && limit > 0, "Finite projection has invalid original physical threshold.");
    // Use an existing admissible residual to redistribute signed occupancy
    // conservatively. These are pressure rate targets, never phase sources.
    // A provisional physical projection supplies their actual circulation;
    // predictor circulation could vanish on a closed tree.
    const auto wide_sums = [&](const std::array<long double, 2>& local)
    {
        std::array<double, 6> send{};
        std::vector<double> received;
        arithmetic_guard(
            [&]
            {
                received.resize(6 * std::size_t(comm->getSize()));
                for (unsigned a = 0; a < 2; ++a)
                {
                    if (!std::isfinite(local[a]))
                        throw std::runtime_error("Nonfinite occupancy target reduction.");
                    int exponent = 0;
                    const auto mantissa = std::frexp(local[a], &exponent);
                    send[3 * a] = double(mantissa);
                    send[3 * a + 1] = double(mantissa - send[3 * a]);
                    send[3 * a + 2] = exponent;
                    if (static_cast<long double>(send[3 * a]) + send[3 * a + 1] != mantissa)
                        throw std::runtime_error("Occupancy target reduction loses represented precision.");
                }
            },
            "Occupancy target reduction exceeds its admitted representation.");
        Teuchos::gatherAll(*comm, 6, send.data(), 6 * comm->getSize(), received.data());
        std::array<CompensatedSum<long double>, 2> sums;
        arithmetic_guard(
            [&]
            {
                for (int rank = 0; rank < comm->getSize(); ++rank)
                    for (unsigned a = 0; a < 2; ++a)
                    {
                        const auto i = 6 * std::size_t(rank) + 3 * a;
                        sums[a] +=
                            std::ldexp(static_cast<long double>(received[i]) + received[i + 1], int(received[i + 2]));
                    }
            },
            "Occupancy target reduction exceeds its arithmetic range.");
        return std::array<long double, 2>{sums[0].value(), sums[1].value()};
    };
    const auto freeze_occupancy_target = [&]
    {
        long double local_capacity = 0, local_floor = 0;
        arithmetic_guard(
            [&]
            {
                const auto lease = mesh_->acquire_execution_view();
                for (std::size_t c = 0; c < n; ++c)
                {
                    // Half of each original allowance remains for solve error.
                    const auto capacity =
                        .5L * std::min({readiness->volume_tolerance * incident_rate[c] / (2 * readiness->max_courant),
                                  limit * volumes_[c],
                                  2 * VoF::algebraic_physical_occupancy_relative_limit * incident_rate[c] /
                                      readiness->max_courant});
                    occupancy_target[c] = capacity;
                    local_capacity = std::max(local_capacity, capacity);
                    local_floor = std::max(local_floor, (32.L + 8.L * mesh_->faces(LO(c)).size()) *
                                                            std::numeric_limits<long double>::epsilon() *
                                                            incident_rate[c] / volumes_[c]);
                }
            },
            "Occupancy pressure target has invalid local capacity.");
        const auto capacity_scale = wide_maximum(local_capacity);
        if (capacity_scale == 0)
        {
            occupancy_target_frozen = true;
            return;
        }
        CompensatedSum<long double> weights, weighted_discrepancy;
        arithmetic_guard(
            [&]
            {
                for (std::size_t c = 0; c < n; ++c)
                {
                    const auto weight = occupancy_target[c] / capacity_scale;
                    const auto relative = readiness->signed_discrepancy[c] /
                                          (VoF::algebraic_physical_occupancy_relative_limit * readiness->volume[c]);
                    weights += weight;
                    weighted_discrepancy += weight * relative;
                }
            },
            "Occupancy pressure target has invalid signed inventory.");
        const auto totals = wide_sums({weights.value(), weighted_discrepancy.value()});
        require(totals[0] > 0 && std::isfinite(totals[0]) && std::isfinite(totals[1]),
            "Occupancy pressure target has no finite compatible weights.");
        const auto mean = totals[1] / totals[0];
        CompensatedSum<long double> target_sum, capacity_sum;
        long double local_target = 0;
        arithmetic_guard(
            [&]
            {
                for (std::size_t c = 0; c < n; ++c)
                {
                    const auto capacity = occupancy_target[c];
                    const auto relative = readiness->signed_discrepancy[c] /
                                          (VoF::algebraic_physical_occupancy_relative_limit * readiness->volume[c]);
                    occupancy_target[c] = capacity * ((relative - mean) / 2);
                    target_sum += occupancy_target[c];
                    capacity_sum += capacity;
                    local_target = std::max(local_target, std::abs(occupancy_target[c]) / volumes_[c]);
                    general_vof::flow_trace(comm->getRank(), "occupancy_pressure_target",
                        [&](auto& out)
                        {
                            out << ",\"cell\":" << mesh_->cell_global_id(LO(c))
                                << ",\"signed_occupancy\":" << readiness->signed_discrepancy[c]
                                << ",\"capacity\":" << capacity << ",\"target_net_rate\":" << occupancy_target[c]
                                << ",\"normalized_mean\":" << mean;
                        });
                }
            },
            "Occupancy pressure target exceeds its arithmetic range.");
        const auto compatible = wide_sums({target_sum.value(), capacity_sum.value()});
        const auto count = mesh_->owned_cell_map()->getGlobalNumElements();
        const auto u = (32.L + 16.L * count + 16.L * comm->getSize()) * std::numeric_limits<long double>::epsilon();
        require(u < .25L, "Occupancy compatibility reduction budget is exhausted.");
        occupancy_compatibility_residual = std::abs(compatible[0]);
        occupancy_compatibility_bound = u / (1 - u) * compatible[1];
        require(occupancy_compatibility_residual <= occupancy_compatibility_bound,
            "Closed-domain occupancy pressure target is incompatible.");
        occupancy_target_maximum = wide_maximum(local_target);
        occupancy_target_limit = std::max(occupancy_target_maximum / 8, wide_maximum(local_floor));
        occupancy_target_frozen = true;
    };
    // Resource capacities are rank-local. Reserve this caller's live footprint;
    // every kernel admission still rejects collectively if any local cap fails.
    GeneralVoFCompatibleProjection coupled(mesh_, mass, budget - static_cast<std::size_t>(payload_units));
    auto peak_kernel_storage = coupled.storage_bytes();
    const std::vector<long double> admitted_face_rhs(nf, 0.L);
    std::fill(rate.begin(), rate.end(), 0.L);
    std::vector<long double> impulse(n);
    auto defect = coupled.residual(rate, impulse, velocity, occupancy_target, force, dt);
    long double rhs_scale = 0;
    for (const auto value : defect.face_rhs)
        rhs_scale = std::max(rhs_scale, std::abs(value));
    rhs_scale = wide_maximum(rhs_scale);
    unsigned degree = 1;
    {
        const auto lease = mesh_->acquire_execution_view();
        for (std::size_t c = 0; c < n; ++c)
            degree = std::max(degree, unsigned(mesh_->faces(LO(c)).size()));
    }
    const long double algebraic_limit =
        wide_maximum((128.L + 32.L * degree) * std::numeric_limits<long double>::epsilon() * rhs_scale);
    diagnostics.pressure_iterations = 0;
    LinearSolverOptions linear;
    linear.backend = LinearSolverBackend::Gmres;
    linear.preconditioner = LinearPreconditioner::None;
    linear.reuse_preconditioner = false;
    linear.tolerance = 1e-8;
    linear.verbosity = 0;
    bool closed = false;
    for (unsigned correction = 0; correction <= 64; ++correction)
    {
        residual = refresh_rates();
        long double row_defect = 0;
        for (const auto value : defect.face_rhs)
            row_defect = std::max(row_defect, std::abs(value));
        row_defect = wide_maximum(row_defect);
        const bool ready = !readiness || readiness_ratio <= readiness_limit ||
                           readiness_step_defect <= static_cast<long double>(readiness->volume_tolerance) / 2;
        if (row_defect <= algebraic_limit && residual <= limit && ready)
        {
            if (cumulative && !occupancy_target_frozen)
            {
                freeze_occupancy_target();
                residual = refresh_rates();
                defect = coupled.residual(rate, impulse, velocity, occupancy_target, force, dt);
            }
            closed = !cumulative ||
                     (cumulative_step_defect <= VoF::algebraic_physical_occupancy_relative_limit &&
                         (occupancy_target_maximum == 0 || occupancy_target_defect <= occupancy_target_limit));
            if (closed)
                break;
        }
        general_vof::flow_trace(comm->getRank(), "compatible_pressure_correction",
            [&](auto& out)
            {
                out << ",\"correction\":" << correction << ",\"dt\":" << dt << ",\"residual\":" << residual
                    << ",\"physical_limit\":" << limit << ",\"momentum_row_defect\":" << row_defect
                    << ",\"momentum_row_limit\":" << algebraic_limit
                    << ",\"iterations\":" << diagnostics.pressure_iterations
                    << ",\"transport_readiness_ratio\":" << readiness_ratio
                    << ",\"transport_readiness_limit\":" << readiness_limit;
            });
        linear.max_iterations = options.max_pressure_iterations - diagnostics.pressure_iterations;
        if (correction == 64 || linear.max_iterations <= 0)
            break;
        const bool momentum_admitted = row_defect <= algebraic_limit;
        if (momentum_admitted)
        {
            long double remaining_constraint = 0;
            for (const auto value : defect.cell_rhs)
                remaining_constraint = std::max(remaining_constraint, std::abs(value));
            if (wide_maximum(remaining_constraint) == 0)
                break; // Keep the complete original-row rejection evidence.
        }
        // A resolved momentum block must not dominate normalization of a much
        // smaller unresolved continuity target. A zero momentum correction RHS
        // preserves that block in exact arithmetic; all original rows and gates
        // are recomputed after accumulation. No residual or target is erased.
        const auto correction_face_rhs = momentum_admitted ? std::span<const long double>(admitted_face_rhs)
                                                          : std::span<const long double>(defect.face_rhs);
        const auto increment = coupled.solve_algebraic(correction_face_rhs, defect.cell_rhs, linear);
        peak_kernel_storage = std::max(peak_kernel_storage, increment.storage_bytes);
        diagnostics.pressure_iterations += increment.linear.iterations;
        arithmetic_guard(
            [&]
            {
                for (std::size_t f = 0; f < nf; ++f)
                    rate[f] += increment.face_rate[f];
                for (std::size_t c = 0; c < n; ++c)
                    impulse[c] += increment.pressure_impulse[c];
            },
            "Compatible projection correction cannot be accumulated.");
        defect = coupled.residual(rate, impulse, velocity, occupancy_target, force, dt);
    }
    if (!closed)
    {
        std::ostringstream message;
        message << std::setprecision(18)
                << "Compatible General VoF projection failed original physical/arithmetic acceptance: residual="
                << residual << ", limit=" << limit << ", iterations=" << diagnostics.pressure_iterations
                << ", momentum_row_defect=" << defect.max_momentum << ", momentum_row_limit=" << algebraic_limit
                << ", transport_readiness_ratio=" << readiness_ratio
                << ", transport_readiness_limit=" << readiness_limit
                << ", transport_step_defect=" << readiness_step_defect
                << ", cumulative_step_defect=" << cumulative_step_defect
                << ", occupancy_target_defect=" << occupancy_target_defect
                << ", occupancy_target_limit=" << occupancy_target_limit;
        throw std::runtime_error(message.str());
    }
    const auto reconstructed = coupled.reconstruct(rate);
    std::vector<Vector> candidate_velocity(n);
    std::vector<double> candidate_pressure(n), candidate_face(geometry_.size());
    std::vector<long double> wall_pressure_offset(provenance && missing_direction ? nf : 0);
    long double wall_normal_residual = 0;
    CompensatedSum<long double> old_energy, new_energy, change_energy, pressure_work, defect_work;
    arithmetic_guard(
        [&]
        {
            for (std::size_t c = 0; c < n; ++c)
            {
                candidate_pressure[c] = double(impulse[c] / dt);
                valid = valid && std::isfinite(candidate_pressure[c]);
                pressure_work += impulse[c] * occupancy_target[c];
                defect_work += std::abs(impulse[c] * defect.cell_rhs[c]);
                for (unsigned a = 0; a < 3; ++a)
                {
                    candidate_velocity[c][a] = double(reconstructed[c][a]);
                    valid = valid && std::isfinite(candidate_velocity[c][a]) &&
                            (reconstructed[c][a] == 0 || candidate_velocity[c][a] != 0);
                    const long double before = velocity[c][a], after = candidate_velocity[c][a],
                                      change = after - before;
                    old_energy += .5L * mass[c] * before * before;
                    new_energy += .5L * mass[c] * after * after;
                    change_energy += .5L * mass[c] * change * change;
                    if (absent_gradient_axis_[a])
                        wall_normal_residual = std::max(wall_normal_residual, std::abs(after) / spans_[c][a]);
                }
            }
            for (std::size_t f = 0; f < nf; ++f)
            {
                const auto& g = geometry_[f];
                candidate_face[f] = g.boundary ? 0 : double(rate[f] / g.area);
                valid = valid && std::isfinite(candidate_face[f]) && (rate[f] == 0 || candidate_face[f] != 0);
                pressure_work += dt * force[f] * rate[f];
                defect_work += std::abs(rate[f] * defect.face_rhs[f]);
                if (provenance && missing_direction && g.boundary)
                {
                    const auto c = std::size_t(g.owner);
                    const auto r = convert(mesh_->face_center_vector(LO(f), g.owner));
                    for (unsigned a = 0; a < 3; ++a)
                        if (absent_gradient_axis_[a])
                            wall_pressure_offset[f] -=
                                mass[c] / volumes_[c] / dt * r[a] * (reconstructed[c][a] - velocity[c][a]);
                }
            }
        },
        "Compatible projection candidate or pressure work is unrepresentable.");
    require(valid, "Compatible projection candidate cannot be represented without range loss.");
    wall_normal_residual = wide_maximum(wall_normal_residual);
    require(wall_normal_residual <= limit, "Compatible projection failed its original wall-normal threshold.");
    const auto energies = wide_sums({old_energy.value(), new_energy.value()});
    const auto work = wide_sums({change_energy.value(), pressure_work.value()});
    const auto residual_work = wide_sums({defect_work.value(), 0})[0];
    const long double identity = energies[1] - energies[0] + work[0] - work[1];
    const long double energy_roundoff =
        128.L * std::numeric_limits<double>::epsilon() * (energies[0] + energies[1] + work[0] + std::abs(work[1]));
    require(std::abs(identity) <= residual_work + energy_roundoff,
        "Compatible projection failed the independent physical pressure-work identity.");
    candidate_face = faces(candidate_face);
    general_vof::flow_trace(comm->getRank(), "compatible_projection_accept",
        [&](auto& out)
        {
            out << ",\"kinetic_energy_before\":" << energies[0] << ",\"kinetic_energy_after\":" << energies[1]
                << ",\"pressure_and_capillary_work\":" << work[1] << ",\"correction_energy\":" << work[0]
                << ",\"energy_identity_defect\":" << identity
                << ",\"energy_identity_bound\":" << residual_work + energy_roundoff
                << ",\"kernel_storage_bytes\":" << peak_kernel_storage << ",\"caller_storage_bytes\":" << payload_units
                << ",\"storage_bytes\":" << peak_kernel_storage + static_cast<std::size_t>(payload_units)
                << ",\"pressure_iterations\":" << diagnostics.pressure_iterations;
        });
    velocity = std::move(candidate_velocity);
    pressure = std::move(candidate_pressure);
    face_velocity = std::move(candidate_face);
    diagnostics.maximum_divergence = double(residual);
    diagnostics.pressure_residual = double(residual / dt);
    if (provenance)
    {
        provenance->arithmetic_mode = ArithmeticMode::Ordinary;
        provenance->face_rate = std::move(rate);
        provenance->face_radius = std::move(radius);
        provenance->cell_radius = std::move(cell_radius);
        provenance->velocity_error.assign(n, {});
        provenance->wall_pressure_offset = std::move(wall_pressure_offset);
        provenance->maximum_wall_normal_residual = double(wall_normal_residual);
        provenance->closure_storage_bytes = 0;
    }
}
