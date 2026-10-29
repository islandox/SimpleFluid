/** @file GeneralVoFFiniteProjection.tcc
 * @brief Compiled finite arithmetic specialization.
 */
template<ArithmeticMode Mode>
void GeneralVoFFlowWorkspace::project_finite(std::vector<double>& face_velocity, std::vector<Vector>& velocity,
    std::vector<double>& pressure, const std::vector<double>& density, const std::vector<double>& fraction,
    const std::vector<double>& kappa, double dt, double sigma, const VoFSolverOptions& options,
    VoFStepDiagnostics& diagnostics, GeneralVoFFiniteProjectionProvenance* provenance,
    std::optional<std::span<const long double>> physical_mass,
    const GeneralVoFFiniteProjectionClosureTopology* closure_topology,
    std::optional<std::span<const long double>> physical_mass_error)
{
    const auto comm = mesh_->owned_cell_map()->getComm();
    const auto n = volumes_.size(), nf = mesh_->num_owned_faces();
    const auto budget = provenance ? provenance->byte_budget : 64 * 1024 * 1024;
    constexpr bool bounded = Mode == ArithmeticMode::BoundedCertification;
    const auto readiness = provenance ? provenance->transport_readiness
                                      : std::optional<GeneralVoFFiniteProjectionProvenance::TransportReadiness>{};
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
                               std::isfinite(readiness->max_courant) && readiness->max_courant > 0));
    for (std::size_t c = 0; c < n; ++c)
        valid = valid && std::isfinite(density[c]) && density[c] > 0 && std::isfinite(fraction[c]) &&
                fraction[c] >= 0 && fraction[c] <= 1 && std::isfinite(kappa[c]) &&
                (!physical_mass || (std::isfinite((*physical_mass)[c]) && (*physical_mass)[c] > 0)) &&
                (!physical_mass_error || (std::isfinite((*physical_mass_error)[c]) && (*physical_mass_error)[c] >= 0 &&
                                             (!bounded || (*physical_mass)[c] > (*physical_mass_error)[c])));
    for (const auto& u : velocity)
        for (double value : u)
            valid = valid && std::isfinite(value);
    for (std::size_t f = 0; f < nf; ++f)
        valid = valid && std::isfinite(face_velocity[f]) && (!geometry_[f].boundary || face_velocity[f] == 0);
    // Fixed-width vector/halo/gradient/force buffers occupy one half; the
    // authenticated sparse closure uses the other half. Backend allocations
    // remain outside this application-payload accounting, as in the exact API.
    const auto cells_count = mesh_->num_cells(), faces_count = mesh_->num_faces();
    const bool missing_direction =
        std::any_of(absent_gradient_axis_.begin(), absent_gradient_axis_.end(), [](bool absent) { return absent; });
    const auto payload_units =
        256 * (static_cast<long double>(cells_count) + faces_count + n + nf + 1) +
        (provenance && missing_direction ? sizeof(long double) * static_cast<long double>(nf) : 0);
    valid = valid && budget > 0 && payload_units <= budget / 2;
    require(valid,
        "Finite General VoF projection has invalid inputs, unsupported FP environment or exhausted payload cap.");
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
        double controls[2] = {readiness->volume_tolerance, readiness->max_courant}, minimum[2]{}, maximum[2]{};
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_MIN, 2, controls, minimum);
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 2, controls, maximum);
        require(minimum[0] == maximum[0] && minimum[1] == maximum[1],
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
    const auto rho = cells(density), fraction_halo = cells(fraction), curvature_halo = cells(kappa);
    const auto grad_f = vectors(gradient(fraction));
    std::vector<double> beta(geometry_.size()), fraction_face(geometry_.size()), curvature_face(geometry_.size());
    std::vector<long double> rate(nf), radius(nf);
    namespace fp = VoF::finite_detail;
    using A = fp::Arithmetic<Mode>;
    namespace arithmetic = vof_projection_fp;
    using Value = arithmetic::Value;
    const auto arithmetic_value = [&](long double value, long double error = 0) { return A::make_value(value, error); };
    std::vector<long double> beta_error(geometry_.size()), fraction_face_error(geometry_.size()),
        curvature_face_error(geometry_.size());
    // These coefficients keep their established binary64 cache and backend
    // interface. Every narrowing retains its error for the canonical wide rate.
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
    const auto record_update = [&](std::size_t f, Value increment)
    {
        const auto update = A::add(arithmetic_value(rate[f], radius[f]), increment);
        rate[f] = update.value;
        radius[f] = update.absolute_error;
    };
    arithmetic_guard(
        [&]
        {
            for (std::size_t f = 0; f < nf; ++f)
            {
                const auto& g = geometry_[f];
                if (g.boundary)
                    continue;
                const auto initial = A::multiply(arithmetic_value(face_velocity[f], 0), arithmetic_value(g.area, 0));
                rate[f] = initial.value;
                radius[f] = initial.absolute_error;
                const auto mass_length =
                    A::add(A::multiply(arithmetic_value(rho[g.owner], 0), arithmetic_value(g.left, 0)),
                        A::multiply(arithmetic_value(rho[g.neighbor], 0), arithmetic_value(g.right, 0)));
                const auto beta_value =
                    arithmetic::binary64<Mode>(A::divide(arithmetic_value(g.distance, 0), mass_length));
                beta[f] = double(beta_value.value);
                beta_error[f] = beta_value.absolute_error;
                const auto left = A::add(arithmetic_value(fraction_halo[g.owner], 0),
                    arithmetic::dot<Mode>(grad_f[g.owner], g.tangent_left));
                const auto right = A::add(arithmetic_value(fraction_halo[g.neighbor], 0),
                    arithmetic::dot<Mode>(grad_f[g.neighbor], g.tangent_right));
                const auto fraction_value = arithmetic::binary64<Mode>(A::divide(
                    A::add(A::multiply(
                               A::multiply(arithmetic_value(rho[g.neighbor], 0), arithmetic_value(g.right, 0)), left),
                        A::multiply(
                            A::multiply(arithmetic_value(rho[g.owner], 0), arithmetic_value(g.left, 0)), right)),
                    mass_length));
                fraction_face[f] = double(fraction_value.value);
                fraction_face_error[f] = fraction_value.absolute_error;
                const auto curvature_value = arithmetic::binary64<Mode>(A::divide(
                    A::add(A::multiply(arithmetic_value(g.right, 0), arithmetic_value(curvature_halo[g.owner], 0)),
                        A::multiply(arithmetic_value(g.left, 0), arithmetic_value(curvature_halo[g.neighbor], 0))),
                    arithmetic_value(g.distance, 0)));
                curvature_face[f] = double(curvature_value.value);
                curvature_face_error[f] = curvature_value.absolute_error;
                auto multiplier = A::multiply(arithmetic_value(dt, 0), beta_value);
                multiplier = A::multiply(multiplier, arithmetic_value(sigma, 0));
                multiplier = A::multiply(multiplier, curvature_value);
                multiplier =
                    A::divide(A::multiply(multiplier, arithmetic_value(g.area, 0)), arithmetic_value(g.distance, 0));
                record_update(f, A::multiply(multiplier, A::subtract(right, left)));
                valid = valid && beta[f] > 0;
            }
        },
        "Finite projection face construction exceeds its supported arithmetic range.");
    require(valid, "Finite projection has an unrepresentable operator coefficient.");
    beta = faces(beta);
    fraction_face = faces(fraction_face);
    curvature_face = faces(curvature_face);
    beta_error = halo_rates(beta_error);
    fraction_face_error = halo_rates(fraction_face_error);
    curvature_face_error = halo_rates(curvature_face_error);
    std::vector<std::array<long double, 3>> force(n), force_error(n), velocity_error(n);
    const auto add_force = [&](std::size_t c, unsigned a, Value contribution)
    {
        const auto update = A::add(arithmetic_value(force[c][a], force_error[c][a]), contribution);
        force[c][a] = update.value;
        force_error[c][a] = update.absolute_error;
    };
    arithmetic_guard(
        [&]
        {
            const auto lease = mesh_->acquire_execution_view();
            for (std::size_t c = 0; c < n; ++c)
                for (auto fid : mesh_->faces(LO(c)))
                {
                    const auto& g = geometry_[fid];
                    Value value;
                    if (g.boundary)
                    {
                        value = A::multiply(A::multiply(arithmetic_value(sigma, 0), arithmetic_value(kappa[c], 0)),
                            arithmetic::dot<Mode>(grad_f[c], convert(mesh_->face_center_vector(fid, LO(c)))));
                        value = A::multiply(value, arithmetic_value(g.area, 0));
                    }
                    else
                    {
                        value = A::multiply(
                            arithmetic_value(g.owner == LO(c) ? 1.L : -1.L, 0), arithmetic_value(g.area, 0));
                        value = A::multiply(A::multiply(value, arithmetic_value(sigma, 0)),
                            arithmetic_value(curvature_face[fid], curvature_face_error[fid]));
                        value = A::multiply(
                            value, A::subtract(arithmetic_value(fraction_face[fid], fraction_face_error[fid]),
                                       arithmetic_value(fraction[c], 0)));
                    }
                    for (unsigned a = 0; a < 3; ++a)
                        add_force(c, a, A::multiply(value, arithmetic_value(g.normal[a], 0)));
                }
        },
        "Finite projection force construction exceeds its supported arithmetic range.");
    std::vector<long double> net(n), cell_radius(n), overlap_rate, overlap_radius;
    long double readiness_ratio = 0, readiness_step_defect = 0;
    const auto refresh_rates = [&]
    {
        overlap_rate = halo_rates(rate);
        overlap_radius = halo_rates(radius);
        std::fill(net.begin(), net.end(), 0);
        std::fill(cell_radius.begin(), cell_radius.end(), 0);
        long double local_divergence = 0;
        long double local_readiness_ratio = 0;
        long double local_step_defect = 0;
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
                    }
                }
            },
            "Finite projection row accumulation exceeds its supported arithmetic range.");
        if (readiness)
        {
            readiness_ratio = wide_maximum(local_readiness_ratio);
            readiness_step_defect = wide_maximum(local_step_defect);
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
    std::vector<long double> wall_pressure_offset(provenance && missing_direction ? nf : 0);
    long double wall_normal_residual = 0;
    if (missing_direction)
    {
        arithmetic_guard(
            [&]
            {
                const auto lease = mesh_->acquire_execution_view();
                for (std::size_t c = 0; c < n; ++c)
                {
                    const auto volume = arithmetic_value(volumes_[c], 0);
                    const auto mass = physical_mass ? arithmetic_value((*physical_mass)[c],
                                                          physical_mass_error ? (*physical_mass_error)[c] : 0)
                                                    : A::multiply(arithmetic_value(density[c], 0), volume);
                    const auto physical_density = A::divide(mass, volume);
                    for (unsigned a = 0; a < 3; ++a)
                    {
                        if (!absent_gradient_axis_[a])
                            continue;
                        if (!wall_reaction_axis_[a])
                        {
                            if (velocity[c][a] != 0)
                                throw std::invalid_argument(
                                    "Unsampled normal motion requires a supported closed-slab pressure mode.");
                            continue;
                        }
                        // Cell centers cannot represent this affine pressure
                        // variation. Its cap values still exert the physical
                        // traction needed to enforce zero wall-normal velocity.
                        // The predictor is the supplied candidate, not gravity:
                        // a resting initialization has no artificial impulse.
                        const auto gradient =
                            A::divide(A::multiply(physical_density, arithmetic_value(velocity[c][a], 0)),
                                arithmetic_value(dt, 0));
                        for (const auto fid : mesh_->faces(LO(c)))
                        {
                            const auto& g = geometry_[fid];
                            if (!g.boundary || g.normal[a] == 0)
                                continue;
                            const auto normal = arithmetic_value(g.normal[a], 0);
                            const auto predictor = A::multiply(arithmetic_value(velocity[c][a], 0), normal);
                            const auto normal_gradient = A::multiply(gradient, normal);
                            const auto correction =
                                A::multiply(A::divide(normal_gradient, physical_density), arithmetic_value(dt, 0));
                            const auto mismatch =
                                A::divide(A::subtract(predictor, correction), arithmetic_value(spans_[c][a], 0));
                            wall_normal_residual = std::max(wall_normal_residual,
                                std::abs(mismatch.value) + (bounded ? mismatch.absolute_error : 0));
                            const auto offset = A::multiply(arithmetic_value(g.left, 0), normal_gradient);
                            if (provenance)
                            {
                                if (fid < 0 || std::size_t(fid) >= nf)
                                    throw std::invalid_argument("A wall pressure face is not owned with its cell.");
                                wall_pressure_offset[fid] = offset.value;
                            }
                            const auto traction = A::negate(A::multiply(offset, arithmetic_value(g.area, 0)));
                            for (unsigned b = 0; b < 3; ++b)
                                add_force(c, b, A::multiply(traction, arithmetic_value(g.normal[b], 0)));
                        }
                    }
                }
            },
            "Finite projection wall pressure requires supported slab geometry and finite physical data.");
        wall_normal_residual = wide_maximum(wall_normal_residual);
        require(std::isfinite(wall_normal_residual) && wall_normal_residual <= limit,
            "Finite wall pressure failed its original physical normal-velocity threshold.");
    }
    std::vector<long double> wide_pressure(n);
    diagnostics.pressure_iterations = 0;
    bool closed = false;
    std::string closure_reason;
    std::size_t closure_storage = 0;
    LinearSolverOptions linear;
    linear.backend = LinearSolverBackend::Cg;
    linear.preconditioner = LinearPreconditioner::Jacobi;
    linear.reuse_preconditioner = true;
    linear.tolerance = 1e-8;
    linear.verbosity = 0;
    for (unsigned correction = 0; correction <= 64; ++correction)
    {
        if (residual <= limit)
        {
            if constexpr (!bounded)
            {
                // Generic Ordinary retains its original gate. Algebraic
                // consumers additionally resolve their relative or finite-step
                // volume budget. The next timestep also bounds the accepted
                // rate defect, avoiding an exact-zero demand on closed trees.
                closed = !readiness || readiness_ratio <= readiness_limit ||
                         readiness_step_defect <= static_cast<long double>(readiness->volume_tolerance) / 2;
                if (closed)
                    break;
            }
            else
                try
                {
                    auto closure = close_general_vof_finite_rate(mesh_, rate, radius, cell_radius, budget / 2,
                        ArithmeticMode::BoundedCertification, closure_topology);
                    arithmetic_guard(
                        [&]
                        {
                            for (std::size_t f = 0; f < nf; ++f)
                            {
                                const auto correction =
                                    A::subtract(arithmetic_value(closure.rate[f], 0), arithmetic_value(rate[f], 0));
                                radius[f] = fp::bound_add(
                                    radius[f], fp::bound_add(std::abs(correction.value), correction.absolute_error));
                            }
                        },
                        "Finite projection closure correction exceeds its arithmetic range.");
                    rate = std::move(closure.rate);
                    closure_storage = closure.admitted_storage_bytes;
                    residual = refresh_rates();
                    closed = residual <= limit;
                }
                catch (const GeneralVoFPhiInfeasible& error)
                {
                    closure_reason = error.what();
                    closed = false;
                }
            if (closed)
                break;
        }
        linear.max_iterations = options.max_pressure_iterations - diagnostics.pressure_iterations;
        if (correction == 64 || linear.max_iterations <= 0)
            break;
        if (correction == 0)
            projection_.refresh(beta, nullptr, dt, gauge_);
        long double local_scale = 0;
        for (auto value : net)
            local_scale = std::max(local_scale, std::abs(value / dt));
        const auto scale = wide_maximum(local_scale);
        require(
            std::isfinite(scale) && scale > 0, "Finite projection cannot resolve a nonzero original rate residual.");
        for (std::size_t c = 0; c < n; ++c)
        {
            const auto rhs = -net[c] / dt / scale;
            projection_.rhs.replaceLocalValue(LO(c), mesh_->cell_global_id(LO(c)) == gauge_ ? 0 : double(rhs));
        }
        const auto stats = projection_.solver.solve_from_zero_with_statistics(
            Teuchos::rcp_implicit_cast<const Pack::operator_type>(projection_.matrix), projection_.rhs,
            projection_.solution, linear);
        diagnostics.pressure_iterations += stats.iterations;
        require(std::isfinite(stats.achieved_tolerance) && std::isfinite(stats.rhs_norm),
            "Finite General VoF pressure correction is nonfinite.");
        const auto data = projection_.solution.getData();
        std::vector<double> p_owned(data.begin(), data.end());
        const auto p = cells(p_owned);
        const auto grad_p = vectors(gradient(p_owned));
        std::vector<double> p_face(geometry_.size());
        std::vector<long double> p_face_error(geometry_.size());
        arithmetic_guard(
            [&]
            {
                for (std::size_t f = 0; f < nf; ++f)
                {
                    const auto& g = geometry_[f];
                    if (g.boundary)
                        continue;
                    const auto left =
                        A::add(arithmetic_value(p[g.owner], 0), arithmetic::dot<Mode>(grad_p[g.owner], g.tangent_left));
                    const auto right = A::add(
                        arithmetic_value(p[g.neighbor], 0), arithmetic::dot<Mode>(grad_p[g.neighbor], g.tangent_right));
                    auto multiplier = A::multiply(arithmetic_value(dt, 0), arithmetic_value(scale, 0));
                    multiplier = A::multiply(multiplier, arithmetic_value(beta[f], beta_error[f]));
                    multiplier = A::divide(
                        A::multiply(multiplier, arithmetic_value(g.area, 0)), arithmetic_value(g.distance, 0));
                    record_update(f, A::negate(A::multiply(multiplier, A::subtract(right, left))));
                    const auto left_weight =
                        A::multiply(arithmetic_value(rho[g.neighbor], 0), arithmetic_value(g.right, 0));
                    const auto right_weight =
                        A::multiply(arithmetic_value(rho[g.owner], 0), arithmetic_value(g.left, 0));
                    const auto interpolated = arithmetic::binary64<Mode>(
                        A::divide(A::add(A::multiply(left_weight, left), A::multiply(right_weight, right)),
                            A::add(left_weight, right_weight)));
                    p_face[f] = double(interpolated.value);
                    p_face_error[f] = interpolated.absolute_error;
                }
            },
            "Finite pressure correction exceeds its arithmetic range.");
        p_face = faces(p_face);
        p_face_error = halo_rates(p_face_error);
        arithmetic_guard(
            [&]
            {
                const auto lease = mesh_->acquire_execution_view();
                for (std::size_t c = 0; c < n; ++c)
                {
                    wide_pressure[c] += scale * p_owned[c];
                    for (auto fid : mesh_->faces(LO(c)))
                    {
                        const auto& g = geometry_[fid];
                        Value value;
                        if (g.boundary)
                            value = A::negate(
                                arithmetic::dot<Mode>(grad_p[c], convert(mesh_->face_center_vector(fid, LO(c)))));
                        else
                            value = A::multiply(arithmetic_value(g.owner == LO(c) ? -1.L : 1.L, 0),
                                A::subtract(
                                    arithmetic_value(p_face[fid], p_face_error[fid]), arithmetic_value(p_owned[c], 0)));
                        value =
                            A::multiply(A::multiply(value, arithmetic_value(g.area, 0)), arithmetic_value(scale, 0));
                        for (unsigned a = 0; a < 3; ++a)
                            add_force(c, a, A::multiply(value, arithmetic_value(g.normal[a], 0)));
                    }
                }
            },
            "Finite pressure force exceeds its supported arithmetic range.");
        residual = refresh_rates();
    }
    if (!closed)
    {
        std::ostringstream message;
        message << std::setprecision(18)
                << "Finite General VoF projection failed original physical/arithmetic acceptance: residual=" << residual
                << ", limit=" << limit << ", iterations=" << diagnostics.pressure_iterations
                << ", closure=" << closure_reason;
        if (readiness)
            message << ", transport_readiness_ratio=" << readiness_ratio
                    << ", transport_readiness_limit=" << readiness_limit
                    << ", transport_step_defect=" << readiness_step_defect
                    << ", transport_step_limit=" << static_cast<long double>(readiness->volume_tolerance) / 2;
        throw std::runtime_error(message.str());
    }
    pressure.assign(n, 0);
    for (std::size_t f = 0; f < nf; ++f)
        face_velocity[f] = geometry_[f].boundary ? 0 : double(rate[f] / geometry_[f].area);
    face_velocity = faces(face_velocity);
    try
    {
        for (std::size_t c = 0; c < n; ++c)
        {
            pressure[c] = double(wide_pressure[c]);
            const auto derived_mass = A::multiply(arithmetic_value(density[c], 0), arithmetic_value(volumes_[c], 0));
            const long double mass = physical_mass ? (*physical_mass)[c] : derived_mass.value;
            const long double mass_error =
                bounded ? (physical_mass_error ? (*physical_mass_error)[c]
                                               : (physical_mass ? 0 : derived_mass.absolute_error))
                        : 0;
            const auto mass_lower = mass - mass_error;
            valid = valid && std::isfinite(mass) && mass_lower > 0 && std::isfinite(pressure[c]);
            for (unsigned a = 0; a < 3; ++a)
            {
                const auto impulse =
                    A::divide(A::multiply(arithmetic_value(force[c][a], force_error[c][a]), arithmetic_value(dt, 0)),
                        arithmetic_value(mass, mass_error));
                const auto update = A::add(arithmetic_value(velocity[c][a], 0), impulse);
                velocity[c][a] = double(update.value);
                velocity_error[c][a] =
                    bounded ? fp::bound_add(update.absolute_error, std::abs(update.value - velocity[c][a])) : 0;
                valid = valid && std::isfinite(velocity[c][a]) && std::isfinite(velocity_error[c][a]) &&
                        (update.value == 0 || velocity[c][a] != 0);
            }
        }
    }
    catch (...)
    {
        valid = false;
    }
    require(valid, "Finite projection candidate cannot be represented without range loss.");
    if (missing_direction)
    {
        arithmetic_guard(
            [&]
            {
                for (std::size_t c = 0; c < n; ++c)
                    for (unsigned a = 0; a < 3; ++a)
                        if (absent_gradient_axis_[a])
                        {
                            const auto normal_speed = A::divide(arithmetic_value(velocity[c][a], velocity_error[c][a]),
                                arithmetic_value(spans_[c][a], 0));
                            wall_normal_residual = std::max(wall_normal_residual,
                                std::abs(normal_speed.value) + (bounded ? normal_speed.absolute_error : 0));
                        }
            },
            "Finite projected wall-normal velocity exceeds its arithmetic range.");
        wall_normal_residual = wide_maximum(wall_normal_residual);
        require(std::isfinite(wall_normal_residual) && wall_normal_residual <= limit,
            "Finite projected cell momentum failed its original wall-normal threshold.");
    }
    diagnostics.maximum_divergence = double(residual);
    diagnostics.pressure_residual = double(residual / dt);
    if (provenance)
    {
        provenance->arithmetic_mode = arithmetic_mode_;
        provenance->face_rate = std::move(rate);
        provenance->face_radius = std::move(radius);
        provenance->cell_radius = std::move(cell_radius);
        provenance->velocity_error = std::move(velocity_error);
        provenance->wall_pressure_offset = std::move(wall_pressure_offset);
        provenance->maximum_wall_normal_residual = double(wall_normal_residual);
        provenance->closure_storage_bytes = closure_storage;
    }
}

template void GeneralVoFFlowWorkspace::project_finite<ArithmeticMode::Ordinary>(std::vector<double>& face_velocity,
    std::vector<GeneralVoFFlowWorkspace::Vector>& velocity, std::vector<double>& pressure,
    const std::vector<double>& density, const std::vector<double>& fraction, const std::vector<double>& kappa,
    double dt, double sigma, const VoFSolverOptions& options, VoFStepDiagnostics& diagnostics,
    GeneralVoFFiniteProjectionProvenance* provenance, std::optional<std::span<const long double>> physical_mass,
    const GeneralVoFFiniteProjectionClosureTopology* closure_topology,
    std::optional<std::span<const long double>> physical_mass_error);

template void GeneralVoFFlowWorkspace::project_finite<ArithmeticMode::BoundedCertification>(
    std::vector<double>& face_velocity, std::vector<GeneralVoFFlowWorkspace::Vector>& velocity,
    std::vector<double>& pressure, const std::vector<double>& density, const std::vector<double>& fraction,
    const std::vector<double>& kappa, double dt, double sigma, const VoFSolverOptions& options,
    VoFStepDiagnostics& diagnostics, GeneralVoFFiniteProjectionProvenance* provenance,
    std::optional<std::span<const long double>> physical_mass,
    const GeneralVoFFiniteProjectionClosureTopology* closure_topology,
    std::optional<std::span<const long double>> physical_mass_error);
