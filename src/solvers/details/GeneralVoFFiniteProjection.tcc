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
    const auto payload_units = 256 * (static_cast<long double>(cells_count) + faces_count + n + nf + 1);
    valid = valid && budget > 0 && payload_units <= budget / 2;
    require(valid,
        "Finite General VoF projection has invalid inputs, unsupported FP environment or exhausted payload cap.");
    int local_requests[3] = {bool(physical_mass), bool(closure_topology), bool(physical_mass_error)},
        minimum_requests[3]{}, maximum_requests[3]{};
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MIN, 3, local_requests, minimum_requests);
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 3, local_requests, maximum_requests);
    require(minimum_requests[0] == maximum_requests[0] && minimum_requests[1] == maximum_requests[1] &&
                minimum_requests[2] == maximum_requests[2],
        "Finite projection requests differ across ranks.");
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
    const auto refresh_rates = [&]
    {
        overlap_rate = halo_rates(rate);
        overlap_radius = halo_rates(radius);
        std::fill(net.begin(), net.end(), 0);
        std::fill(cell_radius.begin(), cell_radius.end(), 0);
        long double local_divergence = 0;
        arithmetic_guard(
            [&]
            {
                const auto lease = mesh_->acquire_execution_view();
                for (std::size_t c = 0; c < n; ++c)
                {
                    fp::SumFor<Mode> row(arithmetic_mode_);
                    for (auto fid : mesh_->faces(LO(c)))
                        row.add(arithmetic_value(
                            (geometry_[fid].owner == LO(c) ? 1 : -1) * overlap_rate[fid], overlap_radius[fid]));
                    const auto sum = row.value();
                    net[c] = sum.value;
                    cell_radius[c] = sum.absolute_error;
                    local_divergence = std::max(local_divergence, std::abs(net[c]) / volumes_[c]);
                }
            },
            "Finite projection row accumulation exceeds its supported arithmetic range.");
        return wide_maximum(local_divergence);
    };
    auto residual = refresh_rates();
    const auto pressure_limit =
        A::add(A::multiply(arithmetic_value(dt, 0), arithmetic_value(options.pressure_absolute_tolerance, 0)),
            A::multiply(arithmetic_value(residual, 0), arithmetic_value(options.pressure_relative_tolerance, 0)));
    const long double limit =
        std::min(static_cast<long double>(options.divergence_tolerance), A::lower(pressure_limit));
    require(std::isfinite(limit) && limit > 0, "Finite projection has invalid original physical threshold.");
    pressure.assign(n, 0);
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
                // The unchanged original physical face-rate residual accepts
                // ordinary FP; no capacity-certified forest repair is requested.
                closed = true;
                break;
            }
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
        throw std::runtime_error(message.str());
    }
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
    diagnostics.maximum_divergence = double(residual);
    diagnostics.pressure_residual = double(residual / dt);
    if (provenance)
    {
        provenance->arithmetic_mode = arithmetic_mode_;
        provenance->face_rate = std::move(rate);
        provenance->face_radius = std::move(radius);
        provenance->cell_radius = std::move(cell_radius);
        provenance->velocity_error = std::move(velocity_error);
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
