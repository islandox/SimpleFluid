/** @file GeneralVoFFiniteProjection.inc
 * @brief Bounded wide-FP canonical rates and physical nonorthogonal projection.
 */
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
    bool valid = Meshes::bounded_geometry_environment() && finite_arithmetic_ && current_geometry() &&
                 velocity.size() == n && density.size() == n && fraction.size() == n && kappa.size() == n &&
                 face_velocity.size() == geometry_.size() && (!physical_mass || physical_mass->size() == n) &&
                 (!physical_mass_error || (physical_mass && physical_mass_error->size() == n)) && std::isfinite(dt) &&
                 dt > 0 && std::isfinite(sigma) && sigma >= 0 && std::numeric_limits<long double>::is_iec559 &&
                 std::numeric_limits<long double>::digits >= std::numeric_limits<double>::digits &&
                 std::numeric_limits<long double>::digits <= 106 && std::fegetround() == FE_TONEAREST;
    valid = valid && std::isfinite(options.pressure_absolute_tolerance) && options.pressure_absolute_tolerance >= 0 &&
            std::isfinite(options.pressure_relative_tolerance) && options.pressure_relative_tolerance >= 0 &&
            std::isfinite(options.divergence_tolerance) && options.divergence_tolerance > 0 &&
            options.max_pressure_iterations > 0;
    for (std::size_t c = 0; c < n; ++c)
        valid = valid && std::isfinite(density[c]) && density[c] > 0 && std::isfinite(fraction[c]) &&
                fraction[c] >= 0 && fraction[c] <= 1 && std::isfinite(kappa[c]) &&
                (!physical_mass || (std::isfinite((*physical_mass)[c]) && (*physical_mass)[c] > 0)) &&
                (!physical_mass_error || (std::isfinite((*physical_mass_error)[c]) && (*physical_mass_error)[c] >= 0 &&
                                             (*physical_mass)[c] > (*physical_mass_error)[c]));
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
    constexpr auto eps = std::numeric_limits<long double>::epsilon();
    constexpr auto double_eps = std::numeric_limits<double>::epsilon();
    const auto outward = [](long double bound)
    { return bound == 0 ? 0.L : std::nextafter(bound, std::numeric_limits<long double>::infinity()); };
    const auto record_update = [&](std::size_t f, long double increment, long double construction_magnitude)
    {
        const auto old = rate[f];
        rate[f] += increment;
        // gamma_64 bounds the explicitly evaluated binary64 face expression;
        // the wide accumulation has a separate outward operation allowance.
        const long double gamma = 64.L * double_eps / (1 - 64.L * double_eps);
        radius[f] =
            outward(radius[f] + gamma * construction_magnitude + 8 * eps * (std::abs(old) + std::abs(increment)));
        valid = valid && std::isfinite(rate[f]) && std::isfinite(radius[f]) &&
                (increment == 0 || construction_magnitude > 0);
    };
    for (std::size_t f = 0; f < nf; ++f)
    {
        const auto& g = geometry_[f];
        if (g.boundary)
            continue;
        rate[f] = static_cast<long double>(face_velocity[f]) * g.area;
        radius[f] = 4 * eps * std::abs(rate[f]);
        const double mass_length = rho[g.owner] * g.left + rho[g.neighbor] * g.right;
        beta[f] = g.distance / mass_length;
        const double left = fraction_halo[g.owner] + dot(grad_f[g.owner], g.tangent_left);
        const double right = fraction_halo[g.neighbor] + dot(grad_f[g.neighbor], g.tangent_right);
        fraction_face[f] = (rho[g.neighbor] * g.right * left + rho[g.owner] * g.left * right) / mass_length;
        curvature_face[f] = (g.right * curvature_halo[g.owner] + g.left * curvature_halo[g.neighbor]) / g.distance;
        const long double multiplier =
            static_cast<long double>(dt) * beta[f] * sigma * curvature_face[f] * g.area / g.distance;
        const long double capillary = multiplier * (static_cast<long double>(right) - left);
        record_update(f, capillary, std::abs(multiplier) * (std::abs(left) + std::abs(right)));
        valid = valid && std::isfinite(beta[f]) && beta[f] > 0;
    }
    require(valid, "Finite projection face construction exceeds its supported range.");
    beta = faces(beta);
    fraction_face = faces(fraction_face);
    curvature_face = faces(curvature_face);
    std::vector<std::array<long double, 3>> force(n), force_error(n), velocity_error(n);
    const long double force_gamma = 64.L * double_eps / (1 - 64.L * double_eps);
    const auto add_force = [&](std::size_t c, unsigned a, long double contribution, long double expression_magnitude)
    {
        const auto old = force[c][a];
        force[c][a] += contribution;
        force_error[c][a] = outward(force_error[c][a] + force_gamma * expression_magnitude +
                                    8 * eps * (std::abs(old) + std::abs(contribution)));
    };
    {
        const auto lease = mesh_->acquire_execution_view();
        for (std::size_t c = 0; c < n; ++c)
            for (auto fid : mesh_->faces(LO(c)))
            {
                const auto& g = geometry_[fid];
                const long double value =
                    g.boundary ? static_cast<long double>(sigma) * kappa[c] *
                                     dot(grad_f[c], convert(mesh_->face_center_vector(fid, LO(c)))) * g.area
                               : static_cast<long double>(g.owner == LO(c) ? 1 : -1) * g.area * sigma *
                                     curvature_face[fid] * (fraction_face[fid] - fraction[c]);
                long double magnitude = 0;
                if (g.boundary)
                {
                    const auto r = convert(mesh_->face_center_vector(fid, LO(c)));
                    for (unsigned a = 0; a < 3; ++a)
                        magnitude += std::abs(static_cast<long double>(grad_f[c][a]) * r[a]);
                    magnitude *= std::abs(static_cast<long double>(sigma) * kappa[c] * g.area);
                }
                else
                    magnitude = std::abs(static_cast<long double>(g.area) * sigma * curvature_face[fid]) *
                                (std::abs(fraction_face[fid]) + std::abs(fraction[c]));
                for (unsigned a = 0; a < 3; ++a)
                    add_force(c, a, value * g.normal[a], magnitude * std::abs(g.normal[a]));
            }
    }
    std::vector<long double> net(n), cell_radius(n), overlap_rate, overlap_radius;
    const auto refresh_rates = [&]
    {
        overlap_rate = halo_rates(rate);
        overlap_radius = halo_rates(radius);
        std::fill(net.begin(), net.end(), 0);
        std::fill(cell_radius.begin(), cell_radius.end(), 0);
        long double local_divergence = 0;
        const auto lease = mesh_->acquire_execution_view();
        for (std::size_t c = 0; c < n; ++c)
        {
            long double sum = 0, compensation = 0, magnitude = 0, error = 0;
            std::size_t degree = 0;
            for (auto fid : mesh_->faces(LO(c)))
            {
                const auto value = (geometry_[fid].owner == LO(c) ? 1 : -1) * overlap_rate[fid];
                const auto next = sum + value;
                compensation += std::abs(sum) >= std::abs(value) ? (sum - next) + value : (value - next) + sum;
                sum = next;
                magnitude += std::abs(value);
                error += overlap_radius[fid];
                ++degree;
            }
            net[c] = sum + compensation;
            cell_radius[c] = outward(error * (1 + 8 * eps * degree) + 16 * eps * magnitude);
            local_divergence = std::max(local_divergence, std::abs(net[c]) / volumes_[c]);
        }
        return wide_maximum(local_divergence);
    };
    auto residual = refresh_rates();
    const long double limit = std::min(static_cast<long double>(options.divergence_tolerance),
        static_cast<long double>(dt) * options.pressure_absolute_tolerance +
            static_cast<long double>(options.pressure_relative_tolerance) * residual);
    require(std::isfinite(limit) && limit > 0, "Finite projection has invalid original physical threshold.");
    pressure.assign(n, 0);
    std::vector<long double> wide_pressure(n);
    diagnostics.pressure_iterations = 0;
    bool closed = false;
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
            try
            {
                auto closure =
                    close_general_vof_finite_rate(mesh_, rate, radius, cell_radius, budget / 2, closure_topology);
                for (std::size_t f = 0; f < nf; ++f)
                    radius[f] = outward(radius[f] + std::abs(closure.correction[f]) +
                                        8 * eps * (std::abs(rate[f]) + std::abs(closure.rate[f])));
                rate = std::move(closure.rate);
                closure_storage = closure.admitted_storage_bytes;
                residual = refresh_rates();
                closed = residual <= limit;
            }
            catch (const GeneralVoFPhiInfeasible&)
            {
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
        for (std::size_t f = 0; f < nf; ++f)
        {
            const auto& g = geometry_[f];
            if (g.boundary)
                continue;
            const double left = p[g.owner] + dot(grad_p[g.owner], g.tangent_left);
            const double right = p[g.neighbor] + dot(grad_p[g.neighbor], g.tangent_right);
            const long double multiplier = static_cast<long double>(dt) * scale * beta[f] * g.area / g.distance;
            record_update(f, -multiplier * (static_cast<long double>(right) - left),
                std::abs(multiplier) * (std::abs(right) + std::abs(left)));
            p_face[f] = (rho[g.neighbor] * g.right * left + rho[g.owner] * g.left * right) /
                        (rho[g.owner] * g.left + rho[g.neighbor] * g.right);
        }
        require(valid, "Finite pressure correction exceeds its arithmetic range.");
        p_face = faces(p_face);
        {
            const auto lease = mesh_->acquire_execution_view();
            for (std::size_t c = 0; c < n; ++c)
            {
                wide_pressure[c] += scale * p_owned[c];
                for (auto fid : mesh_->faces(LO(c)))
                {
                    const auto& g = geometry_[fid];
                    const double value = g.boundary
                                             ? -dot(grad_p[c], convert(mesh_->face_center_vector(fid, LO(c)))) * g.area
                                             : -(g.owner == LO(c) ? 1 : -1) * (p_face[fid] - p_owned[c]) * g.area;
                    long double magnitude = 0;
                    if (g.boundary)
                    {
                        const auto r = convert(mesh_->face_center_vector(fid, LO(c)));
                        for (unsigned a = 0; a < 3; ++a)
                            magnitude += std::abs(static_cast<long double>(grad_p[c][a]) * r[a]);
                    }
                    else
                        magnitude = std::abs(p_face[fid]) + std::abs(p_owned[c]);
                    magnitude *= std::abs(scale) * g.area;
                    for (unsigned a = 0; a < 3; ++a)
                        add_force(c, a, scale * value * g.normal[a], magnitude * std::abs(g.normal[a]));
                }
            }
        }
        residual = refresh_rates();
    }
    if (!closed)
    {
        std::ostringstream message;
        message << std::setprecision(18)
                << "Finite General VoF projection failed original physical/arithmetic acceptance: residual=" << residual
                << ", limit=" << limit << ", iterations=" << diagnostics.pressure_iterations;
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
            const long double mass =
                physical_mass ? (*physical_mass)[c] : static_cast<long double>(density[c]) * volumes_[c];
            const long double mass_error =
                physical_mass_error ? (*physical_mass_error)[c] : (physical_mass ? 0 : 4 * eps * std::abs(mass));
            const auto mass_lower = mass - mass_error;
            valid = valid && std::isfinite(mass) && mass_lower > 0 && std::isfinite(pressure[c]);
            for (unsigned a = 0; a < 3; ++a)
            {
                namespace fp = VoF::finite_detail;
                const auto impulse =
                    fp::divide(fp::multiply({force[c][a], force_error[c][a]}, {dt, 0}), {mass, mass_error});
                const auto update = fp::add({velocity[c][a], 0}, impulse);
                velocity[c][a] = double(update.value);
                velocity_error[c][a] = fp::bound_add(update.absolute_error, std::abs(update.value - velocity[c][a]));
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
        provenance->face_rate = std::move(rate);
        provenance->face_radius = std::move(radius);
        provenance->cell_radius = std::move(cell_radius);
        provenance->velocity_error = std::move(velocity_error);
        provenance->closure_storage_bytes = closure_storage;
    }
}
