/** @file AdaptiveVoFContext.cc
 * @brief Compiled adaptive VoF base operations.
 */
#include "solvers/details/AdaptiveVoFContext.hh"
namespace SimpleFluid::detail::adaptive_vof
{
void require_all(bool valid, const char* message)
{
    const int local = valid ? 1 : 0;
    int global = 0;
    Teuchos::reduceAll(*Tpetra::getDefaultComm(), Teuchos::REDUCE_MIN, 1, &local, &global);
    if (!global)
        throw std::runtime_error(message);
}

double maximum(double local)
{
    if (!std::isfinite(local))
        local = std::numeric_limits<double>::infinity();
    double result = 0.0;
    Teuchos::reduceAll(*Tpetra::getDefaultComm(), Teuchos::REDUCE_MAX, 1, &local, &result);
    return result;
}

double sum(double local)
{
    double result = 0.0;
    Teuchos::reduceAll(*Tpetra::getDefaultComm(), Teuchos::REDUCE_SUM, 1, &local, &result);
    return result;
}

void identical(double value, const char* message)
{
    require_all(std::isfinite(value), message);
    double lo = 0.0, hi = 0.0;
    Teuchos::reduceAll(*Tpetra::getDefaultComm(), Teuchos::REDUCE_MIN, 1, &value, &lo);
    Teuchos::reduceAll(*Tpetra::getDefaultComm(), Teuchos::REDUCE_MAX, 1, &value, &hi);
    require_all(lo == hi, message);
}

void identical_name(const std::string& name)
{
    const auto comm = Tpetra::getDefaultComm();
    int length = comm->getRank() == 0 ? static_cast<int>(name.size()) : 0;
    Teuchos::broadcast(*comm, 0, 1, &length);
    std::string reference(static_cast<std::size_t>(length), '\0');
    if (comm->getRank() == 0)
        reference = name;
    if (length)
        Teuchos::broadcast(*comm, 0, length, reference.data());
    require_all(name == reference, "AdaptiveVoFSolver inventory/checkpoint metadata disagree across ranks.");
}

double dot(const Vector& a, const Vector& b)
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

Vector convert(const MeshUtils::Vec3& v)
{
    return {v.x, v.y, v.z};
}

double mixture(double f, double liquid, double gas)
{
    return liquid * (1.0 - f) + gas * f;
}

double carrier_amount(Carrier carrier, double volume, double fraction, const VoFMaterial& m)
{
    switch (carrier)
    {
        case Carrier::BulkVolume:
            return volume;
        case Carrier::GasVolume:
            return volume * fraction;
        case Carrier::LiquidVolume:
            return volume * (1.0 - fraction);
        case Carrier::MixtureMass:
            return volume * mixture(fraction, m.liquid_density, m.gas_density);
    }
    return std::numeric_limits<double>::quiet_NaN();
}

double carrier_flux(Carrier carrier, double gas, double liquid, const VoFMaterial& m)
{
    switch (carrier)
    {
        case Carrier::BulkVolume:
            return gas + liquid;
        case Carrier::GasVolume:
            return gas;
        case Carrier::LiquidVolume:
            return liquid;
        case Carrier::MixtureMass:
            return m.gas_density * gas + m.liquid_density * liquid;
    }
    return std::numeric_limits<double>::quiet_NaN();
}

void validate_options(const VoFMaterial& m, const VoFSolverOptions& o)
{
    const std::array<double, 15> values{m.liquid_density, m.gas_density, m.liquid_viscosity, m.gas_viscosity,
        m.surface_tension, o.gravity[0], o.gravity[1], o.gravity[2], o.max_courant, o.capillary_safety,
        o.viscous_safety, o.pressure_relative_tolerance, o.pressure_absolute_tolerance, o.divergence_tolerance,
        static_cast<double>(o.max_pressure_iterations)};
    for (double v : values)
        identical(v, "AdaptiveVoFSolver material/options must be finite and identical across ranks.");
    require_all(m.liquid_density > 0 && m.gas_density > 0 && m.liquid_viscosity >= 0 && m.gas_viscosity >= 0 &&
                    m.surface_tension >= 0 && o.max_courant > 0 && o.max_courant <= 0.5 && o.capillary_safety > 0 &&
                    o.capillary_safety <= 1 && o.viscous_safety > 0 && o.viscous_safety <= 1 &&
                    o.pressure_relative_tolerance > 0 && o.pressure_relative_tolerance < 1 &&
                    o.pressure_absolute_tolerance > 0 && o.divergence_tolerance > 0 && o.max_pressure_iterations > 0,
        "AdaptiveVoFSolver has invalid material or numerical controls.");
}

AdaptiveVoFContext::AdaptiveVoFContext(std::shared_ptr<const Hierarchy> h, VoFMaterial m, VoFSolverOptions o)
    : geometry(std::make_shared<VoF::AdaptiveVoF>(std::move(h))), mesh(geometry->mesh()), material(m),
      options(o), projection(mesh), conservative_transport(mesh)
{
    bool valid = true;
    with_geometry(
        [&]
        {
            volume.resize(mesh->num_local_cells());
            spacing.resize(volume.size());
            for (std::size_t c = 0; c < volume.size(); ++c)
            {
                const auto id = static_cast<LO>(c);
                const auto box = geometry->hierarchy()->cell_box(mesh->cell_global_id(id));
                volume[c] = mesh->cell_volume(id);
                for (std::size_t a = 0; a < 3; ++a)
                    spacing[c][a] = box.upper[a] - box.lower[a];
            }
            faces.resize(mesh->num_faces());
            for (std::size_t i = 0; i < faces.size(); ++i)
            {
                const auto id = static_cast<LO>(i);
                auto& f = faces[i];
                f.owner = mesh->owner_cell(id);
                f.neighbor = mesh->neighbor_cell(id);
                f.boundary = mesh->is_geometry_exterior_face(id);
                f.active =
                    f.owner >= 0 && static_cast<std::size_t>(f.owner) < volume.size() &&
                    (f.boundary || (f.neighbor >= 0 && static_cast<std::size_t>(f.neighbor) < volume.size()));
                if (!f.active)
                {
                    if (mesh->is_owned_face(id))
                        valid = false;
                    continue;
                }
                f.area = mesh->face_area(id);
                f.normal = convert(mesh->face_normal(id));
                auto ro = convert(mesh->face_center_vector(id, f.owner));
                f.owner_distance = std::abs(dot(ro, f.normal));
                f.owner_tangent = ro;
                for (std::size_t a = 0; a < 3; ++a)
                    f.owner_tangent[a] -= dot(ro, f.normal) * f.normal[a];
                if (!f.boundary)
                {
                    const auto rn = convert(mesh->face_center_vector(id, f.neighbor));
                    f.neighbor_distance = std::abs(dot(rn, f.normal));
                    f.neighbor_tangent = rn;
                    for (std::size_t a = 0; a < 3; ++a)
                        f.neighbor_tangent[a] -= dot(rn, f.normal) * f.normal[a];
                }
                f.distance = f.owner_distance + f.neighbor_distance;
                valid =
                    valid && std::isfinite(f.distance) && f.distance > 0 && std::isfinite(f.area) && f.area > 0;
            }
            for (std::size_t c = 0; c < owned(); ++c)
            {
                const auto incidence = mesh->faces(static_cast<LO>(c));
                row_capacity = std::max(row_capacity, incidence.size() + 1);
                for (const auto id : incidence)
                    valid = valid && faces[id].active;
            }
        });
    require_all(valid, "AdaptiveVoFSolver native face geometry/overlap is incomplete.");
    GO local = std::numeric_limits<GO>::max();
    for (std::size_t c = 0; c < owned(); ++c)
        local = std::min(local, mesh->cell_global_id(static_cast<LO>(c)));
    Teuchos::reduceAll(*mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_MIN, 1, &local, &gauge);
}

void AdaptiveVoFContext::validate_fraction(const std::vector<double>& f) const
{
    require_all(f.size() == owned() && std::all_of(f.begin(), f.end(), [](double v)
                                           { return std::isfinite(v) && v >= -2e-12 && v <= 1 + 2e-12; }),
        "AdaptiveVoFSolver requires finite owned fractions within retained 2e-12 roundoff of [0,1].");
}

std::vector<double> AdaptiveVoFContext::density(const State& s) const
{
    std::vector<double> rho(owned());
    bool valid = true;
    for (std::size_t c = 0; c < owned(); ++c)
    {
        rho[c] = s.low ? low_mass(*s.low, c) / volume[c]
                       : mixture(s.f[c], material.liquid_density, material.gas_density);
        valid = valid && std::isfinite(rho[c]) && rho[c] > 0;
    }
    require_all(valid, "AdaptiveVoFSolver mixture density is nonpositive/nonfinite.");
    return rho;
}

std::vector<Vector> AdaptiveVoFContext::exchange_vectors(const std::vector<Vector>& values) const
{
    require_all(values.size() == owned(), "AdaptiveVoFSolver vector shape does not match owned cells.");
    return geometry->exchange_cell_vectors(values);
}

std::vector<Vector> AdaptiveVoFContext::gradients(const std::vector<double>& overlap) const
{
    std::vector<Vector> result(owned());
    bool valid = true;
    with_geometry(
        [&]
        {
            for (std::size_t c = 0; c < owned(); ++c)
            {
                std::array<std::array<double, 4>, 3> equation{};
                for (const auto fid : mesh->faces(static_cast<LO>(c)))
                {
                    const auto& f = faces[static_cast<std::size_t>(fid)];
                    if (f.boundary)
                        continue;
                    const auto neighbor = f.owner == static_cast<LO>(c) ? f.neighbor : f.owner;
                    auto r = convert(mesh->cell_center_vector(fid, static_cast<LO>(c)));
                    for (std::size_t a = 0; a < 3; ++a)
                        r[a] /= spacing[c][a];
                    const double weight = 1.0 / dot(r, r), delta = overlap[neighbor] - overlap[c];
                    for (std::size_t a = 0; a < 3; ++a)
                    {
                        equation[a][3] += weight * r[a] * delta;
                        for (std::size_t b = 0; b < 3; ++b)
                            equation[a][b] += weight * r[a] * r[b];
                    }
                }
                bool rank_three = true;
                for (std::size_t a = 0; a < 3; ++a)
                {
                    std::size_t pivot = a;
                    for (std::size_t b = a + 1; b < 3; ++b)
                        if (std::abs(equation[b][a]) > std::abs(equation[pivot][a]))
                            pivot = b;
                    if (std::abs(equation[pivot][a]) < 1e-13)
                    {
                        rank_three = false;
                        break;
                    }
                    std::swap(equation[a], equation[pivot]);
                    const double scale = equation[a][a];
                    for (std::size_t b = a; b < 4; ++b)
                        equation[a][b] /= scale;
                    for (std::size_t row = 0; row < 3; ++row)
                    {
                        if (row == a)
                            continue;
                        const double factor = equation[row][a];
                        for (std::size_t b = a; b < 4; ++b)
                            equation[row][b] -= factor * equation[a][b];
                    }
                }
                valid = valid && rank_three;
                if (rank_three)
                    for (std::size_t a = 0; a < 3; ++a)
                        result[c][a] = equation[a][3] / spacing[c][a];
            }
        });
    require_all(valid, "AdaptiveVoFSolver least-squares gradient stencil is rank deficient.");
    return exchange_vectors(result);
}

std::vector<double> AdaptiveVoFContext::checked_faces(const std::vector<double>& values) const
{
    require_all(values.size() == faces.size(), "AdaptiveVoFSolver face field requires overlap-face ordering.");
    bool valid = true;
    for (std::size_t i = 0; i < mesh->num_owned_faces(); ++i)
        valid = valid && std::isfinite(values[i]) && (!faces[i].boundary || values[i] == 0);
    require_all(valid, "AdaptiveVoFSolver normal face velocities must be finite with impermeable walls.");
    return geometry->exchange_faces(values);
}

std::vector<double> AdaptiveVoFContext::interpolate(const std::vector<Vector>& values) const
{
    const auto overlap = exchange_vectors(values);
    std::array<std::vector<Vector>, 3> gradient;
    for (std::size_t b = 0; b < 3; ++b)
    {
        std::vector<double> component(overlap.size());
        for (std::size_t c = 0; c < component.size(); ++c)
            component[c] = overlap[c][b];
        gradient[b] = gradients(component);
    }
    std::vector<double> result(faces.size(), 0);
    for (std::size_t i = 0; i < mesh->num_owned_faces(); ++i)
    {
        const auto& f = faces[i];
        if (f.boundary)
            continue;
        for (std::size_t b = 0; b < 3; ++b)
        {
            const double a = overlap[f.owner][b] + dot(gradient[b][f.owner], f.owner_tangent);
            const double z = overlap[f.neighbor][b] + dot(gradient[b][f.neighbor], f.neighbor_tangent);
            result[i] += f.normal[b] * (f.neighbor_distance * a + f.owner_distance * z) / f.distance;
        }
    }
    return geometry->exchange_faces(result);
}

std::vector<Vector> AdaptiveVoFContext::average_faces(const std::vector<double>& values) const
{
    std::vector<Vector> result(owned());
    with_geometry(
        [&]
        {
            for (std::size_t c = 0; c < owned(); ++c)
                for (const auto id : mesh->faces(static_cast<LO>(c)))
                {
                    const auto& f = faces[id];
                    const double d = f.owner == static_cast<LO>(c) ? f.owner_distance : f.neighbor_distance;
                    for (std::size_t a = 0; a < 3; ++a)
                        result[c][a] += values[id] * f.normal[a] * f.area * d / volume[c];
                }
        });
    return result;
}

std::vector<double> AdaptiveVoFContext::divergence(const std::vector<double>& velocity) const
{
    std::vector<double> result(owned());
    with_geometry(
        [&]
        {
            for (std::size_t c = 0; c < owned(); ++c)
            {
                detail::CompensatedSum<> net;
                for (const auto fid : mesh->faces(static_cast<LO>(c)))
                {
                    const auto& f = faces[fid];
                    net += (f.owner == static_cast<LO>(c) ? 1.0L : -1.0L) * velocity[fid] * f.area;
                }
                result[c] = static_cast<double>(net.value() / volume[c]);
            }
        });
    return result;
}

double AdaptiveVoFContext::max_abs(const std::vector<double>& values) const
{
    double local = 0;
    for (double v : values)
        local = std::isfinite(v) ? std::max(local, std::abs(v)) : std::numeric_limits<double>::infinity();
    return maximum(local);
}

std::vector<double> AdaptiveVoFContext::extended_curvature(const State& s) const
{
    auto k = s.k;
    std::vector<double> known(owned());
    for (std::size_t c = 0; c < owned(); ++c)
        known[c] = s.f[c] > 2e-12 && s.f[c] < 1 - 2e-12 ? 1 : 0;
    for (int pass = 0; pass < 2; ++pass)
    {
        const auto halo = geometry->exchange_cells(k), mask = geometry->exchange_cells(known);
        with_geometry(
            [&]
            {
                for (std::size_t c = 0; c < owned(); ++c)
                {
                    if (known[c])
                        continue;
                    double total = 0, weight = 0;
                    for (const auto id : mesh->faces(static_cast<LO>(c)))
                    {
                        const auto& f = faces[id];
                        if (f.boundary)
                            continue;
                        const auto n = f.owner == static_cast<LO>(c) ? f.neighbor : f.owner;
                        if (mask[n])
                        {
                            total += f.area * halo[n];
                            weight += f.area;
                        }
                    }
                    if (weight > 0)
                    {
                        k[c] = total / weight;
                        known[c] = 1;
                    }
                }
            });
    }
    return geometry->exchange_cells(k);
}

void AdaptiveVoFContext::project(State& s, const std::vector<double>& rho_owned, double dt,
    const std::vector<Vector>& predictor_change, bool capillary, BoundaryProjection boundary,
    const std::vector<double>* integrated_target ,
    const std::vector<double>* volume_susceptibility ,
    const std::vector<double>* pressure_reference )
{
    require_all(bool(volume_susceptibility) == bool(pressure_reference),
        "AdaptiveVoFSolver volume-pressure response requires a susceptibility and reference pressure.");
    if (volume_susceptibility)
        require_all(volume_susceptibility->size() == owned() && pressure_reference->size() == owned() &&
                        std::all_of(volume_susceptibility->begin(), volume_susceptibility->end(),
                            [](double value) { return std::isfinite(value) && value >= 0; }) &&
                        std::all_of(pressure_reference->begin(), pressure_reference->end(),
                            [](double value) { return std::isfinite(value); }),
            "AdaptiveVoFSolver volume-pressure response has invalid owned storage.");
    const bool compressible = volume_susceptibility && max_abs(*volume_susceptibility) > 0;
    std::array<detail::CompensatedSum<>, 3> pressure_impulse_scale;
    std::vector<std::size_t> pressure_force_terms(s.low ? owned() : 0, 0);
    const auto rho = geometry->exchange_cells(rho_owned);
    const auto fraction = geometry->exchange_cells(s.f);
    const auto grad_f = gradients(fraction);
    const auto k = extended_curvature(s);
    std::vector<double> beta(faces.size()), curvature_face(faces.size()), f_face(faces.size()),
        capillary_face(faces.size());
    for (std::size_t i = 0; i < mesh->num_owned_faces(); ++i)
    {
        const auto& f = faces[i];
        if (f.boundary)
            continue;
        const double mass_length = rho[f.owner] * f.owner_distance + rho[f.neighbor] * f.neighbor_distance;
        beta[i] = f.distance / mass_length;
        const double left = fraction[f.owner] + dot(grad_f[f.owner], f.owner_tangent);
        const double right = fraction[f.neighbor] + dot(grad_f[f.neighbor], f.neighbor_tangent);
        f_face[i] = (rho[f.neighbor] * f.neighbor_distance * left + rho[f.owner] * f.owner_distance * right) /
                    mass_length;
        curvature_face[i] = (f.neighbor_distance * k[f.owner] + f.owner_distance * k[f.neighbor]) / f.distance;
        if (capillary)
            capillary_face[i] = material.surface_tension * curvature_face[i] * (right - left) / f.distance;
    }
    beta = geometry->exchange_faces(beta);
    curvature_face = geometry->exchange_faces(curvature_face);
    f_face = geometry->exchange_faces(f_face);
    capillary_face = geometry->exchange_faces(capillary_face);
    std::vector<Vector> force(owned());
    with_geometry(
        [&]
        {
            for (std::size_t c = 0; c < owned(); ++c)
                for (const auto id : mesh->faces(static_cast<LO>(c)))
                {
                    const auto& f = faces[id];
                    if (f.boundary || !capillary)
                        continue;
                    const double sign = f.owner == static_cast<LO>(c) ? 1 : -1;
                    const double value =
                        sign * f.area * material.surface_tension * curvature_face[id] * (f_face[id] - s.f[c]);
                    for (std::size_t a = 0; a < 3; ++a)
                        force[c][a] += value * f.normal[a];
                }
        });
    for (std::size_t i = 0; i < mesh->num_owned_faces(); ++i)
        s.face_u[i] += dt * beta[i] * capillary_face[i];
    s.face_u = geometry->exchange_faces(s.face_u);
    s.p.assign(owned(), 0);
    auto residual_divergence = [&]
    {
        auto result = divergence(s.face_u);
        if (integrated_target)
            for (std::size_t c = 0; c < owned(); ++c)
                result[c] -= (*integrated_target)[c] / volume[c];
        if (compressible)
            for (std::size_t c = 0; c < owned(); ++c)
                result[c] +=
                    (*volume_susceptibility)[c] * (s.p[c] - (*pressure_reference)[c]) / (dt * volume[c]);
        return result;
    };
    if (integrated_target)
        require_all(integrated_target->size() == owned() &&
                        std::all_of(integrated_target->begin(), integrated_target->end(),
                            [](double v) { return std::isfinite(v); }),
            "AdaptiveVoFSolver continuity target has invalid owned storage.");
    auto div = residual_divergence();
    const double rhs_scale = max_abs(div) / dt;
    require_all(std::isfinite(rhs_scale), "AdaptiveVoFSolver pressure right-hand side is nonfinite.");
    const double tolerance =
        options.pressure_absolute_tolerance + options.pressure_relative_tolerance * rhs_scale;
    const double limit = std::min({options.divergence_tolerance, tolerance * dt, 5e-11 / dt});
    s.diagnostics.pressure_iterations = 0;
    if (max_abs(div) > limit)
    {
        projection.refresh(beta, compressible ? volume_susceptibility : nullptr, dt, gauge);
        const auto& matrix = projection.matrix;
        auto& rhs = projection.rhs;
        auto& solution = projection.solution;
        LinearSolverOptions linear;
        linear.backend = LinearSolverBackend::Cg;
        linear.preconditioner = LinearPreconditioner::Jacobi;
        linear.reuse_preconditioner = true;
        linear.tolerance = 1e-8;
        linear.verbosity = 0;
        // The symmetric two-point matrix preconditions the corrected
        // normal-gradient operator. Actual face divergence, including
        // transverse offsets and the gauge cell, accepts every solve.
        for (int correction = 0; correction < 64 && max_abs(div) > limit; ++correction)
        {
            linear.max_iterations = options.max_pressure_iterations - s.diagnostics.pressure_iterations;
            if (linear.max_iterations <= 0)
                break;
            for (std::size_t c = 0; c < owned(); ++c)
                rhs.replaceLocalValue(
                    static_cast<LO>(c), mesh->cell_global_id(static_cast<LO>(c)) == gauge && !compressible
                                            ? 0
                                            : -div[c] * volume[c] / dt);
            const auto stats = projection.solver.solve_from_zero_with_statistics(
                Teuchos::rcp_implicit_cast<const Pack::operator_type>(matrix), rhs, solution, linear);
            require_all(std::isfinite(stats.achieved_tolerance) && std::isfinite(stats.rhs_norm),
                "AdaptiveVoFSolver pressure algebra returned nonfinite diagnostics.");
            s.diagnostics.pressure_iterations += stats.iterations;
            const auto data = solution.getData();
            std::vector<double> owned_p(data.begin(), data.end());
            require_all(std::all_of(owned_p.begin(), owned_p.end(), [](double v) { return std::isfinite(v); }),
                "AdaptiveVoFSolver pressure correction is nonfinite.");
            const auto p = geometry->exchange_cells(owned_p);
            const auto grad_p = gradients(p);
            std::vector<double> p_face(faces.size()), normal_gradient(faces.size());
            for (std::size_t i = 0; i < mesh->num_owned_faces(); ++i)
            {
                const auto& f = faces[i];
                if (f.boundary)
                    continue;
                const double left = p[f.owner] + dot(grad_p[f.owner], f.owner_tangent);
                const double right = p[f.neighbor] + dot(grad_p[f.neighbor], f.neighbor_tangent);
                normal_gradient[i] = (right - left) / f.distance;
                p_face[i] =
                    (rho[f.neighbor] * f.neighbor_distance * left + rho[f.owner] * f.owner_distance * right) /
                    (rho[f.owner] * f.owner_distance + rho[f.neighbor] * f.neighbor_distance);
                s.face_u[i] -= dt * beta[i] * normal_gradient[i];
            }
            s.face_u = geometry->exchange_faces(s.face_u);
            p_face = geometry->exchange_faces(p_face);
            with_geometry(
                [&]
                {
                    for (std::size_t c = 0; c < owned(); ++c)
                    {
                        s.p[c] += owned_p[c];
                        for (const auto id : mesh->faces(static_cast<LO>(c)))
                        {
                            const auto& f = faces[id];
                            if (s.low)
                                ++pressure_force_terms[c];
                            double value = 0;
                            if (f.boundary)
                            {
                                if (boundary == BoundaryProjection::InitialCell)
                                    value = -dot(grad_p[c],
                                                convert(mesh->face_center_vector(id, static_cast<LO>(c)))) *
                                            f.area;
                            }
                            else
                            {
                                const double sign = f.owner == static_cast<LO>(c) ? 1 : -1;
                                value = -sign * (p_face[id] - owned_p[c]) * f.area;
                            }
                            for (std::size_t a = 0; a < 3; ++a)
                            {
                                force[c][a] += value * f.normal[a];
                                if (s.low && !f.boundary)
                                    pressure_impulse_scale[a] += dt * f.area * std::abs(f.normal[a]) *
                                                                 (std::abs(p_face[id]) + std::abs(owned_p[c]));
                            }
                        }
                    }
                });
            div = residual_divergence();
        }
    }
    const double residual = max_abs(div);
    require_all(std::isfinite(residual) && residual <= limit,
        "AdaptiveVoFSolver corrected pressure/continuity residual exceeded its independent tolerance.");
    double momentum_operation_bound = 0;
    with_geometry(
        [&]
        {
            for (std::size_t c = 0; c < owned(); ++c)
            {
                if (s.low)
                    momentum_operation_bound = std::max(momentum_operation_bound,
                        static_cast<double>(pressure_force_terms[c] +
                                            2 * mesh->faces(static_cast<LO>(c)).size() + 16 +
                                            4 * s.low->dispersed.size()));
                if (boundary == BoundaryProjection::Step)
                    for (const auto id : mesh->faces(static_cast<LO>(c)))
                    {
                        const auto& f = faces[id];
                        if (!f.boundary)
                            continue;
                        const double reaction =
                            -rho_owned[c] * f.owner_distance * f.area * dot(predictor_change[c], f.normal) / dt;
                        for (std::size_t a = 0; a < 3; ++a)
                            force[c][a] += reaction * f.normal[a];
                    }
                for (std::size_t a = 0; a < 3; ++a)
                    s.u[c][a] += dt * force[c][a] / (rho_owned[c] * volume[c]);
            }
        });
    if (s.low)
    {
        for (std::size_t a = 0; a < 3; ++a)
            s.low_pressure_impulse_scale[a] = sum(static_cast<double>(pressure_impulse_scale[a].value()));
        s.low_momentum_operation_bound = maximum(momentum_operation_bound);
    }
    s.diagnostics.maximum_divergence = residual;
    s.diagnostics.pressure_residual = residual / dt;
}

std::vector<Vector> AdaptiveVoFContext::viscous_force(const State& s, Vector* traction_scale ) const
{
    if (traction_scale)
        *traction_scale = {};
    std::vector<Vector> force(owned());
    if (!s.low && material.liquid_viscosity == 0 && material.gas_viscosity == 0)
        return force;
    const auto u = exchange_vectors(s.u);
    const auto f = geometry->exchange_cells(s.f);
    std::vector<double> viscosity;
    if (s.low)
        viscosity = geometry->exchange_cells(low_viscosity(s));
    std::array<std::vector<Vector>, 3> gradient;
    for (std::size_t b = 0; b < 3; ++b)
    {
        std::vector<double> component(u.size());
        for (std::size_t c = 0; c < u.size(); ++c)
            component[c] = u[c][b];
        gradient[b] = gradients(component);
    }
    std::array<std::vector<double>, 3> traction;
    for (auto& values : traction)
        values.assign(faces.size(), 0);
    for (std::size_t i = 0; i < mesh->num_owned_faces(); ++i)
    {
        const auto& face = faces[i];
        const double mu_o = s.low ? viscosity[face.owner]
                                  : mixture(f[face.owner], material.liquid_viscosity, material.gas_viscosity);
        if (face.boundary)
        {
            const double normal = -2 * mu_o * dot(u[face.owner], face.normal) / face.owner_distance;
            for (std::size_t b = 0; b < 3; ++b)
                traction[b][i] = normal * face.normal[b];
            continue;
        }
        const double mu =
            0.5 *
            (mu_o + (s.low ? viscosity[face.neighbor]
                           : mixture(f[face.neighbor], material.liquid_viscosity, material.gas_viscosity)));
        for (std::size_t b = 0; b < 3; ++b)
        {
            const double left = u[face.owner][b] + dot(gradient[b][face.owner], face.owner_tangent);
            const double right = u[face.neighbor][b] + dot(gradient[b][face.neighbor], face.neighbor_tangent);
            const double normal_derivative = (right - left) / face.distance;
            double transpose = 0;
            if (std::abs(face.normal[b]) > 0.5)
                transpose = normal_derivative; // tau_nn = 2 mu du_n/dn.
            else
                for (std::size_t a = 0; a < 3; ++a)
                    transpose += face.normal[a] *
                                 (face.neighbor_distance * gradient[a][face.owner][b] +
                                     face.owner_distance * gradient[a][face.neighbor][b]) /
                                 face.distance;
            traction[b][i] = mu * (normal_derivative + transpose);
        }
    }
    std::vector<Vector> packed_traction(faces.size());
    for (std::size_t id = 0; id < faces.size(); ++id)
        for (std::size_t a = 0; a < 3; ++a)
            packed_traction[id][a] = traction[a][id];
    packed_traction = geometry->exchange_face_vectors(packed_traction);
    for (std::size_t id = 0; id < faces.size(); ++id)
        for (std::size_t a = 0; a < 3; ++a)
            traction[a][id] = packed_traction[id][a];
    if (traction_scale)
    {
        std::array<detail::CompensatedSum<>, 3> local;
        for (std::size_t id = 0; id < mesh->num_owned_faces(); ++id)
            for (std::size_t a = 0; a < 3; ++a)
                local[a] += 2 * faces[id].area * std::abs(traction[a][id]);
        for (std::size_t a = 0; a < 3; ++a)
            (*traction_scale)[a] = sum(static_cast<double>(local[a].value()));
    }
    with_geometry(
        [&]
        {
            for (std::size_t c = 0; c < owned(); ++c)
                for (const auto id : mesh->faces(static_cast<LO>(c)))
                {
                    const auto& f = faces[id];
                    const double sign = f.owner == static_cast<LO>(c) ? 1 : -1;
                    for (std::size_t b = 0; b < 3; ++b)
                        force[c][b] += sign * f.area * traction[b][id];
                }
        });
    return force;
}

double AdaptiveVoFContext::inventory_total(const Inventory& q) const
{
    detail::CompensatedSum<> total;
    for (double value : q.amount)
        total += value;
    return sum(static_cast<double>(total.value()));
}

double AdaptiveVoFContext::inventory_l1(const Inventory& q) const
{
    detail::CompensatedSum<> total;
    for (double value : q.amount)
        total += std::abs(value);
    return sum(static_cast<double>(total.value()));
}

Vector AdaptiveVoFContext::momentum_l1(const State& s) const
{
    std::array<detail::CompensatedSum<>, 3> local;
    for (std::size_t c = 0; c < owned(); ++c)
    {
        const double mass = volume[c] * mixture(s.f[c], material.liquid_density, material.gas_density);
        for (std::size_t a = 0; a < 3; ++a)
            local[a] += static_cast<long double>(mass) * std::abs(s.u[c][a]);
    }
    Vector result{};
    for (std::size_t a = 0; a < 3; ++a)
        result[a] = sum(static_cast<double>(local[a].value()));
    return result;
}

void AdaptiveVoFContext::validate_inventory(const Inventory& q, const State& s) const
{
    require_all(!q.name.empty() && q.name.size() < 1000000,
        "AdaptiveVoFSolver inventory requires a bounded nonempty name.");
    identical_name(q.name);
    identical(
        static_cast<double>(static_cast<int>(q.carrier)), "AdaptiveVoFSolver inventory carriers disagree.");
    identical(q.nonnegative ? 1 : 0, "AdaptiveVoFSolver inventory sign policy disagrees.");
    bool valid =
        q.amount.size() == owned() && static_cast<int>(q.carrier) >= 0 && static_cast<int>(q.carrier) <= 3;
    if (q.amount.size() == owned())
        for (std::size_t c = 0; c < owned(); ++c)
        {
            const double capacity = carrier_amount(q.carrier, volume[c], s.f[c], material);
            valid = valid && std::isfinite(q.amount[c]) && (!q.nonnegative || q.amount[c] >= 0) &&
                    (capacity > 0 || q.amount[c] == 0);
        }
    require_all(valid, "AdaptiveVoFSolver inventory is invalid or has a nonzero amount without its carrier.");
}

std::vector<Inventory> AdaptiveVoFContext::transport_inventories(
    const State& old, const State& next, const VoF::AdaptiveAdvection& transport)
{
    auto result = old.inventories;
    for (auto& q : result)
    {
        std::vector<double> capacity(owned()), flux(faces.size()), zero(owned());
        for (std::size_t c = 0; c < owned(); ++c)
            capacity[c] = std::max(0.0, carrier_amount(q.carrier, volume[c], next.f[c], material));
        for (std::size_t id = 0; id < faces.size(); ++id)
            flux[id] = carrier_flux(q.carrier, transport.gas_volume[id], transport.liquid_volume[id], material);
        // Geometric phase tails can have subnormal positive storage.
        // Equilibrate rows without discarding any inventory; retain
        // the original physical residual and global amount gates.
        try
        {
            auto carried = conservative_transport.solve(q.amount, zero, capacity, flux, q.nonnegative,
                options.max_pressure_iterations, 1e-12, true);
            q.amount = std::move(carried.amount);
        }
        catch (const std::exception& error)
        {
            throw std::runtime_error("AdaptiveVoFSolver passive inventory " + q.name +
                " step=" + std::to_string(next.steps) + ": " + error.what());
        }
        const auto original = std::find_if(
            old.inventories.begin(), old.inventories.end(), [&](const auto& x) { return x.name == q.name; });
        const double before = inventory_total(*original), after = inventory_total(q);
        const double scale = inventory_l1(*original) + inventory_l1(q);
        require_all(std::abs(after - before) <= 2e-10 * std::max(scale, 1e-30),
            "AdaptiveVoFSolver passive transport failed its global inventory gate.");
    }
    return result;
}

void AdaptiveVoFContext::budget(State& s) const
{
    if (s.low)
    {
        budget_low(s);
        return;
    }
    validate_fraction(s.f);
    bool valid = s.u.size() == owned() && s.p.size() == owned() && s.k.size() == owned();
    detail::CompensatedSum<> gas, liquid, kinetic;
    std::array<detail::CompensatedSum<>, 3> momentum;
    double speed = 0;
    if (valid)
        for (std::size_t c = 0; c < owned(); ++c)
        {
            const double rho = mixture(s.f[c], material.liquid_density, material.gas_density);
            gas += static_cast<long double>(s.f[c]) * volume[c];
            liquid += static_cast<long double>(1 - s.f[c]) * volume[c];
            double squared = 0;
            valid = valid && std::isfinite(s.p[c]) && std::isfinite(s.k[c]);
            for (std::size_t a = 0; a < 3; ++a)
            {
                valid = valid && std::isfinite(s.u[c][a]);
                squared += s.u[c][a] * s.u[c][a];
                momentum[a] += static_cast<long double>(rho) * volume[c] * s.u[c][a];
            }
            kinetic += static_cast<long double>(0.5 * rho) * volume[c] * squared;
            speed = std::max(speed, std::sqrt(squared));
        }
    require_all(valid, "AdaptiveVoFSolver candidate state is nonfinite or has invalid shapes.");
    s.face_u = checked_faces(s.face_u);
    s.diagnostics.gas_volume = sum(static_cast<double>(gas.value()));
    s.diagnostics.gas_mass = material.gas_density * s.diagnostics.gas_volume;
    s.diagnostics.liquid_mass = material.liquid_density * sum(static_cast<double>(liquid.value()));
    s.diagnostics.kinetic_energy = sum(static_cast<double>(kinetic.value()));
    s.diagnostics.maximum_velocity = maximum(speed);
    s.diagnostics.maximum_divergence = max_abs(divergence(s.face_u));
    for (std::size_t a = 0; a < 3; ++a)
        s.diagnostics.momentum[a] = sum(static_cast<double>(momentum[a].value()));
    const auto& d = s.diagnostics;
    require_all(
        std::isfinite(d.time) && std::isfinite(d.gas_mass) && std::isfinite(d.liquid_mass) &&
            std::isfinite(d.kinetic_energy) && std::isfinite(d.maximum_velocity) &&
            std::isfinite(d.pressure_residual) && std::isfinite(d.maximum_divergence) &&
            std::all_of(d.momentum.begin(), d.momentum.end(), [](double v) { return std::isfinite(v); }),
        "AdaptiveVoFSolver accepted-state budget overflowed.");
}

double AdaptiveVoFContext::stable_timestep(const State& s) const
{
    if (s.low)
        return stable_low_timestep(s);
    double local_rate = 0, local_h_inverse = 0, viscous_rate = 0, local_gas = 0, local_liquid = 0,
           subface_rate = 0;
    const double nu = std::max(material.liquid_viscosity, material.gas_viscosity) /
                      std::min(material.liquid_density, material.gas_density);
    with_geometry(
        [&]
        {
            for (std::size_t c = 0; c < owned(); ++c)
            {
                std::array<double, 3> low{}, high{};
                for (const auto id : mesh->faces(static_cast<LO>(c)))
                {
                    const auto& f = faces[id];
                    const double sign = f.owner == static_cast<LO>(c) ? 1 : -1;
                    for (std::size_t a = 0; a < 3; ++a)
                        if (std::abs(f.normal[a]) > 0.5)
                        {
                            (sign * f.normal[a] > 0 ? high[a] : low[a]) +=
                                std::abs(s.face_u[id]) * f.area / volume[c];
                            subface_rate = std::max(subface_rate, std::abs(s.face_u[id]) / spacing[c][a]);
                        }
                }
                double rate = 0, diffusion = 0;
                for (std::size_t a = 0; a < 3; ++a)
                {
                    rate += std::max(low[a], high[a]);
                    local_h_inverse = std::max(local_h_inverse, 1.0 / spacing[c][a]);
                    diffusion += 1.0 / (spacing[c][a] * spacing[c][a]);
                }
                local_rate = std::max(local_rate, rate);
                // A conservative factor covers transverse reconstruction at
                // balanced subfaces in the explicit symmetric stress operator.
                viscous_rate = std::max(viscous_rate, 8 * nu * diffusion);
                local_gas = std::max(local_gas, s.f[c]);
                local_liquid = std::max(local_liquid, 1 - s.f[c]);
            }
        });
    const double rate = maximum(std::max(local_rate, subface_rate)), diffusion = maximum(viscous_rate),
                 inverse_h = maximum(local_h_inverse);
    const double gas = maximum(local_gas), liquid = maximum(local_liquid);
    double dt = rate > 0 ? options.max_courant / rate : std::numeric_limits<double>::infinity();
    if (diffusion > 0)
        dt = std::min(dt, options.viscous_safety / diffusion);
    if (material.surface_tension > 0 && gas > 0 && liquid > 0)
        dt =
            std::min(dt, options.capillary_safety * std::sqrt((material.liquid_density + material.gas_density) /
                                                              (2 * std::numbers::pi * material.surface_tension *
                                                                  inverse_h * inverse_h * inverse_h)));
    return dt;
}

AdaptiveVoFContext::State AdaptiveVoFContext::advance(const State& old, double dt)
{
    if (old.low)
        return advance_low(old, dt, {}, nullptr);
    identical(dt, "AdaptiveVoFSolver timestep must be finite and identical across ranks.");
    require_all(dt > 0 && dt <= stable_timestep(old) * (1 + 8 * std::numeric_limits<double>::epsilon()),
        "AdaptiveVoFSolver timestep exceeds its explicit stability limit or is nonpositive.");
    const auto old_rho = density(old);
    const auto transport = geometry->advect(old.f, old.face_u, dt, old.steps % 3);
    State next;
    next.f = transport.fraction;
    next.p = old.p;
    next.k = geometry->curvature(next.f);
    next.u = old.u;
    next.face_u = old.face_u;
    next.steps = old.steps + 1;
    next.diagnostics.time = old.diagnostics.time + dt;
    require_all(std::isfinite(next.diagnostics.time) && next.diagnostics.time > old.diagnostics.time,
        "AdaptiveVoFSolver timestep does not produce finite advancing time.");
    const auto new_rho = density(next);
    std::vector<double> mass_flux(faces.size()), capacity(owned()), zero(owned());
    for (std::size_t id = 0; id < faces.size(); ++id)
        mass_flux[id] = material.gas_density * transport.gas_volume[id] +
                        material.liquid_density * transport.liquid_volume[id];
    for (std::size_t c = 0; c < owned(); ++c)
        capacity[c] = new_rho[c] * volume[c];
    for (std::size_t a = 0; a < 3; ++a)
    {
        std::vector<double> momentum(owned());
        for (std::size_t c = 0; c < owned(); ++c)
            momentum[c] = old_rho[c] * volume[c] * old.u[c][a];
        const auto carried = conservative_transport.solve(momentum, zero, capacity, mass_flux, false,
            options.max_pressure_iterations, 1e-14);
        for (std::size_t c = 0; c < owned(); ++c)
            next.u[c][a] = carried.amount[c] / capacity[c];
    }
    const auto viscous = viscous_force(old);
    std::vector<Vector> change(owned());
    for (std::size_t c = 0; c < owned(); ++c)
        for (std::size_t a = 0; a < 3; ++a)
        {
            next.u[c][a] += dt * (viscous[c][a] / capacity[c] + options.gravity[a]);
            change[c][a] = next.u[c][a] - old.u[c][a];
        }
    const auto face_change = interpolate(change);
    for (std::size_t i = 0; i < next.face_u.size(); ++i)
        next.face_u[i] += face_change[i];
    project(next, new_rho, dt, change, true, BoundaryProjection::Step);
    next.inventories = transport_inventories(old, next, transport);
    budget(next);
    const auto& config = geometry->hierarchy()->config();
    const double domain_volume = config.extent[0] * config.extent[1] * config.extent[2];
    require_all(std::abs(next.diagnostics.gas_volume - old.diagnostics.gas_volume) <= 5e-12 * domain_volume,
        "AdaptiveVoFSolver geometric transport failed its gas-volume gate.");
    if (material.surface_tension == 0)
    {
        const auto old_l1 = momentum_l1(old), new_l1 = momentum_l1(next);
        const double mass = next.diagnostics.gas_mass + next.diagnostics.liquid_mass;
        for (std::size_t a = 0; a < 3; ++a)
            if (config.periodic[a])
            {
                const double impulse = dt * mass * options.gravity[a];
                const double scale = old_l1[a] + new_l1[a] + std::abs(impulse);
                require_all(std::abs(next.diagnostics.momentum[a] - old.diagnostics.momentum[a] - impulse) <=
                                5e-11 * std::max(scale, 1e-30),
                    "AdaptiveVoFSolver periodic momentum failed its external-impulse balance.");
            }
    }
    return next;
}

} // namespace SimpleFluid::detail::adaptive_vof
