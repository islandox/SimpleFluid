/**
 * @file VoFSolver.cc
 * @brief Conservative fixed-grid two-phase flow baseline.
 */
#include "solvers/VoFSolver.hh"

#include "solvers/BelosLinearSolver.hh"
#include "solvers/details/ConservativeVoFTransport.hh"
#include "utils/CompensatedSum.hh"

#include <Teuchos_Array.hpp>
#include <Tpetra_Core.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <map>
#include <numbers>
#include <stdexcept>
#include <string>
#include <utility>

namespace SimpleFluid
{
namespace
{
using Grid = VoF::UniformGrid;
using Vector = VoF::Vector;
using Faces = VoF::FaceField;
using Pack = DefaultTpetraTypes;
using Index = std::array<std::size_t, 3>;
constexpr auto missing = std::numeric_limits<std::size_t>::max();

Faces make_faces(const Grid& g)
{
    Faces result;
    for (std::size_t a = 0; a < 3; ++a)
        result[a].resize(g.face_count(a), 0.0);
    return result;
}

std::size_t cell(const Grid& g, const Index& i)
{
    return g.cell_index(i[0], i[1], i[2]);
}

std::size_t face(const Grid& g, std::size_t a, const Index& i)
{
    return g.face_index(a, i[0], i[1], i[2]);
}

Index face_coordinates(const Grid& g, std::size_t a, std::size_t id)
{
    auto dims = g.cells;
    ++dims[a];
    return {id % dims[0], (id / dims[0]) % dims[1], id / (dims[0] * dims[1])};
}

std::pair<std::size_t, std::size_t> adjacent(const Grid& g, std::size_t a, Index i)
{
    auto left = i;
    auto right = i;
    if (i[a] == 0)
    {
        left[a] = g.cells[a] - 1;
        return {g.periodic[a] ? cell(g, left) : missing, cell(g, right)};
    }
    if (i[a] == g.cells[a])
    {
        --left[a];
        right[a] = 0;
        return {cell(g, left), g.periodic[a] ? cell(g, right) : missing};
    }
    --left[a];
    return {cell(g, left), cell(g, right)};
}

double mixture(double f, double liquid, double gas)
{
    // This form avoids subtracting two nearly equal liquid densities at f=1.
    return liquid * (1.0 - f) + gas * f;
}

void validate_material(const VoFMaterial& m, const VoFSolverOptions& o)
{
    const auto positive = [](double v) { return std::isfinite(v) && v > 0.0; };
    const auto nonnegative = [](double v) { return std::isfinite(v) && v >= 0.0; };
    if (!positive(m.liquid_density) || !positive(m.gas_density) || !nonnegative(m.liquid_viscosity) ||
        !nonnegative(m.gas_viscosity) || !nonnegative(m.surface_tension))
        throw std::invalid_argument(
            "VoFSolver requires positive finite densities and nonnegative finite viscosity/surface tension.");
    if (!positive(o.max_courant) || o.max_courant > 0.5 || !positive(o.capillary_safety) || o.capillary_safety > 1.0 ||
        !positive(o.viscous_safety) || o.viscous_safety > 1.0 || !positive(o.pressure_relative_tolerance) ||
        o.pressure_relative_tolerance >= 1.0 || !positive(o.pressure_absolute_tolerance) ||
        !positive(o.divergence_tolerance) || o.max_pressure_iterations <= 0 ||
        !std::all_of(o.gravity.begin(), o.gravity.end(), [](double v) { return std::isfinite(v); }))
        throw std::invalid_argument("VoFSolver has invalid timestep, projection, or gravity controls.");
}

void validate_fraction(const Grid& g, const std::vector<double>& f)
{
    if (f.size() != g.cell_count() || !std::all_of(f.begin(), f.end(), [](double v)
                                          { return std::isfinite(v) && v >= -2.0e-12 && v <= 1.0 + 2.0e-12; }))
        throw std::invalid_argument(
            "VoFSolver initial fractions must cover the grid and be finite in [0,1] within 2e-12 retained roundoff.");
}

void validate_faces(const Grid& g, const Faces& u)
{
    for (std::size_t a = 0; a < 3; ++a)
    {
        if (u[a].size() != g.face_count(a))
            throw std::invalid_argument("VoFSolver face velocity shape does not match the grid.");
        for (std::size_t id = 0; id < u[a].size(); ++id)
        {
            if (!std::isfinite(u[a][id]))
                throw std::invalid_argument("VoFSolver face velocities must be finite.");
            auto i = face_coordinates(g, a, id);
            if (i[a] != 0 && i[a] != g.cells[a])
                continue;
            if (!g.periodic[a] && u[a][id] != 0.0)
                throw std::invalid_argument("VoFSolver impermeable walls require zero normal face velocity.");
            if (g.periodic[a] && i[a] == g.cells[a])
            {
                i[a] = 0;
                if (u[a][id] != u[a][face(g, a, i)])
                    throw std::invalid_argument("VoFSolver duplicated periodic face velocities must agree exactly.");
            }
        }
    }
}

std::vector<double> divergence(const Grid& g, const Faces& u)
{
    std::vector<double> result(g.cell_count(), 0.0);
    for (std::size_t c = 0; c < result.size(); ++c)
    {
        const auto low = g.cell_coordinates(c);
        for (std::size_t a = 0; a < 3; ++a)
        {
            auto high = low;
            ++high[a];
            result[c] += (u[a][face(g, a, high)] - u[a][face(g, a, low)]) / g.spacing[a];
        }
    }
    return result;
}

double maximum_abs(const std::vector<double>& x)
{
    double result = 0.0;
    for (double v : x)
    {
        if (!std::isfinite(v))
            return std::numeric_limits<double>::infinity();
        result = std::max(result, std::abs(v));
    }
    return result;
}

std::vector<Vector> average_faces(const Grid& g, const Faces& u)
{
    std::vector<Vector> result(g.cell_count());
    for (std::size_t c = 0; c < result.size(); ++c)
    {
        const auto low = g.cell_coordinates(c);
        for (std::size_t a = 0; a < 3; ++a)
        {
            auto high = low;
            ++high[a];
            result[c][a] = 0.5 * (u[a][face(g, a, low)] + u[a][face(g, a, high)]);
        }
    }
    return result;
}

Faces interpolate_cells(const Grid& g, const std::vector<Vector>& u)
{
    auto result = make_faces(g);
    for (std::size_t a = 0; a < 3; ++a)
        for (std::size_t id = 0; id < result[a].size(); ++id)
        {
            const auto [l, r] = adjacent(g, a, face_coordinates(g, a, id));
            if (l != missing && r != missing)
                result[a][id] = 0.5 * (u[l][a] + u[r][a]);
        }
    return result;
}

} // namespace

struct VoFSolver::Impl
{
    struct State
    {
        std::vector<double> f, p, k;
        std::vector<Vector> u;
        Faces face_u;
        VoFStepDiagnostics diagnostics;
        std::size_t steps = 0;
    };

    struct Published
    {
        State state;
        ScalarField f;
        ScalarField p;
        VectorField u;

        Published(State candidate, const SP<const Handle>& mesh)
            : state(std::move(candidate)), f(mesh, "resolvedGasFraction", false), p(mesh, "pressure", false), u(mesh, "velocity", false)
        {
            for (std::size_t c = 0; c < state.f.size(); ++c)
            {
                const auto lid = static_cast<Pack::local_ordinal_type>(c);
                f.set_owned_value(lid, state.f[c]);
                p.set_owned_value(lid, state.p[c]);
                u.set_owned_value(lid, vec3<double>{state.u[c][0], state.u[c][1], state.u[c][2]});
            }
            f.sync_ghosts();
            p.sync_ghosts();
            u.sync_ghosts();
        }
    };

    Grid grid;
    VoFMaterial material;
    VoFSolverOptions options;
    VoF::GeometricVoF geometry;
    SP<const Handle> mesh;
    std::unique_ptr<Published> published;
    BelosLinearSolver<> pressure_solver;
    std::unique_ptr<detail::ConservativeVoFTransport> momentum_transport;

    Impl(Grid g, VoFMaterial m, VoFSolverOptions o) : grid(g), material(m), options(o), geometry(g)
    {
        // This check precedes all communicator-dependent native mesh/field work.
        if (Tpetra::getDefaultComm()->getSize() != 1)
            throw std::invalid_argument(
                "VoFSolver stage 1 supports a single MPI rank only; distributed VoF is not implemented.");
        grid.validate();
        validate_material(material, options);
        Vec3D<ArrReal> edges;
        for (std::size_t a = 0; a < 3; ++a)
        {
            if (grid.periodic[a] && grid.cells[a] < 2)
                throw std::invalid_argument("VoFSolver periodic native mesh axes require at least two cells.");
            edges[a].resize(grid.cells[a] + 1);
            for (std::size_t i = 0; i <= grid.cells[a]; ++i)
                edges[a][i] = grid.origin[a] + static_cast<double>(i) * grid.spacing[a];
        }
        SP<const Meshes::OrthogonalCartesian3D> cartesian = std::make_shared<const Meshes::OrthogonalCartesian3D>(
            edges, Vec3D<bool>{grid.periodic[0], grid.periodic[1], grid.periodic[2]});
        mesh = std::make_shared<Handle>(std::move(cartesian));
        momentum_transport = std::make_unique<detail::ConservativeVoFTransport>(mesh);
        State initial;
        initial.f.resize(grid.cell_count(), 0.0);
        initial.p.resize(grid.cell_count(), 0.0);
        initial.k.resize(grid.cell_count(), 0.0);
        initial.u.resize(grid.cell_count());
        initial.face_u = make_faces(grid);
        budget(initial);
        published = std::make_unique<Published>(std::move(initial), mesh);
    }

    std::vector<double> density(const State& s) const
    {
        std::vector<double> rho(s.f.size());
        for (std::size_t c = 0; c < rho.size(); ++c)
        {
            rho[c] = mixture(s.f[c], material.liquid_density, material.gas_density);
            if (!(rho[c] > 0.0) || !std::isfinite(rho[c]))
                throw std::runtime_error("VoFSolver reconstructed nonpositive or nonfinite density.");
        }
        return rho;
    }

    Faces surface_force(const State& s) const
    {
        auto force = make_faces(grid);
        if (material.surface_tension == 0.0)
            return force;
        // Curvature belongs to interface cells. Extend locally into the pure
        // neighbors used by the face gradient; never average it with pure-cell zero.
        std::vector<double> extended = s.k;
        std::vector<bool> known(s.f.size());
        for (std::size_t c = 0; c < s.f.size(); ++c)
            known[c] = s.f[c] > 2.0e-12 && s.f[c] < 1.0 - 2.0e-12;
        for (int pass = 0; pass < 2; ++pass)
        {
            const auto previous = extended;
            const auto old_known = known;
            for (std::size_t c = 0; c < s.f.size(); ++c)
            {
                if (old_known[c])
                    continue;
                double sum = 0.0;
                int count = 0;
                const auto xyz = grid.cell_coordinates(c);
                for (std::size_t a = 0; a < 3; ++a)
                    for (int direction : {-1, 1})
                    {
                        auto neighbor = xyz;
                        if (direction < 0)
                        {
                            if (neighbor[a] == 0 && !grid.periodic[a])
                                continue;
                            neighbor[a] = (neighbor[a] + grid.cells[a] - 1) % grid.cells[a];
                        }
                        else
                        {
                            if (neighbor[a] + 1 == grid.cells[a] && !grid.periodic[a])
                                continue;
                            neighbor[a] = (neighbor[a] + 1) % grid.cells[a];
                        }
                        const auto n = cell(grid, neighbor);
                        if (old_known[n])
                        {
                            sum += previous[n];
                            ++count;
                        }
                    }
                if (count != 0)
                {
                    extended[c] = sum / static_cast<double>(count);
                    known[c] = true;
                }
            }
        }
        for (std::size_t a = 0; a < 3; ++a)
            for (std::size_t id = 0; id < force[a].size(); ++id)
            {
                const auto [l, r] = adjacent(grid, a, face_coordinates(grid, a, id));
                if (l == missing || r == missing)
                    continue;
                double k = 0.0;
                if (known[l] && known[r])
                    k = 0.5 * (extended[l] + extended[r]);
                else if (known[l])
                    k = extended[l];
                else if (known[r])
                    k = extended[r];
                force[a][id] = material.surface_tension * k * (s.f[r] - s.f[l]) / grid.spacing[a];
            }
        return force;
    }

    void project(State& s, const std::vector<double>& rho, double dt, const Faces& capillary,
        const std::vector<Vector>& predictor_change, bool extrapolate_initial_wall_pressure = false)
    {
        auto beta = make_faces(grid);
        for (std::size_t a = 0; a < 3; ++a)
            for (std::size_t id = 0; id < beta[a].size(); ++id)
            {
                const auto [l, r] = adjacent(grid, a, face_coordinates(grid, a, id));
                if (l == missing || r == missing)
                {
                    beta[a][id] = 1.0 / rho[l == missing ? r : l];
                    continue;
                }
                beta[a][id] = 1.0 / (0.5 * rho[l] + 0.5 * rho[r]);
                s.face_u[a][id] += dt * beta[a][id] * capillary[a][id];
            }
        auto div = divergence(grid, s.face_u);
        const double rhs_max = maximum_abs(div) / dt;
        if (!std::isfinite(rhs_max))
            throw std::runtime_error("VoFSolver pressure right-hand side is nonfinite.");
        const double requested = options.pressure_absolute_tolerance + options.pressure_relative_tolerance * rhs_max;
        const double divergence_limit = std::min({options.divergence_tolerance, requested * dt, 5.0e-11 / dt});
        s.p.assign(s.f.size(), 0.0);
        s.diagnostics.pressure_iterations = 0;
        auto force = capillary;
        if (maximum_abs(div) > divergence_limit)
        {
            const auto map = mesh->owned_cell_map();
            auto matrix = Teuchos::rcp(new Pack::matrix_type(map, 7));
            Pack::vector_type rhs(map, true);
            Pack::vector_type solution(map, true);
            for (std::size_t c = 0; c < s.f.size(); ++c)
            {
                std::map<Pack::global_ordinal_type, double> entries;
                const auto gid = static_cast<Pack::global_ordinal_type>(c);
                if (c == 0)
                    entries[gid] = 1.0;
                else
                {
                    const auto low = grid.cell_coordinates(c);
                    double diagonal = 0.0;
                    for (std::size_t a = 0; a < 3; ++a)
                        for (int side : {0, 1})
                        {
                            auto xyz = low;
                            xyz[a] += static_cast<std::size_t>(side);
                            const auto fid = face(grid, a, xyz);
                            const auto [l, r] = adjacent(grid, a, xyz);
                            const auto n = side == 0 ? l : r;
                            if (n == missing || n == c)
                                continue;
                            const double coefficient = beta[a][fid] / (grid.spacing[a] * grid.spacing[a]);
                            diagonal += coefficient;
                            // Symmetric elimination of the fixed p[0]=0 gauge.
                            if (n != 0)
                                entries[static_cast<Pack::global_ordinal_type>(n)] -= coefficient;
                        }
                    entries[gid] += diagonal;
                }
                Teuchos::Array<Pack::global_ordinal_type> columns;
                Teuchos::Array<double> values;
                for (const auto& [column, value] : entries)
                {
                    columns.push_back(column);
                    values.push_back(value);
                }
                matrix->insertGlobalValues(gid, columns(), values());
            }
            matrix->fillComplete();
            LinearSolverOptions linear;
            linear.backend = LinearSolverBackend::Cg;
            linear.preconditioner = LinearPreconditioner::Jacobi;
            linear.reuse_preconditioner = true;
            // Krylov residuals of A*p lose digits when a nearly constant gas
            // pressure is multiplied by large 1/rho entries. Solve to an
            // attainable internal tolerance, then refine the actual face-flux
            // divergence. The independent physical gate below retains the
            // user's requested accuracy, including the omitted gauge row.
            linear.tolerance = 1.0e-8;
            linear.verbosity = 0;
            for (int refinement = 0; refinement < 4; ++refinement)
            {
                linear.max_iterations = options.max_pressure_iterations - s.diagnostics.pressure_iterations;
                if (linear.max_iterations <= 0)
                    break;
                for (std::size_t c = 1; c < s.f.size(); ++c)
                    rhs.replaceLocalValue(static_cast<Pack::local_ordinal_type>(c), -div[c] / dt);
                const auto stats = pressure_solver.solve_from_zero_with_statistics(
                    Teuchos::rcp_implicit_cast<const Pack::operator_type>(matrix), rhs, solution, linear);
                if (!std::isfinite(stats.achieved_tolerance) || !std::isfinite(stats.rhs_norm))
                    throw std::runtime_error("VoFSolver pressure algebra produced nonfinite diagnostics.");
                s.diagnostics.pressure_iterations += stats.iterations;
                const auto data = solution.getData();
                for (std::size_t c = 0; c < s.p.size(); ++c)
                {
                    if (!std::isfinite(data[c]))
                        throw std::runtime_error("VoFSolver pressure correction is nonfinite.");
                    s.p[c] += data[c];
                }
                for (std::size_t a = 0; a < 3; ++a)
                    for (std::size_t id = 0; id < force[a].size(); ++id)
                    {
                        const auto [l, r] = adjacent(grid, a, face_coordinates(grid, a, id));
                        if (l == missing || r == missing)
                            continue;
                        const double gradient = (data[r] - data[l]) / grid.spacing[a];
                        s.face_u[a][id] -= dt * beta[a][id] * gradient;
                        force[a][id] -= gradient;
                    }
                div = divergence(grid, s.face_u);
                if (maximum_abs(div) <= divergence_limit)
                    break;
            }
        }

        for (std::size_t a = 0; a < 3; ++a)
            for (std::size_t id = 0; id < force[a].size(); ++id)
            {
                const auto [l, r] = adjacent(grid, a, face_coordinates(grid, a, id));
                if (l != missing && r != missing)
                    continue;
                const auto c = l == missing ? r : l;
                if (extrapolate_initial_wall_pressure && grid.cells[a] > 1)
                {
                    // Initial cell data are not an acceleration. Extend
                    // the solved pressure gradient to the wall for the
                    // cell reconstruction: an initially solenoidal field
                    // stays untouched, while a constant normal flow in a
                    // closed one-dimensional box is removed exactly.
                    auto interior = face_coordinates(grid, a, id);
                    interior[a] = l == missing ? 1 : grid.cells[a] - 1;
                    force[a][id] = force[a][face(grid, a, interior)];
                    continue;
                }
                // The normal pressure reaction cancels the extrapolated
                // predictor increment at an impermeable wall. It is needed
                // by the cell momentum balance even though the boundary
                // transport face was already set to zero before assembly.
                force[a][id] = -rho[c] * predictor_change[c][a] / dt;
            }
        for (std::size_t c = 0; c < s.u.size(); ++c)
        {
            const auto low = grid.cell_coordinates(c);
            for (std::size_t a = 0; a < 3; ++a)
            {
                auto high = low;
                ++high[a];
                // Divide a shared face impulse between its two cells in
                // proportion to their mass. Because beta=2/(rhoL+rhoR), their
                // combined momentum increment is exactly dt*force, even across
                // a density jump. This preserves the periodic pressure-force
                // telescoping identity without spuriously amplifying a force
                // carried by liquid inertia in a neighboring light gas cell.
                const auto lo = face(grid, a, low);
                const auto hi = face(grid, a, high);
                s.u[c][a] += dt * 0.5 * (beta[a][lo] * force[a][lo] + beta[a][hi] * force[a][hi]);
            }
        }
        const double residual = maximum_abs(divergence(grid, s.face_u));
        s.diagnostics.pressure_residual = residual / dt;
        if (!std::isfinite(residual) || residual > options.divergence_tolerance || residual * dt > 5.0e-11 ||
            residual / dt > requested)
            throw std::runtime_error(
                "VoFSolver projected physical continuity/pressure residual exceeded its independent tolerance.");
    }

    double velocity_sample(
        const std::vector<Vector>& u, Index xyz, std::size_t a, int direction, std::size_t component) const
    {
        const bool wall = (direction < 0 && xyz[a] == 0) || (direction > 0 && xyz[a] + 1 == grid.cells[a]);
        if (wall && !grid.periodic[a])
            return (a == component ? -1.0 : 1.0) * u[cell(grid, xyz)][component];
        xyz[a] = direction < 0 ? (xyz[a] + grid.cells[a] - 1) % grid.cells[a] : (xyz[a] + 1) % grid.cells[a];
        return u[cell(grid, xyz)][component];
    }

    std::vector<Vector> viscous_force(const State& s) const
    {
        std::vector<Vector> force(s.f.size());
        if (material.liquid_viscosity == 0.0 && material.gas_viscosity == 0.0)
            return force;
        // Cell gradients supply the transposed term of the Newtonian stress.
        std::vector<std::array<Vector, 3>> gradient(s.f.size());
        for (std::size_t c = 0; c < s.f.size(); ++c)
        {
            const auto xyz = grid.cell_coordinates(c);
            for (std::size_t a = 0; a < 3; ++a)
                for (std::size_t b = 0; b < 3; ++b)
                    gradient[c][a][b] = (velocity_sample(s.u, xyz, a, 1, b) - velocity_sample(s.u, xyz, a, -1, b)) /
                                        (2.0 * grid.spacing[a]);
        }
        for (std::size_t a = 0; a < 3; ++a)
            for (std::size_t id = 0; id < grid.face_count(a); ++id)
            {
                const auto xyz = face_coordinates(grid, a, id);
                if (grid.periodic[a] && xyz[a] == grid.cells[a])
                    continue;
                const auto [l, r] = adjacent(grid, a, xyz);
                if (l == missing || r == missing)
                {
                    const auto c = l == missing ? r : l;
                    const double mu = mixture(s.f[c], material.liquid_viscosity, material.gas_viscosity);
                    // Free slip sets tangential traction to zero. Normal
                    // velocity is odd-reflected about the impermeable wall.
                    force[c][a] -= 4.0 * mu * s.u[c][a] / (grid.spacing[a] * grid.spacing[a]);
                    continue;
                }
                const double mu = 0.5 * (mixture(s.f[l], material.liquid_viscosity, material.gas_viscosity) +
                                            mixture(s.f[r], material.liquid_viscosity, material.gas_viscosity));
                for (std::size_t b = 0; b < 3; ++b)
                {
                    const double direct = (s.u[r][b] - s.u[l][b]) / grid.spacing[a];
                    const double transpose = a == b ? direct : 0.5 * (gradient[l][b][a] + gradient[r][b][a]);
                    const double traction_div = mu * (direct + transpose) / grid.spacing[a];
                    force[l][b] += traction_div;
                    force[r][b] -= traction_div;
                }
            }
        return force;
    }

    void budget(State& s) const
    {
        validate_faces(grid, s.face_u);
        detail::CompensatedSum<> gas, liquid, kinetic;
        std::array<detail::CompensatedSum<>, 3> momentum;
        const double volume = grid.cell_volume();
        double maximum_velocity = 0.0;
        for (std::size_t c = 0; c < s.f.size(); ++c)
        {
            if (!std::isfinite(s.f[c]) || s.f[c] < -2.0e-12 || s.f[c] > 1.0 + 2.0e-12 || !std::isfinite(s.p[c]) ||
                !std::isfinite(s.k[c]))
                throw std::runtime_error("VoFSolver candidate contains invalid fraction, pressure, or curvature.");
            const double rho = mixture(s.f[c], material.liquid_density, material.gas_density);
            gas += static_cast<long double>(s.f[c]) * volume;
            liquid += static_cast<long double>(1.0 - s.f[c]) * volume;
            double speed2 = 0.0;
            for (std::size_t a = 0; a < 3; ++a)
            {
                if (!std::isfinite(s.u[c][a]))
                    throw std::runtime_error("VoFSolver candidate has nonfinite velocity.");
                momentum[a] += static_cast<long double>(rho) * s.u[c][a] * volume;
                speed2 += s.u[c][a] * s.u[c][a];
            }
            maximum_velocity = std::max(maximum_velocity, std::sqrt(speed2));
            kinetic += static_cast<long double>(0.5 * rho) * speed2 * volume;
        }
        s.diagnostics.gas_volume = static_cast<double>(gas.value());
        s.diagnostics.gas_mass = static_cast<double>(gas.value() * material.gas_density);
        s.diagnostics.liquid_mass = static_cast<double>(liquid.value() * material.liquid_density);
        s.diagnostics.kinetic_energy = static_cast<double>(kinetic.value());
        s.diagnostics.maximum_velocity = maximum_velocity;
        s.diagnostics.maximum_divergence = maximum_abs(divergence(grid, s.face_u));
        for (std::size_t a = 0; a < 3; ++a)
            s.diagnostics.momentum[a] = static_cast<double>(momentum[a].value());
        if (!std::isfinite(s.diagnostics.gas_volume) || !std::isfinite(s.diagnostics.gas_mass) ||
            !std::isfinite(s.diagnostics.liquid_mass) || !std::isfinite(s.diagnostics.kinetic_energy) ||
            !std::isfinite(s.diagnostics.maximum_velocity) || !std::isfinite(s.diagnostics.maximum_divergence) ||
            !std::all_of(s.diagnostics.momentum.begin(), s.diagnostics.momentum.end(),
                [](double v) { return std::isfinite(v); }))
            throw std::runtime_error("VoFSolver candidate inventory or velocity norm overflowed.");
    }

    double stable_timestep() const
    {
        const auto& s = published->state;
        double maximum_rate = 0.0;
        for (std::size_t c = 0; c < s.f.size(); ++c)
        {
            const auto low = grid.cell_coordinates(c);
            double rate = 0.0;
            for (std::size_t a = 0; a < 3; ++a)
            {
                auto high = low;
                ++high[a];
                rate +=
                    std::max(std::abs(s.face_u[a][face(grid, a, low)]), std::abs(s.face_u[a][face(grid, a, high)])) /
                    grid.spacing[a];
            }
            maximum_rate = std::max(maximum_rate, rate);
        }
        double result =
            maximum_rate > 0.0 ? options.max_courant / maximum_rate : std::numeric_limits<double>::infinity();
        const auto [minf, maxf] = std::minmax_element(s.f.begin(), s.f.end());
        if (material.surface_tension > 0.0 && *minf < 1.0 && *maxf > 0.0)
        {
            const double h = *std::min_element(grid.spacing.begin(), grid.spacing.end());
            result = std::min(
                result, options.capillary_safety * std::sqrt((material.liquid_density + material.gas_density) * h * h *
                                                             h / (2.0 * std::numbers::pi * material.surface_tension)));
        }
        // Arithmetic face viscosity can couple the larger phase viscosity to
        // a small gas-cell inertia. The maximum of the two pure-phase nu values
        // alone is therefore not a safe explicit bound across the interface.
        const double nu = std::max(material.liquid_viscosity, material.gas_viscosity) /
                          std::min(material.liquid_density, material.gas_density);
        if (nu > 0.0)
        {
            double inverse_h2 = 0.0;
            for (double h : grid.spacing)
                inverse_h2 += 1.0 / (h * h);
            result = std::min(result, options.viscous_safety / (4.0 * nu * inverse_h2));
        }
        return result;
    }

    void initialize(
        const std::vector<double>& f, std::vector<Vector> u, Faces faces, bool velocities_were_face_centered)
    {
        validate_fraction(grid, f);
        validate_faces(grid, faces);
        State candidate;
        candidate.f = f;
        candidate.u = std::move(u);
        candidate.face_u = std::move(faces);
        candidate.k = geometry.curvature(f);
        candidate.p.resize(f.size(), 0.0);
        // Face data have already imposed the wall value before averaging to
        // cell centers. Applying another extrapolated-wall impulse would
        // incorrectly change those boundary-cell velocities a second time.
        // For a singleton closed axis, both cell faces are walls and the cell
        // initializer's normal velocity is projected to zero by their impulses.
        const auto predictor = velocities_were_face_centered ? std::vector<Vector>(candidate.u.size()) : candidate.u;
        project(candidate, density(candidate), 1.0, make_faces(grid), predictor, !velocities_were_face_centered);
        budget(candidate);
        auto next = std::make_unique<Published>(std::move(candidate), mesh);
        published.swap(next);
    }

    VoFStepDiagnostics step(double dt)
    {
        if (!(dt > 0.0) || !std::isfinite(dt))
            throw std::invalid_argument("VoFSolver timestep must be positive and finite.");
        if (dt > stable_timestep() * (1.0 + 8.0 * std::numeric_limits<double>::epsilon()))
            throw std::invalid_argument("VoFSolver timestep exceeds its explicit stability limit.");
        const auto& old = published->state;
        const auto old_rho = density(old);
        auto transported = geometry.advect(old.f, old.face_u, dt, old.steps % 3);
        State next;
        next.f = std::move(transported.fraction);
        next.u = old.u;
        next.face_u = old.face_u;
        next.p = old.p;
        next.k = geometry.curvature(next.f);
        next.steps = old.steps + 1;
        next.diagnostics.time = old.diagnostics.time + dt;
        if (!std::isfinite(next.diagnostics.time) || next.diagnostics.time <= old.diagnostics.time)
            throw std::invalid_argument("VoFSolver timestep cannot produce a finite advancing time.");
        const auto new_rho = density(next);
        // The phase sweeps supply final storage and signed integrated mass.
        // Solve their compatible implicit tracer equation for each velocity.
        std::vector<double> mass_flux(mesh->num_faces()), capacity(old.u.size()), zero(old.u.size());
        {
            const auto execution = mesh->acquire_execution_view();
            for (std::size_t id = 0; id < mass_flux.size(); ++id)
            {
                const auto fid = static_cast<Pack::local_ordinal_type>(id);
                if (mesh->is_geometry_exterior_face(fid))
                    continue;
                const auto normal = mesh->face_normal(fid);
                std::size_t a = 0;
                while (std::abs(normal.component(a)) < 0.5)
                    ++a;
                auto xyz = grid.cell_coordinates(mesh->cell_global_id(mesh->owner_cell(fid)));
                if (normal.component(a) > 0)
                    ++xyz[a];
                const auto face_id = face(grid, a, xyz);
                mass_flux[id] = normal.component(a) *
                    (material.gas_density * transported.gas_volume[a][face_id] +
                     material.liquid_density * transported.liquid_volume[a][face_id]);
            }
        }
        for (std::size_t c = 0; c < capacity.size(); ++c)
            capacity[c] = new_rho[c] * grid.cell_volume();
        for (std::size_t b = 0; b < 3; ++b)
        {
            std::vector<double> old_momentum(old.u.size());
            for (std::size_t c = 0; c < old_momentum.size(); ++c)
                old_momentum[c] = old_rho[c] * grid.cell_volume() * old.u[c][b];
            const auto transport = momentum_transport->solve(old_momentum, zero, capacity, mass_flux, false,
                options.max_pressure_iterations, 1e-14);
            for (std::size_t c = 0; c < old.u.size(); ++c)
                next.u[c][b] = transport.amount[c] / capacity[c];
        }
        const auto viscous = viscous_force(old);
        std::vector<Vector> change(old.u.size());
        for (std::size_t c = 0; c < old.u.size(); ++c)
            for (std::size_t b = 0; b < 3; ++b)
            {
                next.u[c][b] += dt * (viscous[c][b] / new_rho[c] + options.gravity[b]);
                change[c][b] = next.u[c][b] - old.u[c][b];
            }
        const auto face_change = interpolate_cells(grid, change);
        for (std::size_t a = 0; a < 3; ++a)
            for (std::size_t id = 0; id < next.face_u[a].size(); ++id)
                next.face_u[a][id] += face_change[a][id];
        project(next, new_rho, dt, surface_force(next), change);
        budget(next);
        const double volume_scale = grid.cell_volume() * static_cast<double>(grid.cell_count());
        if (std::abs(next.diagnostics.gas_volume - old.diagnostics.gas_volume) > 5.0e-12 * volume_scale)
            throw std::runtime_error("VoFSolver gas inventory failed its conservative transport gate.");
        const auto diagnostics = next.diagnostics;
        auto publication = std::make_unique<Published>(std::move(next), mesh);
        published.swap(publication);
        return diagnostics;
    }
};

VoFSolver::VoFSolver(Grid grid, VoFMaterial material, VoFSolverOptions options)
    : impl_(std::make_unique<Impl>(std::move(grid), material, options))
{
}
VoFSolver::~VoFSolver() = default;
VoFSolver::VoFSolver(VoFSolver&&) noexcept = default;
VoFSolver& VoFSolver::operator=(VoFSolver&&) noexcept = default;

void VoFSolver::initialize(const std::vector<double>& f, const std::vector<Vector>& input)
{
    std::vector<Vector> velocity = input;
    if (velocity.empty())
        velocity.resize(impl_->grid.cell_count());
    if (velocity.size() != impl_->grid.cell_count())
        throw std::invalid_argument("VoFSolver initial cell velocity shape does not match the grid.");
    for (const auto& v : velocity)
        for (double component : v)
            if (!std::isfinite(component))
                throw std::invalid_argument("VoFSolver initial cell velocity must be finite.");
    auto faces = interpolate_cells(impl_->grid, velocity);
    impl_->initialize(f, std::move(velocity), std::move(faces), false);
}

void VoFSolver::initialize(const std::vector<double>& f, const Faces& velocity)
{
    validate_faces(impl_->grid, velocity);
    impl_->initialize(f, average_faces(impl_->grid, velocity), velocity, true);
}

VoFStepDiagnostics VoFSolver::step(double dt)
{
    return impl_->step(dt);
}
double VoFSolver::stable_timestep() const
{
    return impl_->stable_timestep();
}
double VoFSolver::time() const noexcept
{
    return impl_->published->state.diagnostics.time;
}
const VoFStepDiagnostics& VoFSolver::diagnostics() const noexcept
{
    return impl_->published->state.diagnostics;
}
const Grid& VoFSolver::grid() const noexcept
{
    return impl_->grid;
}
const VoFMaterial& VoFSolver::material() const noexcept
{
    return impl_->material;
}
const VoFSolverOptions& VoFSolver::options() const noexcept
{
    return impl_->options;
}
const SP<const VoFSolver::Handle>& VoFSolver::mesh() const noexcept
{
    return impl_->mesh;
}
const std::vector<double>& VoFSolver::gas_fraction() const noexcept
{
    return impl_->published->state.f;
}
const std::vector<Vector>& VoFSolver::velocity() const noexcept
{
    return impl_->published->state.u;
}
const std::vector<double>& VoFSolver::pressure() const noexcept
{
    return impl_->published->state.p;
}
const std::vector<double>& VoFSolver::curvature() const noexcept
{
    return impl_->published->state.k;
}
const Faces& VoFSolver::face_velocity() const noexcept
{
    return impl_->published->state.face_u;
}
const VoFSolver::ScalarField& VoFSolver::fraction_field() const noexcept
{
    return impl_->published->f;
}
const VoFSolver::ScalarField& VoFSolver::pressure_field() const noexcept
{
    return impl_->published->p;
}
const VoFSolver::VectorField& VoFSolver::velocity_field() const noexcept
{
    return impl_->published->u;
}

} // namespace SimpleFluid
