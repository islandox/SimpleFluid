/** @file LowMachPressureEquation.cc
 * @brief Owner-oriented variable-density pressure projection with physical continuity checks.
 */
#include "equations/LowMachPressureEquation.hh"
#include "FVM/CellOperators.hh"
#include "equations/CollectiveValidation.hh"
#include "equations/EquationValidation.hh"
#include <Teuchos_Array.hpp>
#include <Teuchos_CommHelpers.hpp>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace SimpleFluid::Reacting
{
struct LowMachPressureEquation::Implementation
{
    SP<const Mesh> mesh;
    std::uint64_t epoch;
    ScalarField correction;
    VectorField gradient;
    FVM::CellGradientCache<Pack, Mesh> gradient_cache;
    BelosLinearSolver<Pack> solver;
    explicit Implementation(SP<const Mesh> grid)
        : mesh(EquationValidation::require_non_null_mesh(std::move(grid), "LowMachPressureEquation")),
          epoch(mesh_geometry_epoch(*mesh)), correction(mesh, 0.0, "low_mach_pressure_correction"),
          gradient(mesh, "low_mach_pressure_gradient"), gradient_cache(mesh)
    {
    }
};
LowMachPressureEquation::LowMachPressureEquation(SP<const Mesh> mesh)
    : d(std::make_unique<Implementation>(std::move(mesh)))
{
}
LowMachPressureEquation::~LowMachPressureEquation() = default;

LowMachPressureEquation::Result LowMachPressureEquation::project(const ScalarField& rho, double dt,
    std::span<const double> target, const FaceField& predictor, FaceField& corrected, ScalarField& pressure,
    VectorField& velocity, const LinearSolverOptions& options, double tolerance)
{
    const auto& mesh = *d->mesh;
    using LO = Pack::local_ordinal_type;
    using GO = Pack::global_ordinal_type;
    collective_detail::require_uniform_value(mesh, dt, "Low-Mach projection time step");
    collective_detail::require_uniform_value(mesh, tolerance, "Low-Mach continuity tolerance");
    collective_detail::collective_local_validation(mesh, "Low-Mach pressure inputs",
        [&]
        {
            if (d->epoch != mesh_geometry_epoch(mesh) || !std::isfinite(dt) || dt <= 0 || !std::isfinite(tolerance) ||
                tolerance <= 0 || target.size() != mesh.num_owned_cells())
                throw std::invalid_argument("Invalid low-Mach projection timestep/target/tolerance.");
            EquationValidation::require_mesh_match(mesh, rho, "LowMachPressureEquation");
            EquationValidation::require_mesh_match(mesh, predictor, "LowMachPressureEquation");
            EquationValidation::require_mesh_match(mesh, corrected, "LowMachPressureEquation");
            EquationValidation::require_mesh_match(mesh, pressure, "LowMachPressureEquation");
            EquationValidation::require_mesh_match(mesh, velocity, "LowMachPressureEquation");
            for (size_t c = 0; c < mesh.num_local_cells(); ++c)
                if (!std::isfinite(rho.local_value(static_cast<LO>(c))) || rho.local_value(static_cast<LO>(c)) <= 0)
                    throw std::invalid_argument("Projection requires synchronized positive actual mixture density.");
            for (double value : target)
                if (!std::isfinite(value))
                    throw std::invalid_argument("Nonfinite continuity target.");
            for (size_t c = 0; c < mesh.num_owned_cells(); ++c)
            {
                const LO cell = static_cast<LO>(c);
                if (!std::isfinite(pressure.local_value(cell)))
                    throw std::invalid_argument("Nonfinite mechanical pressure.");
                for (size_t axis = 0; axis < 3; ++axis)
                    if (!std::isfinite(velocity.local_value(cell).component(axis)))
                        throw std::invalid_argument("Nonfinite projection velocity.");
            }
            for (size_t f = 0; f < mesh.num_owned_faces(); ++f)
            {
                const LO face = static_cast<LO>(f);
                if (!std::isfinite(predictor.local_value(face)))
                    throw std::invalid_argument("Nonfinite predictor volume flux.");
                if (mesh.is_geometry_exterior_face(face) && predictor.local_value(face) != 0)
                    throw std::invalid_argument("Closed low-Mach projection requires zero boundary volume flux.");
            }
        });
    const auto map = mesh.owned_cell_map();
    const auto overlap = mesh.overlap_cell_map();
    const GO gauge = map->getMinAllGlobalIndex();
    auto matrix = Teuchos::rcp(new Pack::matrix_type(map, 8));
    auto rhs = Teuchos::rcp(new Pack::vector_type(map));
    auto rhs_values = rhs->getLocalViewHost(Tpetra::Access::OverwriteAll);
    std::vector<double> coefficients(mesh.num_faces(), 0.0);
    std::vector<std::vector<GO>> columns(mesh.num_owned_cells());
    std::vector<std::vector<double>> entries(mesh.num_owned_cells());
    std::vector<double> diagonal(mesh.num_owned_cells(), 0.0);
    collective_detail::collective_local_validation(mesh, "Low-Mach pressure coefficients",
        [&]
        {
            for (size_t f = 0; f < mesh.num_faces(); ++f)
            {
                const LO face = static_cast<LO>(f);
                if (mesh.is_geometry_exterior_face(face))
                    continue;
                const LO owner = mesh.owner_cell(face), neighbor = mesh.neighbor_cell(face);
                const auto owned = [&](LO cell)
                { return cell >= 0 && static_cast<size_t>(cell) < mesh.num_owned_cells(); };
                // Halo fringe faces need not carry a second local cell. Only
                // incidences contributing to an owned row belong in this matrix.
                if (!owned(owner) && !owned(neighbor))
                    continue;
                if (owner < 0 || neighbor < 0)
                    throw std::invalid_argument("Owned projection incidence lacks its adjacent halo cell.");
                const double distance = mesh.face_cell_center_distance(face), area = mesh.face_area(face);
                if (!std::isfinite(distance) || distance <= 0 || !std::isfinite(area) || area <= 0)
                    throw std::invalid_argument("Projection face geometry is invalid.");
                const double mean_density = 0.5 * rho.local_value(owner) + 0.5 * rho.local_value(neighbor);
                const double coefficient = dt / mean_density * area / distance;
                if (!std::isfinite(coefficient) || coefficient <= 0)
                    throw std::invalid_argument("Projection mobility is not finitely representable and positive.");
                coefficients[f] = coefficient;
                for (int side = 0; side < 2; ++side)
                {
                    const LO cell = side == 0 ? owner : neighbor, other = side == 0 ? neighbor : owner;
                    if (static_cast<size_t>(cell) >= mesh.num_owned_cells())
                        continue;
                    const GO row = map->getGlobalElement(cell), col = overlap->getGlobalElement(other);
                    if (row == gauge)
                        continue;
                    diagonal[cell] += coefficient;
                    if (col != gauge)
                    {
                        columns[cell].push_back(col);
                        entries[cell].push_back(-coefficient);
                    }
                }
            }
        });
    double local_compatibility = 0, local_scale = 0;
    for (size_t c = 0; c < mesh.num_owned_cells(); ++c)
    {
        const LO cell = static_cast<LO>(c);
        const double residual = target[c] - FVM::cell_flux_balance(mesh, predictor, cell);
        local_compatibility += residual;
        local_scale += std::abs(target[c]) + std::abs(FVM::cell_flux_balance(mesh, predictor, cell));
        rhs_values(c, 0) = map->getGlobalElement(cell) == gauge ? 0 : residual;
        columns[c].push_back(map->getGlobalElement(cell));
        entries[c].push_back(map->getGlobalElement(cell) == gauge ? 1 : diagonal[c]);
        matrix->insertGlobalValues(map->getGlobalElement(cell), Teuchos::arrayViewFromVector(columns[c]),
            Teuchos::arrayViewFromVector(entries[c]));
    }
    Result result;
    double scale = 0;
    Teuchos::reduceAll(
        *map->getComm(), Teuchos::REDUCE_SUM, 1, &local_compatibility, &result.global_compatibility_residual);
    Teuchos::reduceAll(*map->getComm(), Teuchos::REDUCE_SUM, 1, &local_scale, &scale);
    if (std::abs(result.global_compatibility_residual) > tolerance * std::max(1.0, scale))
        throw std::runtime_error("Closed low-Mach source target fails global pressure compatibility.");
    matrix->fillComplete();
    d->correction.owned_data().putScalar(0);
    d->solver.notify_operator_values_changed();
    result.linear = d->solver.solve_with_statistics(matrix, *rhs, d->correction.owned_data(), options);
    if (!result.linear.converged)
        throw std::runtime_error("Low-Mach pressure failed its true residual gate.");
    d->correction.sync_ghosts();
    FaceField candidate(d->mesh, 0.0, "candidate_low_mach_flux");
    collective_detail::collective_local_validation(mesh, "Low-Mach projected face flux",
        [&]
        {
            for (size_t f = 0; f < mesh.num_owned_faces(); ++f)
            {
                const LO face = static_cast<LO>(f);
                double flux = predictor.local_value(face);
                if (!mesh.is_geometry_exterior_face(face))
                {
                    const LO owner = mesh.owner_cell(face),
                             neighbor = mesh.opposite_or_periodic_neighbor_cell(face, owner);
                    flux += coefficients[f] * (d->correction.local_value(owner) - d->correction.local_value(neighbor));
                }
                if (!std::isfinite(flux))
                    throw std::invalid_argument("Projected physical face flux is nonfinite.");
                candidate.set_owned_value(face, flux);
            }
        });
    candidate.sync_ghosts();
    double local_maximum = 0;
    for (size_t c = 0; c < mesh.num_owned_cells(); ++c)
        local_maximum =
            std::max(local_maximum, std::abs(FVM::cell_flux_balance(mesh, candidate, static_cast<LO>(c)) - target[c]));
    Teuchos::reduceAll(*map->getComm(), Teuchos::REDUCE_MAX, 1, &local_maximum, &result.maximum_continuity_residual);
    if (result.maximum_continuity_residual > tolerance * std::max(1.0, scale))
        throw std::runtime_error("Low-Mach pressure correction fails physical continuity residual gate.");
    FVM::cell_gradient(
        d->correction, BoundaryConditionMap{}, d->gradient, d->gradient_cache, FVM::CellGradientScheme::LeastSquares);
    std::vector<double> candidate_pressure(mesh.num_owned_cells());
    std::vector<VectorField::vec_type> candidate_velocity(mesh.num_owned_cells());
    collective_detail::collective_local_validation(mesh, "Low-Mach projected cell state",
        [&]
        {
            for (size_t c = 0; c < mesh.num_owned_cells(); ++c)
            {
                const LO cell = static_cast<LO>(c);
                candidate_pressure[c] = pressure.local_value(cell) + d->correction.local_value(cell);
                candidate_velocity[c] =
                    velocity.local_value(cell) - d->gradient.local_value(cell) * (dt / rho.local_value(cell));
                if (!std::isfinite(candidate_pressure[c]))
                    throw std::invalid_argument("Projected pressure is nonfinite.");
                for (size_t axis = 0; axis < 3; ++axis)
                    if (!std::isfinite(candidate_velocity[c].component(axis)))
                        throw std::invalid_argument("Projected velocity is nonfinite.");
            }
        });
    for (size_t c = 0; c < mesh.num_owned_cells(); ++c)
    {
        pressure.set_owned_value(static_cast<LO>(c), candidate_pressure[c]);
        velocity.set_owned_value(static_cast<LO>(c), candidate_velocity[c]);
    }
    pressure.sync_ghosts();
    velocity.sync_ghosts();
    corrected.owned_data().update(1.0, candidate.owned_data(), 0.0);
    corrected.sync_ghosts();
    return result;
}
} // namespace SimpleFluid::Reacting
