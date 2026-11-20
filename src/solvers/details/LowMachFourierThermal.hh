/** Host thermal correction from frozen canonical Fourier contact conductances. */
#pragma once
#include "equations/CollectiveValidation.hh"
#include "solvers/BelosLinearSolver.hh"
#include "solvers/details/LowMachFaceTransportAdapter.hh"
#include <Teuchos_CommHelpers.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <map>

namespace SimpleFluid::Reacting::fourier_thermal_detail
{
using Flow = LowMachMixtureSolver;
using Pack = Flow::Pack;
using Stage = std::shared_ptr<const low_mach_faces::Receipt>;
using GO = Pack::global_ordinal_type;
using LO = Pack::local_ordinal_type;

struct Row
{
    std::map<GO, double> conductances;
    low_mach_faces::FourierCellLinearization caloric;
    double diagonal = 1, defect = 0;
};

SIMPLEFLUID_SOLVERS_LOCAL inline void correct_block(const Flow::Mesh& mesh, double dt, std::span<const Stage> stages,
    std::optional<std::size_t> phase, Flow::State& guess, double maximum_step)
{
    const auto map = mesh.owned_cell_map();
    const auto comm = map->getComm();
    std::vector<Row> rows;
    // Unusable block, largest diagonal, largest temperature defect, negative smallest active diagonal.
    std::array<double, 4> local{0, 0, 0, -std::numeric_limits<double>::infinity()}, global{};
    collective_detail::collective_local_validation(mesh, "Frozen Fourier thermal rows",
        [&]
        {
            rows.resize(guess.cells.size());
            for (const auto& stage : stages)
            {
                if (phase && stage->physical_phase != *phase)
                    continue;
                const auto& caloric = *stage->fourier_thermal;
                for (const auto& face : stage->fourier_faces)
                {
                    const double weight = dt * face.conductance;
                    if (weight == 0)
                        continue;
                    const auto deposit = [&](std::optional<std::size_t> cell, GO other)
                    {
                        if (!cell)
                            return;
                        auto& row = rows.at(*cell);
                        row.conductances[other] += weight;
                        const auto& newest = caloric.at(*cell);
                        // Last incident active stage supplies C and the original temperature operand.
                        row.caloric.capacity = newest.capacity;
                        row.caloric.temperature = newest.temperature;
                        row.caloric.minimum_temperature =
                            std::max(row.caloric.minimum_temperature, newest.minimum_temperature);
                        row.caloric.maximum_temperature =
                            std::min(row.caloric.maximum_temperature, newest.maximum_temperature);
                    };
                    deposit(face.owner, face.neighbor_cell);
                    deposit(face.neighbor, face.owner_cell);
                }
            }
            for (std::size_t c = 0; c < rows.size(); ++c)
            {
                auto& row = rows[c];
                if (row.conductances.empty())
                    continue;
                auto& caloric = row.caloric;
                double recovered = 0, stiffness = 0;
                for (std::size_t a = 0; a < guess.cells[c].phases.size(); ++a)
                {
                    const auto& current = guess.cells[c].phases[a];
                    if ((phase && a != *phase) || current.temperature == 0)
                        continue;
                    recovered = current.temperature;
                    const auto& bounds = current.thermodynamics->bounds();
                    caloric.minimum_temperature = std::max(caloric.minimum_temperature, bounds.minimum_temperature);
                    caloric.maximum_temperature = std::min(caloric.maximum_temperature, bounds.maximum_temperature);
                }
                for (const auto& [other, weight] : row.conductances)
                    stiffness += weight;
                row.diagonal = caloric.capacity + stiffness;
                row.defect = recovered - caloric.temperature;
                if (!(recovered > 0 && std::isfinite(row.defect) && std::isfinite(row.diagonal) &&
                        row.diagonal > stiffness && caloric.temperature >= caloric.minimum_temperature &&
                        caloric.temperature <= caloric.maximum_temperature))
                    local[0] = 1;
                local[1] = std::max(local[1], row.diagonal);
                local[2] = std::max(local[2], std::abs(row.defect));
                local[3] = std::max(local[3], -row.diagonal);
            }
        });
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, static_cast<int>(local.size()), local.data(), global.data());
    // A disappearing separate phase has no intensive unknown. Likewise, keep the recovered guess if
    // the frozen block cannot be represented or its starting point is outside the new material bounds.
    if (global[0] != 0 || global[2] == 0)
        return;

    // Power-of-two scaling is exact for normal operands. Relative residuals remain meaningful even
    // for a temperature defect near roundoff or very small dimensional energy scales.
    const int matrix_exponent = std::ilogb(global[1]), temperature_exponent = std::ilogb(global[2]);
    if (!std::isnormal(std::scalbn(-global[3], -matrix_exponent)))
        return; // The scaled diagonal must support a finite Jacobi inverse on every owner.
    Teuchos::RCP<Pack::matrix_type> matrix;
    Teuchos::RCP<Pack::vector_type> rhs, correction;
    double local_maximum_rhs = 0, maximum_rhs = 0;
    collective_detail::collective_local_validation(mesh, "Frozen Fourier thermal matrix assembly",
        [&]
        {
            std::size_t row_width = 1;
            for (const auto& row : rows)
                row_width = std::max(row_width, 1 + row.conductances.size());
            matrix = Teuchos::rcp(new Pack::matrix_type(map, row_width));
            rhs = Teuchos::rcp(new Pack::vector_type(map));
            correction = Teuchos::rcp(new Pack::vector_type(map));
            auto b = rhs->getLocalViewHost(Tpetra::Access::OverwriteAll);
            for (std::size_t c = 0; c < rows.size(); ++c)
            {
                const auto gid = map->getGlobalElement(static_cast<LO>(c));
                const auto& row = rows[c];
                std::vector<GO> columns{gid};
                std::vector<double> entries;
                if (row.conductances.empty())
                {
                    entries.push_back(1);
                    b(c, 0) = 0;
                }
                else
                {
                    entries.push_back(std::scalbn(row.diagonal, -matrix_exponent));
                    for (const auto& [other, weight] : row.conductances)
                    {
                        columns.push_back(other);
                        entries.push_back(-std::scalbn(weight, -matrix_exponent));
                    }
                    b(c, 0) = std::scalbn(row.caloric.capacity, -matrix_exponent) *
                              std::scalbn(row.defect, -temperature_exponent);
                    local_maximum_rhs = std::max(local_maximum_rhs, std::abs(b(c, 0)));
                }
                matrix->insertGlobalValues(
                    gid, Teuchos::arrayViewFromVector(columns), Teuchos::arrayViewFromVector(entries));
            }
        });
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &local_maximum_rhs, &maximum_rhs);
    if (maximum_rhs < std::sqrt(std::numeric_limits<double>::min()))
        return; // A nonzero physical defect must not disappear in the solver's squared RHS norm.
    matrix->fillComplete();
    BelosLinearSolver<Pack> solver;
    LinearSolverOptions options;
    options.backend = LinearSolverBackend::Cg;
    options.preconditioner = LinearPreconditioner::Jacobi;
    if (!solver.solve_from_zero_with_statistics(matrix, *rhs, *correction, options).converged)
        return;

    std::vector<double> increments;
    double local_step = maximum_step, step = maximum_step;
    collective_detail::collective_local_validation(mesh, "Fourier thermal guess EOS bounds",
        [&]
        {
            increments.resize(rows.size());
            const auto x = correction->getLocalViewHost(Tpetra::Access::ReadOnly);
            for (std::size_t c = 0; c < rows.size(); ++c)
            {
                if (rows[c].conductances.empty())
                    continue;
                const auto& caloric = rows[c].caloric;
                const double delta = std::scalbn(x(c, 0), temperature_exponent);
                increments[c] = delta;
                if (!std::isfinite(delta))
                    local_step = 0;
                else if (delta > 0)
                    local_step = std::min(local_step, (caloric.maximum_temperature - caloric.temperature) / delta);
                else if (delta < 0)
                    local_step = std::min(local_step, (caloric.minimum_temperature - caloric.temperature) / delta);
            }
        });
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MIN, 1, &local_step, &step);
    if (step < maximum_step)
        step = std::nextafter(step, 0.);
    if (step == 0)
        return;
    int local_admissible = 1, admissible = 0;
    for (std::size_t c = 0; c < rows.size(); ++c)
        if (!rows[c].conductances.empty())
        {
            const auto& caloric = rows[c].caloric;
            increments[c] = std::fma(step, increments[c], caloric.temperature);
            if (!(std::isfinite(increments[c]) && increments[c] >= caloric.minimum_temperature &&
                    increments[c] <= caloric.maximum_temperature))
                local_admissible = 0;
        }
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MIN, 1, &local_admissible, &admissible);
    if (!admissible)
        return;
    for (std::size_t c = 0; c < rows.size(); ++c)
        if (!rows[c].conductances.empty())
            for (std::size_t a = 0; a < guess.cells[c].phases.size(); ++a)
                if ((!phase || a == *phase) && guess.cells[c].phases[a].temperature > 0)
                    guess.cells[c].phases[a].temperature = increments[c];
}

SIMPLEFLUID_SOLVERS_LOCAL inline void correct(
    const Flow::Mesh& mesh, double dt, PhaseThermalMode mode, std::span<const Stage> stages, Flow::State& guess, double maximum_step = 1.)
{
    if (stages.empty())
        return;
    if (mode == PhaseThermalMode::Equilibrium)
        return correct_block(mesh, dt, stages, std::nullopt, guess, maximum_step);
    for (auto stage = stages.begin(); stage != stages.end(); ++stage)
    {
        const auto phase = (*stage)->physical_phase;
        if (std::none_of(
                stages.begin(), stage, [=](const auto& previous) { return previous->physical_phase == phase; }))
            correct_block(mesh, dt, stages, phase, guess, maximum_step);
    }
}
} // namespace SimpleFluid::Reacting::fourier_thermal_detail
