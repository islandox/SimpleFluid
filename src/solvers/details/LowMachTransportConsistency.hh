/** @file LowMachTransportConsistency.hh
 * @brief Bounded local backward-Euler upwind balance repair after a global Krylov solve.
 */
#pragma once
#include "FVM/CellOperators.hh"
#include "equations/CollectiveValidation.hh"
#include "solvers/LowMachMixtureSolver.hh"
#include "solvers/details/LowMachSourceStage.hh"
#include "utils/CompensatedSum.hh"
#include <Teuchos_CommHelpers.hpp>
#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>

namespace SimpleFluid::Reacting::transport_consistency_detail
{
using Flow = LowMachMixtureSolver;
using LO = Flow::Pack::local_ordinal_type;
using Scalar = Flow::Traits::scalar_cell_type;
using Face = Flow::Traits::scalar_face_type;
using Sum = SimpleFluid::detail::CompensatedSum<long double>;
enum class Quantity
{
    Species,
    Energy,
    Population
};
struct Result
{
    LinearSolveStatistics linear;
    int refinements = 0;
    double initial_residual = 0, final_residual = 0;
};
SIMPLEFLUID_SOLVERS_LOCAL inline double narrow(long double value)
{
    const double result = static_cast<double>(value);
    if (!std::isfinite(value) || !std::isfinite(result) || (value != 0 && result == 0))
        throw std::invalid_argument("Low-Mach transport packet is nonfinite or unrepresentable.");
    return result;
}
/** The returned amounts, divided by Vnew exactly as the caller does, own all donor traces.
 * Every local sweep spends one remaining iteration of the original scalar solve.
 * This is the existing backward-Euler/upwind/zero-diffusion operator, including ALE;
 * it is not an admission policy for BDF2 or other convection schemes.
 */
SIMPLEFLUID_SOLVERS_LOCAL inline Result enforce(const SP<const Flow::Mesh>& grid, std::span<const double> old_amount,
    const Face& phase_flux, double dt, Scalar& concentration, std::vector<double>& amount,
    const Flow::Pack::matrix_type& matrix, const Flow::Pack::vector_type& rhs, LinearSolveStatistics linear,
    const LinearSolverOptions& options, double physical_tolerance, Quantity quantity,
    const FVM::ALEControlVolumeState* ale)
{
    const auto& mesh = *grid;
    const auto comm = mesh.owned_cell_map()->getComm();
    const bool nonnegative = quantity != Quantity::Energy;
    const auto volume = [&](size_t c)
    { return ale ? ale->new_cell_volumes()[c] : mesh.cell_volume(static_cast<LO>(c)); };
    Result result{linear};
    std::vector<Sum> incoming, outgoing;
    Teuchos::RCP<Flow::Pack::vector_type> residual;
    int local_restart = 0, restart = 0;
    collective_detail::collective_local_validation(mesh, "Low-Mach local transport initial state",
        [&]
        {
            amount.resize(old_amount.size());
            incoming.resize(old_amount.size());
            outgoing.resize(old_amount.size());
            residual = Teuchos::rcp(new Flow::Pack::vector_type(mesh.owned_cell_map()));
            for (size_t c = 0; c < amount.size(); ++c)
            {
                if (!std::isfinite(old_amount[c]) || (nonnegative && old_amount[c] < 0))
                    throw std::invalid_argument("Invalid accepted transport inventory.");
                // Krylov updates authoritative owned storage; its overlap still
                // belongs to the preceding scalar call until the sync below.
                const double value = concentration.value(static_cast<LO>(c));
                amount[c] = value * volume(c);
                if (!std::isfinite(value) || !std::isfinite(amount[c]) || (value != 0 && amount[c] == 0))
                    throw std::invalid_argument("Krylov transport inventory is nonfinite or unrepresentable.");
                local_restart |= nonnegative && amount[c] < 0;
            }
        });
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &local_restart, &restart);
    // A negative Krylov candidate is discarded as an initializer, never clipped.
    // All ranks restart from the same category of accepted finite stocks.
    if (restart)
        std::copy(old_amount.begin(), old_amount.end(), amount.begin());
    // FieldStored construction itself acquires a collective layout; keep that
    // construction on every rank, after the preceding local allocation fence.
    Face packets(grid, "canonical_scalar_transport_packet", false);
    for (;;)
    {
        collective_detail::collective_local_validation(mesh, "Low-Mach transport donor concentrations",
            [&]
            {
                for (size_t c = 0; c < amount.size(); ++c)
                {
                    if (nonnegative && amount[c] < 0)
                        throw std::invalid_argument("Monotone transport produced a negative inventory.");
                    const double trace = amount[c] / volume(c);
                    if (!std::isfinite(trace) || (amount[c] != 0 && trace == 0))
                        throw std::invalid_argument("Returned transport concentration is not representable.");
                    concentration.set_owned_value(static_cast<LO>(c), trace);
                }
            });
        concentration.sync_ghosts();
        collective_detail::collective_local_validation(mesh, "Canonical local transport packets",
            [&]
            {
                for (size_t f = 0; f < mesh.num_owned_faces(); ++f)
                {
                    const LO face = static_cast<LO>(f);
                    const double phi = phase_flux.local_value(face);
                    if (mesh.is_geometry_exterior_face(face))
                    {
                        if (phi != 0)
                            throw std::invalid_argument("Closed phase transport has nonzero relative boundary flux.");
                        packets.set_owned_value(face, 0.);
                        continue;
                    }
                    const LO owner = mesh.owner_cell(face),
                             donor = phi >= 0 ? owner : mesh.opposite_or_periodic_neighbor_cell(face, owner);
                    const double donor_trace = concentration.local_value(donor);
                    const double packet = phi * donor_trace;
                    if (!std::isfinite(packet) || (phi != 0 && donor_trace != 0 && packet == 0))
                        throw std::invalid_argument("Canonical transport face packet is not representable.");
                    packets.set_owned_value(face, packet);
                }
            });
        packets.sync_ghosts();
        double local_error = 0, error = 0;
        collective_detail::collective_local_validation(mesh, "Physical componentwise transport balance",
            [&]
            {
                incoming.assign(amount.size(), Sum{});
                outgoing.assign(amount.size(), Sum{});
                for (size_t c = 0; c < amount.size(); ++c)
                {
                    const LO cell = static_cast<LO>(c);
                    Sum balance, magnitude;
                    for (const LO face : mesh.faces(cell))
                    {
                        const double sign = mesh.owner_cell(face) == cell ? 1. : -1.;
                        const double phi = sign * phase_flux.local_value(face);
                        const double packet = sign * packets.local_value(face);
                        balance += static_cast<long double>(dt) * packet;
                        magnitude += std::abs(static_cast<long double>(dt) * packet);
                        if (phi < 0)
                            incoming[c] += -static_cast<long double>(packet);
                        else
                            outgoing[c] += phi;
                    }
                    Sum defect;
                    defect += amount[c];
                    defect += -static_cast<long double>(old_amount[c]);
                    defect += balance.value();
                    const long double scale = std::abs(static_cast<long double>(amount[c])) +
                                              std::abs(static_cast<long double>(old_amount[c])) + magnitude.value();
                    double cell_error = source_stage_detail::normalized(defect.value(), scale, physical_tolerance);
                    if (quantity == Quantity::Population)
                    {
                        // Preserve the pre-existing independently ordered count-balance gate too.
                        const long double flux =
                            static_cast<long double>(dt) * FVM::cell_flux_balance(mesh, packets, cell);
                        const long double count_defect = static_cast<long double>(amount[c]) - old_amount[c] + flux;
                        cell_error = std::max(
                            cell_error, source_stage_detail::normalized(count_defect,
                                            std::abs(static_cast<long double>(amount[c])) +
                                                std::abs(static_cast<long double>(old_amount[c])) + std::abs(flux),
                                            physical_tolerance));
                    }
                    local_error = std::max(local_error, cell_error);
                }
            });
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &local_error, &error);
        if (result.refinements == 0)
            result.initial_residual = error;
        result.final_residual = error;
        if (error <= 1)
        {
            // Recheck the actual final represented field against the ORIGINAL stored CRS/RHS.
            // Extended row accumulation matches the existing Belos accurate-residual policy.
            collective_detail::collective_local_validation(mesh, "Final local transport stored-CRS residual",
                [&]
                {
                    const auto local = matrix.getLocalMatrixHost();
                    const auto columns = matrix.getColMap();
                    const auto b = rhs.getData(0);
                    auto r = residual->getDataNonConst(0);
                    for (size_t c = 0; c < amount.size(); ++c)
                    {
                        Sum value;
                        value += b[c];
                        for (auto entry = local.graph.row_map(c); entry < local.graph.row_map(c + 1); ++entry)
                        {
                            const auto gid = columns->getGlobalElement(local.graph.entries(entry));
                            const LO column = mesh.overlap_cell_map()->getLocalElement(gid);
                            if (column < 0 || static_cast<size_t>(column) >= mesh.num_local_cells())
                                throw std::invalid_argument(
                                    "Transport CRS column is absent from the native overlap map.");
                            value += -static_cast<long double>(local.values(entry)) * concentration.local_value(column);
                        }
                        r[c] = static_cast<double>(value.value());
                        if (!std::isfinite(r[c]))
                            throw std::invalid_argument("Final transport true residual is nonfinite.");
                    }
                });
            const double rhs_norm = rhs.norm2(), residual_norm = residual->norm2();
            result.linear.rhs_norm = rhs_norm;
            result.linear.achieved_tolerance = rhs_norm > 0 ? residual_norm / rhs_norm : residual_norm;
            result.linear.converged = std::isfinite(rhs_norm) && std::isfinite(residual_norm) &&
                                      std::isfinite(result.linear.achieved_tolerance) &&
                                      result.linear.achieved_tolerance <= options.tolerance;
            if (result.linear.converged && (!restart || result.refinements > 0))
                return result;
        }
        if (result.refinements >= options.max_iterations - linear.iterations)
        {
            std::ostringstream message;
            message.precision(17);
            message << "Low-Mach local transport exhausted the original scalar iteration budget: local balance="
                    << error << ", last checked true residual=" << result.linear.achieved_tolerance
                    << ", Krylov iterations=" << linear.iterations << ", balance refinements=" << result.refinements;
            throw std::runtime_error(message.str());
        }
        collective_detail::collective_local_validation(mesh, "Monotone local transport refinement",
            [&]
            {
                for (size_t c = 0; c < amount.size(); ++c)
                {
                    Sum numerator;
                    numerator += old_amount[c];
                    numerator += static_cast<long double>(dt) * incoming[c].value();
                    const long double denominator = 1 + static_cast<long double>(dt) * outgoing[c].value() / volume(c);
                    if (!(denominator > 0) || !std::isfinite(denominator))
                        throw std::invalid_argument("Monotone transport row is not finitely representable.");
                    amount[c] = narrow(numerator.value() / denominator);
                    if (nonnegative && amount[c] < 0)
                        throw std::invalid_argument("Monotone transport update is negative.");
                }
            });
        ++result.refinements;
    }
}
} // namespace SimpleFluid::Reacting::transport_consistency_detail
