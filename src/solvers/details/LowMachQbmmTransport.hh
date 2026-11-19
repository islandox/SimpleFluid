/** @file LowMachQbmmTransport.hh
 * @brief One positive backward-Euler map and stored-operator checks for a complete native QBMM family.
 */
#pragma once
#include "solvers/details/LowMachTransportConsistency.hh"
#include <Teuchos_DefaultMpiComm.hpp>

namespace SimpleFluid::Reacting::transport_consistency_detail
{
struct BlockResult
{
    std::vector<Result> components;
    std::vector<Flow::QbmmFaceFlux> owned_flux;
    int sweeps = 0;
};

SIMPLEFLUID_SOLVERS_LOCAL inline double transport_concentration(double amount, double volume)
{
    const double trace = amount / volume;
    if (!std::isfinite(trace) || (amount != 0 && trace == 0))
        throw std::invalid_argument("QBMM donor concentration is not representable.");
    if (std::fpclassify(trace) == FP_SUBNORMAL)
        require_transport_roundoff(
            trace, static_cast<long double>(amount) / volume, "QBMM donor concentration loses relative precision.");
    return trace;
}

SIMPLEFLUID_SOLVERS_LOCAL inline double qbmm_closed_budget_tolerance(int ranks)
{
    return (64. + 2. * ranks) * std::numeric_limits<double>::epsilon();
}

/** Use the mesh communicator and keep tiny shape budgets in extended precision. */
SIMPLEFLUID_SOLVERS_LOCAL inline void sum_qbmm_budget(
    const Teuchos::Comm<int>& comm, std::span<const long double> local, std::span<long double> global)
{
    if (comm.getSize() == 1)
    {
        std::copy(local.begin(), local.end(), global.begin());
        return;
    }
#ifdef HAVE_TEUCHOS_MPI
    if (const auto* mpi = dynamic_cast<const Teuchos::MpiComm<int>*>(&comm))
    {
        if (MPI_Allreduce(local.data(), global.data(), static_cast<int>(local.size()), MPI_LONG_DOUBLE, MPI_SUM,
                *mpi->getRawMpiComm()) != MPI_SUCCESS)
            throw std::runtime_error("QBMM native budget reduction failed.");
        return;
    }
#endif
    throw std::invalid_argument("Distributed QBMM transport requires a native MPI communicator.");
}

/** Channels are catalog species, H0, H2, H3 and signed U, in that order.
 * All channels start from their accepted extensive stock and take the SAME number of
 * positive Jacobi sweeps. A scalar cannot choose its own initializer or stop early.
 * In exact arithmetic every iterate is one common nonnegative linear map of the
 * accepted measures. The caller must still validate the represented combined block.
 */
SIMPLEFLUID_SOLVERS_LOCAL inline BlockResult enforce_block(const SP<const Flow::Mesh>& grid,
    const std::vector<std::vector<double>>& old_amount, const Face& phase_flux, double dt,
    std::vector<std::vector<double>>& amount, const Flow::Pack::matrix_type& matrix,
    const Flow::Pack::vector_type& count_rhs, const LinearSolverOptions& options, double physical_tolerance,
    size_t species)
{
    const auto& mesh = *grid;
    const auto comm = mesh.owned_cell_map()->getComm();
    const size_t channels = species + 4, cells = mesh.num_owned_cells();
    BlockResult result;
    std::vector<Scalar> concentrations;
    std::vector<Face> packets;
    std::vector<std::vector<double>> rhs;
    std::vector<std::vector<Sum>> incoming;
    std::vector<double> volume, local_rhs_max, rhs_max, local_error, error;
    std::vector<long double> denominator, local_sums, sums;
    std::vector<LO> overlap_columns;
    Flow::Pack::matrix_type::local_matrix_host_type local_matrix;
    constexpr size_t sum_channels = 4; // RHS norm squared, residual norm squared, stock delta, stock magnitude.
    collective_detail::collective_local_validation(mesh, "QBMM common transport allocation",
        [&]
        {
            if (old_amount.size() != channels || channels > std::numeric_limits<int>::max() / sum_channels)
                throw std::invalid_argument("QBMM common transport channel layout differs.");
            for (const auto& values : old_amount)
                if (values.size() != cells)
                    throw std::invalid_argument("QBMM common transport owned-cell layout differs.");
            amount = old_amount;
            result.components.resize(channels);
            result.owned_flux.resize(mesh.num_owned_faces());
            for (auto& packet : result.owned_flux)
                packet.species_molar_flux.resize(species);
            concentrations.reserve(channels);
            packets.reserve(channels);
            rhs.assign(channels, std::vector<double>(cells));
            incoming.assign(channels, std::vector<Sum>(cells));
            volume.resize(cells);
            denominator.resize(cells);
            local_rhs_max.resize(channels);
            rhs_max.resize(channels);
            local_error.resize(channels);
            error.resize(channels);
            local_sums.resize(sum_channels * channels);
            sums.resize(sum_channels * channels);
            local_matrix = matrix.getLocalMatrixHost();
            const auto columns = matrix.getColMap();
            const auto overlap = mesh.overlap_cell_map();
            overlap_columns.resize(columns->getLocalNumElements());
            for (size_t j = 0; j < overlap_columns.size(); ++j)
            {
                const auto gid = columns->getGlobalElement(static_cast<LO>(j));
                const LO column = overlap->getLocalElement(gid);
                if (column < 0 || static_cast<size_t>(column) >= mesh.num_local_cells())
                    throw std::invalid_argument("QBMM matrix column is absent from the native overlap.");
                overlap_columns[j] = column;
            }
            const auto native_count_rhs = count_rhs.getData(0);
            for (size_t c = 0; c < cells; ++c)
            {
                const LO cell = static_cast<LO>(c);
                volume[c] = mesh.cell_volume(cell);
                Sum outflow;
                for (const LO face : mesh.faces(cell))
                {
                    const double phi = (mesh.owner_cell(face) == cell ? 1. : -1.) * phase_flux.local_value(face);
                    if (phi > 0)
                        outflow += phi;
                }
                denominator[c] = 1 + static_cast<long double>(dt) * outflow.value() / volume[c];
                if (!(denominator[c] > 0) || !std::isfinite(denominator[c]))
                    throw std::invalid_argument("QBMM common transport row is not representable.");
                const double history_coefficient = volume[c] / dt;
                for (size_t q = 0; q < channels; ++q)
                {
                    const double old = old_amount[q][c];
                    if (!std::isfinite(old) || (q + 1 != channels && old < 0))
                        throw std::invalid_argument("QBMM accepted transport channel is invalid.");
                    const double trace = transport_concentration(old, volume[c]);
                    // The native factory supplies the common operator and the count RHS.
                    // Other channels use that same unit-storage BE history construction.
                    const double history = history_coefficient * trace;
                    if (q == species && history != native_count_rhs[c])
                        throw std::invalid_argument("QBMM common history differs from the native stored RHS.");
                    rhs[q][c] = q == species ? native_count_rhs[c] : history;
                    if (!std::isfinite(rhs[q][c]) || (old != 0 && rhs[q][c] == 0))
                        throw std::invalid_argument("QBMM stored transport RHS is not representable.");
                    if (std::fpclassify(rhs[q][c]) == FP_SUBNORMAL)
                        require_transport_roundoff(rhs[q][c], static_cast<long double>(history_coefficient) * trace,
                            "QBMM stored transport RHS loses relative precision.");
                    local_rhs_max[q] = std::max(local_rhs_max[q], std::abs(rhs[q][c]));
                }
            }
        });
    const auto channel_count = static_cast<int>(channels);
    const double budget_tolerance = qbmm_closed_budget_tolerance(comm->getSize());
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, channel_count, local_rhs_max.data(), rhs_max.data());
    for (size_t q = 0; q < channels; ++q)
    {
        concentrations.emplace_back(grid, "qbmm_common_transport_trace", false);
        packets.emplace_back(grid, "qbmm_common_transport_packet", false);
    }
    for (int sweep = 0;; ++sweep)
    {
        collective_detail::collective_local_validation(mesh, "QBMM common transport concentrations",
            [&]
            {
                for (size_t q = 0; q < channels; ++q)
                    for (size_t c = 0; c < cells; ++c)
                    {
                        if (q + 1 != channels && amount[q][c] < 0)
                            throw std::invalid_argument("QBMM common transported channel is invalid.");
                        concentrations[q].set_owned_value(
                            static_cast<LO>(c), transport_concentration(amount[q][c], volume[c]));
                    }
            });
        for (auto& concentration : concentrations)
            concentration.sync_ghosts();
        collective_detail::collective_local_validation(mesh, "QBMM common canonical packets",
            [&]
            {
                for (size_t q = 0; q < channels; ++q)
                    set_owned_donor_packets(
                        mesh, phase_flux, concentrations[q], packets[q], /*require_relative_roundoff=*/true);
            });
        for (auto& packet : packets)
            packet.sync_ghosts();
        collective_detail::collective_local_validation(mesh, "QBMM physical and stored-matrix balances",
            [&]
            {
                for (size_t q = 0; q < channels; ++q)
                {
                    Sum stock_delta, stock_magnitude, rhs_squared, residual_squared;
                    local_error[q] = 0;
                    const long double norm_scale = rhs_max[q] > 0 ? rhs_max[q] : 1;
                    for (size_t c = 0; c < cells; ++c)
                    {
                        const LO cell = static_cast<LO>(c);
                        incoming[q][c] = Sum{};
                        Sum face_balance, face_magnitude;
                        for (const LO face : mesh.faces(cell))
                        {
                            const double sign = mesh.owner_cell(face) == cell ? 1. : -1.;
                            const double packet = sign * packets[q].local_value(face);
                            const double phi = sign * phase_flux.local_value(face);
                            face_balance += static_cast<long double>(dt) * packet;
                            face_magnitude += std::abs(static_cast<long double>(dt) * packet);
                            if (phi < 0)
                                incoming[q][c] += -static_cast<long double>(packet);
                        }
                        Sum defect;
                        defect += amount[q][c];
                        defect += -static_cast<long double>(old_amount[q][c]);
                        stock_delta += defect.value();
                        const long double stock_scale = std::abs(static_cast<long double>(amount[q][c])) +
                                                        std::abs(static_cast<long double>(old_amount[q][c]));
                        stock_magnitude += stock_scale;
                        defect += face_balance.value();
                        double cell_error = source_stage_detail::normalized(
                            defect.value(), stock_scale + face_magnitude.value(), physical_tolerance);
                        if (q == species)
                        {
                            // Retain the count path's independently ordered native balance check.
                            const long double flux =
                                static_cast<long double>(dt) * FVM::cell_flux_balance(mesh, packets[q], cell);
                            cell_error = std::max(
                                cell_error, source_stage_detail::normalized(
                                                static_cast<long double>(amount[q][c]) - old_amount[q][c] + flux,
                                                stock_scale + std::abs(flux), physical_tolerance));
                        }
                        local_error[q] = std::max(local_error[q], cell_error);
                        Sum residual;
                        residual += rhs[q][c];
                        for (auto entry = local_matrix.graph.row_map(c); entry < local_matrix.graph.row_map(c + 1);
                            ++entry)
                        {
                            const LO column = overlap_columns[local_matrix.graph.entries(entry)];
                            residual += -static_cast<long double>(local_matrix.values(entry)) *
                                        concentrations[q].local_value(column);
                        }
                        const long double b = rhs[q][c] / norm_scale, r = residual.value() / norm_scale;
                        rhs_squared += b * b;
                        residual_squared += r * r;
                    }
                    auto out = local_sums.begin() + sum_channels * q;
                    for (const auto& sum : {rhs_squared, residual_squared, stock_delta, stock_magnitude})
                    {
                        if (!std::isfinite(sum.value()))
                            throw std::invalid_argument("QBMM true residual or stock budget is unrepresentable.");
                        *out++ = sum.value();
                    }
                }
            });
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, channel_count, local_error.data(), error.data());
        sum_qbmm_budget(*comm, local_sums, sums);
        bool converged = true;
        for (size_t q = 0; q < channels; ++q)
        {
            const auto* totals = sums.data() + sum_channels * q;
            auto& component = result.components[q];
            if (sweep == 0)
                component.initial_residual = error[q];
            component.final_residual = error[q];
            component.linear.iterations = 0; // All work is the explicitly reported common balance sweeps.
            component.linear.rhs_norm = static_cast<double>(rhs_max[q] * std::sqrt(totals[0]));
            component.linear.achieved_tolerance =
                static_cast<double>(totals[0] > 0 ? std::sqrt(totals[1] / totals[0]) : std::sqrt(totals[1]));
            component.linear.converged = std::isfinite(component.linear.rhs_norm) &&
                                         std::isfinite(component.linear.achieved_tolerance) &&
                                         component.linear.achieved_tolerance <= options.tolerance;
            // A closed family cannot spend the configurable solve/physical tolerance on
            // artificial global loss. Continue the WHOLE map until the signed budget
            // is at the binary64 reduction/packet arithmetic scale, with no SI floor.
            const double global_error = source_stage_detail::normalized(totals[2], totals[3], budget_tolerance);
            converged &= error[q] <= 1 && global_error <= 1 && component.linear.converged;
        }
        if (converged)
        {
            result.sweeps = sweep;
            for (size_t f = 0; f < result.owned_flux.size(); ++f)
            {
                const LO face = static_cast<LO>(f);
                auto& receipt = result.owned_flux[f];
                for (size_t s = 0; s < species; ++s)
                    receipt.species_molar_flux[s] = packets[s].owned_value(face);
                receipt.population_number_flux = packets[species].owned_value(face);
                receipt.shape_flux = {packets[species + 1].owned_value(face), packets[species + 2].owned_value(face)};
                receipt.internal_energy_flux = packets[species + 3].owned_value(face);
            }
            return result;
        }
        if (sweep >= options.max_iterations)
            throw std::runtime_error("QBMM common positive transport exhausted its original scalar iteration budget.");
        collective_detail::collective_local_validation(mesh, "QBMM common positive transport sweep",
            [&]
            {
                for (size_t q = 0; q < channels; ++q)
                    for (size_t c = 0; c < cells; ++c)
                    {
                        Sum numerator;
                        numerator += old_amount[q][c];
                        numerator += static_cast<long double>(dt) * incoming[q][c].value();
                        const long double updated = numerator.value() / denominator[c];
                        amount[q][c] = narrow(updated);
                        if (std::fpclassify(amount[q][c]) == FP_SUBNORMAL)
                            require_transport_roundoff(
                                amount[q][c], updated, "QBMM transported amount loses relative precision.");
                        if (q + 1 != channels && amount[q][c] < 0)
                            throw std::invalid_argument("QBMM positive transport produced a negative channel.");
                    }
            });
    }
}
} // namespace SimpleFluid::Reacting::transport_consistency_detail
