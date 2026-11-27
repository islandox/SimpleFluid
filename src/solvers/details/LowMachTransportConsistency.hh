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
#include <array>
#include <chrono>
#include <cmath>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <sstream>
#include <utility>
#include <vector>

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

SIMPLEFLUID_SOLVERS_LOCAL inline void require_transport_roundoff(double rounded, long double value, const char* message)
{
    if (value == 0 ||
        std::abs((static_cast<long double>(rounded) - value) / value) > std::numeric_limits<double>::epsilon())
        throw std::invalid_argument(message);
}

SIMPLEFLUID_SOLVERS_LOCAL inline double donor_packet(double phi, double trace, bool require_relative_roundoff = false)
{
    const double packet = phi * trace;
    if (!std::isfinite(packet) || (phi != 0 && trace != 0 && packet == 0))
        throw std::invalid_argument("Canonical transport face packet is not representable.");
    if (require_relative_roundoff && std::fpclassify(packet) == FP_SUBNORMAL)
        require_transport_roundoff(
            packet, static_cast<long double>(phi) * trace, "QBMM canonical face packet loses relative precision.");
    return packet;
}

SIMPLEFLUID_SOLVERS_LOCAL inline bool zero_transport_row(const Flow::Mesh& mesh, const Face& phase_flux, LO cell)
{
    return std::ranges::all_of(mesh.faces(cell), [&](LO face) { return phase_flux.local_value(face) == 0.; });
}

SIMPLEFLUID_SOLVERS_LOCAL inline void set_owned_donor_packets(
    const Flow::Mesh& mesh, const Face& phase_flux, const Scalar& concentration, Face& packets)
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
        const LO owner = mesh.owner_cell(face);
        const LO donor = phi >= 0 ? owner : mesh.opposite_or_periodic_neighbor_cell(face, owner);
        packets.set_owned_value(face, donor_packet(phi, concentration.local_value(donor)));
    }
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
                const double candidate = value * volume(c);
                if (!std::isfinite(value) || !std::isfinite(candidate) || (value != 0 && candidate == 0))
                    throw std::invalid_argument("Krylov transport inventory is nonfinite or unrepresentable.");
                // A row without relative flux conserves its extensive stock, including on ALE.
                // The donor and stored-operator checks below still use the resulting concentration.
                amount[c] = candidate != old_amount[c] && zero_transport_row(mesh, phase_flux, static_cast<LO>(c))
                                ? old_amount[c]
                                : candidate;
                local_restart |= nonnegative && candidate < 0;
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
            [&] { set_owned_donor_packets(mesh, phase_flux, concentration, packets); });
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

struct BatchedResult
{
    LinearSolveStatistics linear;
    int total_refinements = 0;
    std::vector<int> column_refinements;
    double initial_residual = 0;
    double final_residual = 0;
    std::vector<double> column_initial_residuals;
    std::vector<double> column_final_residuals;
    std::size_t halo_cell_imports = 0;
    std::size_t halo_face_imports = 0;
    double halo_exchange_seconds = 0;
    double consistency_repair_seconds = 0;
};

struct SIMPLEFLUID_SOLVERS_LOCAL BatchedWorkspace
{
    using Import = Flow::Pack::import_type;
    using MultiVector = Flow::Pack::multi_vector_type;

    Teuchos::RCP<const Flow::Pack::map_type> owned_cell_map;
    Teuchos::RCP<const Flow::Pack::map_type> overlap_cell_map;
    Teuchos::RCP<const Flow::Pack::map_type> owned_face_map;
    Teuchos::RCP<const Flow::Pack::map_type> overlap_face_map;
    std::shared_ptr<Import> cell_import;
    std::shared_ptr<Import> face_import;
    Teuchos::RCP<Flow::Pack::multi_vector_type> conc_owned;
    Teuchos::RCP<Flow::Pack::multi_vector_type> conc_overlap;
    Teuchos::RCP<Flow::Pack::multi_vector_type> face_owned;
    Teuchos::RCP<Flow::Pack::multi_vector_type> face_overlap;
    Teuchos::RCP<Flow::Pack::multi_vector_type> residual_mv;

    void ensure(const Flow::Mesh& mesh, const Face& phase_flux, std::size_t ncols, bool store_residual = false)
    {
        const auto cell_owned_map = mesh.owned_cell_map();
        const auto cell_overlap_map = mesh.overlap_cell_map();
        const auto face_owned_map_now = phase_flux.owned_data().getMap();
        const auto face_overlap_map_now = phase_flux.overlap_data().getMap();
        const auto wrong_width = [ncols](const Teuchos::RCP<MultiVector>& buffer)
        { return buffer.is_null() || buffer->getNumVectors() != ncols; };
        const std::array<int, 5> local{
            !cell_import || owned_cell_map != cell_owned_map || overlap_cell_map != cell_overlap_map,
            !face_import || owned_face_map != face_owned_map_now || overlap_face_map != face_overlap_map_now,
            wrong_width(conc_owned) || wrong_width(conc_overlap),
            wrong_width(face_owned) || wrong_width(face_overlap),
            store_residual && wrong_width(residual_mv)};
        std::array<int, 5> refresh{};
        Teuchos::reduceAll(*cell_owned_map->getComm(), Teuchos::REDUCE_MAX, static_cast<int>(local.size()),
            local.data(), refresh.data());
        const bool new_cells = refresh[0] || refresh[2], new_faces = refresh[1] || refresh[3];
        const bool new_residual = store_residual && (new_cells || refresh[4]);
        if (!new_cells && !new_faces && !new_residual)
            return;

        // Reserve host storage collectively before entering the importer constructors.
        // Keep the previous workspace complete until all replacement buffers are ready.
        std::shared_ptr<std::optional<Import>> cell_storage, face_storage;
        Teuchos::RCP<MultiVector> next_conc_owned, next_conc_overlap, next_face_owned, next_face_overlap, next_residual;
        collective_detail::collective_local_validation(mesh, "Batched transport workspace allocation",
            [&]
            {
                if (refresh[0])
                    cell_storage = std::make_shared<std::optional<Import>>();
                if (refresh[1])
                    face_storage = std::make_shared<std::optional<Import>>();
                if (new_cells)
                {
                    next_conc_owned = Teuchos::rcp(new MultiVector(cell_owned_map, ncols, false));
                    next_conc_overlap = Teuchos::rcp(new MultiVector(cell_overlap_map, ncols, false));
                }
                if (new_faces)
                {
                    next_face_owned = Teuchos::rcp(new MultiVector(face_owned_map_now, ncols, false));
                    next_face_overlap = Teuchos::rcp(new MultiVector(face_overlap_map_now, ncols, false));
                }
                if (new_residual)
                    next_residual = Teuchos::rcp(new MultiVector(cell_owned_map, ncols, false));
            });
        if (cell_storage)
            cell_storage->emplace(cell_owned_map, cell_overlap_map);
        if (face_storage)
            face_storage->emplace(face_owned_map_now, face_overlap_map_now);
        if (cell_storage)
        {
            cell_import = std::shared_ptr<Import>(cell_storage, &cell_storage->value());
            owned_cell_map = cell_owned_map;
            overlap_cell_map = cell_overlap_map;
        }
        if (face_storage)
        {
            face_import = std::shared_ptr<Import>(face_storage, &face_storage->value());
            owned_face_map = face_owned_map_now;
            overlap_face_map = face_overlap_map_now;
        }
        if (new_cells)
        {
            conc_owned = std::move(next_conc_owned);
            conc_overlap = std::move(next_conc_overlap);
            residual_mv = std::move(next_residual);
        }
        else if (new_residual)
            residual_mv = std::move(next_residual);
        if (new_faces)
        {
            face_owned = std::move(next_face_owned);
            face_overlap = std::move(next_face_overlap);
        }
    }
};

SIMPLEFLUID_SOLVERS_LOCAL inline BatchedResult enforce_batch(const SP<const Flow::Mesh>& grid,
    std::span<const std::span<const double>> old_amounts, std::span<const Quantity> quantities,
    const std::vector<bool>& active_columns, const Face& phase_flux, double dt,
    const Flow::Pack::multi_vector_type& solution_owned, std::vector<std::vector<double>>& new_amounts,
    const Flow::Pack::matrix_type& matrix, const Flow::Pack::multi_vector_type& rhs, LinearSolveStatistics linear,
    const LinearSolverOptions& options, double physical_tolerance, const FVM::ALEControlVolumeState* ale,
    BatchedWorkspace& ws)
{
    const auto repair_start = std::chrono::steady_clock::now();
    const auto& mesh = *grid;
    const auto comm = mesh.owned_cell_map()->getComm();
    const std::size_t ncols = old_amounts.size();
    const std::size_t nc = mesh.num_owned_cells();
    const auto volume = [&](size_t c)
    { return ale ? ale->new_cell_volumes()[c] : mesh.cell_volume(static_cast<LO>(c)); };

    BatchedResult result{std::move(linear)};
    result.column_refinements.assign(ncols, 0);
    result.column_initial_residuals.assign(ncols, 0.0);
    result.column_final_residuals.assign(ncols, 0.0);
    if (ncols == 0)
    {
        return result;
    }

    if (quantities.size() != ncols || active_columns.size() != ncols ||
        solution_owned.getNumVectors() != ncols || rhs.getNumVectors() != ncols)
    {
        throw std::invalid_argument("Low-Mach batched transport consistency column count mismatch.");
    }

    ws.ensure(mesh, phase_flux, ncols, true);
    ws.conc_owned->putScalar(0.0);
    ws.conc_overlap->putScalar(0.0);
    ws.face_owned->putScalar(0.0);
    ws.face_overlap->putScalar(0.0);
    new_amounts.resize(ncols);

    std::vector<int> local_restart(ncols, 0);
    std::vector<int> global_restart(ncols, 0);
    std::vector<bool> column_done(ncols, true);
    bool any_active = false;

    // Preserve empty-carrier closed boundary verification even when all columns are zero.
    collective_detail::collective_local_validation(mesh, "Low-Mach batched transport boundary and initial state",
        [&]
        {
            for (std::size_t f = 0; f < mesh.num_owned_faces(); ++f)
            {
                const LO face = static_cast<LO>(f);
                if (mesh.is_geometry_exterior_face(face) && phase_flux.local_value(face) != 0)
                {
                    throw std::invalid_argument("Closed phase transport has nonzero relative boundary flux.");
                }
            }
            for (std::size_t k = 0; k < ncols; ++k)
            {
                if (old_amounts[k].size() != nc)
                {
                    throw std::invalid_argument("Low-Mach batched transport inventory cell count mismatch.");
                }
                new_amounts[k].assign(nc, 0.0);
                const bool nonnegative = quantities[k] != Quantity::Energy;
                if (!active_columns[k])
                {
                    for (std::size_t c = 0; c < nc; ++c)
                    {
                        if (old_amounts[k][c] != 0.0)
                        {
                            throw std::invalid_argument("Inactive transport column has nonzero accepted stock.");
                        }
                    }
                    continue;
                }
                any_active = true;
                column_done[k] = false;
                const auto conc_col = solution_owned.getData(k);
                for (std::size_t c = 0; c < nc; ++c)
                {
                    if (!std::isfinite(old_amounts[k][c]) || (nonnegative && old_amounts[k][c] < 0))
                    {
                        throw std::invalid_argument("Invalid accepted transport inventory.");
                    }
                    const double value = conc_col[c];
                    const double amt = value * volume(c);
                    if (!std::isfinite(value) || !std::isfinite(amt) || (value != 0 && amt == 0))
                    {
                        throw std::invalid_argument("Krylov transport inventory is nonfinite or unrepresentable.");
                    }
                    new_amounts[k][c] =
                        amt != old_amounts[k][c] && zero_transport_row(mesh, phase_flux, static_cast<LO>(c))
                            ? old_amounts[k][c]
                            : amt;
                    local_restart[k] |= nonnegative && amt < 0;
                }
            }
        });

    if (!any_active)
    {
        result.consistency_repair_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - repair_start).count();
        return result;
    }

    Teuchos::reduceAll(
        *comm, Teuchos::REDUCE_MAX, static_cast<int>(ncols), local_restart.data(), global_restart.data());
    for (std::size_t k = 0; k < ncols; ++k)
    {
        if (active_columns[k] && global_restart[k] != 0)
        {
            std::copy(old_amounts[k].begin(), old_amounts[k].end(), new_amounts[k].begin());
        }
    }

    std::vector<std::vector<Sum>> incoming(ncols, std::vector<Sum>(nc));
    std::vector<std::vector<Sum>> outgoing(ncols, std::vector<Sum>(nc));
    std::vector<double> local_errors(ncols, 0.0);
    std::vector<double> global_errors(ncols, 0.0);
    std::vector<std::size_t> pending;
    pending.reserve(ncols);

    for (;;)
    {
        pending.clear();
        for (std::size_t k = 0; k < ncols; ++k)
        {
            if (!column_done[k])
            {
                pending.push_back(k);
            }
        }
        if (pending.empty())
        {
            break;
        }

        collective_detail::collective_local_validation(mesh, "Low-Mach batched transport donor concentrations",
            [&]
            {
                for (const std::size_t k : pending)
                {
                    const bool nonnegative = quantities[k] != Quantity::Energy;
                    auto conc_dst = ws.conc_owned->getDataNonConst(k);
                    for (std::size_t c = 0; c < nc; ++c)
                    {
                        const double amt = new_amounts[k][c];
                        if (nonnegative && amt < 0)
                        {
                            throw std::invalid_argument("Monotone transport produced a negative inventory.");
                        }
                        const double trace = amt / volume(c);
                        if (!std::isfinite(trace) || (amt != 0 && trace == 0))
                        {
                            throw std::invalid_argument("Returned transport concentration is not representable.");
                        }
                        conc_dst[c] = trace;
                    }
                }
            });

        const auto cell_import_start = std::chrono::steady_clock::now();
        ws.conc_overlap->doImport(*ws.conc_owned, *ws.cell_import, Tpetra::REPLACE);
        result.halo_exchange_seconds +=
            std::chrono::duration<double>(std::chrono::steady_clock::now() - cell_import_start).count();
        ++result.halo_cell_imports;

        collective_detail::collective_local_validation(mesh, "Canonical batched local transport packets",
            [&]
            {
                for (const std::size_t k : pending)
                {
                    const auto conc_local = ws.conc_overlap->getData(k);
                    auto face_dst = ws.face_owned->getDataNonConst(k);
                    for (std::size_t f = 0; f < mesh.num_owned_faces(); ++f)
                    {
                        const LO face = static_cast<LO>(f);
                        const LO f_row = phase_flux.owned_row(face);
                        const double phi = phase_flux.local_value(face);
                        if (mesh.is_geometry_exterior_face(face))
                        {
                            face_dst[f_row] = 0.0;
                            continue;
                        }
                        const LO owner = mesh.owner_cell(face);
                        const LO donor = phi >= 0 ? owner : mesh.opposite_or_periodic_neighbor_cell(face, owner);
                        face_dst[f_row] = donor_packet(phi, conc_local[donor]);
                    }
                }
            });

        const auto face_import_start = std::chrono::steady_clock::now();
        ws.face_overlap->doImport(*ws.face_owned, *ws.face_import, Tpetra::REPLACE);
        result.halo_exchange_seconds +=
            std::chrono::duration<double>(std::chrono::steady_clock::now() - face_import_start).count();
        ++result.halo_face_imports;

        std::fill(local_errors.begin(), local_errors.end(), 0.0);
        collective_detail::collective_local_validation(mesh, "Physical batched componentwise transport balance",
            [&]
            {
                for (const std::size_t k : pending)
                {
                    const auto face_local = ws.face_overlap->getData(k);
                    incoming[k].assign(nc, Sum{});
                    outgoing[k].assign(nc, Sum{});
                    double col_error = 0.0;
                    for (std::size_t c = 0; c < nc; ++c)
                    {
                        const LO cell = static_cast<LO>(c);
                        Sum balance, magnitude;
                        double pop_flux_sum = 0.0;
                        for (const LO face : mesh.faces(cell))
                        {
                            const double sign = mesh.owner_cell(face) == cell ? 1.0 : -1.0;
                            const double phi = sign * phase_flux.local_value(face);
                            const double raw_packet = face_local[face];
                            const double packet = sign * raw_packet;
                            pop_flux_sum += packet;
                            balance += static_cast<long double>(dt) * packet;
                            magnitude += std::abs(static_cast<long double>(dt) * packet);
                            if (phi < 0)
                            {
                                incoming[k][c] += -static_cast<long double>(packet);
                            }
                            else
                            {
                                outgoing[k][c] += phi;
                            }
                        }
                        Sum defect;
                        defect += new_amounts[k][c];
                        defect += -static_cast<long double>(old_amounts[k][c]);
                        defect += balance.value();
                        const long double scale = std::abs(static_cast<long double>(new_amounts[k][c])) +
                                                  std::abs(static_cast<long double>(old_amounts[k][c])) +
                                                  magnitude.value();
                        double cell_error =
                            source_stage_detail::normalized(defect.value(), scale, physical_tolerance);
                        if (quantities[k] == Quantity::Population)
                        {
                            const long double flux = static_cast<long double>(dt) * pop_flux_sum;
                            const long double count_defect =
                                static_cast<long double>(new_amounts[k][c]) - old_amounts[k][c] + flux;
                            cell_error = std::max(
                                cell_error,
                                source_stage_detail::normalized(
                                    count_defect,
                                    std::abs(static_cast<long double>(new_amounts[k][c])) +
                                        std::abs(static_cast<long double>(old_amounts[k][c])) + std::abs(flux),
                                    physical_tolerance));
                        }
                        col_error = std::max(col_error, cell_error);
                    }
                    local_errors[k] = col_error;
                }
            });

        Teuchos::reduceAll(
            *comm, Teuchos::REDUCE_MAX, static_cast<int>(ncols), local_errors.data(), global_errors.data());

        for (const std::size_t k : pending)
        {
            if (result.column_refinements[k] == 0)
            {
                result.column_initial_residuals[k] = global_errors[k];
            }
            result.column_final_residuals[k] = global_errors[k];
        }

        // Recheck passing columns against the ORIGINAL CRS (matrix, rhs) column by column.
        std::vector<std::size_t> crs_candidates;
        crs_candidates.reserve(pending.size());
        for (const std::size_t k : pending)
        {
            if (global_errors[k] <= 1.0)
            {
                crs_candidates.push_back(k);
            }
        }
        if (!crs_candidates.empty())
        {
            collective_detail::collective_local_validation(mesh, "Final batched transport stored-CRS residual",
                [&]
                {
                    const auto local = matrix.getLocalMatrixHost();
                    const auto columns = matrix.getColMap();
                    for (const std::size_t k : crs_candidates)
                    {
                        const auto conc_local = ws.conc_overlap->getData(k);
                        const auto b = rhs.getData(k);
                        auto r = ws.residual_mv->getDataNonConst(k);
                        for (std::size_t c = 0; c < nc; ++c)
                        {
                            Sum value;
                            value += b[c];
                            for (auto entry = local.graph.row_map(c); entry < local.graph.row_map(c + 1); ++entry)
                            {
                                const auto gid = columns->getGlobalElement(local.graph.entries(entry));
                                const LO column = mesh.overlap_cell_map()->getLocalElement(gid);
                                if (column < 0 || static_cast<std::size_t>(column) >= mesh.num_local_cells())
                                {
                                    throw std::invalid_argument(
                                        "Transport CRS column is absent from the native overlap map.");
                                }
                                value += -static_cast<long double>(local.values(entry)) * conc_local[column];
                            }
                            r[c] = static_cast<double>(value.value());
                            if (!std::isfinite(r[c]))
                            {
                                throw std::invalid_argument("Final transport true residual is nonfinite.");
                            }
                        }
                    }
                });
            for (const std::size_t k : crs_candidates)
            {
                const double rhs_norm = rhs.getVector(k)->norm2();
                const double residual_norm = ws.residual_mv->getVector(k)->norm2();
                const double achieved = rhs_norm > 0 ? residual_norm / rhs_norm : residual_norm;
                const bool col_conv =
                    std::isfinite(rhs_norm) && std::isfinite(residual_norm) && std::isfinite(achieved) &&
                    (achieved <= options.tolerance ||
                        (options.absolute_tolerance > 0 && residual_norm <= options.absolute_tolerance));
                if (k < result.linear.column_statistics.size())
                {
                    result.linear.column_statistics[k].rhs_norm = rhs_norm;
                    result.linear.column_statistics[k].true_residual_norm = residual_norm;
                    result.linear.column_statistics[k].achieved_tolerance = achieved;
                    result.linear.column_statistics[k].converged = col_conv;
                }
                if (col_conv && (global_restart[k] == 0 || result.column_refinements[k] > 0))
                {
                    // Freeze accepted column so slower columns never alter its returned fields or packets.
                    column_done[k] = true;
                }
            }
        }

        // Check independent remaining iteration budget and refine unconverged columns only.
        std::vector<std::size_t> to_refine;
        for (const std::size_t k : pending)
        {
            if (column_done[k])
            {
                continue;
            }
            const int col_krylov_iters = k < result.linear.column_statistics.size()
                                             ? result.linear.column_statistics[k].iterations
                                             : result.linear.iterations;
            const int remaining_iterations = options.max_iterations - col_krylov_iters;
            if (result.column_refinements[k] >= remaining_iterations)
            {
                std::ostringstream message;
                message.precision(17);
                message << "Low-Mach local transport exhausted the original scalar iteration budget: column=" << k
                        << ", local balance=" << global_errors[k]
                        << ", Krylov iterations=" << col_krylov_iters
                        << ", remaining iterations=" << remaining_iterations
                        << ", balance refinements=" << result.column_refinements[k];
                throw std::runtime_error(message.str());
            }
            to_refine.push_back(k);
        }
        if (to_refine.empty())
        {
            break;
        }
        collective_detail::collective_local_validation(mesh, "Monotone batched local transport refinement",
            [&]
            {
                for (const std::size_t k : to_refine)
                {
                    const bool nonnegative = quantities[k] != Quantity::Energy;
                    for (std::size_t c = 0; c < nc; ++c)
                    {
                        Sum numerator;
                        numerator += old_amounts[k][c];
                        numerator += static_cast<long double>(dt) * incoming[k][c].value();
                        const long double denominator =
                            1 + static_cast<long double>(dt) * outgoing[k][c].value() / volume(c);
                        if (!(denominator > 0) || !std::isfinite(denominator))
                        {
                            throw std::invalid_argument("Monotone transport row is not finitely representable.");
                        }
                        new_amounts[k][c] = narrow(numerator.value() / denominator);
                        if (nonnegative && new_amounts[k][c] < 0)
                        {
                            throw std::invalid_argument("Monotone transport update is negative.");
                        }
                    }
                }
            });
        for (const std::size_t k : to_refine)
        {
            ++result.column_refinements[k];
            ++result.total_refinements;
        }
    }

    double max_rhs = 0.0;
    double max_achieved = 0.0;
    for (std::size_t k = 0; k < ncols; ++k)
    {
        if (active_columns[k])
        {
            result.initial_residual = std::max(result.initial_residual, result.column_initial_residuals[k]);
            result.final_residual = std::max(result.final_residual, result.column_final_residuals[k]);
            if (k < result.linear.column_statistics.size())
            {
                max_rhs = std::max(max_rhs, result.linear.column_statistics[k].rhs_norm);
                max_achieved = std::max(max_achieved, result.linear.column_statistics[k].achieved_tolerance);
            }
        }
    }
    result.linear.converged = true;
    result.linear.rhs_norm = max_rhs;
    result.linear.achieved_tolerance = max_achieved;
    result.consistency_repair_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - repair_start).count();
    return result;
}
} // namespace SimpleFluid::Reacting::transport_consistency_detail
