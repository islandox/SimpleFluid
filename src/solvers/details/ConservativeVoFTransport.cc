/** @file ConservativeVoFTransport.cc
 * @brief Native conservative implicit upwind assembly and per-inventory validation.
 */
#include "solvers/details/ConservativeVoFTransport.hh"
#include "utils/CompensatedSum.hh"

#include <Teuchos_CommHelpers.hpp>
#include <algorithm>
#include <cmath>
#include <map>
#include <stdexcept>
#include <sstream>

namespace SimpleFluid::detail
{
namespace
{
using Pack = DefaultTpetraTypes;
using LO = Pack::local_ordinal_type;
using GO = Pack::global_ordinal_type;
void require(const Pack::map_type& map, bool valid, const char* message)
{
    int local = valid, global = 0;
    Teuchos::reduceAll(*map.getComm(), Teuchos::REDUCE_MIN, 1, &local, &global);
    if (!global)
        throw std::runtime_error(message);
}
std::vector<double> import(const std::vector<double>& values, const Pack::import_type& plan,
    Pack::vector_type& from, Pack::vector_type& to)
{
    {
        auto data = from.getDataNonConst();
        std::copy_n(values.begin(), data.size(), data.begin());
    }
    to.doImport(from, plan, Tpetra::INSERT);
    const auto data = to.getData();
    return {data.begin(), data.end()};
}
} // namespace

ConservativeVoFTransport::ConservativeVoFTransport(SP<const MeshHandle<>> mesh)
    : mesh_(std::move(mesh)), cell_import_(mesh_->owned_cell_map(), mesh_->overlap_cell_map()),
      face_import_(mesh_->owned_face_map(), mesh_->overlap_face_map()),
      rhs_(mesh_->owned_cell_map(), false), solution_(mesh_->owned_cell_map(), false),
      cell_owned_(mesh_->owned_cell_map(), false), cell_overlap_(mesh_->overlap_cell_map(), false),
      face_owned_(mesh_->owned_face_map(), false), face_overlap_(mesh_->overlap_face_map(), false)
{
    const auto execution = mesh_->acquire_execution_view();
    for (std::size_t c = 0; c < mesh_->num_owned_cells(); ++c)
        row_capacity_ = std::max(row_capacity_, mesh_->faces(static_cast<LO>(c)).size() + 1);
}

std::vector<double> ConservativeVoFTransport::cells(const std::vector<double>& owned) const
{
    return import(owned, cell_import_, cell_owned_, cell_overlap_);
}
std::vector<double> ConservativeVoFTransport::faces(const std::vector<double>& overlap) const
{
    return import(overlap, face_import_, face_owned_, face_overlap_);
}

ConservativeVoFTransport::Result ConservativeVoFTransport::solve(const std::vector<double>& old,
    const std::vector<double>& source, const std::vector<double>& capacity, const std::vector<double>& flux,
    bool nonnegative, int maximum_iterations, double tolerance, bool equilibrate_rows)
{
    const auto map = mesh_->owned_cell_map();
    const auto n = mesh_->num_owned_cells();
    require(*map, old.size() == n && source.size() == n && capacity.size() == n && flux.size() == mesh_->num_faces(),
        "Conservative VoF transport has inconsistent storage/flux dimensions.");
    bool valid = true;
    double local_max_flux = 0, global_max_flux = 0;
    for (double f : flux)
    {
        valid = valid && std::isfinite(f);
        local_max_flux = std::max(local_max_flux, std::abs(f));
    }
    for (std::size_t c = 0; c < n; ++c)
        valid = valid && std::isfinite(capacity[c]) && capacity[c] >= 0 && std::isfinite(old[c] + source[c]);
    require(*map, valid, "Conservative VoF transport requires finite nonnegative storage and finite flux/RHS.");
    Teuchos::reduceAll(*map->getComm(), Teuchos::REDUCE_MAX, 1, &local_max_flux, &global_max_flux);
    Result result{std::vector<double>(n), std::vector<double>(flux.size()), std::vector<double>(flux.size())};
    std::vector<double> concentration(n);
    if (global_max_flux == 0)
    {
        for (std::size_t c = 0; c < n; ++c)
        {
            result.amount[c] = old[c] + source[c];
            valid = valid && (capacity[c] > 0 || result.amount[c] == 0) && (!nonnegative || result.amount[c] >= 0);
            concentration[c] = capacity[c] > 0 ? result.amount[c] / capacity[c] : 0;
        }
        require(*map, valid, "A conservative VoF source exhausts inventory or has no carrier.");
    }
    else
    {
        // Exact coefficient equality is collective: an empty-owner rank must
        // follow another rank's numeric refresh. RHS/source/positivity differ
        // freely; only final storage and integrated carrier flux identify A.
        int local_same = numeric_valid_ && equilibrated_ == equilibrate_rows && capacity_ == capacity && flux_ == flux,
            same = 0;
        Teuchos::reduceAll(*map->getComm(), Teuchos::REDUCE_MIN, 1, &local_same, &same);
        if (!same)
        {
            std::string invalid_row;
            numeric_valid_ = false;
            if (equilibrate_rows)
                diagonal_.resize(n);
            const bool first = matrix_.is_null();
            if (first)
                matrix_ = Teuchos::rcp(new Pack::matrix_type(map, row_capacity_));
            else
            {
                if (!matrix_->isFillActive())
                    matrix_->resumeFill();
                matrix_->setAllToScalar(0);
            }
            {
                const auto execution = mesh_->acquire_execution_view();
                for (std::size_t c = 0; c < n; ++c)
                {
                    const auto lid = static_cast<LO>(c);
                    const auto gid = mesh_->cell_global_id(lid);
                    double diagonal = capacity[c];
                    bool incoming = false;
                    std::map<GO, double> row;
                    for (const auto face : mesh_->faces(lid))
                    {
                        if (mesh_->is_geometry_exterior_face(face))
                            continue;
                        const bool owner = mesh_->owner_cell(face) == lid;
                        const double outgoing = flux[face] * (owner ? 1 : -1);
                        // Retain every neighbor in the graph, including zeros:
                        // changing direction must refresh values, not topology.
                        auto& coefficient = row[mesh_->cell_global_id(mesh_->opposite_cell(face, lid))];
                        if (outgoing >= 0)
                            diagonal += outgoing;
                        else
                        {
                            coefficient += outgoing;
                            incoming = true;
                        }
                    }
                    if (diagonal == 0 && !incoming)
                        diagonal = 1;
                    valid = valid && diagonal > 0;
                    if (!(diagonal > 0) && invalid_row.empty())
                    {
                        std::ostringstream message;
                        message.precision(17);
                        message << " gid=" << gid << " capacity=" << capacity[c] << " old=" << old[c]
                                << " source=" << source[c] << " incoming=";
                        for (const auto& [column, value] : row)
                            message << column << ':' << value << ',';
                        invalid_row = message.str();
                    }
                    row[gid] += diagonal;
                    if (equilibrate_rows)
                    {
                        diagonal_[c] = diagonal;
                        // Direct division is essential for subnormal phase
                        // storage: a reciprocal could overflow although every
                        // normalized coefficient and concentration is finite.
                        if (diagonal > 0)
                            for (auto& [column, value] : row)
                                value /= diagonal;
                    }
                    Teuchos::Array<GO> columns;
                    Teuchos::Array<double> coefficients;
                    for (const auto& [column, value] : row)
                    {
                        columns.push_back(column);
                        coefficients.push_back(value);
                    }
                    if (first)
                        matrix_->insertGlobalValues(gid, columns(), coefficients());
                    else
                        matrix_->replaceGlobalValues(gid, columns(), coefficients());
                }
            }
            require(*map, valid,
                ("Conservative VoF transport has a zero-storage incoming-only row." + invalid_row).c_str());
            matrix_->fillComplete();
            solver_.notify_operator_values_changed();
            capacity_ = capacity;
            flux_ = flux;
            equilibrated_ = equilibrate_rows;
            numeric_valid_ = true;
        }
        // A zero-storage isolated row is an identity placeholder, never a
        // physical capacity floor. Validate its RHS independently on every call.
        {
            const auto execution = mesh_->acquire_execution_view();
            for (std::size_t c = 0; c < n; ++c)
            {
                bool connected = false;
                for (const auto face : mesh_->faces(static_cast<LO>(c)))
                    connected = connected || flux[face] != 0;
                valid = valid && (capacity[c] > 0 || connected || old[c] + source[c] == 0);
                const double rhs = equilibrate_rows ? (old[c] + source[c]) / diagonal_[c] : old[c] + source[c];
                valid = valid && std::isfinite(rhs);
                rhs_.replaceLocalValue(static_cast<LO>(c), rhs);
            }
        }
        require(*map, valid, "Conservative VoF inventory has no storage or transport path.");
        LinearSolverOptions linear;
        linear.backend = LinearSolverBackend::Gmres;
        linear.preconditioner = LinearPreconditioner::Jacobi;
        linear.reuse_preconditioner = true;
        linear.tolerance = tolerance;
        if (equilibrate_rows)
        {
            double local_max[2]{}, global_max[2]{};
            {
                const auto execution = mesh_->acquire_execution_view();
                for (std::size_t c = 0; c < n; ++c)
                {
                    bool connected = false;
                    for (const auto face : mesh_->faces(static_cast<LO>(c)))
                        connected = connected || flux[face] != 0;
                    if (capacity[c] > 0 || connected)
                        local_max[0] = std::max(local_max[0], diagonal_[c]);
                    local_max[1] = std::max(local_max[1], std::abs(old[c] + source[c]));
                }
            }
            Teuchos::reduceAll(*map->getComm(), Teuchos::REDUCE_MAX, 2, local_max, global_max);
            if (global_max[1] > 0)
            {
                CompensatedSum<> physical, normalized;
                for (std::size_t c = 0; c < n; ++c)
                {
                    const long double b = (old[c] + source[c]) / global_max[1];
                    const long double bhat = static_cast<long double>((old[c] + source[c]) / diagonal_[c]) *
                                              global_max[0] / global_max[1];
                    physical += b * b;
                    normalized += bhat * bhat;
                }
                double local[2] = {static_cast<double>(physical.value()), static_cast<double>(normalized.value())},
                       global[2]{};
                Teuchos::reduceAll(*map->getComm(), Teuchos::REDUCE_SUM, 2, local, global);
                // ||D*rhat|| <= max(D)*||rhat||. Tighten the scaled solve
                // enough to retain the original unscaled residual target,
                // with a factor four reserve for floating-point accumulation.
                linear.tolerance = 0.25 * tolerance * std::min(1.0, std::sqrt(global[0] / global[1]));
                require(*map, std::isfinite(linear.tolerance) && linear.tolerance > 0,
                    "Conservative VoF row scaling cannot represent its physical residual target.");
            }
        }
        linear.max_iterations = maximum_iterations;
        linear.verbosity = 0;
        const auto stats = solver_.solve_from_zero_with_statistics(
            Teuchos::rcp_implicit_cast<const Pack::operator_type>(matrix_), rhs_, solution_, linear);
        require(*map, stats.converged, "Conservative VoF implicit transport did not converge to its true residual.");
        const auto values = solution_.getData();
        std::copy(values.begin(), values.end(), concentration.begin());
        for (std::size_t c = 0; c < n; ++c)
        {
            result.amount[c] = capacity[c] * concentration[c];
            valid = valid && std::isfinite(result.amount[c]) && (!nonnegative || result.amount[c] >= 0);
        }
        require(*map, valid, "Conservative VoF transport lost inventory positivity or finiteness.");
    }
    const auto halo = cells(concentration);
    {
        const auto execution = mesh_->acquire_execution_view();
        for (std::size_t id = 0; id < mesh_->num_owned_faces(); ++id)
        {
            const auto face = static_cast<LO>(id);
            if (mesh_->is_geometry_exterior_face(face))
                continue;
            const auto owner = mesh_->owner_cell(face), neighbor = mesh_->neighbor_cell(face);
            result.trace[id] = flux[id] == 0 ? 0.5 * (halo[owner] + halo[neighbor])
                                            : halo[flux[id] > 0 ? owner : neighbor];
            result.flux[id] = flux[id] * result.trace[id];
        }
    }
    result.flux = faces(result.flux);
    result.trace = faces(result.trace);
    if (equilibrate_rows)
    {
        // Scaling changes conditioning, never the original convergence test.
        // Normalize before squaring so trace inventories cannot underflow out
        // of their physical residual check.
        double local_scale = 0, scale = 0;
        for (std::size_t c = 0; c < n; ++c)
            local_scale = std::max(local_scale, std::abs(old[c] + source[c]));
        Teuchos::reduceAll(*map->getComm(), Teuchos::REDUCE_MAX, 1, &local_scale, &scale);
        if (scale == 0)
            scale = 1;
        CompensatedSum<> residual_squared, rhs_squared;
        {
            const auto execution = mesh_->acquire_execution_view();
            for (std::size_t c = 0; c < n; ++c)
            {
                CompensatedSum<> residual;
                residual += result.amount[c];
                residual += -(old[c] + source[c]);
                for (const auto face : mesh_->faces(static_cast<LO>(c)))
                    residual += (mesh_->owner_cell(face) == static_cast<LO>(c) ? 1 : -1) * result.flux[face];
                const long double r = residual.value() / scale, b = (old[c] + source[c]) / scale;
                residual_squared += r * r;
                rhs_squared += b * b;
            }
        }
        double local[2] = {static_cast<double>(residual_squared.value()), static_cast<double>(rhs_squared.value())},
               global[2]{};
        Teuchos::reduceAll(*map->getComm(), Teuchos::REDUCE_SUM, 2, local, global);
        require(*map, std::isfinite(global[0]) && std::isfinite(global[1]) &&
                          global[0] <= tolerance * tolerance * (global[1] > 0 ? global[1] : 1),
            "Equilibrated conservative VoF transport failed its original physical residual.");
    }
    return result;
}
} // namespace SimpleFluid::detail
