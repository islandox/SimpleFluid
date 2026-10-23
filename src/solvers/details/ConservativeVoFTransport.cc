/** @file ConservativeVoFTransport.cc
 * @brief Native conservative implicit upwind assembly and per-inventory validation.
 */
#include "solvers/details/ConservativeVoFTransport.hh"
#include "geometry/ExactSweptGeometry.hh"
#include "FVM/details/BoundedPhaseArithmetic.hh"
#include "parallel/SparseOwnerExchange.hh"
#include <bit>
#include <tuple>
#include "solvers/details/ConservativeVoFExactArithmetic.hh"
#include "solvers/details/GeneralVoFExactStorage.hh"
#include "solvers/details/GeneralVoFTransferBinding.hh"
#include "utils/CompensatedSum.hh"

#include <Teuchos_CommHelpers.hpp>
#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>

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
std::vector<double> import(
    const std::vector<double>& values, const Pack::import_type& plan, Pack::vector_type& from, Pack::vector_type& to)
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

struct ConservativeVoFTransport::SparseAssembly
{
    std::vector<double> outgoing;
    std::vector<std::map<GO, double>> incoming;
    std::vector<FaceContribution> faces;
    std::set<LO> exact_faces;
    Teuchos::RCP<const Pack::map_type> donor_map;
    std::unique_ptr<Pack::import_type> donor_import;
};


ConservativeVoFTransport::ConservativeVoFTransport(SP<const MeshHandle<>> mesh)
    : mesh_(std::move(mesh)), cell_import_(mesh_->owned_cell_map(), mesh_->overlap_cell_map()),
      face_import_(mesh_->owned_face_map(), mesh_->overlap_face_map()), rhs_(mesh_->owned_cell_map(), false),
      solution_(mesh_->owned_cell_map(), false), cell_owned_(mesh_->owned_cell_map(), false),
      cell_overlap_(mesh_->overlap_cell_map(), false), face_owned_(mesh_->owned_face_map(), false),
      face_overlap_(mesh_->overlap_face_map(), false)
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
    std::vector<double> forward(flux.size()), reverse(flux.size());
    for (std::size_t f = 0; f < flux.size(); ++f)
    {
        forward[f] = std::max(flux[f], 0.0);
        reverse[f] = std::max(-flux[f], 0.0);
    }
    return solve_streams(
        old, source, capacity, forward, reverse, nonnegative, maximum_iterations, tolerance, equilibrate_rows);
}

ConservativeVoFTransport::Result ConservativeVoFTransport::solve_streams(const std::vector<double>& old,
    const std::vector<double>& source, const std::vector<double>& capacity, const std::vector<double>& flux,
    const std::vector<double>& reverse, bool nonnegative, int maximum_iterations, double tolerance,
    bool equilibrate_rows)
{
    return solve_impl(
        old, source, capacity, flux, reverse, nonnegative, maximum_iterations, tolerance, equilibrate_rows, nullptr);
}

ConservativeVoFTransport::Result ConservativeVoFTransport::solve_transfers(const std::vector<double>& old,
    const std::vector<double>& source, const std::vector<double>& capacity, const std::vector<double>& outgoing,
    const std::vector<Transfer>& incoming, const std::vector<FaceContribution>& contributions, bool nonnegative,
    int maximum_iterations, double tolerance, bool equilibrate_rows, std::size_t byte_budget)
{
    const auto map = mesh_->owned_cell_map();
    const auto n = mesh_->num_owned_cells();
    require(*map, outgoing.size() == n, "Sparse VoF transport outgoing storage shape mismatch.");
    SparseAssembly sparse;
    sparse.outgoing = outgoing;
    sparse.incoming.resize(n);
    sparse.faces = contributions;
    std::set<GO> donors;
    bool valid = true;
    for (double value : outgoing)
        valid = valid && std::isfinite(value) && value >= 0;
    for (const auto& route : incoming)
    {
        const auto row = map->getLocalElement(route.recipient);
        valid = valid && row >= 0 && route.donor != route.recipient && std::isfinite(route.amount) && route.amount >= 0;
        if (row >= 0)
            valid = sparse.incoming[static_cast<std::size_t>(row)].emplace(route.donor, route.amount).second && valid;
        donors.insert(route.donor);
    }
    std::vector<double> magnitude(mesh_->num_faces()), zero(magnitude.size());
    for (const auto& face : contributions)
    {
        valid = valid && face.face >= 0 && static_cast<std::size_t>(face.face) < mesh_->num_owned_faces() &&
                std::isfinite(face.amount);
        if (face.face >= 0 && static_cast<std::size_t>(face.face) < mesh_->num_owned_faces())
        {
            valid = valid && !mesh_->is_geometry_exterior_face(face.face);
            magnitude[face.face] = std::max(magnitude[face.face], std::abs(face.amount));
        }
        donors.insert(face.donor);
        if (face.exact_amount)
        {
            try
            {
                const double view = static_cast<double>(face.exact_amount->enclosure().value);
                valid = valid && std::isfinite(view) && view == face.amount;
                sparse.exact_faces.insert(face.face);
            }
            catch (...)
            {
                valid = false;
            }
        }
    }
    for (const auto& row : sparse.incoming)
        for (const auto& [donor, amount] : row)
            valid = valid && std::isfinite(amount) && amount >= 0;
    require(
        *map, valid, "Sparse VoF transport requires positive recipient-owned routes and canonical owned-face streams.");
    Teuchos::Array<GO> ids(donors.begin(), donors.end());
    sparse.donor_map = Teuchos::rcp(new Pack::map_type(
        Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid(), ids(), map->getIndexBase(), map->getComm()));
    sparse.donor_import = std::make_unique<Pack::import_type>(map, sparse.donor_map);
    // Import a marker as well as traces: a nonexistent donor must never become
    // a silently inserted zero through an unmatched import.
    Pack::vector_type marker(map), authenticated(sparse.donor_map);
    marker.putScalar(1);
    authenticated.putScalar(0);
    authenticated.doImport(marker, *sparse.donor_import, Tpetra::INSERT);
    for (double value : authenticated.getData())
        valid = valid && value == 1;
    require(*map, valid, "Sparse VoF transport references an absent donor.");
    verify_general_vof_transfer_binding(mesh_, outgoing, sparse.incoming, contributions, byte_budget);
    magnitude = faces(magnitude);
    // Sparse donor support may change after a trajectory/regrid. Rebuild its
    // graph explicitly; numeric refresh still notifies the shared solver.
    numeric_valid_ = false;
    matrix_ = Teuchos::null;
    const auto saved_capacity = row_capacity_;
    for (const auto& row : sparse.incoming)
        row_capacity_ = std::max(row_capacity_, row.size() + 1);
    try
    {
        auto result = solve_impl(old, source, capacity, magnitude, zero, nonnegative, maximum_iterations, tolerance,
            equilibrate_rows, &sparse);
        numeric_valid_ = false;
        matrix_ = Teuchos::null;
        row_capacity_ = saved_capacity;
        return result;
    }
    catch (...)
    {
        numeric_valid_ = false;
        matrix_ = Teuchos::null;
        row_capacity_ = saved_capacity;
        throw;
    }
}

ConservativeVoFTransport::Result ConservativeVoFTransport::solve_impl(const std::vector<double>& old,
    const std::vector<double>& source, const std::vector<double>& capacity, const std::vector<double>& flux,
    const std::vector<double>& reverse, bool nonnegative, int maximum_iterations, double tolerance,
    bool equilibrate_rows, const SparseAssembly* sparse)
{
    exact_preparation_.reset();
    const auto map = mesh_->owned_cell_map();
    const auto n = mesh_->num_owned_cells();
    if (sparse && !equilibrate_rows)
        diagonal_.assign(n, 1);
    require(*map,
        old.size() == n && source.size() == n && capacity.size() == n && flux.size() == mesh_->num_faces() &&
            reverse.size() == flux.size(),
        "Conservative VoF transport has inconsistent storage/flux dimensions.");
    require(*map, std::isfinite(tolerance) && tolerance > 0 && maximum_iterations > 0,
        "Conservative VoF transport requires a positive finite tolerance and iteration budget.");
    bool valid = true;
    double local_max_flux = 0, global_max_flux = 0;
    for (double f : flux)
    {
        valid = valid && std::isfinite(f) && f >= 0;
        local_max_flux = std::max(local_max_flux, std::abs(f));
    }
    for (double f : reverse)
    {
        valid = valid && std::isfinite(f) && f >= 0;
        local_max_flux = std::max(local_max_flux, f);
    }
    for (std::size_t c = 0; c < n; ++c)
        valid = valid && std::isfinite(capacity[c]) && capacity[c] >= 0 && std::isfinite(old[c] + source[c]);
    require(*map, valid, "Conservative VoF transport requires finite nonnegative storage and finite flux/RHS.");
    if (sparse)
    {
        for (double value : sparse->outgoing)
            local_max_flux = std::max(local_max_flux, value);
        for (const auto& row : sparse->incoming)
            for (const auto& [donor, value] : row)
                local_max_flux = std::max(local_max_flux, value);
    }
    Teuchos::reduceAll(*map->getComm(), Teuchos::REDUCE_MAX, 1, &local_max_flux, &global_max_flux);
    Result result{std::vector<double>(n), std::vector<double>(flux.size()), std::vector<double>(flux.size())};
    std::vector<double> concentration(n);
    LinearSolverOptions linear;
    linear.backend = LinearSolverBackend::Gmres;
    linear.preconditioner = LinearPreconditioner::Jacobi;
    linear.reuse_preconditioner = true;
    linear.tolerance = tolerance;
    linear.max_iterations = maximum_iterations;
    linear.verbosity = 0;
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
        int local_same = numeric_valid_ && equilibrated_ == equilibrate_rows && capacity_ == capacity &&
                         flux_ == flux && reverse_flux_ == reverse,
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
                    if (sparse)
                    {
                        diagonal += sparse->outgoing[c];
                        for (const auto& [donor, amount] : sparse->incoming[c])
                        {
                            row[donor] -= amount;
                            incoming = incoming || amount > 0;
                        }
                    }
                    else
                        for (const auto face : mesh_->faces(lid))
                        {
                            if (mesh_->is_geometry_exterior_face(face))
                                continue;
                            const bool owner = mesh_->owner_cell(face) == lid;
                            const double outgoing = owner ? flux[face] : reverse[face];
                            const double incoming_flux = owner ? reverse[face] : flux[face];
                            // Both directions remain present even when their net is zero.
                            auto& coefficient = row[mesh_->cell_global_id(mesh_->opposite_cell(face, lid))];
                            diagonal += outgoing;
                            coefficient -= incoming_flux;
                            incoming = incoming || incoming_flux > 0;
                        }
                    if (diagonal == 0 && !incoming)
                        diagonal = 1;
                    valid = valid && std::isfinite(diagonal) && diagonal > 0;
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
            reverse_flux_ = reverse;
            equilibrated_ = equilibrate_rows;
            numeric_valid_ = true;
        }
        // A zero-storage isolated row is an identity placeholder, never a
        // physical capacity floor. Validate its RHS independently on every call.
        {
            const auto execution = mesh_->acquire_execution_view();
            for (std::size_t c = 0; c < n; ++c)
            {
                bool connected = sparse && (sparse->outgoing[c] > 0 || !sparse->incoming[c].empty());
                for (const auto face : mesh_->faces(static_cast<LO>(c)))
                    connected = connected || flux[face] != 0 || reverse[face] != 0;
                valid = valid && (capacity[c] > 0 || connected || old[c] + source[c] == 0);
                const double rhs = equilibrate_rows ? (old[c] + source[c]) / diagonal_[c] : old[c] + source[c];
                valid = valid && std::isfinite(rhs);
                rhs_.replaceLocalValue(static_cast<LO>(c), rhs);
            }
        }
        require(*map, valid, "Conservative VoF inventory has no storage or transport path.");
        if (equilibrate_rows)
        {
            require(*map, std::isfinite(tolerance) && tolerance > 0,
                "Conservative VoF transport requires a positive finite physical tolerance.");
            // A global diagonal bound is sufficient, but can demand a scaled
            // residual below binary64 roundoff (or underflow to zero). Use an
            // attainable inner target and let the original physical ledger
            // accept or request a defect correction. Belos still checks the
            // true scaled residual against every requested inner tolerance.
            linear.tolerance = std::max(tolerance, 64 * std::numeric_limits<double>::epsilon());
        }
        const auto stats = solver_.solve_from_zero_with_statistics(
            Teuchos::rcp_implicit_cast<const Pack::operator_type>(matrix_), rhs_, solution_, linear);
        require(*map, stats.converged, "Conservative VoF implicit transport did not converge to its true residual.");
        result.iterations = stats.iterations;
        const auto values = solution_.getData();
        std::copy(values.begin(), values.end(), concentration.begin());
    }
    const auto publish = [&]
    {
        if (global_max_flux > 0)
            for (std::size_t c = 0; c < n; ++c)
            {
                result.amount[c] = capacity[c] * concentration[c];
                if (sparse)
                    valid = valid && !(capacity[c] > 0 && concentration[c] != 0 && result.amount[c] == 0);
            }
        if (sparse)
            require(*map,
                valid && std::all_of(concentration.begin(), concentration.end(),
                             [](double value) { return std::isfinite(value); }),
                "Sparse VoF physical momentum amount or implicit trace is unrepresentable.");
        if (sparse)
        {
            {
                auto data = cell_owned_.getDataNonConst();
                std::copy(concentration.begin(), concentration.end(), data.begin());
            }
            Pack::vector_type donor(sparse->donor_map);
            donor.doImport(cell_owned_, *sparse->donor_import, Tpetra::INSERT);
            const auto trace = donor.getData();
            std::vector<CompensatedSum<>> amounts(mesh_->num_owned_faces());
            std::map<LO, Meshes::ExactGeometryMeasure> exact;
            bool representable = true;
            try
            {
                for (const auto& face : sparse->faces)
                {
                    const double value = trace[sparse->donor_map->getLocalElement(face.donor)];
                    if (sparse->exact_faces.contains(face.face))
                    {
                        const auto group = face.exact_amount ? *face.exact_amount
                                                             : Meshes::ExactGeometryMeasure::from_value(face.amount);
                        exact[face.face] = exact[face.face] + group.scaled(value);
                    }
                    else
                        amounts[face.face] += static_cast<long double>(face.amount) * value;
                }
                std::fill(result.trace.begin(), result.trace.end(), 0);
                for (std::size_t face = 0; face < amounts.size(); ++face)
                {
                    const auto found = exact.find(LO(face));
                    if (found == exact.end())
                    {
                        const auto value = amounts[face].value();
                        result.flux[face] = static_cast<double>(value);
                        if (!std::isfinite(result.flux[face]) || (value != 0 && result.flux[face] == 0))
                            throw std::runtime_error("Sparse VoF physical face flux cannot be represented.");
                    }
                    else
                        result.flux[face] = found->second.sign() ? found->second.certified().as_double() : 0;
                }
            }
            catch (...)
            {
                representable = false;
            }
            require(*map, representable, "Sparse VoF physical face momentum flux is unrepresentable.");
        }
        else
        {
            const auto halo = cells(concentration);
            {
                const auto execution = mesh_->acquire_execution_view();
                for (std::size_t id = 0; id < mesh_->num_owned_faces(); ++id)
                {
                    const auto face = static_cast<LO>(id);
                    if (mesh_->is_geometry_exterior_face(face))
                        continue;
                    const auto owner = mesh_->owner_cell(face), neighbor = mesh_->neighbor_cell(face);
                    result.flux[id] = flux[id] * halo[owner] - reverse[id] * halo[neighbor];
                    // Preserve the exact donor trace on one-way faces. Multiplying
                    // then dividing by a microscopic flux can erase that trace.
                    if (reverse[id] == 0 && flux[id] > 0)
                        result.trace[id] = halo[owner];
                    else if (flux[id] == 0 && reverse[id] > 0)
                        result.trace[id] = halo[neighbor];
                    else if (flux[id] == 0)
                        result.trace[id] = 0.5 * halo[owner] + 0.5 * halo[neighbor];
                    else
                    {
                        const double scale = std::max(flux[id], reverse[id]);
                        const double a = flux[id] / scale, b = reverse[id] / scale;
                        result.trace[id] = (a / (a + b)) * halo[owner] + (b / (a + b)) * halo[neighbor];
                    }
                }
            }
        }
        result.flux = faces(result.flux);
        result.trace = faces(result.trace);
    };
    publish();
    if (equilibrate_rows || sparse)
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
        std::vector<CompensatedSum<>> accumulated(n);
        for (std::size_t c = 0; c < n; ++c)
            accumulated[c] += concentration[c];
        std::vector<long double> defect(n);
        constexpr int maximum_refinements = 4;
        for (;;)
        {
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
                    defect[c] = -residual.value();
                    const long double r = residual.value() / scale, b = (old[c] + source[c]) / scale;
                    residual_squared += r * r;
                    rhs_squared += b * b;
                }
            }
            double local[2] = {static_cast<double>(residual_squared.value()), static_cast<double>(rhs_squared.value())},
                   global[2]{};
            Teuchos::reduceAll(*map->getComm(), Teuchos::REDUCE_SUM, 2, local, global);
            result.physical_relative_residual = std::sqrt(global[0] / (global[1] > 0 ? global[1] : 1));
            const bool accepted = std::isfinite(global[0]) && std::isfinite(global[1]) &&
                                  global[0] <= tolerance * tolerance * (global[1] > 0 ? global[1] : 1);
            if (accepted)
                break;
            require(*map,
                global_max_flux > 0 && result.physical_refinements < maximum_refinements &&
                    result.iterations < maximum_iterations,
                "Equilibrated conservative VoF transport failed its original physical residual.");

            // Solve D^-1*A*delta = D^-1*(b-A*c) using the actual physical
            // ledger, not a residual of the rounded equilibrated matrix. A
            // common RHS normalization keeps the generic inner norm in range;
            // its own true-residual contract remains unchanged.
            long double local_defect_scale = 0;
            bool finite = true;
            for (std::size_t c = 0; c < n; ++c)
            {
                defect[c] /= diagonal_[c];
                finite = finite && std::isfinite(defect[c]);
                local_defect_scale = std::max(local_defect_scale, std::abs(defect[c]));
            }
            require(*map, finite, "Conservative VoF physical defect is nonfinite.");
            // Teuchos does not serialize long double on every supported
            // backend. Exchange its exponent instead: a shared power of two
            // bounds the normalized RHS by two without narrowing the defect.
            int local_exponent = std::numeric_limits<int>::min(), global_exponent = 0;
            if (local_defect_scale > 0)
                (void) std::frexp(local_defect_scale, &local_exponent);
            Teuchos::reduceAll(*map->getComm(), Teuchos::REDUCE_MAX, 1, &local_exponent, &global_exponent);
            require(*map, global_exponent > std::numeric_limits<int>::min(),
                "Conservative VoF physical defect cannot supply a finite correction.");
            const long double defect_scale = std::scalbn(1.0L, global_exponent - 1);
            require(*map, std::isfinite(defect_scale) && defect_scale > 0,
                "Conservative VoF physical defect cannot supply a finite correction.");
            for (std::size_t c = 0; c < n; ++c)
                rhs_.replaceLocalValue(static_cast<LO>(c), static_cast<double>(defect[c] / defect_scale));
            linear.max_iterations = maximum_iterations - result.iterations;
            const auto stats = solver_.solve_from_zero_with_statistics(
                Teuchos::rcp_implicit_cast<const Pack::operator_type>(matrix_), rhs_, solution_, linear);
            require(
                *map, stats.converged, "Conservative VoF physical defect solve did not converge to its true residual.");
            result.iterations += stats.iterations;
            ++result.physical_refinements;
            {
                const auto correction = solution_.getData();
                for (std::size_t c = 0; c < n; ++c)
                {
                    accumulated[c] += static_cast<long double>(correction[c]) * defect_scale;
                    concentration[c] = static_cast<double>(accumulated[c].value());
                }
            }
            publish();
        }
    }
    for (double amount : result.amount)
        valid = valid && std::isfinite(amount) && (!nonnegative || amount >= 0);
    require(*map, valid, "Conservative VoF transport lost inventory positivity or finiteness.");
    return result;
}
ConservativeVoFTransport::Result ConservativeVoFTransport::solve(const std::vector<long double>& old,
    const std::vector<long double>& source, const std::vector<long double>& capacity,
    const std::vector<long double>& outgoing, const std::vector<FiniteTransfer>& incoming,
    const std::vector<FiniteFaceContribution>& contributions, bool nonnegative,
    int maximum_iterations, double tolerance, std::size_t byte_budget,
    std::span<const long double> old_error, std::span<const long double> source_error,
    std::span<const long double> capacity_error, std::span<const long double> outgoing_error)
{
    namespace fp = VoF::finite_detail;
    using Value = Meshes::BoundedGeometryValue;
    const auto map = mesh_->owned_cell_map();
    const auto comm = map->getComm();
    const auto n = mesh_->num_owned_cells();
    const auto safe = [&](auto&& action)
    {
        bool ok = true;
        try { action(); } catch (...) { ok = false; }
        require(*map, ok, "Finite VoF transport rejected its bounded local arithmetic or metadata.");
    };
    require(*map, old.size() == n && source.size() == n && capacity.size() == n && outgoing.size() == n &&
        maximum_iterations > 0 && std::isfinite(tolerance) && tolerance > 0 && byte_budget > 0,
        "Finite VoF transport has invalid dimensions or controls.");
    const auto error_at = [](std::span<const long double> values, std::size_t i)
    { return values.empty() ? 0.L : values[i]; };
    require(*map, (old_error.empty() || old_error.size() == n) &&
        (source_error.empty() || source_error.size() == n) &&
        (capacity_error.empty() || capacity_error.size() == n) &&
        (outgoing_error.empty() || outgoing_error.size() == n), "Finite momentum input error shape mismatch.");
    std::vector<std::map<GO, Value>> routes(n);
    std::set<GO> donor_ids;
    bool valid = true;
    std::size_t bytes = n * (sizeof(long double) * 12 + sizeof(std::map<GO, Value>));
    safe([&]
    {
        for (const auto& errors : {old_error, source_error, capacity_error, outgoing_error})
            for (auto error : errors)
                valid = valid && std::isfinite(error) && error >= 0;
        for (std::size_t c = 0; c < n; ++c)
            valid = valid && std::isfinite(old[c]) && std::isfinite(source[c]) &&
                std::isfinite(capacity[c]) && capacity[c] > 0 && std::isfinite(outgoing[c]) && outgoing[c] >= 0;
        for (const auto& route : incoming)
        {
            const auto row = map->getLocalElement(route.recipient);
            valid = valid && row >= 0 && route.donor != route.recipient && std::isfinite(route.amount) &&
                route.amount >= 0 && std::isfinite(route.absolute_error) && route.absolute_error >= 0;
            if (row >= 0)
                valid = routes[row].emplace(route.donor, Value{route.amount, route.absolute_error}).second && valid;
            donor_ids.insert(route.donor);
            bytes += sizeof(route) + sizeof(std::pair<const GO, Value>) + 8 * sizeof(void*);
        }
        for (const auto& face : contributions)
        {
            valid = valid && face.face >= 0 && std::size_t(face.face) < mesh_->num_owned_faces() &&
                std::isfinite(face.amount) && std::isfinite(face.absolute_error) && face.absolute_error >= 0;
            if (face.face >= 0 && std::size_t(face.face) < mesh_->num_owned_faces())
                valid = valid && !mesh_->is_geometry_exterior_face(face.face);
            donor_ids.insert(face.donor);
            bytes += sizeof(face) + 128;
        }
        if (bytes > byte_budget)
            throw std::invalid_argument("Finite VoF momentum input capacity exceeds its unchanged byte budget.");
    });
    require(*map, valid, "Finite VoF momentum requires positive routes and canonical internal face streams.");
    // Route raw signed face coefficients once to their incident owned rows.
    // The wire is hi/lo significand plus exponent, independent of ABI padding.
    const auto word = [](std::vector<char>& b, std::uint64_t v)
    {
        for (unsigned a = 0; a < 8; ++a) b.push_back(char(v >> (8 * a)));
    };
    const auto scalar = [&](std::vector<char>& b, long double v)
    {
        int exponent = 0;
        const auto m = std::frexp(v, &exponent);
        const auto hi = static_cast<double>(m), lo = static_cast<double>(m - hi);
        if (!std::isfinite(v) || static_cast<long double>(hi) + lo != m)
            throw std::invalid_argument("Finite momentum scalar exceeds its fixed wire representation.");
        word(b, std::bit_cast<std::uint64_t>(hi));
        word(b, std::bit_cast<std::uint64_t>(lo));
        word(b, std::bit_cast<std::uint64_t>(static_cast<std::int64_t>(exponent)));
    };
    const auto read = [](const std::vector<char>& b, std::size_t& offset)
    {
        std::uint64_t v = 0;
        for (unsigned a = 0; a < 8; ++a) v |= std::uint64_t(static_cast<unsigned char>(b.at(offset++))) << (8 * a);
        return v;
    };
    const auto read_scalar = [&](const std::vector<char>& b, std::size_t& offset)
    {
        const auto hi = std::bit_cast<double>(read(b, offset)), lo = std::bit_cast<double>(read(b, offset));
        const auto exponent = std::bit_cast<std::int64_t>(read(b, offset));
        if (exponent < std::numeric_limits<int>::min() || exponent > std::numeric_limits<int>::max())
            throw std::invalid_argument("Finite momentum exponent is malformed.");
        const auto value = std::scalbn(static_cast<long double>(hi) + lo, int(exponent));
        if (!std::isfinite(value))
            throw std::invalid_argument("Finite momentum wire value is nonfinite.");
        return value;
    };
    std::set<GO> endpoints;
    safe([&]
    {
        const auto lease = mesh_->acquire_execution_view();
        for (const auto& face : contributions)
        {
            endpoints.insert(mesh_->cell_global_id(mesh_->owner_cell(face.face)));
            endpoints.insert(mesh_->cell_global_id(mesh_->neighbor_cell(face.face)));
        }
    });
    Teuchos::Array<GO> endpoint_ids(endpoints.begin(), endpoints.end());
    Teuchos::Array<int> endpoint_owners(endpoint_ids.size(), -1);
    (void) map->getRemoteIndexList(endpoint_ids(), endpoint_owners());
    std::map<GO, int> owner;
    for (std::size_t i = 0; i < std::size_t(endpoint_ids.size()); ++i)
    {
        valid = valid && endpoint_owners[i] >= 0;
        owner.emplace(endpoint_ids[i], endpoint_owners[i]);
    }
    require(*map, valid, "Finite momentum face references a foreign endpoint.");
    std::vector<OwnerPacket> packets;
    safe([&]
    {
        const auto lease = mesh_->acquire_execution_view();
        for (const auto& face : contributions)
            for (int sign : {1, -1})
            {
                const auto row = mesh_->cell_global_id(sign > 0 ? mesh_->owner_cell(face.face) : mesh_->neighbor_cell(face.face));
                OwnerPacket packet;
                packet.peer = owner.at(row);
                word(packet.bytes, std::uint64_t(row));
                word(packet.bytes, std::uint64_t(face.donor));
                word(packet.bytes, std::uint64_t(mesh_->face_global_id(face.face)));
                scalar(packet.bytes, sign * face.amount);
                scalar(packet.bytes, face.absolute_error);
                packets.push_back(std::move(packet));
            }
        if (packets.size() > (byte_budget - std::min(bytes, byte_budget)) / 128)
            throw std::invalid_argument("Finite momentum packet capacity exceeds its unchanged budget.");
    });
    const auto received = exchange_owner_packets(packets, comm, byte_budget - bytes);
    std::vector<std::map<GO, fp::Sum>> raw_sums(n);
    safe([&]
    {
        std::set<std::tuple<int, GO, GO, GO, long double>> seen;
        for (const auto& packet : received.packets)
        {
            std::size_t offset = 0;
            const auto row_gid = GO(read(packet.bytes, offset)), donor = GO(read(packet.bytes, offset));
            const auto face = GO(read(packet.bytes, offset));
            const auto amount = read_scalar(packet.bytes, offset), error = read_scalar(packet.bytes, offset);
            const auto row = map->getLocalElement(row_gid);
            if (row < 0 || offset != packet.bytes.size() || error < 0 || !std::isfinite(amount) ||
                !seen.emplace(packet.peer, row_gid, donor, face, amount).second)
                throw std::invalid_argument("Finite momentum raw coefficient is duplicate or malformed.");
            raw_sums[row][donor].add({amount, error});
            donor_ids.insert(donor);
        }
        for (std::size_t c = 0; c < n; ++c)
        {
            const auto self = map->getGlobalElement(LO(c));
            std::set<GO> columns{self};
            for (const auto& [donor, value] : raw_sums[c]) columns.insert(donor);
            for (const auto& [donor, value] : routes[c]) columns.insert(donor);
            for (auto donor : columns)
            {
                const auto observed = raw_sums[c][donor].value();
                const auto expected = donor == self ? Value{outgoing[c], error_at(outgoing_error, c)} :
                    routes[c].contains(donor) ? fp::negate(routes[c].at(donor)) : Value{};
                const auto defect = fp::subtract(observed, expected);
                if (std::abs(defect.value) > defect.absolute_error)
                    throw std::invalid_argument("Finite momentum positive routes differ from original raw face coefficients.");
            }
        }
    });
    Teuchos::Array<GO> ids(donor_ids.begin(), donor_ids.end());
    const auto donor_map = Teuchos::rcp(new Pack::map_type(
        Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid(), ids(), map->getIndexBase(), comm));
    Pack::import_type donor_import(map, donor_map);
    Pack::vector_type marker(map), donor_trace(donor_map);
    marker.putScalar(1);
    donor_trace.putScalar(0);
    donor_trace.doImport(marker, donor_import, Tpetra::INSERT);
    for (auto value : donor_trace.getData()) valid = valid && value == 1;
    require(*map, valid, "Finite VoF momentum references an absent geometric donor.");
    std::vector<std::map<GO, Value>> raw(n);
    for (std::size_t c = 0; c < n; ++c)
        for (const auto& [donor, sum] : raw_sums[c]) raw[c].emplace(donor, sum.value());
    std::size_t width = 1;
    for (const auto& row : routes) width = std::max(width, row.size() + 1);
    auto matrix = Teuchos::rcp(new Pack::matrix_type(map, width));
    std::vector<long double> diagonal(n);
    safe([&]
    {
        for (std::size_t c = 0; c < n; ++c)
        {
            diagonal[c] = capacity[c] + outgoing[c];
            if (!(diagonal[c] > 0) || !std::isfinite(diagonal[c]))
                throw std::invalid_argument("Finite momentum has a nonpositive row diagonal.");
            Teuchos::Array<GO> columns;
            Teuchos::Array<double> values;
            columns.push_back(map->getGlobalElement(LO(c)));
            values.push_back(1);
            for (const auto& [donor, amount] : routes[c])
            {
                const auto value = static_cast<double>(-amount.value / diagonal[c]);
                if (!std::isfinite(value) || (amount.value != 0 && value == 0))
                    throw std::invalid_argument("Finite momentum row coefficient is outside representable range.");
                columns.push_back(donor);
                values.push_back(value);
            }
            matrix->insertGlobalValues(map->getGlobalElement(LO(c)), columns(), values());
            const auto value = static_cast<double>((old[c] + source[c]) / diagonal[c]);
            if (!std::isfinite(value))
                throw std::invalid_argument("Finite momentum normalized RHS overflows.");
            rhs_.replaceLocalValue(LO(c), value);
        }
    });
    matrix->fillComplete();
    solver_.notify_operator_values_changed();
    LinearSolverOptions linear;
    linear.backend = LinearSolverBackend::Gmres;
    linear.preconditioner = LinearPreconditioner::Jacobi;
    linear.reuse_preconditioner = true;
    linear.tolerance = std::max(tolerance, 64 * std::numeric_limits<double>::epsilon());
    linear.max_iterations = maximum_iterations;
    linear.verbosity = 0;
    Result result;
    std::vector<double> trace(n);
    const auto solve = [&]
    {
        linear.max_iterations = maximum_iterations - result.iterations;
        const auto stats = solver_.solve_from_zero_with_statistics(
            Teuchos::rcp_implicit_cast<const Pack::operator_type>(matrix), rhs_, solution_, linear);
        require(*map, stats.converged, "Finite momentum inner solve failed its true residual.");
        result.iterations += stats.iterations;
    };
    solve();
    std::copy(solution_.getData().begin(), solution_.getData().end(), trace.begin());
    int local_exponent = std::numeric_limits<int>::min(), exponent = 0;
    for (std::size_t c = 0; c < n; ++c)
        if (old[c] + source[c] != 0)
        {
            int e = 0;
            (void) std::frexp(std::abs(old[c] + source[c]), &e);
            local_exponent = std::max(local_exponent, e);
        }
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &local_exponent, &exponent);
    const auto reference_scale = exponent == std::numeric_limits<int>::min() ? 1.L : std::scalbn(1.L, exponent - 1);
    std::vector<long double> defect(n);
    std::vector<CompensatedSum<>> accumulated(n);
    for (std::size_t c = 0; c < n; ++c) accumulated[c] += trace[c];
    for (;;)
    {
        {
            auto values = cell_owned_.getDataNonConst();
            std::copy(trace.begin(), trace.end(), values.begin());
        }
        donor_trace.doImport(cell_owned_, donor_import, Tpetra::INSERT);
        const auto donor_values = donor_trace.getData();
        double local_norm[2]{}, global_norm[2]{};
        safe([&]
        {
            fp::Sum residual_squared, reference_squared;
            for (std::size_t c = 0; c < n; ++c)
            {
                fp::Sum equation;
                equation.add(fp::multiply({capacity[c], error_at(capacity_error, c)}, {trace[c], 0}));
                equation.add({-old[c], error_at(old_error, c)});
                equation.add({-source[c], error_at(source_error, c)});
                for (const auto& [donor, amount] : raw[c])
                {
                    const auto donor_lid = donor_map->getLocalElement(donor);
                    if (donor_lid < 0)
                        throw std::invalid_argument("Finite momentum raw row references absent trace.");
                    equation.add(fp::multiply(amount, {donor_values[donor_lid], 0}));
                }
                const auto residual = equation.value();
                defect[c] = -residual.value;
                const auto upper = (std::abs(residual.value) + residual.absolute_error) / reference_scale;
                const auto rhs = (old[c] + source[c]) / reference_scale;
                residual_squared.add(fp::multiply({upper, 0}, {upper, 0}));
                reference_squared.add(fp::multiply({rhs, 0}, {rhs, 0}));
            }
            local_norm[0] = fp::upper_double(fp::upward(residual_squared.value().value + residual_squared.value().absolute_error));
            const auto lower = std::max(0.L,
                reference_squared.value().value - reference_squared.value().absolute_error);
            local_norm[1] = static_cast<double>(lower);
            if (static_cast<long double>(local_norm[1]) > lower)
                local_norm[1] = std::nextafter(local_norm[1], 0.);
            if (!std::isfinite(local_norm[0]) || !std::isfinite(local_norm[1]))
                throw std::invalid_argument("Finite momentum physical residual reduction overflows.");
        });
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 2, local_norm, global_norm);
        const auto reduction_u = (8.L * comm->getSize() + 16) * std::numeric_limits<double>::epsilon();
        const auto residual_upper = global_norm[0] * (1 + reduction_u);
        const auto reference_lower = global_norm[1] * (1 - reduction_u);
        result.physical_relative_residual = std::sqrt(residual_upper / (reference_lower > 0 ? reference_lower : 1));
        if (std::isfinite(residual_upper) && ((reference_lower > 0 &&
                residual_upper <= static_cast<long double>(tolerance) * tolerance * reference_lower) ||
                (reference_lower == 0 && residual_upper == 0)))
            break;
        require(*map, result.physical_refinements < 4 && result.iterations < maximum_iterations,
            "Finite momentum failed its unchanged original physical residual.");
        int local_defect_exponent = std::numeric_limits<int>::min(), defect_exponent = 0;
        safe([&]
        {
            for (std::size_t c = 0; c < n; ++c)
            {
                defect[c] /= diagonal[c];
                if (!std::isfinite(defect[c]))
                    throw std::invalid_argument("Finite momentum defect is outside qualified range.");
                if (defect[c] != 0)
                {
                    int e = 0;
                    (void) std::frexp(std::abs(defect[c]), &e);
                    local_defect_exponent = std::max(local_defect_exponent, e);
                }
            }
        });
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &local_defect_exponent, &defect_exponent);
        require(*map, defect_exponent != std::numeric_limits<int>::min(),
            "Finite momentum arithmetic bound cannot establish the physical residual target.");
        const auto defect_scale = std::scalbn(1.L, defect_exponent - 1);
        for (std::size_t c = 0; c < n; ++c)
            rhs_.replaceLocalValue(LO(c), static_cast<double>(defect[c] / defect_scale));
        solve();
        const auto correction = solution_.getData();
        for (std::size_t c = 0; c < n; ++c)
        {
            accumulated[c] += correction[c] * defect_scale;
            trace[c] = static_cast<double>(accumulated[c].value());
        }
        ++result.physical_refinements;
    }
    result.cell_trace = trace;
    result.amount.resize(n);
    safe([&]
    {
        for (std::size_t c = 0; c < n; ++c)
        {
            const auto amount = capacity[c] * trace[c];
            result.amount[c] = static_cast<double>(amount);
            if (!std::isfinite(trace[c]) || !std::isfinite(result.amount[c]) ||
                (amount != 0 && result.amount[c] == 0) || (nonnegative && amount < 0))
                throw std::invalid_argument("Finite momentum final publication is outside qualified range.");
        }
    });
    numeric_valid_ = false;
    exact_preparation_.reset();
    return result;
}
#include "solvers/details/ConservativeVoFExactTransport.inc"
} // namespace SimpleFluid::detail
