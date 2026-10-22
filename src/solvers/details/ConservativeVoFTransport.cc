/** @file ConservativeVoFTransport.cc
 * @brief Native conservative implicit upwind assembly and per-inventory validation.
 */
#include "solvers/details/ConservativeVoFTransport.hh"
#include "geometry/ExactSweptGeometry.hh"
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

struct ConservativeVoFTransport::ExactPreparation
{
    using M = Meshes::ExactGeometryMeasure;
    std::vector<M> capacity, outgoing, diagonal;
    std::vector<ExactTransfer> incoming;
    std::vector<ExactFaceContribution> contributions;
    std::vector<std::map<GO, M>> routes;
    std::vector<GeneralVoFOriginalRow> original;
    std::vector<Meshes::CertifiedGeometryValue> coefficient_conversion;
    Teuchos::RCP<const Pack::map_type> owned_map, overlap_map, owned_face_map, overlap_face_map, donor_map;
    std::unique_ptr<Pack::import_type> donor_import;
    Teuchos::RCP<Pack::matrix_type> matrix;
    MeshHandle<>::MultiRegionPtr native_owner;
    std::uint64_t geometry_epoch = 0;
    std::size_t byte_budget = 0, width = 1;
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
ConservativeVoFTransport::Result ConservativeVoFTransport::solve_exact(
    const std::vector<Meshes::ExactGeometryMeasure>& old, const std::vector<Meshes::ExactGeometryMeasure>& source,
    const std::vector<Meshes::ExactGeometryMeasure>& capacity,
    const std::vector<Meshes::ExactGeometryMeasure>& outgoing, const std::vector<ExactTransfer>& incoming,
    const std::vector<ExactFaceContribution>& contributions, bool nonnegative, int maximum_iterations, double tolerance,
    std::size_t byte_budget)
{
    using M = Meshes::ExactGeometryMeasure;
    namespace E = vof_exact;
    const auto map = mesh_->owned_cell_map();
    const auto n = mesh_->num_owned_cells();
    require(*map,
        old.size() == n && source.size() == n && capacity.size() == n && outgoing.size() == n &&
            maximum_iterations > 0 && std::isfinite(tolerance) && tolerance > 0 && byte_budget > 0,
        "Exact VoF input dimensions or controls are invalid.");
    // Semantic controls must agree even on empty owners. Resource budgets may
    // remain rank-local; their failures are synchronized before communication.
    int controls[2] = {maximum_iterations, int(nonnegative)}, minimum[2]{}, maximum[2]{};
    Teuchos::reduceAll(*map->getComm(), Teuchos::REDUCE_MIN, 2, controls, minimum);
    Teuchos::reduceAll(*map->getComm(), Teuchos::REDUCE_MAX, 2, controls, maximum);
    double minimum_tolerance = 0, maximum_tolerance = 0;
    Teuchos::reduceAll(*map->getComm(), Teuchos::REDUCE_MIN, 1, &tolerance, &minimum_tolerance);
    Teuchos::reduceAll(*map->getComm(), Teuchos::REDUCE_MAX, 1, &tolerance, &maximum_tolerance);
    require(*map, minimum[0] == maximum[0] && minimum[1] == maximum[1] && minimum_tolerance == maximum_tolerance,
        "Exact VoF semantic controls differ across ranks.");
    bool valid = true;
    const auto guard = [&](auto work)
    {
        bool accepted = true;
        try
        {
            work();
        }
        catch (...)
        {
            accepted = false;
        }
        require(*map, accepted, "Exact VoF collective storage allocation or arithmetic admission failed.");
    };
    using Domain = GeneralVoFStorageDomain;
    GeneralVoFExactStorage storage(byte_budget);
    std::size_t input_capacity = 0;
    const auto admit_input_capacity = [&](std::size_t count, std::size_t width)
    {
        if (width && count > (byte_budget - input_capacity) / width)
            return false;
        input_capacity += count * width;
        return true;
    };
    bool input_fits = true;
    for (const auto* array : {&old, &source, &capacity, &outgoing})
        input_fits = input_fits && admit_input_capacity(array->capacity(), sizeof(M));
    input_fits = input_fits && admit_input_capacity(incoming.capacity(), sizeof(ExactTransfer)) &&
                 admit_input_capacity(contributions.capacity(), sizeof(ExactFaceContribution));
    require(*map, input_fits, "Exact VoF input container capacities exceed their byte budget.");
    // Immutable exact payload identity is a conservative provenance key: equal
    // freshly recomputed values may miss, but rounded coefficient views never
    // authenticate a different physical ledger. Compare only after cheap input
    // container/control admission, then drop misses before auditing new inputs.
    bool current = false;
    const void* native_identity = nullptr;
    std::uint64_t epoch = 0;
    guard(
        [&]
        {
            const auto lease = mesh_->acquire_execution_view();
            const auto* native = std::get_if<MeshHandle<>::MultiRegionPtr>(&mesh_->variant());
            native_identity = native ? native->get() : nullptr;
            epoch = mesh_->geometry_epoch();
            current = mesh_->owned_cell_map().getRawPtr() == cell_owned_.getMap().getRawPtr() &&
                      mesh_->overlap_cell_map().getRawPtr() == cell_overlap_.getMap().getRawPtr() &&
                      mesh_->owned_face_map().getRawPtr() == face_owned_.getMap().getRawPtr() &&
                      mesh_->overlap_face_map().getRawPtr() == face_overlap_.getMap().getRawPtr();
            for (std::size_t c = 0; c < n; ++c)
                valid = valid && capacity[c].sign() >= 0 && outgoing[c].sign() >= 0;
            for (const auto& route : incoming)
                valid = valid && map->getLocalElement(route.recipient) >= 0 && route.donor != route.recipient &&
                        route.amount.sign() > 0;
            for (const auto& face : contributions)
            {
                const bool owned = face.face >= 0 && std::size_t(face.face) < mesh_->num_owned_faces();
                valid = valid && owned;
                if (owned)
                    valid = valid && !mesh_->is_geometry_exterior_face(face.face);
            }
        });
    require(
        *map, current && valid, "Exact VoF requires current partition, positive routes and owned internal streams.");
    const auto same_measure = [](const M& a, const M& b) { return a.storage_identity() == b.storage_identity(); };
    bool same = exact_preparation_ && exact_preparation_->byte_budget == byte_budget &&
                exact_preparation_->geometry_epoch == epoch &&
                exact_preparation_->native_owner.get() == native_identity &&
                exact_preparation_->owned_map.getRawPtr() == map.getRawPtr() &&
                exact_preparation_->overlap_map.getRawPtr() == mesh_->overlap_cell_map().getRawPtr() &&
                exact_preparation_->owned_face_map.getRawPtr() == mesh_->owned_face_map().getRawPtr() &&
                exact_preparation_->overlap_face_map.getRawPtr() == mesh_->overlap_face_map().getRawPtr();
    if (same)
    {
        const auto& key = *exact_preparation_;
        same = key.capacity.size() == capacity.size() && key.outgoing.size() == outgoing.size() &&
               key.incoming.size() == incoming.size() && key.contributions.size() == contributions.size();
        for (std::size_t c = 0; same && c < n; ++c)
            same = same_measure(key.capacity[c], capacity[c]) && same_measure(key.outgoing[c], outgoing[c]);
        for (std::size_t i = 0; same && i < incoming.size(); ++i)
            same = key.incoming[i].donor == incoming[i].donor && key.incoming[i].recipient == incoming[i].recipient &&
                   same_measure(key.incoming[i].amount, incoming[i].amount);
        for (std::size_t i = 0; same && i < contributions.size(); ++i)
            same = key.contributions[i].face == contributions[i].face &&
                   key.contributions[i].donor == contributions[i].donor &&
                   same_measure(key.contributions[i].amount, contributions[i].amount);
    }
    int local_same = same, global_same = 0;
    Teuchos::reduceAll(*map->getComm(), Teuchos::REDUCE_MIN, 1, &local_same, &global_same);
    const bool cache_hit = global_same;
    if (!cache_hit)
        exact_preparation_.reset();
    guard(
        [&]
        {
            for (const auto* array : {&old, &source, &capacity, &outgoing})
            {
                storage.capacity(Domain::Input, array->capacity(), sizeof(M));
                for (const auto& value : *array)
                    storage.measure(Domain::Input, value);
            }
            storage.capacity(Domain::Input, incoming.capacity(), sizeof(ExactTransfer));
            storage.capacity(Domain::Input, contributions.capacity(), sizeof(ExactFaceContribution));
            for (const auto& route : incoming)
                storage.measure(Domain::Input, route.amount);
            for (const auto& face : contributions)
                storage.measure(Domain::Input, face.amount);
            storage.capacity(Domain::Scratch, n, sizeof(std::map<GO, M>));
            storage.capacity(Domain::Scratch, incoming.size(), sizeof(std::pair<const GO, M>) + 4 * sizeof(void*));
            storage.capacity(Domain::Scratch, incoming.size() + contributions.size(), sizeof(GO) + 4 * sizeof(void*));
        });
    const auto arithmetic_budget = [&]
    {
        // Reserve eight simultaneous bounded scalar/codec temporaries, in
        // addition to all currently retained explicit inputs/work/results.
        const auto available = storage.remaining() / 8;
        if (available < 1024)
            throw std::runtime_error("Exact VoF has no bounded scalar arithmetic reserve.");
        return available;
    };
    std::shared_ptr<ExactPreparation> prepared = exact_preparation_;
    guard(
        [&]
        {
            // Keys retain shared exact payloads; count their real container capacity
            // and let the storage ledger deduplicate payloads against current inputs.
            storage.capacity(Domain::Scratch, 1, sizeof(ExactPreparation));
            storage.capacity(Domain::Scratch, n, 3 * sizeof(M));
            storage.capacity(Domain::Scratch, incoming.size(), sizeof(ExactTransfer));
            storage.capacity(Domain::Scratch, contributions.size(), sizeof(ExactFaceContribution));
            storage.capacity(Domain::Scratch, incoming.size(), sizeof(Meshes::CertifiedGeometryValue));
            if (!prepared)
            {
                prepared = std::make_shared<ExactPreparation>();
                prepared->capacity = capacity;
                prepared->outgoing = outgoing;
                prepared->incoming = incoming;
                prepared->contributions = contributions;
                prepared->byte_budget = byte_budget;
                prepared->geometry_epoch = epoch;
                if (const auto* native = std::get_if<MeshHandle<>::MultiRegionPtr>(&mesh_->variant()))
                    prepared->native_owner = *native;
                prepared->owned_map = map;
                prepared->overlap_map = mesh_->overlap_cell_map();
                prepared->owned_face_map = mesh_->owned_face_map();
                prepared->overlap_face_map = mesh_->overlap_face_map();
                prepared->routes.resize(n);
                prepared->diagonal.resize(n);
                prepared->coefficient_conversion.reserve(incoming.size());
            }
            const auto extra_capacity = [&](std::size_t actual, std::size_t reserved, std::size_t width)
            {
                if (actual < reserved)
                    throw std::runtime_error("Exact VoF preparation capacity audit failed.");
                storage.capacity(Domain::Scratch, actual - reserved, width);
            };
            // The size bounds above admit allocations before they occur. Audit
            // actual retained capacities too, including over-allocation by vectors.
            for (const auto* array : {&prepared->capacity, &prepared->outgoing, &prepared->diagonal})
                extra_capacity(array->capacity(), n, sizeof(M));
            extra_capacity(prepared->incoming.capacity(), incoming.size(), sizeof(ExactTransfer));
            extra_capacity(prepared->contributions.capacity(), contributions.size(), sizeof(ExactFaceContribution));
            extra_capacity(prepared->routes.capacity(), n, sizeof(std::map<GO, M>));
            extra_capacity(
                prepared->coefficient_conversion.capacity(), incoming.size(), sizeof(Meshes::CertifiedGeometryValue));
            for (const auto* array : {&prepared->capacity, &prepared->outgoing})
                for (const auto& value : *array)
                    storage.measure(Domain::Scratch, value);
            for (const auto& route : prepared->incoming)
                storage.measure(Domain::Scratch, route.amount);
            for (const auto& face : prepared->contributions)
                storage.measure(Domain::Scratch, face.amount);
        });
    const auto account_original = [&]
    {
        const auto& original = prepared->original;
        guard(
            [&]
            {
                storage.capacity(Domain::Scratch, original.capacity(), sizeof(GeneralVoFOriginalRow));
                for (const auto& row : original)
                {
                    storage.capacity(Domain::Scratch, row.faces.capacity(), sizeof(GeneralVoFOriginalFaceTerm));
                    storage.capacity(
                        Domain::Scratch, row.coefficients.size(), sizeof(std::pair<const GO, M>) + 4 * sizeof(void*));
                    storage.capacity(Domain::Scratch, row.faces.size(), sizeof(GO) + 4 * sizeof(void*));
                    for (const auto& term : row.faces)
                        storage.measure(Domain::Scratch, term.amount);
                    for (const auto& [gid, value] : row.coefficients)
                        storage.measure(Domain::Scratch, value);
                }
                for (const auto& value : prepared->diagonal)
                    storage.measure(Domain::Scratch, value);
            });
    };
    auto& routes = prepared->routes;
    if (!cache_hit)
    {
        std::set<GO> donors;
        guard(
            [&]
            {
                for (const auto& route : incoming)
                {
                    const auto row = map->getLocalElement(route.recipient);
                    valid = routes[row].emplace(route.donor, route.amount).second && valid;
                    donors.insert(route.donor);
                }
                for (const auto& face : contributions)
                    donors.insert(face.donor);
            });
        require(*map, valid, "Exact VoF requires unique positive routes and owned internal face streams.");
        std::size_t binding_budget = 0;
        guard([&] { binding_budget = arithmetic_budget(); });
        prepared->original =
            verify_general_vof_exact_transfer_binding(mesh_, outgoing, routes, contributions, binding_budget);
        account_original();
        Teuchos::Array<GO> ids;
        guard(
            [&]
            {
                for (const auto& row : prepared->original)
                    for (const auto& term : row.faces)
                        donors.insert(term.donor);
                storage.capacity(Domain::Scratch, donors.size(), sizeof(GO));
                ids.assign(donors.begin(), donors.end());
            });
        prepared->donor_map = Teuchos::rcp(new Pack::map_type(
            Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid(), ids(), map->getIndexBase(), map->getComm()));
        prepared->donor_import = std::make_unique<Pack::import_type>(map, prepared->donor_map);
        Pack::vector_type marker(map), donor(prepared->donor_map);
        marker.putScalar(1);
        donor.putScalar(0);
        donor.doImport(marker, *prepared->donor_import, Tpetra::INSERT);
        for (double value : donor.getData())
            valid = valid && value == 1;
        require(*map, valid, "Exact VoF references an absent donor.");
        for (const auto& row : routes)
            prepared->width = std::max(prepared->width, row.size() + 1);
    }
    else
    {
        account_original();
        guard([&] { storage.capacity(Domain::Scratch, prepared->donor_map->getLocalNumElements(), sizeof(GO)); });
    }
    const auto& original = prepared->original;
    const auto donor_map = prepared->donor_map;
    const auto& donor_import = *prepared->donor_import;
    Pack::vector_type donor(donor_map);
    // Legacy floating/scalar preparation never authenticates an exact operator.
    numeric_valid_ = false;
    matrix_ = Teuchos::null;
    const auto width = prepared->width;
    std::vector<M> diagonal, right, accumulated, amount, residual, defect;
    std::vector<double> concentration;
    std::vector<Meshes::CertifiedGeometryValue> conversion;
    std::size_t conversion_capacity = 0;
    guard(
        [&]
        {
            if (n > (std::numeric_limits<std::size_t>::max() - incoming.size()) / 9)
                throw std::runtime_error("Exact conversion capacity overflow.");
            conversion_capacity = 9 * n + incoming.size();
            storage.capacity(Domain::Scratch, n, 6 * sizeof(M) + sizeof(double));
            storage.capacity(Domain::Result, n, 2 * sizeof(double));
            storage.capacity(
                Domain::Result, mesh_->num_owned_faces(), sizeof(M) + sizeof(Meshes::CertifiedGeometryValue));
            storage.capacity(Domain::Result, conversion_capacity, sizeof(Meshes::CertifiedGeometryValue));
            if (width && n > std::numeric_limits<std::size_t>::max() / width)
                throw std::runtime_error("Exact matrix graph capacity overflow.");
            storage.capacity(Domain::Scratch, n * width, sizeof(double) + sizeof(GO) + sizeof(LO));
            storage.capacity(Domain::Scratch, n + 1, sizeof(std::size_t));
            storage.capacity(Domain::Scratch, width, 2 * (sizeof(GO) + sizeof(double)));
            diagonal.resize(n);
            right.resize(n);
            accumulated.resize(n);
            amount.resize(n);
            residual.resize(n);
            defect.resize(n);
            concentration.resize(n);
            conversion.reserve(conversion_capacity);
            for (const auto* array : {&diagonal, &right, &accumulated, &amount, &residual, &defect})
                for (const auto& value : *array)
                    storage.measure(Domain::Scratch, value);
        });
    if (!cache_hit)
        prepared->matrix = Teuchos::rcp(new Pack::matrix_type(map, width));
    auto matrix = prepared->matrix;
    const auto store = [&](M& target, M value, Domain domain = Domain::Scratch)
    {
        storage.measure(domain, value);
        storage.release(domain, target);
        target = std::move(value);
    };
    try
    {
        std::size_t coefficient = 0;
        for (std::size_t c = 0; c < n; ++c)
        {
            store(
                diagonal[c], cache_hit ? prepared->diagonal[c] : E::add(capacity[c], outgoing[c], arithmetic_budget()));
            store(right[c], E::add(old[c], source[c], arithmetic_budget()));
            if (!capacity[c].sign() && !outgoing[c].sign())
            {
                if (!routes[c].empty() || right[c].sign())
                    throw std::runtime_error("Zero carrier exact row.");
                if (!cache_hit)
                    store(diagonal[c], M::from_value(1));
            }
            if (!cache_hit)
            {
                Teuchos::Array<GO> columns;
                Teuchos::Array<double> coefficients;
                columns.push_back(map->getGlobalElement(LO(c)));
                coefficients.push_back(1);
                for (const auto& [gid, value] : routes[c])
                {
                    Meshes::CertifiedGeometryValue enclosure;
                    coefficients.push_back(
                        E::view(E::divide(E::scale(value, -1, arithmetic_budget()), diagonal[c], arithmetic_budget()),
                            enclosure));
                    columns.push_back(gid);
                    conversion.push_back(enclosure);
                    prepared->coefficient_conversion.push_back(enclosure);
                }
                matrix->insertGlobalValues(map->getGlobalElement(LO(c)), columns(), coefficients());
                store(prepared->diagonal[c], diagonal[c]);
            }
            else
                for (std::size_t i = 0; i < routes[c].size(); ++i)
                    conversion.push_back(prepared->coefficient_conversion[coefficient++]);
            Meshes::CertifiedGeometryValue enclosure;
            rhs_.replaceLocalValue(LO(c), E::view(E::divide(right[c], diagonal[c], arithmetic_budget()), enclosure));
            conversion.push_back(enclosure);
        }
    }
    catch (...)
    {
        valid = false;
    }
    require(*map, valid, "Exact VoF normalization exceeds its arithmetic or representable inner range.");
    if (!cache_hit)
    {
        matrix->fillComplete();
        solver_.notify_operator_values_changed();
    }
    LinearSolverOptions linear;
    linear.backend = LinearSolverBackend::Gmres;
    linear.preconditioner = LinearPreconditioner::Jacobi;
    linear.reuse_preconditioner = true;
    linear.tolerance = std::max(tolerance, 64 * std::numeric_limits<double>::epsilon());
    linear.max_iterations = maximum_iterations;
    linear.verbosity = 0;
    Result result;
    result.exact_input = true;
    const auto solve = [&]
    {
        const auto stats = solver_.solve_from_zero_with_statistics(
            Teuchos::rcp_implicit_cast<const Pack::operator_type>(matrix), rhs_, solution_, linear);
        require(*map, stats.converged, "Exact VoF inner solve failed its true residual.");
        result.iterations += stats.iterations;
    };
    solve();
    guard(
        [&]
        {
            const auto values = solution_.getData();
            for (std::size_t c = 0; c < n; ++c)
            {
                concentration[c] = values[c];
                store(accumulated[c], M::from_value(values[c]));
            }
        });
    guard(
        [&]
        {
            result.exact_flux.resize(mesh_->num_owned_faces());
            result.flux_enclosure.resize(mesh_->num_owned_faces());
            result.amount.resize(n);
            result.cell_trace.resize(n);
            for (const auto& value : result.exact_flux)
                storage.measure(Domain::Result, value);
        });
    for (;;)
    {
        {
            auto data = cell_owned_.getDataNonConst();
            std::copy(concentration.begin(), concentration.end(), data.begin());
        }
        donor.doImport(cell_owned_, donor_import, Tpetra::INSERT);
        const auto traces = donor.getData();

        try
        {
            for (auto& value : result.exact_flux)
                store(value, M{}, Domain::Result);
            for (const auto& face : contributions)
                store(result.exact_flux[face.face],
                    E::add(result.exact_flux[face.face],
                        E::scale(face.amount, traces[donor_map->getLocalElement(face.donor)], arithmetic_budget()),
                        arithmetic_budget()),
                    Domain::Result);
            for (std::size_t f = 0; f < result.exact_flux.size(); ++f)
                result.flux_enclosure[f] = result.exact_flux[f].enclosure();
            for (std::size_t c = 0; c < n; ++c)
            {
                store(amount[c], E::scale(capacity[c], concentration[c], arithmetic_budget()));
                store(residual[c], E::add(amount[c], E::scale(right[c], -1, arithmetic_budget()), arithmetic_budget()));
                for (const auto& term : original[c].faces)
                    store(residual[c],
                        E::add(residual[c],
                            E::scale(term.amount, traces[donor_map->getLocalElement(term.donor)], arithmetic_budget()),
                            arithmetic_budget()));
            }
        }
        catch (...)
        {
            valid = false;
        }
        require(*map, valid, "Exact VoF original physical ledger exceeds its arithmetic budget.");
        std::size_t norm_budget = 0;
        guard([&] { norm_budget = storage.remaining(); });
        const auto norm = E::leased_physical_norm(residual, right, tolerance, *map, norm_budget);
        result.physical_relative_residual = norm.relative_residual;
        if (norm.accepted)
            break;
        require(*map, result.physical_refinements < 4 && result.iterations < maximum_iterations,
            "Exact VoF failed its original physical residual.");
        try
        {
            for (std::size_t c = 0; c < n; ++c)
                store(defect[c],
                    E::divide(E::scale(residual[c], -1, arithmetic_budget()), diagonal[c], arithmetic_budget()));
        }
        catch (...)
        {
            valid = false;
        }
        require(*map, valid, "Exact VoF physical defect exceeds its arithmetic budget.");
        std::size_t maximum_budget = 0;
        guard([&] { maximum_budget = arithmetic_budget(); });
        const auto scale = E::maximum_absolute(defect, *map, maximum_budget);
        require(*map, scale.sign() > 0, "Exact VoF physical defect has no correction.");
        try
        {
            for (std::size_t c = 0; c < n; ++c)
            {
                Meshes::CertifiedGeometryValue enclosure;
                rhs_.replaceLocalValue(LO(c), E::view(E::divide(defect[c], scale, arithmetic_budget()), enclosure));
                conversion.push_back(enclosure);
            }
        }
        catch (...)
        {
            valid = false;
        }
        require(*map, valid, "Exact VoF correction cannot be normalized.");
        linear.max_iterations = maximum_iterations - result.iterations;
        solve();
        ++result.physical_refinements;
        try
        {
            const auto correction = solution_.getData();
            for (std::size_t c = 0; c < n; ++c)
            {
                store(accumulated[c],
                    E::add(accumulated[c], E::scale(scale, correction[c], arithmetic_budget()), arithmetic_budget()));
                Meshes::CertifiedGeometryValue enclosure;
                concentration[c] = E::view(accumulated[c], enclosure);
                conversion.push_back(enclosure);
            }
        }
        catch (...)
        {
            valid = false;
        }
        require(*map, valid, "Exact VoF corrected trace is unrepresentable.");
    }
    result.cell_trace = concentration;
    try
    {
        for (std::size_t c = 0; c < n; ++c)
        {
            if (nonnegative && amount[c].sign() < 0)
                throw std::runtime_error("Negative exact inventory.");
            result.amount[c] = amount[c].sign() ? amount[c].certified().as_double() : 0;
        }
    }
    catch (...)
    {
        valid = false;
    }
    require(*map, valid, "Exact VoF final owned amount is unrepresentable or negative.");
    guard([&] { (void) storage.finish(); });
    result.inner_conversion_enclosures = std::move(conversion);
    exact_preparation_ = std::move(prepared);
    if (cache_hit)
        ++preparation_statistics_.exact_hits;
    else
        ++preparation_statistics_.exact_builds;
    return result;
}
} // namespace SimpleFluid::detail
