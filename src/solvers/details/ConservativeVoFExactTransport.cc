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

#include "solvers/details/ConservativeVoFTransportSupport.hh"
namespace SimpleFluid::detail
{
using namespace transport_support;
/** Optional rational backend in the Exact namespace. */
namespace Exact
{
struct TransportPreparation
{
    using ExactTransfer = detail::ConservativeVoFTransport::ExactTransfer;
    using ExactFaceContribution = detail::ConservativeVoFTransport::ExactFaceContribution;
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

ConservativeVoFTransport::Result ConservativeVoFTransport::solve(detail::ConservativeVoFTransport& self,
    const std::vector<Meshes::ExactGeometryMeasure>& old, const std::vector<Meshes::ExactGeometryMeasure>& source,
    const std::vector<Meshes::ExactGeometryMeasure>& capacity,
    const std::vector<Meshes::ExactGeometryMeasure>& outgoing, const std::vector<ExactTransfer>& incoming,
    const std::vector<ExactFaceContribution>& contributions, bool nonnegative, int maximum_iterations, double tolerance,
    std::size_t byte_budget)
{
    using M = Meshes::ExactGeometryMeasure;
    namespace E = vof_exact;
    const auto map = self.mesh_->owned_cell_map();
    const auto n = self.mesh_->num_owned_cells();
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
            const auto lease = self.mesh_->acquire_execution_view();
            const auto* native = std::get_if<MeshHandle<>::MultiRegionPtr>(&self.mesh_->variant());
            native_identity = native ? native->get() : nullptr;
            epoch = self.mesh_->geometry_epoch();
            current = self.mesh_->owned_cell_map().getRawPtr() == self.cell_owned_.getMap().getRawPtr() &&
                      self.mesh_->overlap_cell_map().getRawPtr() == self.cell_overlap_.getMap().getRawPtr() &&
                      self.mesh_->owned_face_map().getRawPtr() == self.face_owned_.getMap().getRawPtr() &&
                      self.mesh_->overlap_face_map().getRawPtr() == self.face_overlap_.getMap().getRawPtr();
            for (std::size_t c = 0; c < n; ++c)
                valid = valid && capacity[c].sign() >= 0 && outgoing[c].sign() >= 0;
            for (const auto& route : incoming)
                valid = valid && map->getLocalElement(route.recipient) >= 0 && route.donor != route.recipient &&
                        route.amount.sign() > 0;
            for (const auto& face : contributions)
            {
                const bool owned = face.face >= 0 && std::size_t(face.face) < self.mesh_->num_owned_faces();
                valid = valid && owned;
                if (owned)
                    valid = valid && !self.mesh_->is_geometry_exterior_face(face.face);
            }
        });
    require(
        *map, current && valid, "Exact VoF requires current partition, positive routes and owned internal streams.");
    const auto same_measure = [](const M& a, const M& b) { return a.storage_identity() == b.storage_identity(); };
    bool same = self.exact_preparation_ && self.exact_preparation_->byte_budget == byte_budget &&
                self.exact_preparation_->geometry_epoch == epoch &&
                self.exact_preparation_->native_owner.get() == native_identity &&
                self.exact_preparation_->owned_map.getRawPtr() == map.getRawPtr() &&
                self.exact_preparation_->overlap_map.getRawPtr() == self.mesh_->overlap_cell_map().getRawPtr() &&
                self.exact_preparation_->owned_face_map.getRawPtr() == self.mesh_->owned_face_map().getRawPtr() &&
                self.exact_preparation_->overlap_face_map.getRawPtr() == self.mesh_->overlap_face_map().getRawPtr();
    if (same)
    {
        const auto& key = *self.exact_preparation_;
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
        self.exact_preparation_.reset();
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
    std::shared_ptr<ExactPreparation> prepared = self.exact_preparation_;
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
                if (const auto* native = std::get_if<MeshHandle<>::MultiRegionPtr>(&self.mesh_->variant()))
                    prepared->native_owner = *native;
                prepared->owned_map = map;
                prepared->overlap_map = self.mesh_->overlap_cell_map();
                prepared->owned_face_map = self.mesh_->owned_face_map();
                prepared->overlap_face_map = self.mesh_->overlap_face_map();
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
            verify_general_vof_exact_transfer_binding(self.mesh_, outgoing, routes, contributions, binding_budget);
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
    self.numeric_valid_ = false;
    self.matrix_ = Teuchos::null;
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
                Domain::Result, self.mesh_->num_owned_faces(), sizeof(M) + sizeof(Meshes::CertifiedGeometryValue));
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
            self.rhs_.replaceLocalValue(
                LO(c), E::view(E::divide(right[c], diagonal[c], arithmetic_budget()), enclosure));
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
        self.solver_.notify_operator_values_changed();
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
        const auto stats = self.solver_.solve_from_zero_with_statistics(
            Teuchos::rcp_implicit_cast<const Pack::operator_type>(matrix), self.rhs_, self.solution_, linear);
        require(*map, stats.converged, "Exact VoF inner solve failed its true residual.");
        result.iterations += stats.iterations;
    };
    solve();
    guard(
        [&]
        {
            const auto values = self.solution_.getData();
            for (std::size_t c = 0; c < n; ++c)
            {
                concentration[c] = values[c];
                store(accumulated[c], M::from_value(values[c]));
            }
        });
    guard(
        [&]
        {
            result.exact_flux.resize(self.mesh_->num_owned_faces());
            result.flux_enclosure.resize(self.mesh_->num_owned_faces());
            result.amount.resize(n);
            result.cell_trace.resize(n);
            for (const auto& value : result.exact_flux)
                storage.measure(Domain::Result, value);
        });
    for (;;)
    {
        {
            auto data = self.cell_owned_.getDataNonConst();
            std::copy(concentration.begin(), concentration.end(), data.begin());
        }
        donor.doImport(self.cell_owned_, donor_import, Tpetra::INSERT);
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
                self.rhs_.replaceLocalValue(
                    LO(c), E::view(E::divide(defect[c], scale, arithmetic_budget()), enclosure));
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
            const auto correction = self.solution_.getData();
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
    self.exact_preparation_ = std::move(prepared);
    if (cache_hit)
        ++self.preparation_statistics_.exact_hits;
    else
        ++self.preparation_statistics_.exact_builds;
    return result;
}
} // namespace Exact
} // namespace SimpleFluid::detail
