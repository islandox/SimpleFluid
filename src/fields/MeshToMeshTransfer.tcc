/** @file MeshToMeshTransfer.tcc
 * @brief Implements conservative cell-field mapping between supported meshes.
 */
#pragma once

#include "fields/MeshToMeshTransfer.hh"
#include "fields/details/MeshTransferGeometry.hh"
#include "fields/details/MeshTransferSearch.hh"

#include <Teuchos_CommHelpers.hpp>
#include <Teuchos_DefaultMpiComm.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>
#include <optional>
#include <vector>

namespace SimpleFluid
{
namespace mesh_transfer_detail
{

inline void require_all(const Teuchos::Comm<int>& comm, bool valid,
                        const char* message)
{
    const int local = valid ? 1 : 0;
    int global = 0;
    Teuchos::reduceAll(comm, Teuchos::REDUCE_MIN, 1, &local, &global);
    if (!global) throw std::invalid_argument(message);
}

inline bool same_communicator(const Teuchos::Comm<int>& a,
                              const Teuchos::Comm<int>& b)
{
    if (&a == &b) return true;
#ifdef HAVE_TEUCHOS_MPI
    const auto* ma = dynamic_cast<const Teuchos::MpiComm<int>*>(&a);
    const auto* mb = dynamic_cast<const Teuchos::MpiComm<int>*>(&b);
    if (ma && mb)
    {
        int comparison = MPI_UNEQUAL;
        MPI_Comm_compare(*ma->getRawMpiComm(), *mb->getRawMpiComm(), &comparison);
        return comparison == MPI_IDENT || comparison == MPI_CONGRUENT;
    }
#endif
    return a.getSize() == 1 && b.getSize() == 1;
}

template<class Map>
bool same_map_locally(const Map& actual, const Map& expected)
{
    return same_communicator(*actual.getComm(), *expected.getComm())
        && actual.getGlobalNumElements() == expected.getGlobalNumElements()
        && actual.getIndexBase() == expected.getIndexBase()
        && actual.locallySameAs(expected);
}

template<class MultiVector>
bool finite(const MultiVector& values)
{
    // getData selects the logical column even for nonconstant-stride subviews.
    for (size_t col = 0; col < values.getNumVectors(); ++col)
    {
        const auto data = values.getData(col);
        for (size_t row = 0; row < values.getLocalLength(); ++row)
            if (!std::isfinite(data[row])) return false;
    }
    return true;
}

} // namespace mesh_transfer_detail

template<TpetraTypePack Pack>
MeshToMeshTransfer<Pack>::MeshToMeshTransfer(
    SP<const mesh_type> source, SP<const mesh_type> target,
    MeshToMeshTransferOptions options)
    : MeshToMeshTransfer(*Tpetra::getDefaultComm(),
                         std::move(source), std::move(target), options)
{
}

template<TpetraTypePack Pack>
MeshToMeshTransfer<Pack>::MeshToMeshTransfer(
    const typename Pack::comm_type& construction_comm,
    SP<const mesh_type> source, SP<const mesh_type> target,
    MeshToMeshTransferOptions options)
    : d_source(std::move(source)), d_target(std::move(target)),
      d_source_epoch(0), d_target_epoch(0), d_options(options)
{
    using namespace mesh_transfer_detail;
    using GO = typename Pack::global_ordinal_type;
    using LO = typename Pack::local_ordinal_type;
    using Scalar = typename Pack::scalar_type;
    // Congruent MPI communicators have different collective contexts. Never
    // choose a fallback based on which rank-local endpoint happens to be null.
    const auto comm = Teuchos::rcpFromRef(construction_comm);
    require_all(*comm, bool(d_source) && bool(d_target),
                "Mesh transfer requires source and target meshes on every rank.");
    const auto source_map = d_source->owned_cell_map();
    const auto target_map = d_target->owned_cell_map();
    require_all(*comm, !source_map.is_null() && !target_map.is_null(),
                "Mesh transfer requires source and target cell maps on every rank.");
    d_source_map = source_map;
    d_target_map = target_map;
    const auto retain_geometry = [](const auto& native) -> SP<const void> { return native; };
    d_source_geometry = std::visit(retain_geometry, d_source->variant());
    d_target_geometry = std::visit(retain_geometry, d_target->variant());
    require_all(*comm, same_communicator(*comm, *source_map->getComm())
        && same_communicator(*comm, *target_map->getComm()),
        "Mesh transfer requires source and target communicators congruent to the construction context.");

    const bool conservative = options.method == MeshToMeshTransferMethod::ConservativeCellAverage;
    const auto source_kind = geometry_kind(*d_source), target_kind = geometry_kind(*d_target);
    const bool partial = options.coverage_mode == MeshToMeshCoverageMode::AllowPartial;
    require_all(*comm,
        (options.coverage_mode == MeshToMeshCoverageMode::RequireFull || partial)
        && (!partial || conservative)
        && (options.method == MeshToMeshTransferMethod::NearestCell
         || options.method == MeshToMeshTransferMethod::InverseDistance || conservative)
        && options.neighbors > 0
        && options.neighbors <= static_cast<size_t>(std::numeric_limits<int>::max())
        && options.max_distance >= 0 && !std::isnan(options.max_distance)
        && std::isfinite(options.coverage_tolerance)
        && options.coverage_tolerance >= 0 && options.coverage_tolerance <= 1e-6,
        "Invalid mesh transfer options.");
    const std::array<double, 7> local_options{
        static_cast<double>(options.method), static_cast<double>(options.neighbors),
        options.max_distance, options.coverage_tolerance,
        static_cast<double>(options.coverage_mode),
        conservative ? static_cast<double>(source_kind) : 0.0,
        conservative ? static_cast<double>(target_kind) : 0.0};
    std::array<double, 7> minimum{}, maximum{};
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MIN, 7, local_options.data(), minimum.data());
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 7, local_options.data(), maximum.data());
    require_all(*comm, minimum == maximum,
                "Mesh transfer options must agree on every rank.");
    require_all(*comm, !conservative
        || (source_kind == GeometryKind::Cartesian && target_kind == GeometryKind::Cartesian)
        || (source_kind == GeometryKind::Cylindrical && target_kind == GeometryKind::Cylindrical)
        || (source_kind == GeometryKind::Cylindrical && target_kind == GeometryKind::PolygonPrism)
        || (source_kind == GeometryKind::PolygonPrism && target_kind == GeometryKind::Cylindrical)
        || (source_kind == GeometryKind::Cylindrical && target_kind == GeometryKind::Composite)
        || (source_kind == GeometryKind::Composite && target_kind == GeometryKind::Cylindrical),
        "Conservative transfer requires two Cartesian meshes, two cylindrical meshes, "
        "or one cylindrical mesh and supported convex XY/Z or angular RZ polygon cells.");
    // Both checks are collective and must execute even if the first fails.
    const bool source_unique = source_map->isOneToOne();
    const bool target_unique = target_map->isOneToOne();
    require_all(*comm, source_unique && target_unique
        && source_map->getGlobalNumElements() > 0 && target_map->getGlobalNumElements() > 0
        && source_map->getGlobalNumElements() <= static_cast<size_t>(std::numeric_limits<int>::max() / 12),
        "Mesh transfer requires nonempty one-to-one cell maps within the geometry gather limit.");

    bool valid = true;
    std::vector<Geometry> source_geometry, target_geometry;
    try
    {
        source_geometry = geometry(*d_source, conservative, valid);
        target_geometry = geometry(*d_target, conservative, valid);
        d_source_epoch = d_source->geometry_epoch();
        d_target_epoch = d_target->geometry_epoch();
    }
    catch (const std::exception&)
    {
        valid = false;
    }
    size_t source_coordinates = 0;
    for (const auto& g : source_geometry) source_coordinates += serialized_size(g);
    valid = valid && source_coordinates <= static_cast<size_t>(std::numeric_limits<int>::max());
    require_all(*comm, valid, "Mesh transfer requires finite positive native volumes and supported "
        "geometry; polygon cells must be supported convex XY/Z or angular RZ extrusions within the geometry gather limit.");

    std::vector<GO> donor_ids;
    std::vector<Geometry> donors;
    donor_ids.reserve(source_map->getGlobalNumElements());
    donors.reserve(source_map->getGlobalNumElements());
    size_t source_offset = 0;
    for (int rank = 0; rank < comm->getSize(); ++rank)
    {
        int count = static_cast<int>(source_geometry.size());
        Teuchos::broadcast(*comm, rank, 1, &count);
        int coordinate_count = static_cast<int>(source_coordinates);
        Teuchos::broadcast(*comm, rank, 1, &coordinate_count);
        std::vector<GO> ids(count);
        std::vector<double> coordinates;
        if (comm->getRank() == rank)
        {
            coordinates.reserve(coordinate_count);
            for (int c = 0; c < count; ++c)
            {
                ids[c] = source_map->getGlobalElement(static_cast<LO>(c));
                append_geometry(source_geometry[c], coordinates);
            }
        }
        else coordinates.resize(coordinate_count);
        if (count)
        {
            Teuchos::broadcast(*comm, rank, count, ids.data());
            Teuchos::broadcast(*comm, rank, coordinate_count, coordinates.data());
        }
        if (rank == comm->getRank()) source_offset = donors.size();
        donor_ids.insert(donor_ids.end(), ids.begin(), ids.end());
        size_t coordinate_offset = 0;
        for (int c = 0; c < count; ++c) donors.push_back(read_geometry(coordinates, coordinate_offset));
    }

    // Donor records are immutable for this transfer and may have crossed MPI.
    // Validate them once before the candidate loop uses the internal kernel.
    if (conservative)
    {
        for (const auto& donor : donors)
        {
            if (donor.kind != GeometryKind::RZAngular) continue;
            try { rz_overlap_detail::validate(donor.polygon, donor[6], donor[7]); }
            catch (const std::invalid_argument&) { valid = false; }
        }
        require_all(*comm, valid, "Mesh transfer received invalid RZ angular geometry.");
    }

    std::vector<std::vector<GO>> columns(target_geometry.size());
    std::vector<std::vector<Scalar>> weights(target_geometry.size());
    std::vector<double> source_coverage(conservative ? donors.size() : 0, 0);
    std::vector<std::pair<double, size_t>> distances(conservative ? 0 : donors.size());
    // Geometry remains replicated, but exact intersections only visit BVH
    // candidates rather than every source cell for each target cell.
    std::optional<OverlapCandidates> search;
    if (conservative) search.emplace(donors);
    std::vector<size_t> candidates;
    const auto covered_is_valid = [&](double covered, double volume) {
        const double residual = covered / volume - 1.0;
        return std::isfinite(covered) && covered >= 0
            && residual <= options.coverage_tolerance
            && (partial || std::abs(residual) <= options.coverage_tolerance);
    };
    if (conservative)
    {
        for (const auto& g : target_geometry) d_coverage.target_cell_volumes.push_back(g[3]);
        d_coverage.target_covered_volumes.resize(target_geometry.size());
        d_coverage.target_uncovered_volumes.resize(target_geometry.size());
    }
    size_t max_entries = 0;
    for (size_t row = 0; row < target_geometry.size(); ++row)
    {
        const auto& target_cell = target_geometry[row];
        if (conservative)
        {
            double coverage = 0;
            search->find(target_cell, candidates);
            for (const size_t s : candidates)
            {
                const double overlap = intersection(target_cell, donors[s]);
                valid = valid && std::isfinite(overlap) && overlap >= 0;
                if (overlap <= 0) continue;
                columns[row].push_back(donor_ids[s]);
                weights[row].push_back(static_cast<Scalar>(overlap));
                coverage += overlap;
                source_coverage[s] += overlap;
            }
            valid = valid && covered_is_valid(coverage, target_cell[3]);
            d_coverage.target_covered_volumes[row] = coverage;
            d_coverage.target_uncovered_volumes[row] = target_cell[3] - coverage;
        }
        else
        {
            for (size_t s = 0; s < donors.size(); ++s)
            {
                const double distance = std::hypot(target_cell[0] - donors[s][0],
                    target_cell[1] - donors[s][1], target_cell[2] - donors[s][2]);
                distances[s] = {distance, s};
            }
            const size_t count = options.method == MeshToMeshTransferMethod::NearestCell
                ? 1 : std::min(options.neighbors, donors.size());
            std::partial_sort(distances.begin(), distances.begin() + count, distances.end(),
                [&](const auto& a, const auto& b) {
                    return a.first < b.first
                        || (a.first == b.first && donor_ids[a.second] < donor_ids[b.second]);
                });
            const double nearest = distances.front().first;
            valid = valid && std::isfinite(nearest) && nearest <= options.max_distance;
            if (!std::isfinite(nearest)) continue;
            const size_t used = nearest == 0 ? 1 : count;
            double sum = 0;
            for (size_t s = 0; s < used; ++s)
            {
                const double ratio = nearest == 0 ? 1 : nearest / distances[s].first;
                const double weight = ratio * ratio;
                columns[row].push_back(donor_ids[distances[s].second]);
                weights[row].push_back(static_cast<Scalar>(weight));
                sum += weight;
            }
            for (auto& weight : weights[row]) weight /= sum;
        }
        max_entries = std::max(max_entries, columns[row].size());
    }
    if (conservative)
    {
        std::vector<double> global_coverage(donors.size());
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, static_cast<int>(donors.size()),
                          source_coverage.data(), global_coverage.data());
        for (size_t s = 0; s < donors.size(); ++s)
            valid = valid && covered_is_valid(global_coverage[s], donors[s][3]);
        for (size_t s = 0; s < source_geometry.size(); ++s)
        {
            const double volume = source_geometry[s][3];
            const double covered = global_coverage[source_offset + s];
            d_coverage.source_cell_volumes.push_back(volume);
            d_coverage.source_covered_volumes.push_back(covered);
            d_coverage.source_uncovered_volumes.push_back(volume - covered);
        }
        std::array<double, 5> local_volumes{
            std::accumulate(d_coverage.source_cell_volumes.begin(), d_coverage.source_cell_volumes.end(), 0.0),
            std::accumulate(d_coverage.target_cell_volumes.begin(), d_coverage.target_cell_volumes.end(), 0.0),
            std::accumulate(d_coverage.target_covered_volumes.begin(), d_coverage.target_covered_volumes.end(), 0.0),
            std::accumulate(d_coverage.source_uncovered_volumes.begin(), d_coverage.source_uncovered_volumes.end(), 0.0),
            std::accumulate(d_coverage.target_uncovered_volumes.begin(), d_coverage.target_uncovered_volumes.end(), 0.0)};
        std::array<double, 5> global_volumes{};
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 5, local_volumes.data(), global_volumes.data());
        for (double volume : global_volumes) valid = valid && std::isfinite(volume);
        d_coverage.source_volume = global_volumes[0];
        d_coverage.target_volume = global_volumes[1];
        d_coverage.overlap_volume = global_volumes[2];
        d_coverage.uncovered_source_volume = global_volumes[3];
        d_coverage.uncovered_target_volume = global_volumes[4];
    }
    require_all(*comm, valid, conservative
        ? (partial ? "Conservative mesh transfer encountered invalid or excess overlap coverage."
                   : "Conservative mesh transfer requires full source and target volume coverage.")
        : "Mesh transfer target has no finite donor within max_distance.");

    d_weights = Teuchos::rcp(new typename Pack::matrix_type(target_map, max_entries));
    for (size_t row = 0; row < columns.size(); ++row)
        d_weights->insertGlobalValues(target_map->getGlobalElement(static_cast<LO>(row)),
            Teuchos::arrayViewFromVector(columns[row]), Teuchos::arrayViewFromVector(weights[row]));
    d_weights->fillComplete(source_map, target_map);
}

template<TpetraTypePack Pack>
void MeshToMeshTransfer<Pack>::validate_meshes(
    const mesh_type& source, const mesh_type& target) const
{
    mesh_transfer_detail::require_all(*d_source_map->getComm(),
        &source == d_source.get() && &target == d_target.get(),
        "Mesh transfer fields must use the original source and target mesh handles.");
}

template<TpetraTypePack Pack>
void MeshToMeshTransfer<Pack>::validate_geometry() const
{
    using namespace mesh_transfer_detail;
    int stale = 0;
    try
    {
        stale = d_source_epoch != d_source->geometry_epoch()
            || d_target_epoch != d_target->geometry_epoch()
            || d_source_geometry.get() != d_source->geometry_identity()
            || d_target_geometry.get() != d_target->geometry_identity()
            || !same_map_locally(*d_source->owned_cell_map(), *d_source_map)
            || !same_map_locally(*d_target->owned_cell_map(), *d_target_map);
    }
    catch (const std::exception&)
    {
        stale = 1;
    }
    int any_stale = 0;
    Teuchos::reduceAll(*d_source_map->getComm(), Teuchos::REDUCE_MAX, 1, &stale, &any_stale);
    if (any_stale)
        throw std::runtime_error("Mesh transfer geometry or maps changed; rebuild the transfer.");
}

template<TpetraTypePack Pack>
void MeshToMeshTransfer<Pack>::validate_data(
    const multi_vector_type& source, const multi_vector_type& target) const
{
    using namespace mesh_transfer_detail;
    validate_geometry();
    const auto comm = d_source_map->getComm();
    // Local comparisons plus fixed collectives handle rank-local mistakes.
    const auto source_map = source.getMap();
    const auto target_map = target.getMap();
    require_all(*comm, !source_map.is_null() && !target_map.is_null()
        && same_map_locally(*source_map, *d_source_map)
        && same_map_locally(*target_map, *d_target_map)
        && source.getNumVectors() == target.getNumVectors()
        && source.getNumVectors() > 0
        && source.getNumVectors() <= static_cast<size_t>(std::numeric_limits<int>::max()),
        "Mesh transfer requires matching cell maps and source/target component counts.");
    int count = static_cast<int>(source.getNumVectors()), min_count = 0, max_count = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MIN, 1, &count, &min_count);
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &count, &max_count);
    require_all(*comm, min_count == max_count,
                "Mesh transfer component counts must agree on every rank.");
    require_all(*comm, finite(source), "Mesh transfer source values must be finite.");
}

template<TpetraTypePack Pack>
const MeshToMeshCoverage& MeshToMeshTransfer<Pack>::coverage() const
{
    validate_geometry();
    mesh_transfer_detail::require_all(*d_source_map->getComm(),
        d_options.method == MeshToMeshTransferMethod::ConservativeCellAverage,
        "Coverage information requires conservative mesh transfer.");
    return d_coverage;
}

template<TpetraTypePack Pack>
void MeshToMeshTransfer<Pack>::apply(
    const multi_vector_type& source, multi_vector_type& target) const
{
    using namespace mesh_transfer_detail;
    if (d_options.method == MeshToMeshTransferMethod::ConservativeCellAverage)
    {
        require_all(*d_source_map->getComm(),
            d_options.coverage_mode == MeshToMeshCoverageMode::RequireFull,
            "Partial conservative transfer requires project() with an explicit quantity and report.");
        static_cast<void>(project(source, target, MeshToMeshQuantity::Intensive));
        return;
    }
    validate_data(source, target);
    multi_vector_type result(d_target_map, source.getNumVectors());
    d_weights->apply(source, result);
    require_all(*d_source_map->getComm(), finite(result),
                "Mesh transfer produced non-finite target values.");
    Tpetra::deep_copy(target, result);
}

template<TpetraTypePack Pack>
MeshToMeshConservationReport MeshToMeshTransfer<Pack>::project(
    const multi_vector_type& source, multi_vector_type& target,
    MeshToMeshQuantity quantity) const
{
    using namespace mesh_transfer_detail;
    const auto comm = d_source_map->getComm();
    require_all(*comm, d_options.method == MeshToMeshTransferMethod::ConservativeCellAverage
        && (quantity == MeshToMeshQuantity::Intensive || quantity == MeshToMeshQuantity::Extensive),
        "project() requires conservative mesh transfer and a valid quantity representation.");
    int representation = static_cast<int>(quantity), minimum = 0, maximum = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MIN, 1, &representation, &minimum);
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &representation, &maximum);
    require_all(*comm, minimum == maximum, "Mesh transfer quantity must agree on every rank.");
    validate_data(source, target);
    const auto count = source.getNumVectors();
    multi_vector_type density(d_source_map, count);
    for (size_t col = 0; col < count; ++col)
    {
        const auto input = source.getData(col);
        auto output = density.getDataNonConst(col);
        for (size_t row = 0; row < source.getLocalLength(); ++row)
            output[row] = quantity == MeshToMeshQuantity::Extensive
                ? input[row] / d_coverage.source_cell_volumes[row] : input[row];
    }
    require_all(*comm, finite(density), "Mesh transfer source density is non-finite.");
    multi_vector_type inventory(d_target_map, count);
    d_weights->apply(density, inventory);
    require_all(*comm, finite(inventory), "Mesh transfer produced non-finite target inventories.");
    multi_vector_type result(d_target_map, count);
    MeshToMeshConservationReport report;
    report.source_integral.resize(count);
    report.transferred_integral.resize(count);
    report.uncovered_source_integral.resize(count);
    report.target_integral.resize(count);
    report.transfer_error.resize(count);
    report.conservation_error.resize(count);
    std::vector<double> local_source(count), local_transferred(count),
                        local_uncovered(count), local_target(count);
    for (size_t col = 0; col < count; ++col)
    {
        const auto input = source.getData(col);
        const auto rho = density.getData(col);
        for (size_t row = 0; row < source.getLocalLength(); ++row)
        {
            local_source[col] += quantity == MeshToMeshQuantity::Extensive
                ? input[row] : input[row] * d_coverage.source_cell_volumes[row];
            local_transferred[col] += rho[row] * d_coverage.source_covered_volumes[row];
            local_uncovered[col] += rho[row] * d_coverage.source_uncovered_volumes[row];
        }
        const auto amounts = inventory.getData(col);
        auto output = result.getDataNonConst(col);
        for (size_t row = 0; row < target.getLocalLength(); ++row)
        {
            output[row] = quantity == MeshToMeshQuantity::Intensive
                ? amounts[row] / d_coverage.target_cell_volumes[row] : amounts[row];
            // Report the integral of the actual returned representation.
            local_target[col] += quantity == MeshToMeshQuantity::Intensive
                ? output[row] * d_coverage.target_cell_volumes[row] : output[row];
        }
    }
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, static_cast<int>(count),
                      local_source.data(), report.source_integral.data());
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, static_cast<int>(count),
                      local_transferred.data(), report.transferred_integral.data());
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, static_cast<int>(count),
                      local_uncovered.data(), report.uncovered_source_integral.data());
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, static_cast<int>(count),
                      local_target.data(), report.target_integral.data());
    bool valid = finite(result);
    for (size_t col = 0; col < count; ++col)
    {
        report.transfer_error[col] = report.target_integral[col] - report.transferred_integral[col];
        report.conservation_error[col] = report.target_integral[col]
            + report.uncovered_source_integral[col] - report.source_integral[col];
        valid = valid && std::isfinite(report.source_integral[col])
            && std::isfinite(report.transferred_integral[col])
            && std::isfinite(report.uncovered_source_integral[col])
            && std::isfinite(report.target_integral[col])
            && std::isfinite(report.transfer_error[col])
            && std::isfinite(report.conservation_error[col]);
    }
    require_all(*comm, valid, "Mesh transfer produced non-finite target values or conservation diagnostics.");
    Tpetra::deep_copy(target, result);
    return report;
}

} // namespace SimpleFluid
