/** @file LocalGeometry.cc
 * @brief Indexed conservative interface reach and owner-authenticated local cuts.
 */
#include "FVM/details/GeneralVoF/LocalGeometry.hh"
#include "FVM/details/GeneralVoF/WorkCapture.hh"
#include "FVM/details/GeneralVoF/AggregateBudgets.hh"
#include "FVM/details/GeneralVoF/FaceWorkspaceBudget.hh"
#include "FVM/details/GeneralVoF/LocalSpatialCache.hh"
#include "FVM/details/GeneralVoF/OwnerPlaneRequests.hh"
#include "FVM/details/GeneralVoF/PhaseAllocation.hh"
#include "geometry/PLIC/PlanarGeometry.hh"
#include "geometry/swept_volume/OrdinaryAggregate.hh"
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace SimpleFluid::VoF::general_vof_detail
{
namespace
{
constexpr auto mode = ArithmeticMode::Ordinary;
using Point = PLIC::Point<mode>;
using Geometry = PLIC::ConvexCellGeometry<mode>;
using Math = fp::Arithmetic<mode>;
using local_spatial_detail::Box;
using Wide = std::array<long double, 3>;
/** Local diagnostic only: no collective, allocation, or exception may hide the
 * original geometry failure. Witness capacity is deliberately not called RSS. */
struct AggregateFaceTrace
{
    const SimpleFluid::geometry::swept_volume::OrdinaryAggregateContext& progress;
    const FaceWorkspaceBudget& workspace;
    std::size_t outer_entry, original_budget, aggregate_entry, initial_remaining_budget;
    int rank;
    std::uint64_t face;
    bool enabled = false, completed = false;

    AggregateFaceTrace(const SimpleFluid::geometry::swept_volume::OrdinaryAggregateContext& snapshot,
        const FaceWorkspaceBudget& storage, std::size_t budget, int owner, std::uint64_t identity) noexcept
        : progress(snapshot), workspace(storage), outer_entry(storage.retained_entry()), original_budget(budget),
          aggregate_entry(outer_entry), initial_remaining_budget(storage.remaining_bytes()), rank(owner), face(identity)
    {
        const auto* value = std::getenv("SIMPLEFLUID_VOF_AGGREGATE_TRACE");
        enabled = value && std::strcmp(value, "1") == 0;
    }

    ~AggregateFaceTrace() noexcept
    {
        if (!enabled)
            return;
        const auto& stats = progress.statistics;
        const auto aggregate_bytes = progress.bytes;
        const auto reported_total = workspace.retained_bytes() + workspace.scratch_bytes();
        const auto preparation_bytes = workspace.scratch_bytes() - aggregate_bytes;
        using Stage = SimpleFluid::geometry::swept_volume::OrdinaryAggregateStage;
        const auto* stage = progress.stage == Stage::Source            ? "source"
                            : progress.stage == Stage::DonorValidation ? "donor_validation"
                            : progress.stage == Stage::Coverage        ? "coverage"
                                                                       : "metric";
        (void) std::fprintf(stderr,
            "{\"event\":\"vof_aggregate_face\",\"rank\":%d,\"face\":%llu,\"status\":\"%s\",\"stage\":\"%s\","
            "\"outer_charged_bytes_at_entry\":%zu,\"aggregate_outer_bytes_at_entry\":%zu,"
            "\"face_charged_bytes\":%zu,\"aggregate_charged_bytes\":%zu,\"outer_reported_total\":%zu,"
            "\"face_preparation_charged_bytes\":%zu,"
            "\"retained_bytes_at_entry\":%zu,\"retained_bytes_after\":%zu,\"face_scratch_charged_bytes\":%zu,"
            "\"combined_storage_peak_bytes\":%zu,\"total_construction_charged_bytes\":%zu,\"total_aggregate_"
            "operations\":%zu,"
            "\"original_overall_budget\":%zu,\"face_initial_remaining_budget\":%zu,"
            "\"stage_charged_bytes\":{\"source\":%zu,\"donor_validation\":%zu,\"coverage\":%zu,\"metric\":%zu},"
            "\"aggregate_requested_failed_charge_bytes\":%zu,\"vertices_classified\":%zu,\"vertices_copied\":%zu,"
            "\"copied_vertex_bytes\":%zu,"
            "\"active_vertices_seen\":%zu,\"stored_vertices_seen\":%zu,\"peak_active_vertices\":%zu,"
            "\"peak_stored_vertices\":%zu,\"peak_remainders\":%zu,\"witness_admissions\":%zu,"
            "\"peak_live_witnesses\":%zu,\"peak_live_witness_capacity_bytes\":%zu,"
            "\"source_pieces\":%zu,\"pair_tests\":%zu,\"operations\":%zu}\n",
            rank, static_cast<unsigned long long>(face), completed ? "success" : "failure", stage, outer_entry,
            aggregate_entry, reported_total - outer_entry, aggregate_bytes, reported_total, preparation_bytes,
            outer_entry, workspace.retained_after(), workspace.scratch_bytes(), workspace.peak_bytes(),
            workspace.construction_bytes(), workspace.operations(), original_budget, initial_remaining_budget,
            stats.stage_charged_bytes[0], stats.stage_charged_bytes[1], stats.stage_charged_bytes[2],
            stats.stage_charged_bytes[3], stats.failed_charge_bytes, stats.vertices_classified, stats.vertices_copied,
            stats.copied_vertex_bytes, stats.active_vertices_seen, stats.stored_vertices_seen,
            stats.peak_active_vertices, stats.peak_stored_vertices, stats.peak_remainders, stats.witnesses,
            stats.peak_live_witnesses, stats.peak_live_witness_bytes, stats.source_pieces, stats.pair_tests,
            stats.operations);
    }
};
bool needs_interface(const Cell& cell)
{
    return cell.declaration == GeneralVoFPhaseGeometry::Infer || cell.requires_interface ||
           (cell.retained_interface && !cell.retained_interface->pure_gas && !cell.retained_interface->pure_carrier);
}
} // namespace
LocalGeometryResult prepare_local_sweeps(const GeneralVoF::Data& data, const OrdinaryTraceMap& traces,
    const OrdinaryFluxMap& flux, const std::map<std::uint64_t, long double>& face_budgets,
    const PLIC::GeometryPacketContext& context)
{
    namespace failure = ::SimpleFluid::detail::vof_failure;
    failure::Scope operation_context("interface_local_geometry");
    operation_context.record.arithmetic_mode = mode;
    operation_context.record.mark(failure::Field::Arithmetic);
    operation_context.record.byte_budget = data.options.byte_budget;
    operation_context.record.mark(failure::Field::ByteBudget);
    const auto comm = data.hierarchy->communicator();
    LocalGeometryResult result;
    SimpleFluid::geometry::swept_volume::Work work{data.options.sweep};
    work.arithmetic_mode = mode;
    work.volume_tolerance = data.options.reconstruction.volume_tolerance;
    work.limits.byte_budget = std::min(work.limits.byte_budget, data.options.byte_budget);
    FaceWorkspaceTotals workspace_totals;
    struct Pending
    {
        const AMR::AdaptiveTile* tile;
        SimpleFluid::geometry::swept_volume::Sweep<mode> recipe;
        std::vector<std::size_t> candidates;
        bool geometric = false;
    };
    std::vector<Pending> pending;
    std::map<std::uint64_t, std::size_t> local_authorities;
    std::vector<const Cell*> cells;
    std::shared_ptr<const LocalSpatialCache> spatial;
    std::set<Key> requested;
    std::size_t capture_bytes = 0, source_capture_remaining = 0;
    const auto charge_capture = [&](std::size_t bytes)
    {
        work.charge(bytes);
        if (bytes > source_capture_remaining)
            throw PLIC::GeometryError(PLIC::GeometryErrorCode::ResourceLimit,
                "General VoF work capture exceeds the source retained byte budget.");
        source_capture_remaining -= bytes;
        capture_bytes += bytes;
    };
    wire::collective(*comm,
        [&]
        {
            current(data);
            if (data.options.arithmetic_mode != mode)
                throw std::invalid_argument("Local VoF geometry requires Ordinary arithmetic.");
            spatial = retained_spatial_cache(data, work, result.diagnostics,
                [](const GeneralVoF::Data& source, std::size_t bytes) { retained_budget<mode>(source, bytes); });
            for (const auto& identity : spatial->cells)
                work.push(cells, &data.cells.at(identity.key));
            result.diagnostics.indexed_cells = cells.size();
            if (data.options.record_balance_work)
            {
                retained_budget<mode>(data, {}, 0, &source_capture_remaining);
                charge_capture(sizeof(GeneralVoFWorkSample) + 2 * sizeof(void*));
                result.balance_work = std::make_shared<GeneralVoFWorkSample>();
                auto& capture = *result.balance_work;
                charge_capture(data.topology->order.size() * sizeof(GeneralVoFWorkCell));
                capture.cells.reserve(data.topology->order.size());
                if (capture.cells.capacity() > data.topology->order.size())
                    charge_capture(
                        (capture.cells.capacity() - data.topology->order.size()) * sizeof(GeneralVoFWorkCell));
                for (const auto& key : data.topology->order)
                {
                    const auto& cell = data.cells.at(key);
                    charge_capture(owner_plane_key_storage_bytes(key));
                    capture.cells.push_back({key, cell.revision, cell.owner});
                }
                std::sort(capture.cells.begin(), capture.cells.end(),
                    [](const auto& a, const auto& b) { return a.key < b.key; });
                const auto count = std::count_if(data.topology->tiles.begin(), data.topology->tiles.end(),
                    [&](const auto& tile)
                    { return !tile.exterior && tile.second && tile.first_owner == comm->getRank(); });
                charge_capture(count * sizeof(GeneralVoFWorkFace));
                capture.faces.reserve(count);
                if (capture.faces.capacity() > static_cast<std::size_t>(count))
                    charge_capture((capture.faces.capacity() - count) * sizeof(GeneralVoFWorkFace));
                for (const auto& tile : data.topology->tiles)
                    if (!tile.exterior && tile.second && tile.first_owner == comm->getRank())
                    {
                        charge_capture(tile.identity.size() + owner_plane_key_storage_bytes(tile.first) +
                            owner_plane_key_storage_bytes(*tile.second));
                        capture.faces.push_back({tile.identity,
                            {tile.first, data.cells.at(tile.first).revision, tile.first_owner},
                            {*tile.second, data.cells.at(*tile.second).revision, tile.second_owner}});
                    }
            }
            std::size_t capture_face = 0;
            for (const auto& tile : data.topology->tiles)
            {
                if (tile.exterior || !tile.second || tile.first_owner != comm->getRank())
                    continue;
                auto* face_sample = result.balance_work ? &result.balance_work->faces[capture_face++] : nullptr;
                WorkObservation setup_observation(face_sample ? &face_sample->setup : nullptr);
                failure::Scope face_context("interface_local_reach");
                face_context.record.face = tile.face_gid;
                face_context.record.mark(failure::Field::Face);
                work.charge(2 * tile.node_gids.size() * sizeof(Point) + sizeof(Pending));
                auto recipe = [&]
                {
                    SimpleFluid::geometry::swept_volume::OrdinaryAggregateContext progress;
                    FaceWorkspaceBudget workspace(work, progress, workspace_totals);
                    return workspace.checked_action(
                        [&]
                        {
                            auto& scratch = workspace.setup_work();
                            // prepare_sweep retains the point arrays charged above, but
                            // its unique-node set and triangle buffer end with the call.
                            scratch.charge(tile.node_gids.size() * (sizeof(std::uint64_t) + 4 * sizeof(void*) +
                                                                       3 * sizeof(std::array<Point, 3>)));
                            return SimpleFluid::geometry::swept_volume::prepare_sweep<mode>(tile.face_gid,
                                tile.node_gids, traces, flux.at(tile.face_gid), scratch,
                                face_budgets.at(tile.face_gid));
                        });
                }();
                // These sealed centers own provenance beyond the inline point
                // arrays. Their lifetime follows the pending recipe, not the
                // source/metric workspace that later borrows them.
                work.charge(PLIC::point_storage_bytes(recipe.center) + PLIC::point_storage_bytes(recipe.backcenter));
                Box bounds;
                for (const auto& p : recipe.current)
                    bounds.add(p.value);
                for (const auto& p : recipe.departure)
                    bounds.add(p.value);
                bounds.add(recipe.center.value);
                bounds.add(recipe.backcenter.value);
                {
                    SimpleFluid::geometry::swept_volume::OrdinaryAggregateContext progress;
                    FaceWorkspaceBudget workspace(work, progress, workspace_totals);
                    workspace.checked_action(
                        [&]
                        {
                            auto& scratch = workspace.setup_work();
                            for (const auto index : spatial->frontier_index.query(bounds, scratch, result.diagnostics))
                                prove_retained(data, *spatial, spatial->frontiers[index], recipe, scratch);
                        });
                }
                auto candidates = spatial->donor_index.query(bounds, work, result.diagnostics);
                result.diagnostics.donor_candidates += candidates.size();
                bool geometric = false;
                std::optional<GeneralVoFPhaseGeometry> pure;
                const auto classify = [&](const Cell& cell)
                {
                    geometric |= needs_interface(cell);
                    if (cell.declaration != GeneralVoFPhaseGeometry::Infer)
                    {
                        if (pure && *pure != cell.declaration)
                            geometric = true;
                        pure = cell.declaration;
                    }
                };
                classify(data.cells.at(tile.first));
                classify(data.cells.at(*tile.second));
                for (const auto candidate : candidates)
                    classify(*cells[candidate]);
                // A zero net control can still have nonzero cancelling signed pieces.
                bool zero_effect = true;
                {
                    SimpleFluid::geometry::swept_volume::OrdinaryAggregateContext progress;
                    FaceWorkspaceBudget workspace(work, progress, workspace_totals);
                    workspace.checked_action(
                        [&]
                        {
                            auto& scratch = workspace.setup_work();
                            scratch.charge(3 * tile.node_gids.size() * sizeof(std::array<Point, 3>));
                            for (const auto& triangle :
                                SimpleFluid::geometry::swept_volume::sweep_boundary(
                                    tile.node_gids, recipe.current, recipe.departure, recipe.center, recipe.backcenter))
                            {
                                scratch.test();
                                const auto v =
                                    SimpleFluid::geometry::swept_volume::cone_determinant(recipe.center, triangle);
                                zero_effect &= v.value == 0 && v.absolute_error == 0;
                            }
                        });
                }
                geometric &= !zero_effect;
                if (geometric)
                    for (const auto candidate : candidates)
                    {
                        work.charge(
                            sizeof(Key) + 4 * sizeof(void*) + AMR::encode_adaptive_key(cells[candidate]->key).size());
                        requested.insert(cells[candidate]->key);
                        const auto gid = cells[candidate]->gid;
                        if (!local_authorities.contains(gid))
                            work.charge(sizeof(decltype(local_authorities)::value_type) + 4 * sizeof(void*));
                        auto& count = local_authorities[gid];
                        if (count > std::numeric_limits<std::size_t>::max() - 2)
                            throw std::invalid_argument("Aggregate source authority count overflows.");
                        count += 2; // A face may retain both signed source groups.
                    }
                setup_observation.finish();
                if (face_sample)
                {
                    face_sample->geometric = geometric;
                    if (geometric)
                    {
                        charge_capture(candidates.size() * sizeof(GeneralVoFWorkDonor));
                        face_sample->donors.reserve(candidates.size());
                        if (face_sample->donors.capacity() > candidates.size())
                            charge_capture(
                                (face_sample->donors.capacity() - candidates.size()) * sizeof(GeneralVoFWorkDonor));
                        for (auto candidate : candidates)
                        {
                            const auto& cell = *cells[candidate];
                            charge_capture(owner_plane_key_storage_bytes(cell.key));
                            face_sample->donors.push_back({cell.key, cell.revision, cell.owner});
                        }
                    }
                }
                work.push(pending, Pending{&tile, std::move(recipe), std::move(candidates), geometric});
            }
            result.diagnostics.plane_requests = requested.size();
            if (result.balance_work)
            {
                std::size_t remote = 0;
                for (const auto& key : requested)
                    remote += data.cells.at(key).owner != comm->getRank();
                charge_capture(remote * sizeof(GeneralVoFWorkReceiver));
                auto& receivers = result.balance_work->receivers;
                receivers.reserve(remote);
                if (receivers.capacity() > remote)
                    charge_capture((receivers.capacity() - remote) * sizeof(GeneralVoFWorkReceiver));
                for (const auto& key : requested)
                    if (const auto& cell = data.cells.at(key); cell.owner != comm->getRank())
                    {
                        charge_capture(owner_plane_key_storage_bytes(key));
                        receivers.push_back({{key, cell.revision, cell.owner}, comm->getRank()});
                    }
                const auto actual = balance_work_storage_bytes(*result.balance_work);
                if (actual > capture_bytes)
                    charge_capture(actual - capture_bytes);
                result.balance_work->retained_bytes = actual;
            }
        });
    const auto authority_counts = aggregate_authority_counts(data, local_authorities, context);
    // These retained owners coexist with import buffers and optional peer-only
    // reconstructions. Reserve their entire returned set before cache admission.
    wire::collective(*comm,
        [&]
        {
            work.charge(authority_counts.size() * (sizeof(decltype(authority_counts)::value_type) + 4 * sizeof(void*)));
            for (const auto& key : requested)
                work.charge(owner_plane_cache_entry_bytes(key));
        });
    const auto planes = import_requested_planes(
        data, requested, context, work, result.diagnostics.owner_plane_requests, result.balance_work.get());
    wire::collective(*comm,
        [&]
        {
            std::map<Key, PLIC::ConvexGeometryPtr<mode>> geometries;
            std::map<Key, PhaseLabel<mode>> labels;
            for (const auto& key : requested)
            {
                const auto& cell = data.cells.at(key);
                work.charge(sizeof(typename decltype(geometries)::value_type) +
                            sizeof(typename decltype(labels)::value_type) + 8 * sizeof(void*) + sizeof(Geometry));
                auto geometry = Geometry::from_dynamic(cell.finite_geometry);
                const auto& plane = planes.at(key);
                if ((cell.phase.gas_volume == 0 || cell.phase.carrier_volume == 0) && !plane.pure_gas &&
                    !plane.pure_carrier)
                    throw std::invalid_argument(
                        "Local VoF absent phase requires pure geometry for normalized history transport.");
                labels.emplace(key, phase_label<mode>(cell, *geometry, plane, work.volume_tolerance));
                geometries.emplace(key, std::move(geometry));
            }
            std::size_t capture_face = 0;
            for (auto& entry : pending)
            {
                auto* face_sample = result.balance_work ? &result.balance_work->faces[capture_face++] : nullptr;
                WorkObservation aggregate_observation(face_sample ? &face_sample->aggregate : nullptr);
                const auto& tile = *entry.tile;
                failure::Scope face_context("interface_local_partition");
                face_context.record.face = tile.face_gid;
                face_context.record.mark(failure::Field::Face);
                face_context.record.volume_budget = face_budgets.at(tile.face_gid);
                face_context.record.mark(failure::Field::VolumeBudget);
                face_context.record.bytes_used = work.bytes;
                face_context.record.mark(failure::Field::BytesUsed);
                const auto make_face = [&]
                {
                    LocalFaceGeometry face;
                    face.face = tile.face_gid;
                    face.geometric = entry.geometric;
                    face.chain = {tile.face_gid, entry.recipe.control, entry.recipe.control, Math::make_value(mode, 0),
                        entry.recipe.ordinary_volume_budget};
                    return face;
                };
                if (!entry.geometric)
                {
                    ++result.diagnostics.algebraic_faces;
                    work.push(result.owned_faces, make_face());
                    continue;
                }
                ++result.diagnostics.geometric_faces;
                SimpleFluid::geometry::swept_volume::OrdinaryAggregateContext progress;
                // Accounting is retired after every face-local container, on
                // success and exception. Committed preparation storage survives.
                FaceWorkspaceBudget workspace(work, progress, workspace_totals);
                AggregateFaceTrace aggregate_trace(
                    progress, workspace, work.limits.byte_budget, comm->getRank(), tile.face_gid);
                workspace.checked_action(
                    [&]
                    {
                        auto& face_work = workspace.setup_work();
                        auto face = make_face();
                        // Source identities follow the face recipe, including its private
                        // corrected center. Equal rounded coordinates never merge identities.
                        const auto& recipe = entry.recipe;
                        const auto n = tile.node_gids.size();
                        std::vector<SimpleFluid::geometry::swept_volume::OrdinaryAggregateTetra> tetrahedra;
                        const auto append = [&](std::array<std::size_t, 3> ids)
                        {
                            const auto point = [&](std::size_t id) -> const Point&
                            {
                                if (id == 0)
                                    return recipe.center;
                                if (id == 1)
                                    return recipe.backcenter;
                                return id < n + 2 ? recipe.current.at(id - 2) : recipe.departure.at(id - n - 2);
                            };
                            SimpleFluid::geometry::swept_volume::OrdinaryAggregateTetra tetra;
                            tetra.vertex_ids = {0, ids[0], ids[1], ids[2]};
                            for (unsigned i = 0; i < 4; ++i)
                                tetra.vertices[i] = point(tetra.vertex_ids[i]);
                            const auto orientation = PLIC::orientation<mode>(tetra.vertices).value;
                            // A proved contact has zero measure. Every nonzero source
                            // cone remains in the immutable batch, however thin it is.
                            if (orientation == 0)
                                return;
                            tetra.sign = orientation > 0 ? 1 : -1;
                            face_work.push(tetrahedra, std::move(tetra));
                        };
                        for (std::size_t i = 0; i < n; ++i)
                        {
                            const auto j = (i + 1) % n;
                            append({1, n + 2 + j, n + 2 + i});
                            if (tile.node_gids[i] < tile.node_gids[j])
                            {
                                append({2 + i, n + 2 + i, n + 2 + j});
                                append({2 + i, n + 2 + j, 2 + j});
                            }
                            else
                            {
                                append({2 + j, 2 + i, n + 2 + i});
                                append({2 + j, n + 2 + i, n + 2 + j});
                            }
                        }
                        std::vector<SimpleFluid::geometry::swept_volume::OrdinaryAggregateDonor> donors;
                        std::map<std::size_t, std::array<long double, 2>> phase_quotas;
                        const auto quota =
                            face_budgets.at(tile.face_gid) / (2 * std::max<std::size_t>(1, entry.candidates.size()));
                        for (const auto i : entry.candidates)
                        {
                            const auto& cell = *cells[i];
                            const auto& plane = planes.at(cell.key);
                            SimpleFluid::geometry::swept_volume::OrdinaryAggregateDonor donor;
                            donor.id = i;
                            donor.geometry = geometries.at(cell.key);
                            donor.pure_gas = plane.pure_gas;
                            donor.pure_carrier = plane.pure_carrier;
                            if (!plane.pure_gas && !plane.pure_carrier)
                                donor.gas = plane.gas;
                            donor.bulk_budget = quota;
                            const auto& label = labels.at(cell.key);
                            const auto phase_quota = [&](bool bulk, PhaseValue phase)
                            {
                                const auto source = bulk ? label.raw_source.bulk.value : phase.value;
                                const auto total = std::nextafter(source * work.volume_tolerance, 0.L);
                                const auto allocated = std::nextafter(total / authority_counts.at(cell.gid), 0.L);
                                if (source > 0 && !(allocated > 0))
                                    throw std::invalid_argument("Aggregate independent phase quota is unresolved.");
                                return std::min(quota, allocated);
                            };
                            face_work.charge(sizeof(decltype(phase_quotas)::value_type) + 4 * sizeof(void*));
                            phase_quotas.emplace(i, std::array{phase_quota(label.gas_bulk, label.raw_source.gas),
                                                        phase_quota(label.carrier_bulk, label.raw_source.carrier)});
                            // The candidate must resolve both positive phases and the
                            // original face allowance. Its donor-phase allowance is
                            // consumed only by the final shared correction below.
                            donor.gas_budget = donor.carrier_budget = quota;
                            face_work.push(donors, std::move(donor));
                        }
                        auto sweep_limits = work.limits;
                        sweep_limits.byte_budget = workspace.remaining_bytes();
                        aggregate_trace.aggregate_entry = work.bytes + face_work.bytes;
                        aggregate_trace.initial_remaining_budget = sweep_limits.byte_budget;
                        sweep_limits.maximum_pair_tests -=
                            std::min(work.pair_tests + face_work.pair_tests, sweep_limits.maximum_pair_tests);
                        auto limits = geometry_limits(mode, sweep_limits.byte_budget);
                        limits.maximum_pair_tests = sweep_limits.maximum_pair_tests;
                        limits.maximum_operations = data.options.maximum_face_geometry_operations;
                        SimpleFluid::geometry::swept_volume::OrdinaryAggregateResult aggregate;
                        try
                        {
                            const SimpleFluid::geometry::swept_volume::OrdinaryAggregateSource source(
                                tetrahedra, sweep_limits, limits, &progress);
                            aggregate = source.measure(donors, &progress);
                        }
                        catch (...)
                        {
                            using Stage = SimpleFluid::geometry::swept_volume::OrdinaryAggregateContext::Stage;
                            const auto stage = progress.stage == Stage::Source     ? "interface_aggregate_source"
                                               : progress.stage == Stage::Coverage ? "interface_aggregate_coverage"
                                               : progress.stage == Stage::Metric   ? "interface_aggregate_metric"
                                                                                   : "interface_aggregate_donors";
                            failure::Scope diagnostic(stage);
                            diagnostic.record.bytes_used = work.bytes + workspace.scratch_bytes();
                            diagnostic.record.pair_tests = work.pair_tests + face_work.pair_tests + progress.pairs;
                            diagnostic.record.pair_budget = work.limits.maximum_pair_tests;
                            diagnostic.record.retry = progress.retry;
                            for (const auto field : {failure::Field::BytesUsed, failure::Field::PairTests,
                                     failure::Field::PairBudget, failure::Field::Retry})
                                diagnostic.record.mark(field);
                            if (progress.source < tetrahedra.size())
                            {
                                diagnostic.record.construction = progress.source;
                                diagnostic.record.mark(failure::Field::Construction);
                                for (unsigned i = 0; i < 4; ++i)
                                    for (unsigned k = 0; k < 3; ++k)
                                        diagnostic.record.simplex[3 * i + k] =
                                            tetrahedra[progress.source].vertices[i].value[k];
                                diagnostic.record.mark(failure::Field::Simplex);
                            }
                            if (progress.donor < cells.size())
                            {
                                diagnostic.record.donor = cells[progress.donor]->gid;
                                diagnostic.record.mark(failure::Field::Donor);
                            }
                            if (progress.cut != SimpleFluid::geometry::swept_volume::OrdinaryAggregateContext::absent)
                            {
                                diagnostic.record.cut = progress.cut;
                                diagnostic.record.mark(failure::Field::Cut);
                            }
                            throw;
                        }
                        // The aggregate reservation includes its returned groups and
                        // remains live while retained output is copied below. Only
                        // actual retained output growth enters preparation Work.
                        result.diagnostics.swept_pieces += aggregate.statistics.source_pieces;
                        result.diagnostics.aggregate_groups += aggregate.groups.size();
                        result.diagnostics.aggregate_witnesses += aggregate.statistics.witnesses;
                        result.diagnostics.cancelled_boundaries += aggregate.statistics.cancelled_boundaries;
                        result.diagnostics.aggregate_retries += aggregate.statistics.retries;
                        face.chain.constructed = Math::make_value(mode, aggregate.signed_source_volume);
                        face.chain.signed_difference = Math::subtract(face.chain.constructed, recipe.control);
                        if (std::abs(face.chain.signed_difference.value) > face_budgets.at(tile.face_gid))
                            throw std::invalid_argument(
                                "Interface aggregate source chain exceeds original face allowance.");
                        for (const auto& group : aggregate.groups)
                        {
                            const auto& cell = *cells.at(group.donor);
                            failure::Scope donor_context("interface_local_aggregate_measure");
                            donor_context.record.donor = cell.gid;
                            donor_context.record.mark(failure::Field::Donor);
                            // Unresolved approximations require a separate conservative
                            // ledger debit. Until one is present, reject rather than
                            // interpreting numerical quality estimates as enclosures.
                            if (group.approximation_bound.bulk != 0 || group.approximation_bound.gas != 0 ||
                                group.approximation_bound.carrier != 0)
                                throw std::invalid_argument(
                                    "Interface aggregate approximation has no conservative ledger debit.");
                            const auto& label = labels.at(cell.key);
                            const std::array errors{label.gas_bulk ? group.numerical_error_estimate.bulk
                                                                   : group.numerical_error_estimate.gas,
                                label.carrier_bulk ? group.numerical_error_estimate.bulk
                                                   : group.numerical_error_estimate.carrier};
                            for (unsigned phase = 0; phase < 2; ++phase)
                            {
                                const auto available = phase_quotas.at(group.donor)[phase];
                                const auto error = errors[phase];
                                if (error > available)
                                {
                                    const auto ratio = std::nextafter(std::nextafter(available, 0.L) / error, 0.L);
                                    if (!(ratio > 0) || !std::isfinite(ratio))
                                        throw std::invalid_argument(
                                            "Aggregate final phase-quality ratio is unresolved.");
                                    face.quality_theta = std::min(face.quality_theta, ratio);
                                }
                            }
                            const PhaseValues raw{Math::make_value(mode, group.nominal.bulk),
                                Math::make_value(mode, group.nominal.gas),
                                Math::make_value(mode, group.nominal.carrier)};
                            const auto amount = label.apply(raw), history = label.history(raw);
                            const auto allocation = label.allocation(raw);
                            ++result.diagnostics.plane_measurements;
                            const auto inventory = phase_inventory<mode>(cell, *geometries.at(cell.key));
                            const auto share = Math::divide(amount.bulk, inventory.bulk).value;
                            SimpleFluid::detail::CompensatedSum<> source_defect, cut_closure;
                            source_defect += inventory.gas.value;
                            source_defect += inventory.carrier.value;
                            source_defect += -inventory.bulk.value;
                            cut_closure += amount.gas.value;
                            cut_closure += amount.carrier.value;
                            cut_closure += -amount.bulk.value;
                            cut_closure += -source_defect.value() * share;
                            cut_closure += -allocation.value;
                            workspace.push_retained(face.groups,
                                LocalGeometricGroup{{tile.face_gid, cell.gid, cell.owner, group.sign, amount}, share,
                                    allocation.value, cut_closure.value(), history.gas.value, history.carrier.value,
                                    group.gas_positive, group.carrier_positive});
                        }
                        const auto old_face_capacity = result.owned_faces.capacity();
                        workspace.reserve_retained_slot(result.owned_faces);
                        const auto surviving_bytes =
                            face.groups.capacity() * sizeof(LocalGeometricGroup) +
                            (result.owned_faces.capacity() - old_face_capacity) * sizeof(LocalFaceGeometry);
                        workspace.validate_retained_commit(surviving_bytes);
                        static_assert(std::is_nothrow_move_constructible_v<LocalFaceGeometry>);
                        // Capacity has been admitted and reserved. No allocation or
                        // throwing physical check occurs between publication and commit.
                        result.owned_faces.push_back(std::move(face));
                        workspace.commit_retained(surviving_bytes);
                        aggregate_trace.completed = true;
                    });
            }
            std::sort(result.owned_faces.begin(), result.owned_faces.end(),
                [&](const auto& a, const auto& b)
                {
                    work.test();
                    return a.face < b.face;
                });
        });
    return result;
}
} // namespace SimpleFluid::VoF::general_vof_detail
