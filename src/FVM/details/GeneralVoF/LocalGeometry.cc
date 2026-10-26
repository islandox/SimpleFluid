/** @file LocalGeometry.cc
 * @brief Indexed conservative interface reach and owner-authenticated local cuts.
 */
#include "FVM/details/GeneralVoF/LocalGeometry.hh"
#include "FVM/details/GeneralVoF/AggregateBudgets.hh"
#include "FVM/details/GeneralVoF/PhaseAllocation.hh"
#include "geometry/PLIC/OrdinaryAggregate.hh"
#include "geometry/PLIC/PlanarGeometry.hh"
#include <numeric>

namespace SimpleFluid::VoF::general_vof_detail
{
namespace
{
constexpr auto mode = ArithmeticMode::Ordinary;
using Point = PLIC::Point<mode>;
using Geometry = PLIC::ConvexCellGeometry<mode>;
using Math = fp::Arithmetic<mode>;
using Wide = std::array<long double, 3>;
constexpr auto infinity = std::numeric_limits<long double>::infinity();
struct Box
{
    Wide low{infinity, infinity, infinity}, high{-infinity, -infinity, -infinity};
    void add(const Wide& p)
    {
        for (unsigned axis = 0; axis < 3; ++axis)
        {
            if (!std::isfinite(p[axis]))
                throw std::invalid_argument("Local VoF index coordinate is nonfinite.");
            low[axis] = std::min(low[axis], std::nextafter(p[axis], -infinity));
            high[axis] = std::max(high[axis], std::nextafter(p[axis], infinity));
        }
    }
    void add(const Box& b)
    {
        for (unsigned axis = 0; axis < 3; ++axis)
        {
            low[axis] = std::min(low[axis], b.low[axis]);
            high[axis] = std::max(high[axis], b.high[axis]);
        }
    }
    bool overlaps(const Box& b) const
    {
        for (unsigned axis = 0; axis < 3; ++axis)
            if (high[axis] < b.low[axis] || b.high[axis] < low[axis])
                return false;
        return true;
    }
};
/** Balanced median index. Spatial ordering only schedules conservative AABB tests;
 * every returned list is restored to its canonical source order. */
class Index
{
    struct Node
    {
        Box box;
        std::size_t begin = 0, end = 0, left = 0, right = 0;
    };
    const std::vector<Box>& boxes_;
    fp::Work& work_;
    GeneralVoFTransportDiagnostics& diagnostics_;
    std::vector<std::size_t> order_;
    std::vector<Node> nodes_;
    std::size_t build(std::size_t begin, std::size_t end)
    {
        const auto here = nodes_.size();
        nodes_.push_back({});
        Box box;
        for (auto i = begin; i < end; ++i)
        {
            work_.test();
            box.add(boxes_[order_[i]]);
        }
        nodes_[here] = {box, begin, end, 0, 0};
        if (end - begin > 4)
        {
            unsigned axis = 0;
            for (unsigned a = 1; a < 3; ++a)
                if (box.high[a] - box.low[a] > box.high[axis] - box.low[axis])
                    axis = a;
            const auto middle = begin + (end - begin) / 2;
            std::nth_element(order_.begin() + begin, order_.begin() + middle, order_.begin() + end,
                [&](auto a, auto b)
                {
                    work_.test();
                    const auto x = boxes_[a].low[axis] / 2 + boxes_[a].high[axis] / 2;
                    const auto y = boxes_[b].low[axis] / 2 + boxes_[b].high[axis] / 2;
                    return x != y ? x < y : a < b;
                });
            const auto left = build(begin, middle), right = build(middle, end);
            nodes_[here].left = left;
            nodes_[here].right = right;
        }
        return here;
    }
    void visit(std::size_t i, const Box& box, std::vector<std::size_t>& result)
    {
        work_.test();
        ++diagnostics_.index_node_visits;
        const auto& node = nodes_[i];
        if (!node.box.overlaps(box))
            return;
        if (node.left)
        {
            visit(node.left, box, result);
            visit(node.right, box, result);
        }
        else
            for (auto j = node.begin; j < node.end; ++j)
            {
                work_.test();
                if (boxes_[order_[j]].overlaps(box))
                    work_.push(result, order_[j]);
            }
    }

public:
    Index(const std::vector<Box>& boxes, fp::Work& work, GeneralVoFTransportDiagnostics& diagnostics)
        : boxes_(boxes), work_(work), diagnostics_(diagnostics)
    {
        if (boxes.size() > std::numeric_limits<std::size_t>::max() / (2 * sizeof(Node) + sizeof(std::size_t)))
            throw std::invalid_argument("Local VoF index capacity overflows.");
        work_.charge(boxes.size() * (2 * sizeof(Node) + sizeof(std::size_t)));
        order_.resize(boxes.size());
        std::iota(order_.begin(), order_.end(), 0);
        nodes_.reserve(2 * boxes.size());
        if (!boxes.empty())
            build(0, boxes.size());
    }
    std::vector<std::size_t> query(const Box& box)
    {
        ++diagnostics_.index_queries;
        std::vector<std::size_t> result;
        if (!nodes_.empty())
            visit(0, box, result);
        std::sort(result.begin(), result.end(),
            [&](auto a, auto b)
            {
                work_.test();
                return a < b;
            });
        return result;
    }
};
// Reuse the public Ordinary predicate: it bounds represented-coordinate
// evaluation, handles underflow, and scopes/restores the caller's FP environment.
// Plain points intentionally carry no borrowed incidence certificates.
Point represented(const Wide& p)
{
    return Point(p);
}
struct Frontier
{
    const AMR::AdaptiveTile* tile;
    bool first;
};
void prove_retained(const Frontier& frontier, const fp::Sweep<mode>& recipe, fp::Work& work)
{
    const auto& points = frontier.tile->geometry->points();
    if (points.size() < 3)
        throw std::invalid_argument("Local VoF malformed frontier polygon.");
    std::size_t third = 2;
    // A probe differing in one coordinate resolves a projected cross component.
    // Its displacement is at least one, avoiding an artificial subnormal probe.
    const auto a = represented(points[0]);
    for (; third < points.size(); ++third)
    {
        const auto b = represented(points[third - 1]), c = represented(points[third]);
        bool nonzero = false;
        for (unsigned axis = 0; axis < 3 && !nonzero; ++axis)
        {
            work.test();
            auto probe = a.value;
            probe[axis] = std::abs(probe[axis]) >= 1 ? 0 : probe[axis] >= 0 ? -1 : 1;
            nonzero = PLIC::orientation<mode>({a, b, c, Point(probe)}).value != 0;
        }
        if (nonzero)
            break;
    }
    if (third == points.size())
        throw std::invalid_argument("Local VoF frontier support is unresolved.");
    const auto b = represented(points[third - 1]), c = represented(points[third]);
    const auto check = [&](const Point& p)
    {
        work.test();
        const auto distance = PLIC::orientation<mode>({a, b, c, Point(p.value)}).value;
        if ((frontier.first ? distance : -distance) > 0)
            throw std::invalid_argument("Local VoF corrected sweep may leave the authenticated retained shell.");
    };
    for (const auto& p : recipe.current)
        check(p);
    for (const auto& p : recipe.departure)
        check(p);
    check(recipe.center);
    check(recipe.backcenter);
}
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
    fp::Work work{data.options.sweep};
    work.arithmetic_mode = mode;
    work.volume_tolerance = data.options.reconstruction.volume_tolerance;
    work.limits.byte_budget = std::min(work.limits.byte_budget, data.options.byte_budget);
    struct Pending
    {
        const AMR::AdaptiveTile* tile;
        fp::Sweep<mode> recipe;
        std::vector<std::size_t> candidates;
        bool geometric = false;
    };
    std::vector<Pending> pending;
    std::map<std::uint64_t, std::size_t> local_authorities;
    std::vector<const Cell*> cells;
    std::vector<Box> cell_boxes, frontier_boxes;
    std::vector<Frontier> frontiers;
    std::set<Key> requested;
    wire::collective(*comm,
        [&]
        {
            current(data);
            if (data.options.arithmetic_mode != mode)
                throw std::invalid_argument("Local VoF geometry requires Ordinary arithmetic.");
            for (const auto& [key, cell] : data.cells)
            {
                work.test();
                if (!cell.finite_geometry || cell.finite_geometry->arithmetic_mode() != mode)
                    throw std::invalid_argument("Local VoF retained native geometry is missing or has another mode.");
                work.push(cells, &cell);
            }
            if (!data.topology->order.empty() && cells.empty())
                throw std::invalid_argument("Local VoF owned cells have no retained native geometry.");
            std::sort(cells.begin(), cells.end(),
                [&](auto a, auto b)
                {
                    work.test();
                    return a->key != b->key ? a->key < b->key : a->gid < b->gid;
                });
            for (const auto* cell : cells)
            {
                Box box;
                for (const auto& p : cell->finite_geometry->points())
                {
                    work.test();
                    box.add(p.value);
                }
                work.push(cell_boxes, box);
            }
            result.diagnostics.indexed_cells = cells.size();
            std::vector<const AMR::AdaptiveTile*> frontier_tiles;
            for (const auto& tile : data.topology->frontier_tiles)
            {
                work.test();
                const bool first = data.cells.contains(tile.first),
                           second = tile.second && data.cells.contains(*tile.second);
                if (first != second)
                    work.push(frontier_tiles, &tile);
            }
            std::sort(frontier_tiles.begin(), frontier_tiles.end(),
                [&](auto a, auto b)
                {
                    work.test();
                    return a->face_gid < b->face_gid;
                });
            for (const auto* tile : frontier_tiles)
            {
                // Periodic images require a separate authenticated spatial index.
                if (!tile->second_image_identity.empty())
                    throw std::invalid_argument("Local VoF retained frontier has an unsupported periodic image.");
                Box box;
                for (const auto& p : tile->geometry->points())
                {
                    work.test();
                    box.add(p);
                }
                work.push(frontier_boxes, box);
                work.push(frontiers, Frontier{tile, data.cells.contains(tile->first)});
            }
            Index donor_index(cell_boxes, work, result.diagnostics);
            Index frontier_index(frontier_boxes, work, result.diagnostics);
            for (const auto& tile : data.topology->tiles)
            {
                if (tile.exterior || !tile.second || tile.first_owner != comm->getRank())
                    continue;
                failure::Scope face_context("interface_local_reach");
                face_context.record.face = tile.face_gid;
                face_context.record.mark(failure::Field::Face);
                work.charge(2 * tile.node_gids.size() * sizeof(Point) + sizeof(Pending));
                auto recipe = fp::prepare_sweep<mode>(tile.face_gid, tile.node_gids, traces, flux.at(tile.face_gid),
                    work, face_budgets.at(tile.face_gid));
                Box bounds;
                for (const auto& p : recipe.current)
                    bounds.add(p.value);
                for (const auto& p : recipe.departure)
                    bounds.add(p.value);
                bounds.add(recipe.center.value);
                bounds.add(recipe.backcenter.value);
                for (const auto index : frontier_index.query(bounds))
                    prove_retained(frontiers[index], recipe, work);
                auto candidates = donor_index.query(bounds);
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
                work.charge(3 * tile.node_gids.size() * sizeof(std::array<Point, 3>));
                for (const auto& triangle : fp::sweep_boundary(
                         tile.node_gids, recipe.current, recipe.departure, recipe.center, recipe.backcenter))
                {
                    work.test();
                    const auto v = fp::cone_determinant(recipe.center, triangle);
                    zero_effect &= v.value == 0 && v.absolute_error == 0;
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
                work.push(pending, Pending{&tile, std::move(recipe), std::move(candidates), geometric});
            }
            result.diagnostics.plane_requests = requested.size();
        });
    const auto authority_counts = aggregate_authority_counts(data, local_authorities, context);
    const auto planes = import_requested_planes(data, requested, context);
    wire::collective(*comm,
        [&]
        {
            work.charge(authority_counts.size() * (sizeof(decltype(authority_counts)::value_type) + 4 * sizeof(void*)));
            std::map<Key, PLIC::ConvexGeometryPtr<mode>> geometries;
            std::map<Key, PhaseLabel<mode>> labels;
            work.charge(
                planes.size() * (sizeof(std::map<Key, PLIC::PolyhedralInterface>::value_type) + 4 * sizeof(void*)));
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
            for (auto& entry : pending)
            {
                const auto& tile = *entry.tile;
                failure::Scope face_context("interface_local_partition");
                face_context.record.face = tile.face_gid;
                face_context.record.mark(failure::Field::Face);
                face_context.record.volume_budget = face_budgets.at(tile.face_gid);
                face_context.record.mark(failure::Field::VolumeBudget);
                face_context.record.bytes_used = work.bytes;
                face_context.record.mark(failure::Field::BytesUsed);
                LocalFaceGeometry face;
                face.face = tile.face_gid;
                face.geometric = entry.geometric;
                face.chain = {tile.face_gid, entry.recipe.control, entry.recipe.control, Math::make_value(mode, 0),
                    entry.recipe.ordinary_volume_budget};
                if (!entry.geometric)
                {
                    ++result.diagnostics.algebraic_faces;
                    work.push(result.owned_faces, std::move(face));
                    continue;
                }
                ++result.diagnostics.geometric_faces;
                // Source identities follow the face recipe, including its private
                // corrected center. Equal rounded coordinates never merge identities.
                const auto& recipe = entry.recipe;
                const auto n = tile.node_gids.size();
                std::vector<PLIC::OrdinaryAggregateTetra> tetrahedra;
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
                    PLIC::OrdinaryAggregateTetra tetra;
                    tetra.vertex_ids = {0, ids[0], ids[1], ids[2]};
                    for (unsigned i = 0; i < 4; ++i)
                        tetra.vertices[i] = point(tetra.vertex_ids[i]);
                    const auto orientation = PLIC::orientation<mode>(tetra.vertices).value;
                    // A proved contact has zero measure. Every nonzero source
                    // cone remains in the immutable batch, however thin it is.
                    if (orientation == 0)
                        return;
                    tetra.sign = orientation > 0 ? 1 : -1;
                    work.push(tetrahedra, std::move(tetra));
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
                std::vector<PLIC::OrdinaryAggregateDonor> donors;
                std::map<std::size_t, std::array<long double, 2>> phase_quotas;
                const auto quota =
                    face_budgets.at(tile.face_gid) / (2 * std::max<std::size_t>(1, entry.candidates.size()));
                for (const auto i : entry.candidates)
                {
                    const auto& cell = *cells[i];
                    const auto& plane = planes.at(cell.key);
                    PLIC::OrdinaryAggregateDonor donor;
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
                    work.charge(sizeof(decltype(phase_quotas)::value_type) + 4 * sizeof(void*));
                    phase_quotas.emplace(i, std::array{phase_quota(label.gas_bulk, label.raw_source.gas),
                                                phase_quota(label.carrier_bulk, label.raw_source.carrier)});
                    // The candidate must resolve both positive phases and the
                    // original face allowance. Its donor-phase allowance is
                    // consumed only by the final shared correction below.
                    donor.gas_budget = donor.carrier_budget = quota;
                    work.push(donors, std::move(donor));
                }
                auto sweep_limits = work.limits;
                sweep_limits.byte_budget -= std::min(work.bytes, sweep_limits.byte_budget);
                sweep_limits.maximum_pair_tests -= std::min(work.pair_tests, sweep_limits.maximum_pair_tests);
                auto limits = geometry_limits(mode, sweep_limits.byte_budget);
                limits.maximum_pair_tests = sweep_limits.maximum_pair_tests;
                PLIC::OrdinaryAggregateContext progress;
                PLIC::OrdinaryAggregateResult aggregate;
                try
                {
                    const PLIC::OrdinaryAggregateSource source(tetrahedra, sweep_limits, limits, &progress);
                    aggregate = source.measure(donors, &progress);
                }
                catch (...)
                {
                    using Stage = PLIC::OrdinaryAggregateContext::Stage;
                    const auto stage = progress.stage == Stage::Source     ? "interface_aggregate_source"
                                       : progress.stage == Stage::Coverage ? "interface_aggregate_coverage"
                                       : progress.stage == Stage::Metric   ? "interface_aggregate_metric"
                                                                           : "interface_aggregate_donors";
                    failure::Scope diagnostic(stage);
                    diagnostic.record.bytes_used = work.bytes + progress.bytes;
                    diagnostic.record.pair_tests = work.pair_tests + progress.pairs;
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
                                diagnostic.record.simplex[3 * i + k] = tetrahedra[progress.source].vertices[i].value[k];
                        diagnostic.record.mark(failure::Field::Simplex);
                    }
                    if (progress.donor < cells.size())
                    {
                        diagnostic.record.donor = cells[progress.donor]->gid;
                        diagnostic.record.mark(failure::Field::Donor);
                    }
                    if (progress.cut != PLIC::OrdinaryAggregateContext::absent)
                    {
                        diagnostic.record.cut = progress.cut;
                        diagnostic.record.mark(failure::Field::Cut);
                    }
                    throw;
                }
                work.charge(aggregate.statistics.storage_bytes);
                work.pair_tests += aggregate.statistics.pair_tests;
                result.diagnostics.swept_pieces += aggregate.statistics.source_pieces;
                result.diagnostics.aggregate_groups += aggregate.groups.size();
                result.diagnostics.aggregate_witnesses += aggregate.statistics.witnesses;
                result.diagnostics.cancelled_boundaries += aggregate.statistics.cancelled_boundaries;
                result.diagnostics.aggregate_retries += aggregate.statistics.retries;
                face.chain.constructed = Math::make_value(mode, aggregate.signed_source_volume);
                face.chain.signed_difference = Math::subtract(face.chain.constructed, recipe.control);
                if (std::abs(face.chain.signed_difference.value) > face_budgets.at(tile.face_gid))
                    throw std::invalid_argument("Interface aggregate source chain exceeds original face allowance.");
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
                    const std::array errors{
                        label.gas_bulk ? group.numerical_error_estimate.bulk : group.numerical_error_estimate.gas,
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
                                throw std::invalid_argument("Aggregate final phase-quality ratio is unresolved.");
                            face.quality_theta = std::min(face.quality_theta, ratio);
                        }
                    }
                    const PhaseValues raw{Math::make_value(mode, group.nominal.bulk),
                        Math::make_value(mode, group.nominal.gas), Math::make_value(mode, group.nominal.carrier)};
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
                    work.push(
                        face.groups, LocalGeometricGroup{{tile.face_gid, cell.gid, cell.owner, group.sign, amount},
                                         share, allocation.value, cut_closure.value(), history.gas.value,
                                         history.carrier.value, group.gas_positive, group.carrier_positive});
                }
                work.push(result.owned_faces, std::move(face));
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
