/** @file LocalGeometry.cc
 * @brief Indexed conservative interface reach and owner-authenticated local cuts.
 */
#include "FVM/details/GeneralVoF/LocalGeometry.hh"
#include "FVM/details/GeneralVoF/PhaseAllocation.hh"
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
                    }
                work.push(pending, Pending{&tile, std::move(recipe), std::move(candidates), geometric});
            }
            result.diagnostics.plane_requests = requested.size();
        });
    const auto planes = import_requested_planes(data, requested, context);
    wire::collective(*comm,
        [&]
        {
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
                if ((cell.phase.gas_volume == 0 || cell.phase.carrier_volume == 0) &&
                    !plane.pure_gas && !plane.pure_carrier)
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
                auto sweep = fp::materialize_sweep<mode>(std::move(entry.recipe), tile.node_gids, work);
                result.diagnostics.swept_pieces += sweep.pieces.size();
                face.chain.constructed = sweep.constructed_volume;
                face.chain.signed_difference = sweep.signed_control_difference;
                std::vector<PLIC::ConvexGeometryPtr<mode>> donors;
                for (const auto i : entry.candidates)
                    work.push(donors, geometries.at(cells[i]->key));
                auto limits = geometry_limits(mode, work.limits.byte_budget);
                limits.maximum_pair_tests =
                    work.limits.maximum_pair_tests - std::min(work.pair_tests, work.limits.maximum_pair_tests);
                const PLIC::ConvexPartition<mode> partition(donors, limits);
                work.charge(partition.storage_bytes());
                struct Group
                {
                    PhaseSum<mode> amount;
                    SimpleFluid::detail::CompensatedSum<> share, allocation, closure, gas_history, carrier_history;
                    bool gas = false, carrier = false;
                };
                std::map<std::pair<std::size_t, int>, Group> groups;
                for (std::size_t ordinal = 0; ordinal < sweep.pieces.size(); ++ordinal)
                {
                    const auto& piece = sweep.pieces[ordinal];
                    failure::Scope piece_context("interface_local_donor_partition");
                    piece_context.record.construction = ordinal;
                    piece_context.record.mark(failure::Field::Construction);
                    work.test();
                    const auto& simplex = piece.geometry->dynamic_geometry()->points();
                    for (std::size_t vertex = 0; vertex < std::min<std::size_t>(simplex.size(), 4); ++vertex)
                        for (unsigned axis = 0; axis < 3; ++axis)
                            piece_context.record.simplex[3 * vertex + axis] = simplex[vertex].value[axis];
                    piece_context.record.mark(failure::Field::Simplex);
                    for (const auto& intersection : partition.partition(*piece.geometry, work.limits.maximum_atoms))
                    {
                        work.test();
                        const auto candidate = entry.candidates.at(intersection.donor);
                        const auto& cell = *cells[candidate];
                        failure::Scope donor_context("interface_local_phase_measure");
                        donor_context.record.donor = cell.gid;
                        donor_context.record.mark(failure::Field::Donor);
                        work.atom(intersection.geometry);
                        const auto& label = labels.at(cell.key);
                        std::array<bool, 2> presence{};
                        PLIC::DynamicGeometryValue allocation;
                        const auto amount =
                            label.apply(*intersection.geometry, planes.at(cell.key), &presence, &allocation);
                        const auto history = label.history(*intersection.geometry, planes.at(cell.key));
                        ++result.diagnostics.plane_measurements;
                        const auto source = phase_inventory<mode>(cell, *geometries.at(cell.key));
                        const auto share = Math::divide(amount.bulk, source.bulk).value;
                        SimpleFluid::detail::CompensatedSum<> source_defect, cut_closure;
                        source_defect += source.gas.value;
                        source_defect += source.carrier.value;
                        source_defect += -source.bulk.value;
                        cut_closure += amount.gas.value;
                        cut_closure += amount.carrier.value;
                        cut_closure += -amount.bulk.value;
                        cut_closure += -source_defect.value() * share;
                        cut_closure += -allocation.value;
                        const auto key = std::pair{candidate, piece.sign};
                        if (!groups.contains(key))
                            work.charge(sizeof(typename decltype(groups)::value_type) + 4 * sizeof(void*));
                        auto& group = groups[key];
                        group.amount.add(amount);
                        group.share += share;
                        group.allocation += allocation.value;
                        group.closure += cut_closure.value();
                        group.gas_history += history.gas.value;
                        group.carrier_history += history.carrier.value;
                        group.gas |= presence[0];
                        group.carrier |= presence[1];
                    }
                }
                for (const auto& [key, group] : groups)
                {
                    const auto& cell = *cells[key.first];
                    work.push(face.groups,
                        LocalGeometricGroup{{tile.face_gid, cell.gid, cell.owner, key.second, group.amount.value()},
                            group.share.value(), group.allocation.value(), group.closure.value(),
                            group.gas_history.value(), group.carrier_history.value(), group.gas, group.carrier});
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
