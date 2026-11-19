#include "geometry/details/NativeGeometryScratchAllocator.hh"
#include "geometry/mesh/AMR/AdaptiveBoundaryProof.hh"
#include "geometry/PLIC/GeometryAllocationLedger.hh"
#include "geometry/PLIC/details/GeometryAllocationOwnerAccess.hh"
#include "geometry/PLIC/details/NativePlanarOwnerAccess.hh"
#include "geometry/PLIC/details/GeometryIdentityAccess.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveWire.hh"
#include <algorithm>
#include <limits>
#include <set>
#include <stdexcept>
#include <vector>
namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
namespace
{
struct ConstructionLease
{
    Budget& budget;
    std::size_t bytes;
    ConstructionLease(Budget& value, std::size_t count) : budget(value), bytes(0) { add(count); }
    void add(std::size_t count)
    {
        budget.add(count);
        bytes += count;
    }
    ~ConstructionLease() { budget.used -= bytes; }
};
template<class T, class Allocator> void reserve(std::vector<T, Allocator>& values, std::size_t count, Budget& budget)
{
    if (count > values.max_size() || count > budget.limit / sizeof(T))
        throw std::invalid_argument("Physical boundary proof capacity exceeds its budget.");
    budget.add(count * sizeof(T));
    values.reserve(count);
    if (values.capacity() != count)
        throw std::invalid_argument("Physical boundary proof capacity differs from admission.");
}
} // namespace
static void certify_boundary_impl(const PLIC::ConvexGeometryPtr<ArithmeticMode::Ordinary>& source,
    std::span<const SupportFace> certified_source_faces, std::span<const SupportFace> pieces, std::size_t byte_budget,
    AdaptiveBoundaryProofStatistics* statistics, std::span<const SupportFace> whole_faces,
    const geometry_detail::NativeConstructionAdmission* admission = nullptr)
{
    const auto state = admission ? admission->state() : std::shared_ptr<const geometry_detail::NativeAllocationState>{};
    const PLIC::GeometryLimits limits;
    const auto source_count = source ? source->num_faces() : certified_source_faces.size();
    if ((!source && certified_source_faces.empty()) || !byte_budget || pieces.size() < 4 ||
        pieces.size() > limits.maximum_faces || source_count > limits.maximum_faces ||
        (!whole_faces.empty() && whole_faces.size() != pieces.size()) ||
        std::any_of(pieces.begin(), pieces.end(), [](const auto& face) { return !face; }))
        throw std::invalid_argument("Incomplete or excessive physical boundary proof.");
    Budget budget{byte_budget};
    std::size_t work = 0;
    const auto use_work = [&](std::size_t count)
    {
        if (count > limits.maximum_pair_tests - work)
            throw std::invalid_argument("Physical boundary proof pair-work cap exhausted.");
        work += count;
    };
    const auto product_work = [&](std::size_t first, std::size_t second)
    {
        if (first && second > (limits.maximum_pair_tests - work) / first)
            throw std::invalid_argument("Physical boundary proof pair-work cap exhausted.");
        use_work(first * second);
    };
    using WidePoint = PLIC::PlanarFace<ArithmeticMode::Ordinary>::WidePoint;
    for (const auto& piece : pieces)
    {
        use_work(piece->points().size());
    }
    struct Observation
    {
        Budget& budget;
        AdaptiveBoundaryProofStatistics* value;
        const std::size_t& work;
        ~Observation()
        {
            if (value)
            {
                value->peak_bytes = budget.peak;
                value->pair_tests = work;
            }
        }
    } observation{budget, statistics, work};
    if (statistics)
        *statistics = {source_count, pieces.size(), 0};
    PLIC::GeometryAllocationLedger owners;
    std::size_t retained = 0;
    if (admission)
        PLIC::details::GeometryAllocationOwnerAccess::begin_native_bookkeeping(
            owners, *admission, retained, byte_budget);
    const auto retain = [&](const auto& value)
    {
        const auto available = budget.limit - budget.used;
        const auto maximum = std::numeric_limits<std::size_t>::max();
        const auto limit = admission ? byte_budget : available > maximum - retained ? maximum : retained + available;
        const auto before = retained;
        try
        {
            if constexpr (requires { owners.retain_planar_face(value, retained, limit); })
                owners.retain_planar_face(value, retained, limit);
            else
                owners.retain(value, retained, limit);
        }
        catch (...)
        {
            budget.add(retained - before);
            throw;
        }
        budget.add(retained - before);
    };
    if (source)
        retain(source);

    for (const auto& piece : pieces)
        retain(piece);
    for (const auto& whole : whole_faces)
        if (whole)
            retain(whole);
    ConstructionLease vertex_memory(budget, 0);
    std::set<WidePoint, std::less<WidePoint>, geometry_detail::NativeGeometryScratchAllocator<WidePoint>> all_vertices(
        std::less<WidePoint>{}, geometry_detail::NativeGeometryScratchAllocator<WidePoint>(state));
    for (const auto& piece : pieces)
        for (const auto& point : piece->points())
            if (!all_vertices.contains(point))
            {
                if (all_vertices.size() == limits.maximum_vertices)
                    throw std::invalid_argument("Physical boundary proof vertex cap exhausted.");
                vertex_memory.add(sizeof(WidePoint) + 4 * sizeof(void*));
                all_vertices.insert(point);
            }
    geometry_detail::NativeGeometryScratchVector<SupportFace> source_faces{
        geometry_detail::NativeGeometryScratchAllocator<SupportFace>(state)};
    reserve(source_faces, source_count, budget);
    using Row = geometry_detail::NativeGeometryScratchVector<SupportFace>;
    geometry_detail::NativeGeometryScratchVector<Row> coverage{
        geometry_detail::NativeGeometryScratchAllocator<Row>(state)};
    reserve(coverage, source_count, budget);
    for (std::size_t i = 0; i < source_count; ++i)
        coverage.emplace_back(geometry_detail::NativeGeometryScratchAllocator<SupportFace>(state));
    for (std::size_t face = 0; face < source_count; ++face)
    {
        if (source)
        {
            ConstructionLease construction(budget,
                PLIC::PlanarFace<ArithmeticMode::Ordinary>::construction_capacity_bound(source->faces()[face].size()));
            auto original = admission ? PLIC::details::NativePlanarOwnerAccess::cell_face(*source, face, *admission)
                                      : source->face_geometry(face);
            retain(original);
            source_faces.push_back(std::move(original));
        }
        else
        {
            if (!certified_source_faces[face])
                throw std::invalid_argument("Null authenticated native source face.");
            retain(certified_source_faces[face]);
            source_faces.push_back(certified_source_faces[face]);
        }
    }
    for (std::size_t piece_index = 0; piece_index < pieces.size(); ++piece_index)
    {
        const auto& piece = pieces[piece_index];
        auto matched = source_faces.size();
        for (std::size_t face = 0; face < source_faces.size(); ++face)
        {
            ConstructionLease construction(budget, support_construction_capacity_bound(SupportConstructionKind::Pair,
                                                       source_faces[face], std::span<const SupportFace>(&piece, 1)));
            // Equal support identities do not independently prove physical
            // coplanarity of received coordinates. Check each actual vertex.
            product_work(2 * (source_faces[face]->points().size() + 1), piece->points().size());
            if (!std::all_of(piece->points().begin(), piece->points().end(),
                    [&](const auto& point)
                    {
                        return source_faces[face]->contains_point(PLIC::details::GeometryIdentityAccess::point(point));
                    }))
                continue;
            if (matched != source_faces.size() || source_faces[face]->oppositely_oriented(*piece))
                throw std::invalid_argument("Ambiguous or inward physical boundary piece.");
            matched = face;
        }
        if (matched == source_faces.size())
            throw std::invalid_argument("Physical boundary piece leaves its original source faces.");
        if (!whole_faces.empty() && whole_faces[piece_index])
        {
            const auto& whole = whole_faces[piece_index];
            const auto& original = source_faces[matched];
            ConstructionLease construction(budget, support_construction_capacity_bound(SupportConstructionKind::Pair,
                                                       original, std::span<const SupportFace>(&whole, 1)));
            const auto contains_vertices = [&](const SupportFace& target, const SupportFace& candidate)
            {
                product_work(2 * (target->points().size() + 1), candidate->points().size());
                return std::all_of(candidate->points().begin(), candidate->points().end(), [&](const auto& point)
                    { return target->contains_point(PLIC::details::GeometryIdentityAccess::point(point)); });
            };
            if (!contains_vertices(original, whole) || !contains_vertices(whole, original))
                throw std::invalid_argument("Native whole-face certificate differs from the original source face.");
        }
        budget.append_capacity(coverage[matched]);
        coverage[matched].push_back(piece);
    }
    for (std::size_t face = 0; face < source_faces.size(); ++face)
    {
        const auto& row = coverage[face];
        const auto source_points = source_faces[face]->points().size();
        std::size_t edge_count = source_points;
        ConstructionLease row_memory(budget, 0);
        std::set<WidePoint, std::less<WidePoint>, geometry_detail::NativeGeometryScratchAllocator<WidePoint>> vertices(
            std::less<WidePoint>{}, geometry_detail::NativeGeometryScratchAllocator<WidePoint>(state));
        const auto insert_points = [&](const auto& points)
        {
            for (const auto& point : points)
                if (!vertices.contains(point))
                {
                    row_memory.add(sizeof(WidePoint) + 4 * sizeof(void*));
                    vertices.insert(point);
                }
        };
        insert_points(source_faces[face]->points());
        for (const auto& piece : row)
        {
            product_work(source_points, piece->points().size()); // Repeated containment inside partition proof.
            edge_count += piece->points().size();
            insert_points(piece->points());
        }
        // These are the existing partition kernel's pair/edge-point visits;
        // one cell-wide allowance prevents per-face resets from raising a cap.
        const auto pairs = row.empty() ? 0 : row.size() * (row.size() - 1) / 2;
        use_work(pairs);
        product_work(edge_count, vertices.size());
        ConstructionLease construction(
            budget, support_construction_capacity_bound(SupportConstructionKind::Partition, source_faces[face], row));
        if (admission)
            PLIC::details::NativePlanarOwnerAccess::certify_partition(*source_faces[face], row, *admission);
        else
            source_faces[face]->certify_partition(row);
    }
}
void certify_adaptive_physical_boundary(const PLIC::ConvexGeometryPtr<ArithmeticMode::Ordinary>& source,
    std::span<const SupportFace> pieces, std::size_t byte_budget, AdaptiveBoundaryProofStatistics* statistics,
    std::span<const SupportFace> whole_faces)
{
    certify_boundary_impl(source, {}, pieces, byte_budget, statistics, whole_faces);
}
void certify_adaptive_physical_boundary_faces(std::span<const SupportFace> source, std::span<const SupportFace> pieces,
    std::size_t byte_budget, AdaptiveBoundaryProofStatistics* statistics)
{
    certify_boundary_impl({}, source, pieces, byte_budget, statistics, {});
}
void certify_adaptive_physical_boundary_native(const PLIC::ConvexGeometryPtr<ArithmeticMode::Ordinary>& source,
    std::span<const SupportFace> pieces, std::size_t byte_budget,
    const geometry_detail::NativeConstructionAdmission& admission, std::span<const SupportFace> whole_faces)
{
    certify_boundary_impl(source, {}, pieces, byte_budget, nullptr, whole_faces, &admission);
}
void certify_adaptive_physical_boundary_faces_native(std::span<const SupportFace> source,
    std::span<const SupportFace> pieces, std::size_t byte_budget,
    const geometry_detail::NativeConstructionAdmission& admission)
{
    certify_boundary_impl({}, source, pieces, byte_budget, nullptr, {}, &admission);
}
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
