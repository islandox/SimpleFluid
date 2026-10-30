/** @file SweptGeometryInternal.hh
 * @brief Private exact signed sweeps, trajectories and departure partitions.
 */
#pragma once

#include "geometry/PLIC/GeometryAllocationLedger.hh"
#include "geometry/PLIC/details/exact/ConvexCellGeometryInternal.hh"
#include "geometry/PLIC/details/exact/GeometryIdentity.hh"
#include "geometry/swept_volume/SweptGeometry.hh"

namespace SimpleFluid::PLIC
{
// Signed transport constructions share this compiled unit's exact arithmetic.
struct SIMPLEFLUID_LOCAL GeometryMeasure<ArithmeticMode::Exact>::Impl
{
    Q value;
};

namespace exact_geometry_detail SIMPLEFLUID_LOCAL
{
P sparse_support_normal(const Poly& poly, const Face& face);
// Axis bounds of admitted convex shells are attained at vertices. Equality
// means only a zero-volume contact; any positive penetration survives exactly.
// Fixed rational locals avoid retaining an additional per-fragment cache.
bool axis_separates(CertifiedPolyRef first, CertifiedPolyRef second);
// Both references carry immutable strict convex-shell admission. Every face is
// a nonzero convex planar polygon with outward winding. A nonzero fan triangle
// therefore supplies the same outward supporting halfspace as its full area
// vector, without summing all polygon cross products. No derived/raw Poly can
// enter this certificate without the existing private admission bridge.
bool support_plane_separates(const P& normal, const P& anchor, const Poly& other, ClipScratch* scratch = nullptr);
bool support_separates(CertifiedPolyRef boundary, CertifiedPolyRef other);
bool support_separates(
    CertifiedPolyRef boundary, CertifiedPolyRef other, std::span<const Plane> support_planes, ClipScratch& scratch);
std::size_t exact_integer_bits(I value);
void exact_measure_arithmetic_preflight(const Q& a, const Q& b, bool division, std::size_t budget);
void exact_storage_add(std::size_t& bytes, std::size_t count, std::size_t size = 1);
void exact_scalar_storage(std::size_t& bytes, const Q& q);
} // namespace exact_geometry_detail SIMPLEFLUID_LOCAL
} // namespace SimpleFluid::PLIC

namespace SimpleFluid::geometry::swept_volume
{
using namespace ::SimpleFluid::PLIC;
using namespace ::SimpleFluid::PLIC::exact_geometry_detail;
namespace exact_swept_detail SIMPLEFLUID_LOCAL
{
}
using namespace exact_swept_detail;

struct SIMPLEFLUID_LOCAL ConvexFragment<ArithmeticMode::Exact>::Impl
{
    explicit Impl(CertifiedPoly value) : poly(std::move(value)) {}
    CertifiedPoly poly;
};

struct SIMPLEFLUID_LOCAL SweptFace<ArithmeticMode::Exact>::Impl
{
    std::vector<SignedSweepSimplex<ArithmeticMode::Exact>> pieces;
    std::vector<std::array<PLIC::PointIdentity<ArithmeticMode::Exact>, 3>> edges;
    Q volume, original;
    P center;
    PlanarFacePtr<ArithmeticMode::Exact> native;
    std::vector<NodeTrajectory<ArithmeticMode::Exact>> traces;
};

struct SIMPLEFLUID_LOCAL DonorExport<ArithmeticMode::Exact>::Impl
{
    std::vector<ConvexFragmentPtr<ArithmeticMode::Exact>> exported, retained;
};

template<> class SIMPLEFLUID_LOCAL SweepAccess<ArithmeticMode::Exact>
{
public:
    static const Poly& poly(const ConvexCellGeometry<ArithmeticMode::Exact>& geometry) { return geometry.impl_->poly; }
    static const Poly& poly(const ConvexFragment<ArithmeticMode::Exact>& fragment)
    {
        return fragment.impl_->poly.get();
    }
    // Native cells are immutable after full admission (make/decode/product
    // construction or the strict public clip/intersection path). No raw Poly
    // or decoded packet can acquire a certified reference through this bridge.
    static CertifiedPolyRef certified(const ConvexCellGeometry<ArithmeticMode::Exact>& geometry);
    static CertifiedPolyRef certified(const ConvexFragment<ArithmeticMode::Exact>& fragment)
    {
        return fragment.impl_->poly.ref();
    }
    static CertifiedPoly copy(CertifiedPolyRef poly) { return CertifiedPoly(poly.get()); }
    static ConvexFragmentPtr<ArithmeticMode::Exact> fragment(Poly poly)
    {
        return fragment(CertifiedPoly::admit(std::move(poly)));
    }
    static ConvexFragmentPtr<ArithmeticMode::Exact> fragment(CertifiedPolyRef poly)
    {
        return fragment(CertifiedPoly(poly.get()));
    }
    static ConvexFragmentPtr<ArithmeticMode::Exact> fragment(CertifiedPoly poly);
    template<class Budget>
    static std::optional<CertifiedPoly> intersect(CertifiedPolyRef a, CertifiedPolyRef b, Budget& budget)
    {
        if (axis_separates(a, b) || support_separates(a, b) || support_separates(b, a))
            return {};
        ClipScratch scratch{budget.limits.byte_budget - budget.bytes};
        auto result = certified_intersection(a, b, scratch);
        budget.cache(scratch);
        return result;
    }
    template<class Budget>
    static std::optional<CertifiedPoly> clip(CertifiedPolyRef poly, const Plane& plane, Budget& budget)
    {
        ClipScratch scratch{budget.limits.byte_budget - budget.bytes};
        auto result = certified_clip(poly, plane, scratch);
        budget.cache(scratch);
        return result;
    }
    static const Polygon& polygon(const PlanarFace<ArithmeticMode::Exact>& face) { return face.impl_->polygon; }
    static GeometryMeasure<ArithmeticMode::Exact> measure(Q value);
    static const Q& value(const GeometryMeasure<ArithmeticMode::Exact>& value)
    {
        static const Q zero;
        return value.impl_ ? value.impl_->value : zero;
    }
    static ConvexGeometryPtr<ArithmeticMode::Exact> geometry(Poly poly);
    static PlanarFacePtr<ArithmeticMode::Exact> retain(const PlanarFace<ArithmeticMode::Exact>& face);
    static ConvexGeometryPtr<ArithmeticMode::Exact> retain(const ConvexCellGeometry<ArithmeticMode::Exact>& cell);
};

namespace exact_swept_detail SIMPLEFLUID_LOCAL
{
PLIC::PointIdentity<ArithmeticMode::Exact> swept_point(const P& point);

struct SweepBudget
{
    SweepLimits limits;
    std::size_t bytes = 0, tests = 0;
    std::size_t cache_peak = 0, cache_simultaneous_peak = 0, cache_fallbacks = 0;
    GeometryAllocationLedger allocations;
    explicit SweepBudget(SweepLimits l);
    template<class Identity> void retain_identity(const Identity& identity)
    {
        allocations.retain(identity, bytes, limits.byte_budget);
    }
    void add(std::size_t n);
    void cache(const ClipScratch& scratch);
    void array(std::size_t n, std::size_t size);
    void test();
    void atom(std::size_t n);
    void poly(const Poly& p);
};

template<class T>
void reserve_support_planes(std::vector<T>& values, std::size_t count, ClipScratch& scratch, const char* description)
{
    if (count > scratch.limit / sizeof(T))
        throw ClipCacheExhausted(std::string("Optional ") + description + " exceeds its live capacity lease.");
    scratch.add(count * sizeof(T));
    values.reserve(count);
    if (values.capacity() < count)
        throw std::logic_error("Optional support-plane vector allocated below its requested capacity.");
    if (values.capacity() > count)
        scratch.add((values.capacity() - count) * sizeof(T));
}

struct SupportPlaneRange
{
    std::size_t begin, end;
};
/** Certification-local derived planes. The source references remain owned by
 * immutable admitted pieces; this cache never enters a fragment, identity or
 * retained-storage certificate. */
class SupportPlaneCache
{
    ClipCacheLease lease_;
    ClipScratch& scratch_;
    std::vector<Plane> planes_;
    std::vector<SupportPlaneRange> ranges_;

public:
    SupportPlaneCache(std::span<const ConvexFragmentPtr<ArithmeticMode::Exact>> pieces, ClipScratch& scratch);
    SupportPlaneCache(const SupportPlaneCache&) = delete;
    SupportPlaneCache& operator=(const SupportPlaneCache&) = delete;
    std::span<const Plane> for_piece(std::size_t piece) const;
};
std::string cover_scalar(const Q& q, std::size_t maximum_bits = 8192);
std::string cover_point(const P& p);
std::string sweep_cover_context(const SweptFace<ArithmeticMode::Exact>& face,
    const SignedSweepSimplex<ArithmeticMode::Exact>& piece, std::span<const SweepDonor<ArithmeticMode::Exact>> donors,
    std::size_t simplex);
bool cached_pair_intersects(CertifiedPolyRef first, std::size_t first_index, CertifiedPolyRef second,
    std::size_t second_index, const SupportPlaneCache& cache, ClipScratch& scratch);
// Only immutable admitted convex shells can enter this predicate. For each
// outward convex planar face, its first nonzero fan triangle is a positive
// multiple of the full area normal. It therefore defines exactly the same
// closed halfspace, including equality and arbitrarily thin outside slivers.
// Construct one supporting normal per face, then test every piece vertex;
// avoid rebuilding and summing the donor's polygon normals for each piece.
bool certified_cover_contained(
    std::span<const ConvexFragmentPtr<ArithmeticMode::Exact>> pieces, CertifiedPolyRef outer);
void certify_sweep_partition(CertifiedPolyRef expected_certificate,
    std::span<const ConvexFragmentPtr<ArithmeticMode::Exact>> pieces, SweepBudget& budget);

template<class Geometry> Q sweep_volume(const Geometry& cell)
{
    const auto& p = SweepAccess<ArithmeticMode::Exact>::poly(cell);
    return exact_volume(p, p.vertices[0]);
}
std::optional<SignedSweepSimplex<ArithmeticMode::Exact>> swept_tetra(
    const P& origin, const P& a, const P& b, const P& c);

using SweptTriangle = std::array<P, 3>;
std::vector<SweptTriangle> swept_boundary(const Polygon& current, const Polygon& previous, const P& center,
    const P& backcenter, std::vector<std::array<PLIC::PointIdentity<ArithmeticMode::Exact>, 3>>* sides);
Q signed_boundary_volume(const P& origin, const std::vector<SweptTriangle>& triangles);
void sweep_donors(std::span<const SweepDonor<ArithmeticMode::Exact>> donors, SweepBudget& budget);

struct SweepAtom
{
    ConvexFragmentPtr<ArithmeticMode::Exact> geometry;
    std::vector<int> coefficient;
};
std::vector<SweepAtom> overlay_sweep(std::vector<SweepAtom> atoms,
    const SignedSweepSimplex<ArithmeticMode::Exact>& piece, std::size_t column, int sign, SweepBudget& budget);
PhaseMeasures<ArithmeticMode::Exact> sweep_phases(
    std::span<const ConvexFragmentPtr<ArithmeticMode::Exact>> atoms, const GeometryHalfspace& halfspace);

struct TrajectoryArithmetic
{
    TrajectoryLimits limits;
    std::size_t operations = 0, bytes = 0;
    explicit TrajectoryArithmetic(TrajectoryLimits value);
    void charge(std::size_t count);
    std::array<std::size_t, 2> bits(const Q& q) const;
    Q retain(Q q);
    Q binary(const Q& a, const Q& b, unsigned operation);
    P subtract(const P& a, const P& b) { return {binary(a[0], b[0], 1), binary(a[1], b[1], 1), binary(a[2], b[2], 1)}; }
    P scale(const P& a, const Q& b) { return {binary(a[0], b, 2), binary(a[1], b, 2), binary(a[2], b, 2)}; }
    Q inner(const P& a, const P& b);
    P cross_product(const P& a, const P& b);
};
bool trajectory_zero(const P& p);
P bounded_polygon_vector(const Polygon& polygon, TrajectoryArithmetic& arithmetic);
void target_sweep_shell(
    const SweepRecipient<ArithmeticMode::Exact>& target, SweepBudget& budget, std::map<P, P>& shared);

// Private certified positive departure decomposition.
// Its caller MUST first finish sweep_donors,
// target_sweep_shell and every existing signed-simplex donor-coverage proof.
// No public header/template exposure. All missing DERIVED certificates return
// nullopt; original-chain corruption and resource failures throw.
using DepartureTriangle = std::array<P, 3>;

using DepartureTetraKey = std::array<P, 4>;

using DepartureTriangleChain = std::map<DepartureTriangle, std::int64_t>;

using DepartureEdgeChain = std::map<std::array<P, 2>, std::int64_t>;
void departure_charge_q(const Q& q, SweepBudget& budget);
void departure_charge_point(const P& p, SweepBudget& budget);

template<class Key> void departure_charge_key(const Key& key, SweepBudget& budget)
{
    budget.add(sizeof(Key) + sizeof(std::int64_t) + 4 * sizeof(void*));
    for (const auto& p : key)
        departure_charge_point(p, budget);
}

template<class Key>
void departure_coefficient(
    std::map<Key, std::int64_t>& chain, const Key& key, std::int64_t coefficient, SweepBudget& budget)
{
    if (!coefficient)
        return;
    auto it = chain.find(key);
    if (it == chain.end())
    {
        departure_charge_key(key, budget);
        chain.emplace(key, coefficient);
        return;
    }
    const auto old = it->second;
    if ((coefficient > 0 && old > std::numeric_limits<std::int64_t>::max() - coefficient) ||
        (coefficient < 0 && old < std::numeric_limits<std::int64_t>::min() - coefficient))
        throw std::invalid_argument("Departure chain multiplicity overflow.");
    it->second += coefficient;
    if (!it->second)
        chain.erase(it);
}
void departure_edge(DepartureEdgeChain& chain, const P& a, const P& b, int sign, SweepBudget& budget);
void departure_triangle(
    DepartureTriangleChain& chain, const P& a, const P& b, const P& c, int sign, SweepBudget& budget);
void departure_triangle(DepartureTriangleChain& chain, const DepartureTriangle& t, int sign, SweepBudget& budget);
void departure_boundary_edges(DepartureEdgeChain& edges, const DepartureTriangle& t, SweepBudget& budget);

template<class T> void departure_push(std::vector<T>& values, T value, SweepBudget& budget)
{
    if (values.size() == values.capacity())
    {
        const auto capacity = values.capacity();
        if (capacity > std::numeric_limits<std::size_t>::max() / 2)
            throw std::invalid_argument("Departure vector capacity overflow.");
        const auto next = capacity ? 2 * capacity : 1;
        // Cumulative construction accounting includes both old and newly
        // requested capacities; a failed optional attempt never refunds work.
        budget.array(next, sizeof(T));
        values.reserve(next);
    }
    values.push_back(std::move(value));
}
// Authenticate C(B) against the original immutable signed-sweep representation.
// For the explicitly CLOSED boundary B, the polyhedral-current homotopy identity
// is d C(B)=B-C(dB)=B. A det==0 cone is a zero THREE-current. Its own formal
// triangle boundary need not be an empty identical-triangle multiset; we do not
// make that false assertion or rely on volume/moments to prove chain equality.
void departure_authenticate_cones(const SweptFace<ArithmeticMode::Exact>& face, const Polygon& current, const P& center,
    const std::vector<DepartureTriangle>& boundary, SweepBudget& budget);
void departure_admit_boundary_storage(
    const Polygon& current, const Polygon& previous, const P& center, const P& backcenter, SweepBudget& budget);
// Unconditional entry proof, BEFORE any optional count/topology/kernel exit.
void departure_authenticate_recipient_sweeps(const SweepRecipient<ArithmeticMode::Exact>& target, SweepBudget& budget);

struct DepartureFanCandidate
{
    std::size_t recipient = 0;
    std::vector<ConvexFragmentPtr<ArithmeticMode::Exact>> tetrahedra;
};
// No interpolation of absent coarse-edge nodes. Exact shared-side cancellation
// is a prerequisite; a valid hanging-edge configuration lacking that OPTIONAL
// derived topology simply falls back to the signed overlay.
std::optional<DepartureFanCandidate> departure_fan_candidate(
    const SweepRecipient<ArithmeticMode::Exact>& target, std::size_t recipient, SweepBudget& budget);
std::optional<std::vector<TransportAtom<ArithmeticMode::Exact>>> optional_departure_fan_partition(
    std::span<const SweepDonor<ArithmeticMode::Exact>> donors,
    std::span<const SweepRecipient<ArithmeticMode::Exact>> recipients, SweepBudget& budget);
} // namespace exact_swept_detail SIMPLEFLUID_LOCAL
} // namespace SimpleFluid::geometry::swept_volume
