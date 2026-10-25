/**
 * @file ConvexCellGeometryInternal.hh
 * @brief Exact represented-input convex clipping and moment certificates.
 */

#pragma once

#include "SimpleFluidExport.hh"
#include "utils/ExactRational.hh"
#include "geometry/PLIC/ConvexCellGeometry.hh"
#include "geometry/PLIC/ConvexGeometry.hh"
#include "geometry/PLIC/ExtrudedConstruction.hh"
#include "geometry/PLIC/GeometryInstrumentation.hh"
#include "geometry/PLIC/PlanarGeometry.hh"
#include "geometry/PLIC/PolyhedralComplex.hh"
#include "geometry/PLIC/SweptGeometry.hh"
#include "geometry/PLIC/details/exact/CertifiedSupportFilter.hh"
#include "geometry/PLIC/details/exact/SupportSign.hh"
#include "geometry/PLIC/details/ordinary/NativeGeometrySupport.hh"
#include "geometry/mesh/MultiRegionMesh.hh"
#include <algorithm>
#include <bit>
#include <boost/multiprecision/cpp_int.hpp>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iterator>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>

namespace SimpleFluid::PLIC
{
using Meshes::RegionLayout;
using Meshes::SemiStructMeshTopo;
using Meshes::SemiStructuredIndexer;
namespace exact_geometry_detail
{
}
using namespace exact_geometry_detail;
template<> class SweepAccess<ArithmeticMode::Exact>;
} // namespace SimpleFluid::PLIC
namespace SimpleFluid::PLIC
{

namespace exact_geometry_detail SIMPLEFLUID_LOCAL
{
using Q = boost::multiprecision::cpp_rational;

using I = boost::multiprecision::cpp_int;

using P = std::array<Q, 3>;

using Face = std::vector<std::size_t>;
extern thread_local bool count_exact_geometry;
extern thread_local BackendStatistics<ArithmeticMode::Exact> exact_geometry_counts;
void sample_exact_measure_result(const Q& value);

struct Poly
{
    std::vector<P> vertices;
    std::vector<Face> faces;
};

struct Plane
{
    P normal, anchor;
    Q offset = 0;
};

struct Product
{
    std::vector<P> base;
    Q lower, upper;
};
using ::SimpleFluid::detail::exact_rational::exact;
using ::SimpleFluid::detail::exact_rational::exact_wide;
P point(GeometryPoint p);
P add(const P& a, const P& b);
P sub(const P& a, const P& b);
P mul(const P& a, const Q& b);
Q dot(const P& a, const P& b);
P cross(const P& a, const P& b);
bool zero(const P& p);
Q distance(const Plane& plane, const P& p);

template<class Storage = long double>
GeometryValue<ArithmeticMode::Exact> measured(const Q& value, bool allow_underflow_interval = false)
{
    const long double v = value.convert_to<Storage>();
    if (!std::isfinite(v) || (v == 0 && value != 0 && !allow_underflow_interval))
        throw std::runtime_error("Positive/nonzero exact geometry is not representable in admitted measure storage.");
    Q error = value - exact_wide(v);
    if (error < 0)
        error = -error;
    if (error == 0)
        return {v, 0, value != 0};
    // Outward rounding in Storage, including a subnormal enclosure when the
    // exact first moment lies below that format's minimum positive value.
    const long double bound = std::nextafter(error.convert_to<Storage>(), std::numeric_limits<Storage>::infinity());
    if (!std::isfinite(bound) || exact_wide(bound) < error)
        throw std::runtime_error("Geometry conversion error could not be certified.");
    return {v, bound, value != 0};
}
GeometryValue<ArithmeticMode::Exact> square_root(const Q& squared);
P area_vector(const Poly& poly, const Face& face);

struct Moments
{
    Q volume;
    P first;
};
Moments exact_moments(const Poly& poly, const P& origin);
Q exact_volume(const Poly& poly, const P& origin);
std::vector<Plane> planes(const Poly& poly);
// Native/public convex outputs retain numeric admission. Only opaque exact
// transport fragments defer conversion until their measures are aggregated.
void validate(const Poly& poly, bool measure_outputs = true);

struct Cut
{
    std::optional<Poly> poly;
    Face cap;
    std::vector<GeometryValue<ArithmeticMode::Exact>> apertures;
    std::vector<std::size_t> face_sources;
};

class ClipCacheExhausted : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};

struct ClipScratch
{
    std::size_t limit, used = 0, peak = 0, fallbacks = 0;
    void add(std::size_t bytes);
    void scalar(const Q& q);
};

struct ClipCacheLease
{
    ClipScratch& scratch;
    std::size_t previous;
    explicit ClipCacheLease(ClipScratch& value) : scratch(value), previous(value.used) {}
    ClipCacheLease(const ClipCacheLease&) = delete;
    ClipCacheLease& operator=(const ClipCacheLease&) = delete;
    ~ClipCacheLease() { scratch.used = previous; }
};

struct UncachedPlaneDistances
{
    struct Sample
    {
        Q value;
        int sign() const { return value < 0 ? -1 : value > 0 ? 1 : 0; }
        const Q& exact() const { return value; }
    };
    const Poly& poly;
    const Plane& plane;
    Q input(std::size_t i) const { return distance(plane, poly.vertices[i]); }
    Sample sample(std::size_t i) const { return {input(i)}; }
    int sign(std::size_t i) const
    {
        const Q value = input(i);
        return value < 0 ? -1 : value > 0 ? 1 : 0;
    }
    void append(int) {}
    bool output_zero(const Poly& output, std::size_t i) const { return distance(plane, output.vertices[i]) == 0; }
};

struct CachedPlaneDistances
{
    struct Sample
    {
        CachedPlaneDistances& distances;
        std::size_t index;
        int sign() const { return distances.sign(index); }
        const Q& exact() const { return distances.input(index); }
    };
    struct Entry
    {
        Q value;
        // Two means not yet classified. A filter never supplies exact zero.
        int sign = 2;
        bool exact = false;
    };
    // Declared first, destroyed last: release the admitted capacity only after
    // both cache vectors have died, including constructor exception unwinding.
    ClipCacheLease lease;
    ClipScratch& scratch;
    const Poly& poly;
    const Plane& plane;
    detail::CertifiedFilterPlane filter;
    std::vector<Entry> input_values;
    std::vector<int> output_signs;
    CachedPlaneDistances(const Poly& input, const Plane& halfspace, ClipScratch& budget);
    const Q& input(std::size_t i);
    int sign(std::size_t i);
    Sample sample(std::size_t i) { return {*this, i}; }
    void append(int sign);
    bool output_zero(const Poly&, std::size_t i) const { return output_signs[i] == 0; }
};

class CertifiedPoly;

class CertifiedPolyRef
{
public:
    const Poly& get() const noexcept { return *poly_; }

private:
    friend class CertifiedPoly;
    friend class ::SimpleFluid::PLIC::SweepAccess<ArithmeticMode::Exact>;
    explicit CertifiedPolyRef(const Poly& poly) : poly_(&poly) {}
    const Poly* poly_;
};

bool axis_separates(CertifiedPolyRef, CertifiedPolyRef);

bool support_separates(CertifiedPolyRef, CertifiedPolyRef);

std::optional<CertifiedPoly> certified_clip(CertifiedPolyRef, const Plane&, ClipScratch&);

std::optional<CertifiedPoly> certified_clip_uncached(CertifiedPolyRef, const Plane&);

std::optional<CertifiedPoly> certified_intersection(CertifiedPolyRef, CertifiedPolyRef, ClipScratch&);

std::optional<CertifiedPoly> certified_intersection(CertifiedPolyRef, std::span<const Plane>, ClipScratch&);

/** Only immutable fully admitted cells, full raw admission, and the exact
 * halfspace preservation proof below can mint this private certificate. */
class CertifiedPoly
{
public:
    static CertifiedPoly admit(Poly poly)
    {
        validate(poly, false);
        return CertifiedPoly(std::move(poly));
    }
    CertifiedPolyRef ref() const noexcept { return CertifiedPolyRef(poly_); }
    const Poly& get() const noexcept { return poly_; }

private:
    friend class ::SimpleFluid::PLIC::SweepAccess<ArithmeticMode::Exact>;
    friend std::optional<CertifiedPoly> certified_clip(CertifiedPolyRef, const Plane&, ClipScratch&);
    friend std::optional<CertifiedPoly> certified_clip_uncached(CertifiedPolyRef, const Plane&);
    friend std::optional<CertifiedPoly> certified_intersection(CertifiedPolyRef, CertifiedPolyRef, ClipScratch&);
    friend std::optional<CertifiedPoly> certified_intersection(CertifiedPolyRef, std::span<const Plane>, ClipScratch&);
    explicit CertifiedPoly(Poly poly) : poly_(std::move(poly)) {}
    Poly poly_;
};

template<class Distances, class Check>
Cut clipped_impl(const Poly& input, const Plane&, bool measure_outputs, Distances& distances, Check check)
{
    Cut result;
    if (measure_outputs)
        result.apertures.resize(input.faces.size());
    bool negative = false, positive = false;
    for (std::size_t i = 0; i < input.vertices.size(); ++i)
    {
        const auto d = distances.sign(i);
        negative = negative || d < 0;
        positive = positive || d > 0;
    }
    if (!negative && positive)
    {
        // Zero 3D measure does not erase a represented face contact.
        for (std::size_t f = 0; measure_outputs && f < input.faces.size(); ++f)
            if (std::all_of(
                    input.faces[f].begin(), input.faces[f].end(), [&](auto i) { return distances.sign(i) == 0; }))
            {
                const auto area = area_vector(input, input.faces[f]);
                if (measure_outputs)
                    result.apertures[f] = square_root(dot(area, area));
            }
        return result;
    }
    if (!positive)
    {
        result.poly = input;
        for (std::size_t f = 0; measure_outputs && f < input.faces.size(); ++f)
        {
            result.face_sources.push_back(f);
            const auto area = area_vector(input, input.faces[f]);
            if (measure_outputs)
                result.apertures[f] = square_root(dot(area, area));
        }
        return result;
    }
    Poly output;
    std::map<P, std::size_t> indices;
    auto index = [&](const P& p, int sign)
    {
        const auto [it, inserted] = indices.emplace(p, output.vertices.size());
        if (inserted)
        {
            distances.append(sign);
            output.vertices.push_back(p);
        }
        return it->second;
    };
    std::map<std::size_t, std::size_t> cap_next;
    for (std::size_t f = 0; f < input.faces.size(); ++f)
    {
        Face face;
        auto append = [&](const P& p, int sign)
        {
            const auto n = index(p, sign);
            if (face.empty() || face.back() != n)
                face.push_back(n);
        };
        const auto& original = input.faces[f];
        for (std::size_t i = 0; i < original.size(); ++i)
        {
            const auto& a = input.vertices[original[i]];
            const auto& b = input.vertices[original[(i + 1) % original.size()]];
            // The uncached samples own one exact distance per endpoint, as in
            // the original edge loop. Cached samples only reference entries
            // and preserve lazy construction until a crossing needs the value.
            const auto da = distances.sample(original[i]);
            const auto db = distances.sample(original[(i + 1) % original.size()]);
            const auto sa = da.sign();
            const auto sb = db.sign();
            if (sa <= 0)
                append(a, sa);
            if ((sa < 0 && sb > 0) || (sa > 0 && sb < 0))
            {
                // Certified signs decide topology only. Intersections always
                // use the original, unscaled exact rational distances.
                const auto& va = da.exact();
                const auto& vb = db.exact();
                append(add(a, mul(sub(b, a), Q(va / (va - vb)))), 0);
            }
        }
        if (face.size() > 1 && face.front() == face.back())
            face.pop_back();
        if (face.size() < 3)
            continue;
        const auto area = area_vector(output, face);
        if (zero(area))
            continue;
        if (measure_outputs)
            result.apertures[f] = square_root(dot(area, area));
        for (std::size_t i = 0; i < face.size(); ++i)
        {
            const auto a = face[i], b = face[(i + 1) % face.size()];
            if (distances.output_zero(output, a) && distances.output_zero(output, b))
                if (!cap_next.emplace(b, a).second)
                    throw std::runtime_error("Clipped cap has unresolved/nonmanifold edge incidence.");
        }
        output.faces.push_back(std::move(face));
        if (measure_outputs)
            result.face_sources.push_back(f);
    }
    if (cap_next.size() < 3)
        throw std::runtime_error("Positive convex cut has an unresolved cap.");
    auto next = cap_next.begin()->first;
    const auto first = next;
    do
    {
        if (result.cap.size() >= cap_next.size() || !cap_next.contains(next))
            throw std::runtime_error("Clipped cap is not one closed cycle.");
        result.cap.push_back(next);
        next = cap_next.at(next);
    } while (next != first);
    if (result.cap.size() != cap_next.size())
        throw std::runtime_error("Clipped cap has disconnected cycles.");
    output.faces.push_back(result.cap);
    if (measure_outputs)
        result.face_sources.push_back(std::numeric_limits<std::size_t>::max());
    // Face tangencies can introduce vertices unused by every positive-area face.
    Poly compact;
    std::map<std::size_t, std::size_t> compact_ids;
    for (const auto& face : output.faces)
    {
        Face mapped;
        for (const auto old : face)
        {
            const auto [it, inserted] = compact_ids.emplace(old, compact.vertices.size());
            if (inserted)
                compact.vertices.push_back(output.vertices[old]);
            mapped.push_back(it->second);
        }
        compact.faces.push_back(std::move(mapped));
    }
    result.cap = compact.faces.back();
    check(compact, measure_outputs);
    result.poly = std::move(compact);
    return result;
}
Cut clipped(const Poly& input, const Plane& plane, bool measure_outputs = true);
void clipped_incidence(const Poly& poly, const Plane& plane);
std::optional<CertifiedPoly> certified_clip(CertifiedPolyRef input, const Plane& plane, ClipScratch& scratch);
std::optional<CertifiedPoly> certified_clip_uncached(CertifiedPolyRef input, const Plane& plane);
std::optional<CertifiedPoly> certified_intersection(CertifiedPolyRef a, CertifiedPolyRef b, ClipScratch& scratch);
std::optional<CertifiedPoly> certified_intersection(
    CertifiedPolyRef a, std::span<const Plane> clipper_planes, ClipScratch& scratch);
std::optional<Poly> intersection(const Poly& a, const Poly& b, bool measure_outputs = true);
bool contained(const Poly& a, const Poly& b);
std::optional<Product> product_description(const Poly& poly);
Moments product_moments(const Product& product, const P& origin);
Poly product_poly(const Product& product);
std::optional<Product> product_intersection(const Product& a, const Product& b);

struct WireWriter
{
    std::vector<char> data;
    void word(std::uint64_t value);
    void real(double value);
    void integer(I value);
    void rational(const Q& value);
};

struct WireReader
{
    std::span<const char> data;
    std::size_t position = 0;
    std::uint64_t word();
    std::size_t count();
    double real();
    I integer();
    Q rational();
    void finish();
};
NativeConvexShell canonical_native_shell(
    std::span<const GeometryPoint> vertices, const std::vector<Face>& outward_faces);
} // namespace exact_geometry_detail SIMPLEFLUID_LOCAL
} // namespace SimpleFluid::PLIC
namespace SimpleFluid::PLIC
{

template<> void enable_backend_statistics<ArithmeticMode::Exact>(bool enabled);
template<> void reset_backend_statistics<ArithmeticMode::Exact>();
template<> BackendStatistics<ArithmeticMode::Exact> backend_statistics<ArithmeticMode::Exact>();

struct SIMPLEFLUID_LOCAL ConvexCellGeometry<ArithmeticMode::Exact>::Impl
{
    Poly poly;
    GeometryPoint origin;
    std::optional<Product> product;
};

namespace exact_geometry_detail
{
struct SIMPLEFLUID_LOCAL ExtrudedBaseData
{
    std::vector<PointRecipe> recipes;
    std::vector<std::size_t> active;
    std::vector<P> points, centers, deltas, edge_centers;
    std::vector<Q> areas;
    std::vector<MeshUtils::Vec3> display;
    std::vector<char> topology;
};

struct SIMPLEFLUID_LOCAL ExtrudedAxisValues
{
    std::vector<ScalarRecipe> zrecipes;
    std::vector<Q> z;
};

struct SIMPLEFLUID_LOCAL ExtrudedAxisData
{
    std::shared_ptr<ExtrudedAxisValues> values = std::make_shared<ExtrudedAxisValues>();
    std::vector<std::size_t> zactive;
    std::vector<double> display;
};

} // namespace exact_geometry_detail

struct SIMPLEFLUID_LOCAL ExtrudedConstruction<ArithmeticMode::Exact>::Impl
{
    std::shared_ptr<const ExtrudedTopology> topology;
    std::shared_ptr<ExtrudedBaseData> base = std::make_shared<ExtrudedBaseData>();
    std::shared_ptr<ExtrudedAxisData> axis = std::make_shared<ExtrudedAxisData>();
    std::size_t begin = 0, layers = 0;
};
} // namespace SimpleFluid::PLIC
namespace SimpleFluid::PLIC
{

namespace exact_geometry_detail SIMPLEFLUID_LOCAL
{
std::vector<char> topology_descriptor(const ExtrudedTopology& topology);
MeshUtils::Vec3 rounded_point(const P& p);

using Polygon = std::vector<P>;
P polygon_vector(const Polygon& polygon);
Polygon minimal_polygon(Polygon polygon);
Polygon polygon_intersection(Polygon polygon, const Polygon& clipper, const P& normal);

struct FaceMoment
{
    Q area;
    P first;
};
FaceMoment polygon_moments(const Polygon& polygon, std::size_t axis, const P& origin);
P polygon_center(const Polygon& polygon);
void surface_write(WireWriter& w, const PolyhedralSurfaceUse& surface);
PolyhedralSurfaceUse surface_read(WireReader& r);
void cell_key_write(WireWriter& w, const ::SimpleFluid::Meshes::AMR::AdaptiveCellKey& key);
::SimpleFluid::Meshes::AMR::AdaptiveCellKey cell_key_read(WireReader& r);

using ComplexID = std::uint64_t;

constexpr ComplexID absent_cell = std::numeric_limits<ComplexID>::max();

struct ComplexCellRow
{
    ::SimpleFluid::Meshes::AMR::AdaptiveCellKey key;
    ConvexGeometryPtr<ArithmeticMode::Exact> geometry;
    double volume = 0;
    MeshUtils::Vec3 center{};
    std::vector<ComplexID> faces, nodes;
};

struct ComplexFaceRow
{
    PolyhedralSurfaceUse surface;
    ComplexID owner = absent_cell, neighbor = absent_cell;
    std::vector<ComplexID> nodes;
    double area = 0;
    MeshUtils::Vec3 vector{}, center{};
};

struct ComplexNodeRow
{
    std::uint64_t region = 0;
    P point;
};
} // namespace exact_geometry_detail SIMPLEFLUID_LOCAL
} // namespace SimpleFluid::PLIC
namespace SimpleFluid::PLIC
{

struct SIMPLEFLUID_LOCAL PolyhedralComplex<ArithmeticMode::Exact>::Impl
{
    ComplexID global_cells = 0, global_faces = 0, global_nodes = 0;
    std::map<ComplexID, ComplexCellRow> cells;
    std::map<ComplexID, ComplexFaceRow> faces;
    std::map<ComplexID, ComplexNodeRow> nodes;
    std::vector<ComplexID> cell_ids, resident_ids, face_ids, node_ids;
};
} // namespace SimpleFluid::PLIC
namespace SimpleFluid::PLIC
{

namespace exact_geometry_detail SIMPLEFLUID_LOCAL
{
void complex_validate(PolyhedralComplex<ArithmeticMode::Exact>::Impl& impl);
void write_point_recipes(WireWriter& w, const ExtrudedBaseData& base);
void read_point_recipes(WireReader& r, std::vector<PointRecipe>& points, std::vector<std::size_t>& active);
void read_axis_parts(std::span<const char> packet, std::vector<ScalarRecipe>& roots, std::vector<std::size_t>& active);
} // namespace exact_geometry_detail SIMPLEFLUID_LOCAL
} // namespace SimpleFluid::PLIC
namespace SimpleFluid::PLIC
{

struct SIMPLEFLUID_LOCAL PlanarFace<ArithmeticMode::Exact>::Impl
{
    Polygon polygon;
};
} // namespace SimpleFluid::PLIC
namespace SimpleFluid::PLIC
{

namespace exact_geometry_detail SIMPLEFLUID_LOCAL
{
void validate_planar_polygon(const Polygon& polygon);
P authority_point(std::span<const char> packet);
P displacement_point(std::span<const char> packet);
std::vector<char> displacement_packet(const P& p);
std::vector<char> point_packet(const P& p);
} // namespace exact_geometry_detail SIMPLEFLUID_LOCAL
} // namespace SimpleFluid::PLIC
namespace SimpleFluid::PLIC
{

template<> std::vector<char> displacement<ArithmeticMode::Exact>(GeometryPoint from, GeometryPoint to);
template<>
std::vector<char> displacement_difference<ArithmeticMode::Exact>(
    std::span<const char> first, std::span<const char> second);
template<> GeometryPoint displacement_display<ArithmeticMode::Exact>(std::span<const char> packet);
template<>
std::vector<char> point_translated<ArithmeticMode::Exact>(
    std::span<const char> identity, std::span<const char> displacement);
template<>
std::vector<char> periodic_point<ArithmeticMode::Exact>(
    std::span<const char> identity, GeometryPoint lower, GeometryPoint upper, std::array<bool, 3> periodic);
template<> GeometryPoint point_display<ArithmeticMode::Exact>(std::span<const char> identity);
template<>
std::array<GeometryValue<ArithmeticMode::Exact>, 3> numerical_point<ArithmeticMode::Exact>(
    std::span<const char> identity);

struct SIMPLEFLUID_LOCAL EdgePointIndex<ArithmeticMode::Exact>::Impl
{
    P origin, direction;
    std::size_t axis = 0;
    std::map<Q, std::vector<char>> points;
};
template<> std::vector<char> planar_support<ArithmeticMode::Exact>(const PlanarFace<ArithmeticMode::Exact>& face);
template<> int planar_orientation<ArithmeticMode::Exact>(const PlanarFace<ArithmeticMode::Exact>& face);
template<>
std::vector<PlanarFacePtr<ArithmeticMode::Exact>> subtract_planar_faces<ArithmeticMode::Exact>(
    const PlanarFace<ArithmeticMode::Exact>& source, std::span<const PlanarFacePtr<ArithmeticMode::Exact>> removed);

// Signed transport constructions share this compiled unit's exact arithmetic.
struct SIMPLEFLUID_LOCAL GeometryMeasure<ArithmeticMode::Exact>::Impl
{
    Q value;
};

struct SIMPLEFLUID_LOCAL ConvexFragment<ArithmeticMode::Exact>::Impl
{
    explicit Impl(CertifiedPoly value) : poly(std::move(value)) {}
    CertifiedPoly poly;
};

struct SIMPLEFLUID_LOCAL SweptFace<ArithmeticMode::Exact>::Impl
{
    std::vector<SignedSweepSimplex<ArithmeticMode::Exact>> pieces;
    std::vector<std::array<std::vector<char>, 3>> edges;
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
} // namespace SimpleFluid::PLIC
namespace SimpleFluid::PLIC
{

namespace exact_geometry_detail SIMPLEFLUID_LOCAL
{
std::vector<char> swept_point(const P& point);

struct SweepBudget
{
    SweepLimits limits;
    std::size_t bytes = 0, tests = 0;
    std::size_t cache_peak = 0, cache_simultaneous_peak = 0, cache_fallbacks = 0;
    explicit SweepBudget(SweepLimits l);
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
P sparse_support_normal(const Poly& poly, const Face& face);

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
    const P& backcenter, std::vector<std::array<std::vector<char>, 3>>* sides);
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
} // namespace exact_geometry_detail SIMPLEFLUID_LOCAL
} // namespace SimpleFluid::PLIC
namespace SimpleFluid::PLIC
{

template<>
GeometryMeasure<ArithmeticMode::Exact> point_projection<ArithmeticMode::Exact>(
    std::span<const char> identity, GeometryPoint normal, GeometryPoint anchor);
}
namespace SimpleFluid::PLIC
{

namespace exact_geometry_detail SIMPLEFLUID_LOCAL
{
std::size_t exact_integer_bits(I value);
void exact_measure_arithmetic_preflight(const Q& a, const Q& b, bool division, std::size_t budget);
void exact_storage_add(std::size_t& bytes, std::size_t count, std::size_t size = 1);
void exact_scalar_storage(std::size_t& bytes, const Q& q);

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
} // namespace exact_geometry_detail SIMPLEFLUID_LOCAL
} // namespace SimpleFluid::PLIC
namespace SimpleFluid::PLIC
{

template<> const void* backend_statistics_domain<ArithmeticMode::Exact>();
}
