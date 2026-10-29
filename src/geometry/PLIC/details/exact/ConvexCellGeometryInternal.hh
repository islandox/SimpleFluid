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
} // namespace SimpleFluid::PLIC
namespace SimpleFluid::PLIC
{

namespace exact_geometry_detail SIMPLEFLUID_LOCAL
{
using Q = boost::multiprecision::cpp_rational;

using I = boost::multiprecision::cpp_int;

using P = std::array<Q, 3>;

using Face = std::vector<std::size_t>;
extern constinit thread_local bool count_exact_geometry;
extern constinit thread_local BackendStatistics<ArithmeticMode::Exact> exact_geometry_counts;
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
    friend class ::SimpleFluid::geometry::swept_volume::SweepAccess<ArithmeticMode::Exact>;
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
    friend class ::SimpleFluid::geometry::swept_volume::SweepAccess<ArithmeticMode::Exact>;
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

} // namespace SimpleFluid::PLIC
namespace SimpleFluid::PLIC
{

template<> const void* backend_statistics_domain<ArithmeticMode::Exact>();
}
