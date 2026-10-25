/** @file ExtrudedConstructionInternal.hh
 * @brief Ordinary affine recipe storage and native extrusion metrics.
 */
#pragma once
#include "geometry/PLIC/ConvexGeometry.hh"
#include "geometry/PLIC/ExtrudedConstruction.hh"
#include "geometry/PLIC/details/ordinary/NativeGeometrySupport.hh"
#include "geometry/PLIC/details/ordinary/NativeSupportAccess.hh"
#include "geometry/mesh/ExtrudedRegionProviders.hh"
#include <algorithm>
#include <cmath>
#include <set>

namespace SimpleFluid::PLIC::ordinary_extrusion_detail
{
using Scalar = long double;
inline constexpr std::uint64_t recipe_packet_version = 2;
using P = details::ordinary_native::WidePoint;
using details::ordinary_native::add;
using details::ordinary_native::cross;
using details::ordinary_native::dot;
using details::ordinary_native::mul;
using details::ordinary_native::sub;
using WireWriter = details::ordinary_native::Writer;
struct WireReader : details::ordinary_native::Reader
{
    explicit WireReader(std::span<const char> input) : Reader{input} {}
    double real() { return GeometryValue<ArithmeticMode::Ordinary>{Reader::real()}.as_double(); }
};
inline P point(GeometryPoint p)
{
    for (auto x : p)
        details::ordinary_native::finite(x);
    return {p[0], p[1], p[2]};
}
inline Scalar represented(double x)
{
    details::ordinary_native::finite(x);
    return x;
}
inline bool zero(const P& p)
{
    return p == P{};
}
inline GeometryValue<ArithmeticMode::Ordinary> measured(Scalar value)
{
    details::ordinary_native::finite(value);
    return {value};
}
inline GeometryValue<ArithmeticMode::Ordinary> positive_measure(Scalar value)
{
    if (!(value > 0))
        details::ordinary_native::reject("nonpositive native geometric measure after arithmetic.");
    return measured(value);
}
inline GeometryValue<ArithmeticMode::Ordinary> square_root(Scalar value)
{
    if (!(value > 0))
        details::ordinary_native::reject("nonpositive squared native measure.");
    return measured(std::sqrt(value));
}
struct Product
{
    std::vector<P> base;
    Scalar lower = 0, upper = 1;
};
struct Moments
{
    Scalar volume = 0;
    P first{};
};
inline Moments product_moments(const Product& product, const P& origin)
{
    if (product.base.size() < 3 || !(product.upper > product.lower))
        details::ordinary_native::reject("invalid native extrusion product.");
    const auto anchor = product.base.front();
    Scalar area = 0;
    P first{};
    for (std::size_t i = 1; i + 1 < product.base.size(); ++i)
    {
        const auto b = sub(product.base[i], anchor), c = sub(product.base[i + 1], anchor);
        const auto triangle = cross(b, c)[2] / 2;
        area += triangle;
        first = add(first, mul(add(b, c), triangle / 3));
    }
    if (!(area > 0))
        details::ordinary_native::reject("native extrusion has nonpositive base area.");
    auto center = add(anchor, mul(first, 1 / area));
    center[2] = product.lower + (product.upper - product.lower) / 2;
    const auto volume = area * (product.upper - product.lower);
    return {measured(volume).value, mul(sub(center, origin), volume)};
}
inline ConvexGeometryPtr<ArithmeticMode::Ordinary> product_geometry(const Product& product)
{
    const auto n = product.base.size();
    std::vector<Point<ArithmeticMode::Ordinary>> vertices;
    vertices.reserve(2 * n);
    for (const auto z : {product.lower, product.upper})
        for (auto p : product.base)
        {
            p[2] = z;
            vertices.emplace_back(p);
        }
    std::vector<std::vector<std::size_t>> faces;
    std::vector<std::size_t> bottom, top;
    for (std::size_t i = 0; i < n; ++i)
    {
        bottom.push_back(n - 1 - i);
        top.push_back(n + i);
    }
    faces.push_back(std::move(bottom));
    faces.push_back(std::move(top));
    for (std::size_t i = 0; i < n; ++i)
        faces.push_back({i, (i + 1) % n, (i + 1) % n + n, i + n});
    return ConvexCellGeometry<ArithmeticMode::Ordinary>::make_points(vertices, faces);
}
inline ConvexGeometryPtr<ArithmeticMode::Ordinary> product_poly(const Product& product)
{
    return product_geometry(product);
}
inline bool product_intersection(const Product& a, const Product& b)
{
    return bool(PlanarFace<ArithmeticMode::Ordinary>::make_wide(a.base)->intersection(
        *PlanarFace<ArithmeticMode::Ordinary>::make_wide(b.base)));
}
std::vector<char> topology_descriptor(const Meshes::ExtrudedTopology&);
MeshUtils::Vec3 rounded_point(const P&);
struct ExtrudedBaseData
{
    std::vector<PointRecipe> recipes;
    std::vector<std::size_t> active;
    std::vector<P> points, centers, deltas, edge_centers;
    std::vector<Scalar> areas;
    std::vector<std::vector<char>> edge_supports;
    std::vector<P> support_normals;
    std::vector<MeshUtils::Vec3> display;
    std::vector<char> topology;
};
struct ExtrudedAxisValues
{
    std::vector<ScalarRecipe> zrecipes;
    std::vector<Scalar> z;
};
struct ExtrudedAxisData
{
    std::shared_ptr<ExtrudedAxisValues> values = std::make_shared<ExtrudedAxisValues>();
    std::vector<std::size_t> zactive;
    std::vector<double> display;
};
void write_point_recipes(WireWriter&, const ExtrudedBaseData&);
void read_point_recipes(WireReader&, std::vector<PointRecipe>&, std::vector<std::size_t>&);
void read_axis_parts(std::span<const char>, std::vector<ScalarRecipe>&, std::vector<std::size_t>&);
} // namespace SimpleFluid::PLIC::ordinary_extrusion_detail

namespace SimpleFluid::PLIC
{
using namespace ordinary_extrusion_detail;
using Meshes::SemiStructMeshTopo;
using Meshes::SemiStructuredIndexer;
struct ExtrudedConstruction<ArithmeticMode::Ordinary>::Impl
{
    std::shared_ptr<const Meshes::ExtrudedTopology> topology;
    std::shared_ptr<ExtrudedBaseData> base = std::make_shared<ExtrudedBaseData>();
    std::shared_ptr<ExtrudedAxisData> axis = std::make_shared<ExtrudedAxisData>();
    std::size_t begin = 0, layers = 0;
};
} // namespace SimpleFluid::PLIC
