/** @file FilteredSurfaceGeometry.cc
 * @brief Finite scalar-tetrahedron clipping and exact-level sections.
 */
#include "FVM/details/GeneralVoF/FilteredSurfaceGeometry.hh"
#include "utils/CompensatedSum.hh"
#include <algorithm>
#include <cmath>

namespace SimpleFluid::VoF::general_vof_detail::filtered_surface_kernel
{
long double admit_published_volume(
    std::span<const GeneralVoFSurfacePatch> patches, Point origin, long double target, long double allowance)
{
    const auto frame = wide(origin);
    SimpleFluid::detail::CompensatedSum<> published_volume;
    for (const auto& patch : patches)
    {
        const auto& polygon = patch.vertices;
        if (polygon.size() < 3)
            reject(Reason::VolumeConstraint, "Filtered published polygon has fewer than three vertices.");
        const auto a = sub(wide(polygon.front()), frame);
        for (std::size_t j = 1; j + 1 < polygon.size(); ++j)
            published_volume += dot(a, cross(sub(wide(polygon[j]), frame), sub(wide(polygon[j + 1]), frame))) / 6;
    }
    const auto measured = published_volume.value();
    if (!(measured > 0) || !std::isfinite(measured) || std::abs(measured - target) > allowance)
        reject(Reason::VolumeConstraint,
            "Filtered published binary64 surface does not preserve its component geometric volume.");
    return measured;
}
[[noreturn]] void reject(Reason reason, const char* message)
{
    throw GeneralVoFSurfaceResolutionError(reason, message);
}
Wide wide(const Point& p)
{
    return {p[0], p[1], p[2]};
}
Wide sub(const Wide& a, const Wide& b)
{
    return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}
Wide cross(const Wide& a, const Wide& b)
{
    return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}
long double dot(const Wide& a, const Wide& b)
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
long double norm(const Wide& a)
{
    return std::sqrt(dot(a, a));
}
long double volume(const Wide& a, const Wide& b, const Wide& c, const Wide& d)
{
    return std::abs(dot(sub(b, a), cross(sub(c, a), sub(d, a)))) / 6;
}
Cut intersection(const Vertex& a, const Vertex& b, long double iso)
{
    if (a.q == iso)
        return {{a.id, a.id}, a.p};
    if (b.q == iso)
        return {{b.id, b.id}, b.p};
    // Evaluate shared edges in one immutable orientation as well as assigning
    // shared IDs; adjacent tetrahedra then publish identical coordinates.
    const auto& first = a.id < b.id ? a : b;
    const auto& second = a.id < b.id ? b : a;
    const auto t = (iso - first.q) / (second.q - first.q);
    Cut result{{first.id, second.id}, {}};
    for (unsigned k = 0; k < 3; ++k)
        result.p[k] = first.p[k] + t * (second.p[k] - first.p[k]);
    return result;
}
long double clipped_volume(const Tetra& t, long double iso)
{
    std::array<unsigned, 4> in{}, out{};
    unsigned ni = 0, no = 0;
    for (unsigned j = 0; j < 4; ++j)
        (t.v[j].q > iso ? in[ni++] : out[no++]) = j;
    const auto full = volume(t.v[0].p, t.v[1].p, t.v[2].p, t.v[3].p);
    if (ni == 0)
        return 0;
    if (ni == 4)
        return full;
    if (ni == 1 || ni == 3)
    {
        const auto s = ni == 1 ? in[0] : out[0];
        const auto& other = ni == 1 ? out : in;
        const auto a = intersection(t.v[s], t.v[other[0]], iso).p;
        const auto b = intersection(t.v[s], t.v[other[1]], iso).p;
        const auto c = intersection(t.v[s], t.v[other[2]], iso).p;
        const auto small = volume(t.v[s].p, a, b, c);
        return ni == 1 ? small : full - small;
    }
    const auto& a = t.v[in[0]];
    const auto& b = t.v[in[1]];
    const auto p = intersection(a, t.v[out[0]], iso).p;
    const auto q = intersection(a, t.v[out[1]], iso).p;
    const auto r = intersection(b, t.v[out[0]], iso).p;
    const auto s = intersection(b, t.v[out[1]], iso).p;
    return volume(a.p, b.p, p, q) + volume(b.p, p, q, s) + volume(b.p, p, r, s);
}
std::vector<Cut> section(const Tetra& t, long double iso, Wide& outward)
{
    std::vector<Cut> result;
    unsigned positive = 0, zero = 0;
    for (const auto& v : t.v)
    {
        positive += v.q > iso;
        if (v.q == iso)
        {
            ++zero;
            result.push_back({{v.id, v.id}, v.p});
        }
    }
    if (zero == 4)
        reject(Reason::AmbiguousTopology, "Filtered isovalue coincides with an unresolved tetrahedral plateau.");
    if (!positive)
        return {};
    for (unsigned a = 0; a < 4; ++a)
        for (unsigned b = a + 1; b < 4; ++b)
            if ((t.v[a].q > iso && t.v[b].q < iso) || (t.v[a].q < iso && t.v[b].q > iso))
                result.push_back(intersection(t.v[a], t.v[b], iso));
    std::sort(result.begin(), result.end(), [](const Cut& a, const Cut& b) { return a.id < b.id; });
    result.erase(std::unique(result.begin(), result.end(), [](const Cut& a, const Cut& b) { return a.id == b.id; }),
        result.end());
    // Fewer than three immutable intersections is structurally a point/edge,
    // not a discarded positive-area computational fragment.
    if (result.size() < 3)
        return {};
    const auto e1 = sub(t.v[1].p, t.v[0].p), e2 = sub(t.v[2].p, t.v[0].p), e3 = sub(t.v[3].p, t.v[0].p);
    const auto c1 = cross(e2, e3), c2 = cross(e3, e1), c3 = cross(e1, e2);
    const auto det = dot(e1, c1);
    for (unsigned k = 0; k < 3; ++k)
        outward[k] =
            -((t.v[1].q - t.v[0].q) * c1[k] + (t.v[2].q - t.v[0].q) * c2[k] + (t.v[3].q - t.v[0].q) * c3[k]) / det;
    const auto n = norm(outward);
    if (!(n > 0) || !std::isfinite(n))
        reject(Reason::AmbiguousTopology, "Filtered tetrahedron has unresolved normal.");
    for (auto& x : outward)
        x /= n;
    Wide center = result.front().p;
    for (const auto& c : result)
        for (unsigned k = 0; k < 3; ++k)
            center[k] += (c.p[k] - result.front().p[k]) / result.size();
    auto u = sub(result.front().p, center);
    const auto un = norm(u);
    if (!(un > 0))
        reject(Reason::AmbiguousTopology, "Filtered polygon collapsed.");
    for (auto& x : u)
        x /= un;
    const auto v = cross(outward, u);
    std::sort(result.begin(), result.end(),
        [&](const Cut& a, const Cut& b)
        {
            return std::atan2(dot(sub(a.p, center), v), dot(sub(a.p, center), u)) <
                   std::atan2(dot(sub(b.p, center), v), dot(sub(b.p, center), u));
        });
    return result;
}
std::optional<std::size_t> match_full_facet(FullFacetMap& facets, const std::vector<CutId>& ids, std::size_t patch)
{
    if (ids.size() != 3 || !std::all_of(ids.begin(), ids.end(), [](const CutId& id) { return id.first == id.second; }))
        return {};
    std::array<NodeId, 3> facet{ids[0].first, ids[1].first, ids[2].first};
    unsigned inversions = 0;
    for (unsigned a = 0; a < 3; ++a)
        for (unsigned b = a + 1; b < 3; ++b)
            inversions += facet[b] < facet[a];
    const auto orientation = inversions % 2 ? -1 : 1;
    std::sort(facet.begin(), facet.end());
    auto& record = facets[facet];
    if (record.incidence == 0)
    {
        record = {patch, orientation, 1};
        return {};
    }
    if (record.incidence != 1 || record.orientation == orientation)
        reject(Reason::AmbiguousTopology, "Filtered exact facet has excessive or equally oriented gas-side incidence.");
    ++record.incidence;
    return record.patch;
}
} // namespace SimpleFluid::VoF::general_vof_detail::filtered_surface_kernel
