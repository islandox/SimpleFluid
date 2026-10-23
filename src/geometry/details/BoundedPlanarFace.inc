/** @file BoundedPlanarFace.inc
 * @brief Directed projected face intersection on authenticated source supports.
 * Included by the bounded kernel to share its environment, arithmetic and caps.
 */
namespace SimpleFluid::Meshes
{
namespace
{
struct PlanarVertex
{
    std::array<Interval, 2> coordinates;
    Point physical;
    std::vector<BoundedPlaneAuthority> supports;
    std::size_t token = 0;
};
using PlanarPolygon = std::vector<PlanarVertex>;
struct PlanarLine
{
    PlanarVertex first, second;
    std::vector<BoundedPlaneAuthority> supports;
};
Interval planar_turn(const PlanarVertex& a, const PlanarVertex& b, const PlanarVertex& c)
{
    const auto x = b.coordinates[0] - a.coordinates[0], y = b.coordinates[1] - a.coordinates[1];
    return x * (c.coordinates[1] - a.coordinates[1]) - y * (c.coordinates[0] - a.coordinates[0]);
}
std::vector<BoundedPlaneAuthority> planar_common(
    const std::vector<BoundedPlaneAuthority>& first, const std::vector<BoundedPlaneAuthority>& second)
{
    std::vector<BoundedPlaneAuthority> result;
    for (const auto& key : first)
        if (std::find(second.begin(), second.end(), key) != second.end())
            result.push_back(key);
    return result;
}
void planar_charge(const PlanarVertex& vertex)
{
    active_work->charge(1, sizeof(PlanarVertex));
    active_work->charge(vertex.supports.capacity(), sizeof(BoundedPlaneAuthority));
    for (const auto& key : vertex.supports)
        active_work->charge(key.key.capacity(), sizeof(std::uint64_t));
}
PlanarLine planar_line(const PlanarVertex& first, const PlanarVertex& second, const BoundedPlaneAuthority& primary)
{
    PlanarLine line{first, second, planar_common(first.supports, second.supports)};
    std::erase(line.supports, primary);
    const auto dx = second.coordinates[0] - first.coordinates[0], dy = second.coordinates[1] - first.coordinates[1];
    if (!((dx.lo > 0 || dx.hi < 0) || (dy.lo > 0 || dy.hi < 0)))
        reject("planar face edge has unresolved positive length.");
    return line;
}
Interval planar_distance(const PlanarVertex& vertex, const PlanarLine& edge)
{
    for (const auto& key : edge.supports)
        if (std::find(vertex.supports.begin(), vertex.supports.end(), key) != vertex.supports.end())
            return Interval{};
    return planar_turn(edge.first, edge.second, vertex);
}
Interval planar_area(const PlanarPolygon& polygon)
{
    if (polygon.size() < 3)
        return Interval{};
    Interval area;
    long double estimate = 0, correction = 0;
    for (std::size_t i = 1; i + 1 < polygon.size(); ++i)
    {
        const auto term = planar_turn(polygon[0], polygon[i], polygon[i + 1]);
        area = area + term;
        const auto next = estimate + term.estimate;
        correction += std::abs(estimate) >= std::abs(term.estimate) ? (estimate - next) + term.estimate
                                                                    : (term.estimate - next) + estimate;
        estimate = next;
    }
    area = area / Interval(2);
    area.estimate = (estimate + correction) / 2;
    return area;
}
void planar_convex(const PlanarPolygon& polygon, int winding, const BoundedPlaneAuthority& primary)
{
    for (std::size_t e = 0; e < polygon.size(); ++e)
    {
        const auto edge = planar_line(polygon[e], polygon[(e + 1) % polygon.size()], primary);
        for (std::size_t v = 0; v < polygon.size(); ++v)
        {
            active_work->pair();
            if (v == e || v == (e + 1) % polygon.size())
                continue;
            if (sign(planar_distance(polygon[v], edge)) * winding < 0)
                reject("planar face loop is nonconvex or incorrectly ordered.");
        }
    }
}
void planar_push(PlanarPolygon& polygon, PlanarVertex vertex)
{
    if (!polygon.empty() && polygon.back().token == vertex.token)
        return;
    if (polygon.size() == active_work->limits.maximum_vertices)
        reject("planar overlap vertex cap exhausted.");
    planar_charge(vertex);
    polygon.push_back(std::move(vertex));
}
PlanarVertex planar_crossing(
    const PlanarVertex& a, const PlanarVertex& b, Interval da, Interval db, const PlanarLine& edge, std::size_t& token)
{
    if (da.zero())
        return a;
    if (db.zero())
        return b;
    const auto fraction = da / (da - db);
    if (fraction.hi < 0 || fraction.lo > 1)
        reject("planar clipping intersection is outside its certified crossing edge.");
    PlanarVertex result;
    for (unsigned k = 0; k < 2; ++k)
        result.coordinates[k] = a.coordinates[k] + fraction * (b.coordinates[k] - a.coordinates[k]);
    for (unsigned k = 0; k < 3; ++k)
        result.physical[k] = a.physical[k] + fraction * (b.physical[k] - a.physical[k]);
    result.supports = planar_common(a.supports, b.supports);
    for (const auto& key : edge.supports)
        if (std::find(result.supports.begin(), result.supports.end(), key) == result.supports.end())
            result.supports.push_back(key);
    result.token = ++token;
    return result;
}
bool planar_line_contact(const PlanarPolygon& polygon, const BoundedPlaneAuthority& primary)
{
    if (polygon.size() < 3)
        return true;
    auto common = polygon.front().supports;
    for (std::size_t i = 1; i < polygon.size(); ++i)
        common = planar_common(common, polygon[i].supports);
    std::erase(common, primary);
    return !common.empty();
}
} // namespace
BoundedPlanarFaceOverlap bounded_planar_face_overlap(std::span<const BoundedPoint> first,
    std::span<const BoundedPoint> second, const BoundedPlaneAuthority& support, BoundedGeometryLimits limits)
{
    Operation scope(limits);
    ++statistics.intersections;
    if (first.size() < 3 || second.size() < 3 || first.size() > limits.maximum_vertices ||
        second.size() > limits.maximum_vertices || support.key.empty())
        reject("planar overlap requires admitted nondegenerate source loops and a support authority.");
    for (const auto loop : {first, second})
        for (const auto& vertex : loop)
        {
            const auto& keys = vertex.supports();
            if (std::find(keys.begin(), keys.end(), support) == keys.end())
                reject("planar overlap point lacks authenticated common source support.");
        }
    const auto origin = first.front().value;
    std::vector<Point> source_first, source_second;
    long double maximum_coordinate = 0;
    const std::array loops{first, second};
    const std::array outputs{&source_first, &source_second};
    for (unsigned which = 0; which < 2; ++which)
    {
        const auto loop = loops[which];
        auto& output = *outputs[which];
        active_work->charge(loop.size(), sizeof(Point));
        for (const auto& vertex : loop)
        {
            auto p = point(vertex);
            for (unsigned a = 0; a < 3; ++a)
            {
                p[a] = p[a] - Interval(origin[a]);
                maximum_coordinate = std::max({maximum_coordinate, std::abs(p[a].lo), std::abs(p[a].hi)});
            }
            output.push_back(p);
        }
    }
    if (!(maximum_coordinate > 0))
        reject("planar overlap source loops collapse to a point.");
    int exponent = 0;
    (void) std::frexp(maximum_coordinate, &exponent);
    for (auto* loop : {&source_first, &source_second})
        for (auto& vertex : *loop)
            vertex = scaled(vertex, -exponent);
    const auto project = [&](const std::vector<Point>& points, std::span<const BoundedPoint> originals,
                             unsigned dropped, std::size_t& token)
    {
        PlanarPolygon polygon;
        polygon.reserve(points.size());
        for (std::size_t i = 0; i < points.size(); ++i)
        {
            const unsigned a = (dropped + 1) % 3, b = (dropped + 2) % 3;
            PlanarVertex vertex{{points[i][a], points[i][b]}, points[i], originals[i].supports(), ++token};
            planar_charge(vertex);
            polygon.push_back(std::move(vertex));
        }
        return polygon;
    };
    unsigned dropped = 0;
    long double best = 0;
    for (unsigned axis = 0; axis < 3; ++axis)
    {
        std::size_t token = 0;
        const auto area = planar_area(project(source_first, first, axis, token));
        const auto certain = area.lo > 0 ? area.lo : area.hi < 0 ? -area.hi : 0;
        if (certain > best)
        {
            best = certain;
            dropped = axis;
        }
    }
    if (!(best > 0))
        reject("planar overlap source area sign is unresolved.");
    std::size_t token = 0;
    auto first_polygon = project(source_first, first, dropped, token);
    const auto second_polygon = project(source_second, second, dropped, token);
    auto first_area = planar_area(first_polygon), second_area = planar_area(second_polygon);
    const auto first_winding = sign(first_area), second_winding = sign(second_area);
    if (!first_winding || !second_winding)
        reject("planar overlap has zero-area input.");
    planar_convex(first_polygon, first_winding, support);
    planar_convex(second_polygon, second_winding, support);
    if (first_winding < 0)
        first_area = -first_area;
    if (second_winding < 0)
        second_area = -second_area;
    BoundedPlanarFaceOverlap result;
    result.orientation = first_winding * second_winding;
    result.dropped_axis = dropped;
    result.first_projected_area = bounded(scaled(first_area, 2 * exponent));
    result.second_projected_area = bounded(scaled(second_area, 2 * exponent));
    for (std::size_t e = 0; e < second_polygon.size() && !first_polygon.empty(); ++e)
    {
        const auto edge = planar_line(second_polygon[e], second_polygon[(e + 1) % second_polygon.size()], support);
        PlanarPolygon clipped;
        for (std::size_t i = 0; i < first_polygon.size(); ++i)
        {
            const auto& a = first_polygon[i];
            const auto& b = first_polygon[(i + 1) % first_polygon.size()];
            const auto da = planar_distance(a, edge), db = planar_distance(b, edge);
            const auto sa = sign(da) * second_winding, sb = sign(db) * second_winding;
            if (sb >= 0)
            {
                if (sa < 0)
                    planar_push(clipped, planar_crossing(a, b, da, db, edge, token));
                planar_push(clipped, b);
            }
            else if (sa >= 0)
                planar_push(clipped, planar_crossing(a, b, da, db, edge, token));
        }
        if (clipped.size() > 1 && clipped.front().token == clipped.back().token)
            clipped.pop_back();
        first_polygon = std::move(clipped);
    }
    if (planar_line_contact(first_polygon, support))
        return result;
    auto overlap = planar_area(first_polygon);
    const auto overlap_winding = sign(overlap);
    if (overlap_winding != first_winding)
        reject("planar overlap lost its certified polygon orientation.");
    if (overlap_winding < 0)
        overlap = -overlap;
    result.overlap_projected_area = bounded(scaled(overlap, 2 * exponent));
    const Point anchor{Interval(origin[0]), Interval(origin[1]), Interval(origin[2])};
    result.overlap_vertices.reserve(first_polygon.size());
    active_work->charge(first_polygon.size(), sizeof(BoundedPoint));
    for (const auto& vertex : first_polygon)
    {
        auto exported = bounded(scaled(vertex.physical, exponent) + anchor);
        BoundedGeometryAccess::seal(exported, vertex.supports);
        active_work->charge(1, BoundedGeometryAccess::storage_bytes(exported));
        result.overlap_vertices.push_back(std::move(exported));
    }
    const auto over_first = overlap / first_area, over_second = overlap / second_area;
    if (over_first.lo > 1 || over_second.lo > 1)
        reject("planar overlap area exceeds an input polygon.");
    result.overlap_over_first = bounded(result.orientation > 0 ? over_first : -over_first);
    result.overlap_over_second = bounded(result.orientation > 0 ? over_second : -over_second);
    return result;
}
} // namespace SimpleFluid::Meshes
