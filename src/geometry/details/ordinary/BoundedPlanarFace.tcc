/** @file ordinary/BoundedPlanarFace.tcc
 * @brief Ordinary planar-face clipping and nominal overlap measures.
 */
namespace SimpleFluid::Meshes::bounded_geometry_detail
{
namespace ordinary_kernel SIMPLEFLUID_LOCAL
{
template<ArithmeticMode Mode> Scalar planar_turn(const PlanarVertex& a, const PlanarVertex& b, const PlanarVertex& c)
{
    static_assert(Mode == kernel_mode);
    const auto x = b.coordinates[0] - a.coordinates[0], y = b.coordinates[1] - a.coordinates[1];
    return x * (c.coordinates[1] - a.coordinates[1]) - y * (c.coordinates[0] - a.coordinates[0]);
}

template<ArithmeticMode Mode>
std::vector<BoundedPlaneAuthority> planar_common(
    const std::vector<BoundedPlaneAuthority>& first, const std::vector<BoundedPlaneAuthority>& second)
{
    static_assert(Mode == kernel_mode);
    std::vector<BoundedPlaneAuthority> result;
    for (const auto& key : first)
        if (std::find(second.begin(), second.end(), key) != second.end())
            result.push_back(key);
    return result;
}

template<ArithmeticMode Mode> void planar_charge(const PlanarVertex& vertex)
{
    static_assert(Mode == kernel_mode);
    active_work->charge(1, sizeof(PlanarVertex));
    active_work->charge(vertex.supports.capacity(), sizeof(BoundedPlaneAuthority));
    for (const auto& key : vertex.supports)
        active_work->charge(key.key.capacity(), sizeof(std::uint64_t));
}

template<ArithmeticMode Mode>
PlanarLine planar_line(const PlanarVertex& first, const PlanarVertex& second, const BoundedPlaneAuthority& primary)
{
    static_assert(Mode == kernel_mode);
    PlanarLine line{first, second, planar_common<Mode>(first.supports, second.supports)};
    std::erase(line.supports, primary);
    const auto dx = second.coordinates[0] - first.coordinates[0], dy = second.coordinates[1] - first.coordinates[1];
    if (!((dx.value > 0 || dx.value < 0) || (dy.value > 0 || dy.value < 0)))
        reject("planar face edge has unresolved positive length.");
    return line;
}

template<ArithmeticMode Mode> Scalar planar_distance(const PlanarVertex& vertex, const PlanarLine& edge)
{
    static_assert(Mode == kernel_mode);
    for (const auto& key : edge.supports)
        if (std::find(vertex.supports.begin(), vertex.supports.end(), key) != vertex.supports.end())
            return Scalar{};
    return planar_turn<Mode>(edge.first, edge.second, vertex);
}

template<ArithmeticMode Mode> Scalar planar_area(const PlanarPolygon& polygon)
{
    static_assert(Mode == kernel_mode);
    if (polygon.size() < 3)
        return Scalar{};
    long double sum = 0, correction = 0;
    for (std::size_t i = 1; i + 1 < polygon.size(); ++i)
    {
        const auto term = planar_turn<Mode>(polygon[0], polygon[i], polygon[i + 1]);
        ArithmeticPolicy<Mode>::scalar_operation();
        const auto next = sum + term.value;
        operation();
        correction += std::abs(sum) >= std::abs(term.value) ? (sum - next) + term.value : (term.value - next) + sum;
        finite(next);
        sum = next;
    }
    ArithmeticPolicy<Mode>::scalar_operation();
    operation();
    return Scalar((sum + correction) / 2);
}

template<ArithmeticMode Mode>
void planar_convex(const PlanarPolygon& polygon, int winding, const BoundedPlaneAuthority& primary)
{
    static_assert(Mode == kernel_mode);
    for (std::size_t e = 0; e < polygon.size(); ++e)
    {
        const auto edge = planar_line<Mode>(polygon[e], polygon[(e + 1) % polygon.size()], primary);
        for (std::size_t v = 0; v < polygon.size(); ++v)
        {
            active_work->pair();
            if (v == e || v == (e + 1) % polygon.size())
                continue;
            if (sign<Mode>(planar_distance<Mode>(polygon[v], edge)) * winding < 0)
                reject("planar face loop is nonconvex or incorrectly ordered.");
        }
    }
}

template<ArithmeticMode Mode> void planar_push(PlanarPolygon& polygon, PlanarVertex vertex)
{
    static_assert(Mode == kernel_mode);
    if (!polygon.empty() && polygon.back().token == vertex.token)
        return;
    if (polygon.size() == active_work->limits.maximum_vertices)
        reject("planar overlap vertex cap exhausted.");
    planar_charge<Mode>(vertex);
    polygon.push_back(std::move(vertex));
}

template<ArithmeticMode Mode>
PlanarVertex planar_crossing(
    const PlanarVertex& a, const PlanarVertex& b, Scalar da, Scalar db, const PlanarLine& edge, std::size_t& token)
{
    static_assert(Mode == kernel_mode);
    if (da.zero())
        return a;
    if (db.zero())
        return b;
    const auto fraction = da / (da - db);
    if (fraction.value < 0 || fraction.value > 1)
        reject("planar clipping intersection is outside its certified crossing edge.");
    PlanarVertex result;
    for (unsigned k = 0; k < 2; ++k)
        result.coordinates[k] = a.coordinates[k] + fraction * (b.coordinates[k] - a.coordinates[k]);
    for (unsigned k = 0; k < 3; ++k)
        result.physical[k] = a.physical[k] + fraction * (b.physical[k] - a.physical[k]);
    result.supports = planar_common<Mode>(a.supports, b.supports);
    for (const auto& key : edge.supports)
        if (std::find(result.supports.begin(), result.supports.end(), key) == result.supports.end())
            result.supports.push_back(key);
    result.token = ++token;
    return result;
}

template<ArithmeticMode Mode>
bool planar_line_contact(const PlanarPolygon& polygon, const BoundedPlaneAuthority& primary)
{
    static_assert(Mode == kernel_mode);
    if (polygon.size() < 3)
        return true;
    auto common = polygon.front().supports;
    for (std::size_t i = 1; i < polygon.size(); ++i)
        common = planar_common<Mode>(common, polygon[i].supports);
    std::erase(common, primary);
    return !common.empty();
}

template<ArithmeticMode Mode>
BoundedPlanarFaceOverlap bounded_planar_face_overlap(std::span<const BoundedPoint> first,
    std::span<const BoundedPoint> second, const BoundedPlaneAuthority& support, BoundedGeometryLimits limits)
{
    static_assert(Mode == kernel_mode);
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
            auto p = point<Mode>(vertex);
            for (unsigned a = 0; a < 3; ++a)
            {
                p[a] = p[a] - Scalar(origin[a]);
                maximum_coordinate = std::max(maximum_coordinate, std::abs(p[a].value));
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
            vertex = scaled<Mode>(vertex, -exponent);
    const auto project = [&](const std::vector<Point>& points, std::span<const BoundedPoint> originals,
                             unsigned dropped, std::size_t& token)
    {
        PlanarPolygon polygon;
        polygon.reserve(points.size());
        for (std::size_t i = 0; i < points.size(); ++i)
        {
            const unsigned a = (dropped + 1) % 3, b = (dropped + 2) % 3;
            PlanarVertex vertex{{points[i][a], points[i][b]}, points[i], originals[i].supports(), ++token};
            planar_charge<Mode>(vertex);
            polygon.push_back(std::move(vertex));
        }
        return polygon;
    };
    unsigned dropped = 0;
    long double best = 0;
    for (unsigned axis = 0; axis < 3; ++axis)
    {
        std::size_t token = 0;
        const auto area = planar_area<Mode>(project(source_first, first, axis, token));
        const auto certain = area.value > 0 ? area.value : area.value < 0 ? -area.value : 0;
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
    auto first_area = planar_area<Mode>(first_polygon), second_area = planar_area<Mode>(second_polygon);
    const auto first_winding = sign<Mode>(first_area), second_winding = sign<Mode>(second_area);
    if (!first_winding || !second_winding)
        reject("planar overlap has zero-area input.");
    planar_convex<Mode>(first_polygon, first_winding, support);
    planar_convex<Mode>(second_polygon, second_winding, support);
    if (first_winding < 0)
        first_area = -first_area;
    if (second_winding < 0)
        second_area = -second_area;
    BoundedPlanarFaceOverlap result;
    result.overlap_projected_area = bounded<Mode>(Scalar{});
    result.overlap_over_first = bounded<Mode>(Scalar{});
    result.overlap_over_second = bounded<Mode>(Scalar{});
    result.orientation = first_winding * second_winding;
    result.dropped_axis = dropped;
    result.first_projected_area = bounded<Mode>(scaled<Mode>(first_area, 2 * exponent));
    result.second_projected_area = bounded<Mode>(scaled<Mode>(second_area, 2 * exponent));
    for (std::size_t e = 0; e < second_polygon.size() && !first_polygon.empty(); ++e)
    {
        const auto edge =
            planar_line<Mode>(second_polygon[e], second_polygon[(e + 1) % second_polygon.size()], support);
        PlanarPolygon clipped;
        for (std::size_t i = 0; i < first_polygon.size(); ++i)
        {
            const auto& a = first_polygon[i];
            const auto& b = first_polygon[(i + 1) % first_polygon.size()];
            const auto da = planar_distance<Mode>(a, edge), db = planar_distance<Mode>(b, edge);
            const auto sa = sign<Mode>(da) * second_winding, sb = sign<Mode>(db) * second_winding;
            if (sb >= 0)
            {
                if (sa < 0)
                    planar_push<Mode>(clipped, planar_crossing<Mode>(a, b, da, db, edge, token));
                planar_push<Mode>(clipped, b);
            }
            else if (sa >= 0)
                planar_push<Mode>(clipped, planar_crossing<Mode>(a, b, da, db, edge, token));
        }
        if (clipped.size() > 1 && clipped.front().token == clipped.back().token)
            clipped.pop_back();
        first_polygon = std::move(clipped);
    }
    if (planar_line_contact<Mode>(first_polygon, support))
        return result;
    auto overlap = planar_area<Mode>(first_polygon);
    const auto overlap_winding = sign<Mode>(overlap);
    if (overlap_winding != first_winding)
        reject("planar overlap lost its certified polygon orientation.");
    if (overlap_winding < 0)
        overlap = -overlap;
    result.overlap_projected_area = bounded<Mode>(scaled<Mode>(overlap, 2 * exponent));
    const Point anchor{Scalar(origin[0]), Scalar(origin[1]), Scalar(origin[2])};
    result.overlap_vertices.reserve(first_polygon.size());
    active_work->charge(first_polygon.size(), sizeof(BoundedPoint));
    for (const auto& vertex : first_polygon)
    {
        auto exported = bounded<Mode>(scaled<Mode>(vertex.physical, exponent) + anchor);
        BoundedGeometryAccess::seal(exported, vertex.supports);
        active_work->charge(1, BoundedGeometryAccess::storage_bytes(exported));
        result.overlap_vertices.push_back(std::move(exported));
    }
    const auto over_first = overlap / first_area, over_second = overlap / second_area;
    if (over_first.value > 1 || over_second.value > 1)
        reject("planar overlap area exceeds an input polygon.");
    result.overlap_over_first = bounded<Mode>(result.orientation > 0 ? over_first : -over_first);
    result.overlap_over_second = bounded<Mode>(result.orientation > 0 ? over_second : -over_second);
    return result;
}

template Scalar planar_turn<kernel_mode>(const PlanarVertex& a, const PlanarVertex& b, const PlanarVertex& c);
template std::vector<BoundedPlaneAuthority> planar_common<kernel_mode>(
    const std::vector<BoundedPlaneAuthority>& first, const std::vector<BoundedPlaneAuthority>& second);
template void planar_charge<kernel_mode>(const PlanarVertex& vertex);
template PlanarLine planar_line<kernel_mode>(
    const PlanarVertex& first, const PlanarVertex& second, const BoundedPlaneAuthority& primary);
template Scalar planar_distance<kernel_mode>(const PlanarVertex& vertex, const PlanarLine& edge);
template Scalar planar_area<kernel_mode>(const PlanarPolygon& polygon);
template void planar_convex<kernel_mode>(
    const PlanarPolygon& polygon, int winding, const BoundedPlaneAuthority& primary);
template void planar_push<kernel_mode>(PlanarPolygon& polygon, PlanarVertex vertex);
template PlanarVertex planar_crossing<kernel_mode>(
    const PlanarVertex& a, const PlanarVertex& b, Scalar da, Scalar db, const PlanarLine& edge, std::size_t& token);
template bool planar_line_contact<kernel_mode>(const PlanarPolygon& polygon, const BoundedPlaneAuthority& primary);
template BoundedPlanarFaceOverlap bounded_planar_face_overlap<kernel_mode>(std::span<const BoundedPoint> first,
    std::span<const BoundedPoint> second, const BoundedPlaneAuthority& support, BoundedGeometryLimits limits);
} // namespace ordinary_kernel SIMPLEFLUID_LOCAL
} // namespace SimpleFluid::Meshes::bounded_geometry_detail
