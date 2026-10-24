namespace
{
struct NumericalWriter
{
    std::vector<char> bytes;
    std::size_t limit;
    explicit NumericalWriter(std::size_t budget) : limit(budget) {}
    void word(std::uint64_t x)
    {
        if (bytes.size() > limit || limit - bytes.size() < 8)
            reject("numerical packet exceeds storage cap.");
        for (unsigned byte = 0; byte < 8; ++byte)
            bytes.push_back(static_cast<char>((x >> (8 * byte)) & 255));
    }
    void real(long double x)
    {
        finite(x);
        word(std::signbit(x) && x != 0 ? 1 : 0);
        int exponent = 0;
        auto fraction = std::frexp(std::abs(x), &exponent);
        word(static_cast<std::uint64_t>(static_cast<std::int64_t>(exponent)));
        constexpr unsigned chunks = (std::numeric_limits<long double>::digits + 31) / 32;
        for (unsigned i = 0; i < chunks; ++i)
        {
            fraction = std::scalbn(fraction, 32);
            const auto chunk = static_cast<std::uint64_t>(fraction);
            word(chunk);
            fraction -= static_cast<long double>(chunk);
        }
        if (fraction != 0)
            reject("wider-FP significand cannot be serialized losslessly.");
    }
    void interval(Interval x)
    {
        real(x.lo);
        real(x.hi);
    }
    void coordinates(const Point& p)
    {
        for (const auto x : p)
            interval(x);
    }
    void frame(const Frame& f)
    {
        for (const auto x : f.origin)
            real(x);
        word(static_cast<std::uint64_t>(static_cast<std::int64_t>(f.exponent)));
    }
    void authorities(const std::vector<BoundedPlaneAuthority>& keys)
    {
        word(keys.size());
        for (const auto& key : keys)
        {
            word(key.key.size());
            for (const auto x : key.key)
                word(x);
        }
    }
};
struct NumericalReader
{
    std::span<const char> bytes;
    std::size_t cursor = 0;
    explicit NumericalReader(std::span<const char> input) : bytes(input) {}
    std::uint64_t word()
    {
        if (cursor > bytes.size() || bytes.size() - cursor < 8)
            reject("truncated numerical geometry packet.");
        std::uint64_t x = 0;
        for (unsigned byte = 0; byte < 8; ++byte)
            x |= static_cast<std::uint64_t>(static_cast<unsigned char>(bytes[cursor++])) << (8 * byte);
        return x;
    }
    std::size_t count(std::size_t maximum, std::size_t element)
    {
        const auto n = word();
        if (n > maximum || (element && n > (bytes.size() - cursor) / element))
            reject("numerical packet count exceeds remaining bytes/resource cap.");
        active_work->charge(static_cast<std::size_t>(n), element);
        return static_cast<std::size_t>(n);
    }
    long double real()
    {
        const auto negative = word();
        const auto exponent_word = word();
        const auto exponent = std::bit_cast<std::int64_t>(exponent_word);
        if (negative > 1 || exponent < std::numeric_limits<long double>::min_exponent - 1 ||
            exponent > std::numeric_limits<long double>::max_exponent)
            reject("invalid numerical packet scalar exponent/sign.");
        constexpr unsigned chunks = (std::numeric_limits<long double>::digits + 31) / 32;
        long double significand = 0;
        for (unsigned i = 0; i < chunks; ++i)
        {
            const auto chunk = word();
            if (chunk > std::numeric_limits<std::uint32_t>::max())
                reject("invalid numerical packet significand.");
            significand = std::scalbn(significand, 32) + static_cast<long double>(chunk);
        }
        const auto value = std::scalbn(significand, static_cast<int>(exponent) - 32 * chunks);
        finite(value);
        if (significand != 0 && value == 0)
            reject("numerical packet loses nonzero coordinates.");
        return negative ? -value : value;
    }
    Interval interval()
    {
        const auto lo = real(), hi = real();
        return {lo, hi};
    }
    Point coordinates() { return {interval(), interval(), interval()}; }
    Frame frame()
    {
        Frame f;
        for (auto& x : f.origin)
        {
            const auto original = real();
            x = static_cast<double>(original);
            if (!std::isfinite(x) || static_cast<long double>(x) != original)
                reject("source frame anchor is not its original binary64 value.");
        }
        const auto exponent = std::bit_cast<std::int64_t>(word());
        if (exponent < -32768 || exponent > 32768)
            reject("invalid local scaling exponent.");
        f.exponent = static_cast<int>(exponent);
        return f;
    }
    std::vector<BoundedPlaneAuthority> authorities(
        const std::set<BoundedPlaneAuthority>& registry, BoundedGeometryLimits limits)
    {
        std::vector<BoundedPlaneAuthority> result;
        const auto n = count(limits.maximum_faces, sizeof(std::uint64_t));
        for (std::size_t i = 0; i < n; ++i)
        {
            BoundedPlaneAuthority key;
            const auto words = count(limits.byte_budget / sizeof(std::uint64_t), sizeof(std::uint64_t));
            if (!words)
                reject("empty numerical support authority.");
            for (std::size_t j = 0; j < words; ++j)
                key.key.push_back(word());
            if (!registry.contains(key))
                reject("foreign or absent original source support authority.");
            result.push_back(std::move(key));
        }
        std::sort(result.begin(), result.end());
        if (std::adjacent_find(result.begin(), result.end()) != result.end())
            reject("duplicate support authority in numerical packet.");
        return result;
    }
};
} // namespace
std::vector<char> BoundedConvexGeometry::encode(
    const BoundedGeometryPacketContext& context, std::size_t byte_budget) const
{
    Operation scope(impl_->limits);
    NumericalWriter writer(std::min(byte_budget, impl_->limits.byte_budget));
    writer.word(0x5346424f554e4431ULL);
    writer.word(1);
    writer.word(std::numeric_limits<long double>::digits);
    writer.word(std::numeric_limits<long double>::max_exponent);
    for (const auto pin : context.publication)
        writer.word(pin);
    writer.frame(impl_->frame);
    std::map<PlanePtr, std::size_t> plane_ids;
    std::vector<PlanePtr> planes;
    for (const auto& vertex : impl_->vertices)
        for (const auto& plane : vertex.incident)
            if (!plane_ids.contains(plane))
            {
                plane_ids.emplace(plane, planes.size());
                planes.push_back(plane);
            }
    writer.word(planes.size());
    for (const auto& plane : planes)
    {
        writer.frame(plane->frame);
        writer.coordinates(plane->normal);
        writer.coordinates(plane->anchor);
        writer.interval(plane->offset);
        writer.word(plane->input ? 1 : 0);
        if (plane->input)
        {
            for (const auto x : plane->input->normal)
                writer.real(x);
            for (const auto x : plane->input->anchor)
                writer.real(x);
            writer.real(plane->input->offset);
        }
        auto keys = plane->authorities;
        std::sort(keys.begin(), keys.end());
        writer.authorities(keys);
    }
    writer.word(impl_->vertices.size());
    for (const auto& vertex : impl_->vertices)
    {
        writer.coordinates(vertex.coordinates);
        writer.word(vertex.incident.size());
        for (const auto& plane : vertex.incident)
            writer.word(plane_ids.at(plane));
        auto keys = vertex.authorities;
        std::sort(keys.begin(), keys.end());
        writer.authorities(keys);
    }
    writer.word(impl_->faces.size());
    for (std::size_t f = 0; f < impl_->faces.size(); ++f)
    {
        const auto& face = impl_->faces[f];
        writer.word(face.size());
        for (const auto index : face)
            writer.word(index);
        writer.word(plane_ids.at(impl_->surfaces[f].plane));
        writer.word(impl_->surfaces[f].orientation > 0 ? 1 : 0);
    }
    active_work->charge(writer.bytes.size(), sizeof(char));
    return std::move(writer.bytes);
}
BoundedConvexGeometryPtr BoundedConvexGeometry::decode(std::span<const char> packet,
    const BoundedGeometryPacketContext& expected, std::span<const BoundedConvexGeometryPtr> authenticated_sources,
    BoundedGeometryLimits limits)
{
    Operation scope(limits);
    if (packet.size() > limits.byte_budget || authenticated_sources.empty())
        reject("numerical decode requires capped payload and authenticated source registry.");
    std::set<BoundedPlaneAuthority> registry;
    std::map<BoundedPlaneAuthority, PlanePtr> registry_planes;
    for (const auto& source : authenticated_sources)
    {
        if (!source)
            reject("null authenticated source registry entry.");
        for (const auto& vertex : source->impl_->vertices)
            for (const auto& plane : vertex.incident)
                for (const auto& key : plane->authorities)
                {
                    if (registry.insert(key).second)
                    {
                        active_work->charge(2, sizeof(BoundedPlaneAuthority) + 4 * sizeof(void*));
                        active_work->charge(2 * key.key.size(), sizeof(std::uint64_t));
                        registry_planes.emplace(key, plane);
                    }
                }
    }
    NumericalReader reader(packet);
    if (reader.word() != 0x5346424f554e4431ULL || reader.word() != 1 ||
        reader.word() != std::numeric_limits<long double>::digits ||
        reader.word() != std::numeric_limits<long double>::max_exponent)
        reject("unknown or incompatible numerical geometry packet.");
    for (const auto pin : expected.publication)
        if (reader.word() != pin)
            reject("stale or foreign numerical geometry publication/trial pins.");
    auto result = std::make_shared<Impl>();
    result->frame = reader.frame();
    result->limits = limits;
    std::vector<PlanePtr> planes;
    const auto plane_count = reader.count(limits.maximum_faces + limits.maximum_vertices, sizeof(PlanePtr));
    for (std::size_t i = 0; i < plane_count; ++i)
    {
        auto plane = std::make_shared<Plane>();
        plane->frame = reader.frame();
        plane->normal = reader.coordinates();
        plane->anchor = reader.coordinates();
        plane->offset = reader.interval();
        if (!nonzero(plane->normal))
            reject("numerical packet plane normal is unresolved.");
        const auto has_input = reader.word();
        if (has_input > 1)
            reject("invalid numerical plane provenance tag.");
        if (has_input)
        {
            GeometryHalfspace h;
            for (auto& x : h.normal)
            {
                const auto input = reader.real();
                x = static_cast<double>(input);
                if (static_cast<long double>(x) != input)
                    reject("plane input provenance is not original binary64.");
            }
            for (auto& x : h.anchor)
            {
                const auto input = reader.real();
                x = static_cast<double>(input);
                if (static_cast<long double>(x) != input)
                    reject("plane input anchor is not original binary64.");
            }
            const auto offset = reader.real();
            h.offset = static_cast<double>(offset);
            if (static_cast<long double>(h.offset) != offset || !nonzero(point(h.normal)))
                reject("invalid original plane input.");
            const auto expected_normal = point(h.normal);
            const auto expected_anchor = scaled(point(h.anchor) - point(plane->frame.origin), -plane->frame.exponent);
            const auto expected_offset = scaled(Interval(h.offset), -plane->frame.exponent);
            if (!same(plane->normal, expected_normal) || !same(plane->anchor, expected_anchor) ||
                plane->offset.lo != expected_offset.lo || plane->offset.hi != expected_offset.hi)
                reject("plane expression disagrees with its original input provenance.");
            plane->input = h;
        }
        plane->authorities = reader.authorities(registry, limits);
        // Known opaque identity is only an index. Its supplied numerical
        // expression must remain compatible with the pinned original support.
        for (const auto& key : plane->authorities)
        {
            const auto& original = registry_planes.at(key);
            const auto reference_anchor = transform(original->anchor, original->frame, plane->frame);
            const auto offset_residual = dot(plane->normal, reference_anchor - plane->anchor) - plane->offset;
            if (offset_residual.lo > 0 || offset_residual.hi < 0)
                reject("numerical source support expression differs from its registry.");
            const auto parallel = cross(plane->normal, original->normal);
            for (const auto component : parallel)
                if (component.lo > 0 || component.hi < 0)
                    reject("numerical source normal differs from its registry.");
        }
        planes.push_back(std::move(plane));
    }
    const auto vertices = reader.count(limits.maximum_vertices, sizeof(Vertex));
    if (vertices < 4)
        reject("numerical packet has too few vertices.");
    for (std::size_t i = 0; i < vertices; ++i)
    {
        Vertex vertex;
        vertex.coordinates = reader.coordinates();
        const auto count = reader.count(plane_count, sizeof(std::uint64_t));
        std::set<std::size_t> distinct;
        for (std::size_t j = 0; j < count; ++j)
        {
            const auto index = reader.word();
            if (index >= planes.size() || !distinct.insert(static_cast<std::size_t>(index)).second)
                reject("invalid or repeated numerical vertex/plane incidence.");
            vertex.incident.push_back(planes[index]);
        }
        vertex.authorities = reader.authorities(registry, limits);
        for (const auto& plane : vertex.incident)
        {
            const auto raw =
                dot(plane->normal, transform(vertex.coordinates, result->frame, plane->frame) - plane->anchor) -
                plane->offset;
            if (raw.lo > 0 || raw.hi < 0)
                reject("numerical vertex misses its claimed original construction support.");
        }
        for (const auto& key : vertex.authorities)
        {
            const auto& original = registry_planes.at(key);
            const auto raw = dot(original->normal,
                                 transform(vertex.coordinates, result->frame, original->frame) - original->anchor) -
                             original->offset;
            if (raw.lo > 0 || raw.hi < 0)
                reject("numerical point claims a foreign source support incidence.");
        }
        result->vertices.push_back(std::move(vertex));
    }
    const auto faces = reader.count(limits.maximum_faces, sizeof(Face));
    if (faces < 4)
        reject("numerical packet has too few faces.");
    std::map<std::pair<std::size_t, std::size_t>, std::pair<unsigned, int>> edges;
    for (std::size_t f = 0; f < faces; ++f)
    {
        Face face;
        const auto n = reader.count(vertices, sizeof(std::uint64_t));
        if (n < 3)
            reject("numerical packet face is degenerate.");
        std::set<std::size_t> distinct;
        for (std::size_t j = 0; j < n; ++j)
        {
            const auto index = reader.word();
            if (index >= vertices || !distinct.insert(static_cast<std::size_t>(index)).second)
                reject("invalid numerical face loop.");
            face.push_back(static_cast<std::size_t>(index));
        }
        const auto plane = reader.word(), outward = reader.word();
        if (plane >= planes.size() || outward > 1)
            reject("invalid numerical face support/orientation.");
        result->surfaces.push_back({planes[plane], outward ? 1 : -1});
        for (std::size_t j = 0; j < face.size(); ++j)
        {
            if (!incident(result->vertices[face[j]], planes[plane]))
                reject("numerical face lacks original vertex/support incidence.");
            const auto a = face[j], b = face[(j + 1) % face.size()];
            auto& edge = edges[std::minmax(a, b)];
            ++edge.first;
            edge.second += a < b ? 1 : -1;
        }
        result->faces.push_back(std::move(face));
    }
    if (reader.cursor != packet.size())
        reject("trailing numerical geometry packet bytes.");
    for (const auto& [edge, count] : edges)
        if (count.first != 2 || count.second != 0)
            reject("numerical packet shell is not a closed oriented manifold.");
    validate_connected_boundary(result->vertices.size(), result->faces);
    for (std::size_t f = 0; f < result->faces.size(); ++f)
    {
        const auto& surface = result->surfaces[f];
        for (const auto& vertex : result->vertices)
            if (sign(distance(vertex, result->frame, surface.plane) * Interval(surface.orientation)) > 0)
                reject("numerical packet violates convex containment.");
        const auto& face = result->faces[f];
        Point normal;
        for (std::size_t j = 1; j + 1 < face.size(); ++j)
            normal = normal + cross(result->vertices[face[j]].coordinates - result->vertices[face[0]].coordinates,
                                  result->vertices[face[j + 1]].coordinates - result->vertices[face[0]].coordinates);
        if (sign(dot(normal, surface.plane->normal) * Interval(surface.orientation)) <= 0)
            reject("numerical packet face winding is not outward.");
        for (std::size_t j = 0; j < face.size(); ++j)
        {
            const auto& a = result->vertices[face[j]].coordinates;
            const auto& b = result->vertices[face[(j + 1) % face.size()]].coordinates;
            for (const auto index : face)
                if (dot(cross(b - a, result->vertices[index].coordinates - a), normal).hi < 0)
                    reject("numerical packet contains a nonconvex face loop.");
        }
    }
    finish(*result);
    return BoundedConvexGeometryPtr(new BoundedConvexGeometry(std::move(result)));
}
