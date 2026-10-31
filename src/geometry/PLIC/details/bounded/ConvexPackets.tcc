/** Compiled certification geometry kernels. Included only by the owning implementation unit.
 * Directed interval arithmetic and retained certificates remain confined to this
 * provider; Mode templates retain the compiled implementation boundary.
 */
namespace SimpleFluid::PLIC::details
{
namespace certification_kernel SIMPLEFLUID_LOCAL
{
template<ArithmeticMode Mode>
std::vector<char> GeometryKernel<Mode>::encode(const GeometryPacketContext& context, std::size_t byte_budget) const
{
    static_assert(Mode == kernel_mode);
    Operation scope(impl_->limits);
    if (impl_->partition_proof_only)
        reject("private partition witness cannot be serialized.");
    if (!(impl_->volume.lo() > 0))
        reject("numerical packet requires a quantitative positive-volume certificate.");
    NumericalWriter<Mode> writer(std::min(byte_budget, impl_->limits.byte_budget));
    writer.tag(GeometryTag::FiniteGeometry);
    writer.byte(4);
    writer.byte(static_cast<std::uint8_t>(arithmetic_mode()));
    writer.word(std::numeric_limits<long double>::digits);
    writer.word(std::numeric_limits<long double>::max_exponent);
    for (const auto pin : context.publication)
        writer.word(pin);
    writer.frame(impl_->frame);
    writer.frame(Frame{impl_->reference_origin, 0});
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
        writer.byte(plane->input ? 1 : 0);
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
        std::vector<PlaneAuthority> keys = vertex.authorities.values();
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
        writer.byte(impl_->surfaces[f].orientation > 0 ? 1 : 0);
    }
    active_work->charge(writer.bytes.size(), sizeof(char));
    return std::move(writer.bytes);
}

template<ArithmeticMode Mode>
DynamicConvexGeometryPtr GeometryKernel<Mode>::decode(std::span<const char> packet,
    const GeometryPacketContext& expected, std::span<const DynamicConvexGeometryPtr> authenticated_sources,
    GeometryLimits limits)
{
    static_assert(Mode == kernel_mode);
    Operation scope(limits);
    if (packet.size() > limits.byte_budget || authenticated_sources.empty())
        reject("numerical decode requires capped payload and authenticated source registry.");
    PlaneAuthorityRegistry registry;
    std::map<PlaneAuthority, PlanePtr> registry_planes;
    for (const auto& source : authenticated_sources)
    {
        if (!source)
            reject("null authenticated source registry entry.");
        require_mode(limits.arithmetic_mode, source->arithmetic_mode());
        for (const auto& vertex : checked_geometry_storage<Mode>(source->impl_)->vertices)
            for (const auto& plane : vertex.incident)
                for (const auto& key : plane->authorities)
                {
                    if (registry.insert(key).second)
                    {
                        active_work->charge(2, sizeof(PlaneAuthority) + 4 * sizeof(void*));
                        active_work->allocations.retain(key, active_work->bytes, active_work->limits.byte_budget);
                        registry_planes.emplace(key, plane);
                    }
                }
    }
    NumericalReader<Mode> reader(packet);
    if (reader.tag() != GeometryTag::FiniteGeometry || reader.byte() != 4 ||
        reader.byte() != static_cast<std::uint8_t>(limits.arithmetic_mode) ||
        reader.word() != std::numeric_limits<long double>::digits ||
        reader.word() != std::numeric_limits<long double>::max_exponent)
        reject("unknown or incompatible numerical geometry packet.");
    for (const auto pin : expected.publication)
        if (reader.word() != pin)
            reject("stale or foreign numerical geometry publication/trial pins.");
    auto result = std::make_shared<Impl>();
    result->numerical_only = true;
    result->frame = reader.frame();
    const auto reference = reader.frame();
    if (reference.exponent != 0)
        reject("invalid public moment reference frame.");
    result->reference_origin = reference.origin;
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
        if (!nonzero<Mode>(plane->normal))
            reject("numerical packet plane normal is unresolved.");
        const auto has_input = reader.byte();
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
            if (static_cast<long double>(h.offset) != offset || !nonzero<Mode>(point<Mode>(h.normal)))
                reject("invalid original plane input.");
            const auto expected_normal = point<Mode>(h.normal);
            const auto expected_anchor =
                scaled<Mode>(point<Mode>(h.anchor) - point<Mode>(plane->frame.origin), -plane->frame.exponent);
            const auto expected_offset = scaled<Mode>(Interval(h.offset), -plane->frame.exponent);
            if (!same<Mode>(plane->normal, expected_normal) || !same<Mode>(plane->anchor, expected_anchor) ||
                plane->offset.lo() != expected_offset.lo() || plane->offset.hi() != expected_offset.hi())
                reject("plane expression disagrees with its original input provenance.");
            plane->input = h;
        }
        plane->authorities = reader.authorities(registry, limits);
        // Known opaque identity is only an index. Its supplied numerical
        // expression must remain compatible with the pinned original support.
        for (const auto& key : plane->authorities)
        {
            const auto& original = registry_planes.at(key);
            const auto reference_anchor = transform<Mode>(original->anchor, original->frame, plane->frame);
            const auto offset_residual = dot<Mode>(plane->normal, reference_anchor - plane->anchor) - plane->offset;
            if (offset_residual.lo() > 0 || offset_residual.hi() < 0)
                reject("numerical source support expression differs from its registry.");
            const auto parallel = cross<Mode>(plane->normal, original->normal);
            for (const auto component : parallel)
                if (component.lo() > 0 || component.hi() < 0)
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
        vertex.numerical_only = true;
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
            const auto raw = raw_distance<Mode>(vertex, result->frame, plane);
            if (raw.lo() > 0 || raw.hi() < 0)
                reject("numerical vertex misses its claimed original construction support.");
        }
        for (const auto& key : vertex.authorities)
        {
            const auto& original = registry_planes.at(key);
            const auto raw = raw_distance<Mode>(vertex, result->frame, original);
            if (raw.lo() > 0 || raw.hi() < 0)
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
        const auto plane = reader.word();
        const auto outward = reader.byte();
        if (plane >= planes.size() || outward > 1)
            reject("invalid numerical face support/orientation.");
        result->surfaces.push_back({planes[plane], outward ? 1 : -1});
        for (std::size_t j = 0; j < face.size(); ++j)
        {
            if (!incident<Mode>(result->vertices[face[j]], planes[plane]))
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
    validate_connected_boundary<Mode>(result->vertices.size(), result->faces);
    for (std::size_t f = 0; f < result->faces.size(); ++f)
    {
        const auto& surface = result->surfaces[f];
        for (const auto& vertex : result->vertices)
            if (sign<Mode>(distance<Mode>(vertex, result->frame, surface.plane) * Interval(surface.orientation)) > 0)
                reject("numerical packet violates convex containment.");
        const auto& face = result->faces[f];
        Point normal;
        for (std::size_t j = 1; j + 1 < face.size(); ++j)
            normal = normal + cross<Mode>(result->vertices[face[j]].coordinates - result->vertices[face[0]].coordinates,
                                  result->vertices[face[j + 1]].coordinates - result->vertices[face[0]].coordinates);
        if (sign<Mode>(dot<Mode>(normal, surface.plane->normal) * Interval(surface.orientation)) <= 0)
            reject("numerical packet face winding is not outward.");
        for (std::size_t j = 0; j < face.size(); ++j)
        {
            const auto& a = result->vertices[face[j]].coordinates;
            const auto& b = result->vertices[face[(j + 1) % face.size()]].coordinates;
            for (const auto index : face)
                if (dot<Mode>(cross<Mode>(b - a, result->vertices[index].coordinates - a), normal).hi() < 0)
                    reject("numerical packet contains a nonconvex face loop.");
        }
    }
    finish<Mode>(*result);
    return DynamicConvexGeometryPtr(new DynamicConvexGeometry(std::move(result)));
}

template<ArithmeticMode Mode> void NumericalWriter<Mode>::byte(std::uint8_t value)
{
    static_assert(Mode == kernel_mode);
    if (bytes.size() >= limit)
        reject("numerical packet exceeds storage cap.");
    bytes.push_back(static_cast<char>(value));
}

template<ArithmeticMode Mode> void NumericalWriter<Mode>::word(std::uint64_t x)
{
    static_assert(Mode == kernel_mode);
    if (bytes.size() > limit || limit - bytes.size() < 8)
        reject("numerical packet exceeds storage cap.");
    for (unsigned byte = 0; byte < 8; ++byte)
        bytes.push_back(static_cast<char>((x >> (8 * byte)) & 255));
}

template<ArithmeticMode Mode> void NumericalWriter<Mode>::real(long double x)
{
    static_assert(Mode == kernel_mode);
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

template<ArithmeticMode Mode> void NumericalWriter<Mode>::interval(Interval x)
{
    static_assert(Mode == kernel_mode);
    real(x.lo());
    real(x.hi());
}

template<ArithmeticMode Mode> void NumericalWriter<Mode>::coordinates(const Point& p)
{
    static_assert(Mode == kernel_mode);
    for (const auto x : p)
        interval(x);
}

template<ArithmeticMode Mode> void NumericalWriter<Mode>::frame(const Frame& f)
{
    static_assert(Mode == kernel_mode);
    for (const auto x : f.origin)
        real(x);
    word(static_cast<std::uint64_t>(static_cast<std::int64_t>(f.exponent)));
}

template<ArithmeticMode Mode> void NumericalWriter<Mode>::authorities(const std::vector<PlaneAuthority>& keys)
{
    static_assert(Mode == kernel_mode);
    word(keys.size());
    for (const auto& key : keys)
    {
        word(key.word_count());
        key.emit_words(this,
            [](void* context, std::uint64_t value) { static_cast<NumericalWriter<Mode>*>(context)->word(value); });
    }
}

template<ArithmeticMode Mode> std::uint8_t NumericalReader<Mode>::byte()
{
    static_assert(Mode == kernel_mode);
    if (cursor >= bytes.size())
        reject("truncated numerical geometry tag.");
    return static_cast<std::uint8_t>(bytes[cursor++]);
}

template<ArithmeticMode Mode> std::uint64_t NumericalReader<Mode>::word()
{
    static_assert(Mode == kernel_mode);
    if (cursor > bytes.size() || bytes.size() - cursor < 8)
        reject("truncated numerical geometry packet.");
    std::uint64_t x = 0;
    for (unsigned byte = 0; byte < 8; ++byte)
        x |= static_cast<std::uint64_t>(static_cast<unsigned char>(bytes[cursor++])) << (8 * byte);
    return x;
}

template<ArithmeticMode Mode> std::size_t NumericalReader<Mode>::count(std::size_t maximum, std::size_t element)
{
    static_assert(Mode == kernel_mode);
    const auto n = word();
    if (n > maximum || (element && n > (bytes.size() - cursor) / element))
        reject("numerical packet count exceeds remaining bytes/resource cap.");
    active_work->charge(static_cast<std::size_t>(n), element);
    return static_cast<std::size_t>(n);
}

template<ArithmeticMode Mode> long double NumericalReader<Mode>::real()
{
    static_assert(Mode == kernel_mode);
    const auto negative = word();
    const auto exponent_word = word();
    const auto exponent = std::bit_cast<std::int64_t>(exponent_word);
    const auto minimum_exponent = std::numeric_limits<long double>::min_exponent - 1;
    if (negative > 1 || exponent < minimum_exponent || exponent > std::numeric_limits<long double>::max_exponent)
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

template<ArithmeticMode Mode> Interval NumericalReader<Mode>::interval()
{
    static_assert(Mode == kernel_mode);
    const auto lo = real(), hi = real();
    return {lo, hi};
}

template<ArithmeticMode Mode> Frame NumericalReader<Mode>::frame()
{
    static_assert(Mode == kernel_mode);
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

template<ArithmeticMode Mode>
std::vector<PlaneAuthority> NumericalReader<Mode>::authorities(
    const PlaneAuthorityRegistry& registry, GeometryLimits limits)
{
    static_assert(Mode == kernel_mode);
    std::vector<PlaneAuthority> result;
    const auto n = count(limits.maximum_faces, sizeof(std::uint64_t));
    // count() charges the wire-count lower bound. Admit the complete retained
    // handle buffer separately from each temporary serialized word buffer.
    active_work->charge(n, sizeof(PlaneAuthority) - sizeof(std::uint64_t));
    result.reserve(n);
    if (result.capacity() > n)
        active_work->charge(result.capacity() - n, sizeof(PlaneAuthority));
    for (std::size_t i = 0; i < n; ++i)
    {
        const auto words = count(limits.byte_budget / sizeof(std::uint64_t), sizeof(std::uint64_t));
        if (!words)
            reject("empty numerical support authority.");
        std::vector<std::uint64_t> serialized;
        serialized.reserve(words);
        if (serialized.capacity() > words)
            active_work->charge(serialized.capacity() - words, sizeof(std::uint64_t));
        for (std::size_t j = 0; j < words; ++j)
            serialized.push_back(word());
        auto authenticated = OriginalSupportAuthorityAccess::authenticate(serialized, registry);
        active_work->allocations.retain(authenticated, active_work->bytes, active_work->limits.byte_budget);
        result.push_back(std::move(authenticated));
    }
    std::sort(result.begin(), result.end());
    if (std::adjacent_find(result.begin(), result.end()) != result.end())
        reject("duplicate support authority in numerical packet.");
    return result;
}

template std::vector<char> GeometryKernel<kernel_mode>::encode(
    const GeometryPacketContext& context, std::size_t byte_budget) const;
template DynamicConvexGeometryPtr GeometryKernel<kernel_mode>::decode(std::span<const char> packet,
    const GeometryPacketContext& expected, std::span<const DynamicConvexGeometryPtr> authenticated_sources,
    GeometryLimits limits);
template void NumericalWriter<kernel_mode>::byte(std::uint8_t);
template std::uint8_t NumericalReader<kernel_mode>::byte();
template void NumericalWriter<kernel_mode>::word(std::uint64_t x);
template void NumericalWriter<kernel_mode>::real(long double x);
template void NumericalWriter<kernel_mode>::interval(Interval x);
template void NumericalWriter<kernel_mode>::coordinates(const Point& p);
template void NumericalWriter<kernel_mode>::frame(const Frame& f);
template void NumericalWriter<kernel_mode>::authorities(const std::vector<PlaneAuthority>& keys);
template std::uint64_t NumericalReader<kernel_mode>::word();
template std::size_t NumericalReader<kernel_mode>::count(std::size_t maximum, std::size_t element);
template long double NumericalReader<kernel_mode>::real();
template Interval NumericalReader<kernel_mode>::interval();
template Frame NumericalReader<kernel_mode>::frame();
template std::vector<PlaneAuthority> NumericalReader<kernel_mode>::authorities(
    const PlaneAuthorityRegistry& registry, GeometryLimits limits);
} // namespace certification_kernel SIMPLEFLUID_LOCAL
} // namespace SimpleFluid::PLIC::details
