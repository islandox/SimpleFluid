/** @file ordinary/ConvexPackets.tcc
 * @brief Ordinary geometry packets with authenticated metadata and scalar coordinates.
 */
namespace SimpleFluid::PLIC::details
{
namespace ordinary_kernel SIMPLEFLUID_LOCAL
{
template<ArithmeticMode Mode>
std::vector<char> GeometryKernel<Mode>::encode(const GeometryPacketContext& context, std::size_t byte_budget) const
{
    static_assert(Mode == kernel_mode);
    Operation scope(impl_->limits);
    if (impl_->partition_proof_only)
        reject("private partition witness cannot be serialized.");
    if (!(impl_->volume.value > 0))
        reject("numerical packet requires a quantitative positive-volume certificate.");
    NumericalWriter<Mode> writer(std::min(byte_budget, impl_->limits.byte_budget));
    writer.word(0x5346424f554e4431ULL);
    writer.word(3);
    writer.word(static_cast<std::uint64_t>(arithmetic_mode()));
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
        writer.scalar(plane->offset);
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

template<ArithmeticMode Mode>
DynamicConvexGeometryPtr GeometryKernel<Mode>::decode(std::span<const char> packet,
    const GeometryPacketContext& expected, std::span<const DynamicConvexGeometryPtr> authenticated_sources,
    GeometryLimits limits)
{
    static_assert(Mode == kernel_mode);
    Operation scope(limits);
    if (packet.size() > limits.byte_budget || authenticated_sources.empty())
        reject("numerical decode requires capped payload and authenticated source registry.");
    std::set<PlaneAuthority> registry;
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
                        active_work->charge(2 * key.key.size(), sizeof(std::uint64_t));
                        registry_planes.emplace(key, plane);
                    }
                }
    }
    NumericalReader<Mode> reader(packet);
    if (reader.word() != 0x5346424f554e4431ULL || reader.word() != 3 ||
        reader.word() != static_cast<std::uint64_t>(limits.arithmetic_mode) ||
        reader.word() != std::numeric_limits<long double>::digits ||
        reader.word() != std::numeric_limits<long double>::max_exponent)
        reject("unknown or incompatible numerical geometry packet.");
    for (const auto pin : expected.publication)
        if (reader.word() != pin)
            reject("stale or foreign numerical geometry publication/trial pins.");
    auto result = std::make_shared<Impl>();
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
        plane->offset = reader.scalar();
        if (!nonzero<Mode>(plane->normal))
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
            if (static_cast<long double>(h.offset) != offset || !nonzero<Mode>(point<Mode>(h.normal)))
                reject("invalid original plane input.");
            const auto expected_normal = point<Mode>(h.normal);
            const auto expected_anchor =
                scaled<Mode>(point<Mode>(h.anchor) - point<Mode>(plane->frame.origin), -plane->frame.exponent);
            const auto expected_offset = scaled<Mode>(Scalar(h.offset), -plane->frame.exponent);
            if (!same<Mode>(plane->normal, expected_normal) || !same<Mode>(plane->anchor, expected_anchor) ||
                plane->offset.value != expected_offset.value)
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
            if (offset_residual.value > 0 || offset_residual.value < 0)
                reject("numerical source support expression differs from its registry.");
            const auto parallel = cross<Mode>(plane->normal, original->normal);
            for (const auto component : parallel)
                if (component.value > 0 || component.value < 0)
                    reject("numerical source normal differs from its registry.");
            // The packet carries no rank witness or source-point tokens. Only
            // this pinned original support may supply its existing witness,
            // and only for the identical (possibly reversed) normal values.
            if (original->normal_error && std::isfinite(*original->normal_error) && *original->normal_error >= 0)
            {
                bool identical = true, reversed = true;
                for (std::size_t axis = 0; axis < 3; ++axis)
                {
                    active_work->pair();
                    identical = identical && plane->normal[axis].value == original->normal[axis].value;
                    reversed = reversed && plane->normal[axis].value == -original->normal[axis].value;
                }
                if (identical || reversed)
                    plane->normal_error = std::max(plane->normal_error.value_or(0), *original->normal_error);
            }
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
    finish<Mode>(*result);
    return DynamicConvexGeometryPtr(new DynamicConvexGeometry(std::move(result)));
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

template<ArithmeticMode Mode> void NumericalWriter<Mode>::scalar(Scalar x)
{
    static_assert(Mode == kernel_mode);
    real(x.value);
    real(x.value);
}

template<ArithmeticMode Mode> void NumericalWriter<Mode>::coordinates(const Point& p)
{
    static_assert(Mode == kernel_mode);
    for (const auto x : p)
        scalar(x);
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
        word(key.key.size());
        for (const auto x : key.key)
            word(x);
    }
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
    const auto minimum_exponent =
        std::numeric_limits<long double>::min_exponent - std::numeric_limits<long double>::digits;
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

template<ArithmeticMode Mode> Scalar NumericalReader<Mode>::scalar()
{
    static_assert(Mode == kernel_mode);
    const auto lo = real(), hi = real();
    if (lo != hi)
        reject("ordinary packet contains an interval certificate.");
    return Scalar(lo);
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
    const std::set<PlaneAuthority>& registry, GeometryLimits limits)
{
    static_assert(Mode == kernel_mode);
    std::vector<PlaneAuthority> result;
    const auto n = count(limits.maximum_faces, sizeof(std::uint64_t));
    for (std::size_t i = 0; i < n; ++i)
    {
        PlaneAuthority key;
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

template std::vector<char> GeometryKernel<kernel_mode>::encode(
    const GeometryPacketContext& context, std::size_t byte_budget) const;
template DynamicConvexGeometryPtr GeometryKernel<kernel_mode>::decode(std::span<const char> packet,
    const GeometryPacketContext& expected, std::span<const DynamicConvexGeometryPtr> authenticated_sources,
    GeometryLimits limits);
template void NumericalWriter<kernel_mode>::word(std::uint64_t x);
template void NumericalWriter<kernel_mode>::real(long double x);
template void NumericalWriter<kernel_mode>::scalar(Scalar x);
template void NumericalWriter<kernel_mode>::coordinates(const Point& p);
template void NumericalWriter<kernel_mode>::frame(const Frame& f);
template void NumericalWriter<kernel_mode>::authorities(const std::vector<PlaneAuthority>& keys);
template std::uint64_t NumericalReader<kernel_mode>::word();
template std::size_t NumericalReader<kernel_mode>::count(std::size_t maximum, std::size_t element);
template long double NumericalReader<kernel_mode>::real();
template Scalar NumericalReader<kernel_mode>::scalar();
template Frame NumericalReader<kernel_mode>::frame();
template std::vector<PlaneAuthority> NumericalReader<kernel_mode>::authorities(
    const std::set<PlaneAuthority>& registry, GeometryLimits limits);
} // namespace ordinary_kernel SIMPLEFLUID_LOCAL
} // namespace SimpleFluid::PLIC::details
