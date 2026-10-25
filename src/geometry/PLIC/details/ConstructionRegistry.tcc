/** @file ConstructionRegistry.tcc
 * @brief Version 4 finite construction replay, rooted in independent authority.
 */
namespace SimpleFluid::PLIC
{
namespace construction_detail SIMPLEFLUID_LOCAL
{
constexpr std::uint64_t magic = 0x5346424f554e4431ULL;
[[noreturn]] void fail(GeometryErrorCode code, const char* message)
{
    ++details::statistics.rejections;
    throw GeometryError(code, message);
}
template<ArithmeticMode Mode>
using Kernel = std::conditional_t<Mode == ArithmeticMode::Ordinary, details::ordinary_kernel::GeometryKernel<Mode>,
    details::certification_kernel::GeometryKernel<Mode>>;
template<ArithmeticMode Mode>
using Writer = std::conditional_t<Mode == ArithmeticMode::Ordinary, details::ordinary_kernel::NumericalWriter<Mode>,
    details::certification_kernel::NumericalWriter<Mode>>;
template<ArithmeticMode Mode>
using Reader = std::conditional_t<Mode == ArithmeticMode::Ordinary, details::ordinary_kernel::NumericalReader<Mode>,
    details::certification_kernel::NumericalReader<Mode>>;
template<ArithmeticMode Mode> GeometryLimits limits_for(GeometryLimits limits)
{
    limits.arithmetic_mode = Mode;
    return limits;
}
void charge(std::size_t& bytes, std::size_t amount, std::size_t limit)
{
    if (amount > limit - std::min(bytes, limit))
        fail(GeometryErrorCode::ResourceLimit, "Construction registry byte budget exhausted.");
    bytes += amount;
}
template<class T> void append(std::vector<T>& values, T value)
{
    if (values.size() == values.capacity())
    {
        const auto old = values.capacity();
        if (old > std::numeric_limits<std::size_t>::max() / 2)
            fail(GeometryErrorCode::ResourceLimit, "Construction dependency capacity overflows.");
        const auto capacity = old ? 2 * old : 1;
        details::active_work->charge(capacity, sizeof(T));
        values.reserve(capacity);
    }
    values.push_back(std::move(value));
}
template<ArithmeticMode Mode> void write_header(Writer<Mode>& writer, const GeometryPacketContext& context)
{
    writer.word(magic);
    writer.word(4);
    writer.word(static_cast<std::uint64_t>(Mode));
    writer.word(std::numeric_limits<long double>::digits);
    writer.word(std::numeric_limits<long double>::max_exponent);
    for (const auto pin : context.publication)
        writer.word(pin);
}
template<ArithmeticMode Mode> void read_header(Reader<Mode>& reader, const GeometryPacketContext& expected)
{
    if (reader.word() != magic || reader.word() != 4 || reader.word() != static_cast<std::uint64_t>(Mode) ||
        reader.word() != std::numeric_limits<long double>::digits ||
        reader.word() != std::numeric_limits<long double>::max_exponent)
        fail(GeometryErrorCode::ProvenanceMismatch, "Incompatible construction packet mode or floating-point format.");
    for (const auto pin : expected.publication)
        if (reader.word() != pin)
            fail(GeometryErrorCode::ProvenanceMismatch, "Stale or foreign construction publication/trial pins.");
}
template<ArithmeticMode Mode> void write_blob(Writer<Mode>& writer, std::span<const char> value)
{
    writer.word(value.size());
    if (value.size() > writer.limit - std::min(writer.bytes.size(), writer.limit))
        fail(GeometryErrorCode::ResourceLimit, "Construction packet byte budget exhausted.");
    writer.bytes.insert(writer.bytes.end(), value.begin(), value.end());
}
template<ArithmeticMode Mode> std::vector<char> read_blob(Reader<Mode>& reader, std::size_t cap)
{
    const auto size = reader.count(cap, 1);
    if (size > reader.bytes.size() - reader.cursor)
        fail(GeometryErrorCode::ProvenanceMismatch, "Truncated construction packet byte field.");
    std::vector<char> result(reader.bytes.begin() + reader.cursor, reader.bytes.begin() + reader.cursor + size);
    reader.cursor += size;
    return result;
}
template<ArithmeticMode Mode> void write_key(Writer<Mode>& writer, const ConstructionRootKey& key)
{
    writer.word(static_cast<std::uint64_t>(key.kind));
    writer.word(key.entity);
    writer.word(key.revision);
    writer.word(key.slot);
    write_blob<Mode>(writer, key.identity);
}
template<ArithmeticMode Mode> ConstructionRootKey read_key(Reader<Mode>& reader, std::size_t cap)
{
    const auto kind = reader.word();
    if (kind < 1 || kind > 4)
        fail(GeometryErrorCode::ProvenanceMismatch, "Unknown construction root kind.");
    ConstructionRootKey key{static_cast<ConstructionRootKey::Kind>(kind), reader.word(), reader.word(), reader.word(),
        read_blob<Mode>(reader, cap)};
    if (!key.identity.empty() || (key.kind == ConstructionRootKey::Kind::Native && key.slot != 0) ||
        (key.kind == ConstructionRootKey::Kind::Sweep && key.revision != 0) ||
        (key.kind == ConstructionRootKey::Kind::InputPlane && (key.revision != 0 || key.slot != 0)))
        fail(GeometryErrorCode::ProvenanceMismatch, "Noncanonical construction root declaration.");
    return key;
}
bool same_input(const GeometryHalfspace& a, const GeometryHalfspace& b)
{
    return a.normal == b.normal && a.anchor == b.anchor && a.offset == b.offset;
}
std::array<double, 7> input_key(const GeometryHalfspace& plane)
{
    return {plane.normal[0], plane.normal[1], plane.normal[2], plane.anchor[0], plane.anchor[1], plane.anchor[2],
        plane.offset};
}
struct Reference
{
    std::size_t node = 0;
    unsigned side = 0;
};
struct PlaneReference
{
    ConstructionRootKey key;
    std::size_t face = 0;
    details::Frame frame;
    bool input = false;
};
struct Record
{
    enum class Kind : std::uint64_t
    {
        Root = 1,
        Clip = 2,
        Rebase = 3
    };
    Kind kind = Kind::Root;
    ConstructionRootKey root;
    Reference parent;
    PlaneReference plane;
    int orientation = 1;
    bool measure_section = false, construct_outside = true, partition_proof = false;
};
template<ArithmeticMode Mode> void write_reference(Writer<Mode>& writer, Reference reference)
{
    writer.word(reference.node);
    writer.word(reference.side);
}
template<ArithmeticMode Mode> Reference read_reference(Reader<Mode>& reader, std::size_t before)
{
    const auto node = reader.word(), side = reader.word();
    if (node >= before || side > 1)
        fail(GeometryErrorCode::ProvenanceMismatch, "Forward, cyclic or invalid construction dependency.");
    return {static_cast<std::size_t>(node), static_cast<unsigned>(side)};
}
} // namespace construction_detail SIMPLEFLUID_LOCAL

template<ArithmeticMode Mode> struct ConstructionRegistry<Mode>::Impl
{
    GeometryPacketContext context;
    GeometryLimits limits;
    std::size_t bytes = sizeof(Impl);
    std::map<ConstructionRootKey, ConvexGeometryPtr<Mode>> geometries;
    std::map<ConstructionRootKey, GeometryHalfspace> planes;
    std::map<const DynamicConvexGeometry::Impl*, ConstructionRootKey> geometry_roots;
    std::map<const details::GeometryConstruction*, ConstructionRootKey> node_roots;
    std::map<const void*, construction_detail::PlaneReference> plane_roots;
    std::map<std::array<double, 7>, ConstructionRootKey> input_roots;
};

template<ArithmeticMode Mode>
ConstructionRegistry<Mode>::ConstructionRegistry(GeometryPacketContext context, GeometryLimits limits)
    : impl_(std::make_shared<Impl>())
{
    static_assert(Mode == ArithmeticMode::Ordinary || Mode == ArithmeticMode::BoundedCertification);
    if (!context.publication[3])
        construction_detail::fail(
            GeometryErrorCode::ProvenanceMismatch, "Construction registry requires a trial identity.");
    impl_->context = context;
    impl_->limits = construction_detail::limits_for<Mode>(limits);
    if (impl_->bytes > limits.byte_budget)
        construction_detail::fail(GeometryErrorCode::ResourceLimit, "Construction registry byte budget exhausted.");
}
template<ArithmeticMode Mode>
void ConstructionRegistry<Mode>::register_geometry(ConstructionRootKey key, ConvexGeometryPtr<Mode> geometry)
{
    if (!geometry)
        construction_detail::fail(GeometryErrorCode::ProvenanceMismatch, "Null authenticated construction root.");
    const auto& source = DynamicGeometryAccess::implementation(*geometry->dynamic_geometry());
    details::require_mode(source->limits.arithmetic_mode, Mode);
    if (source->numerical_only)
        construction_detail::fail(GeometryErrorCode::ProvenanceMismatch,
            "Numerical-only packet geometry is not an independently authenticated construction root.");
    if (source->partition_proof_only)
        construction_detail::fail(
            GeometryErrorCode::ProvenanceMismatch, "A private witness cannot be a construction root.");
    if (const auto found = impl_->geometries.find(key); found != impl_->geometries.end())
    {
        if (found->second->storage_identity() != geometry->storage_identity())
            construction_detail::fail(
                GeometryErrorCode::ProvenanceMismatch, "Conflicting authenticated construction root.");
        return;
    }
    const auto stored = details::checked_geometry_storage<Mode>(source);
    auto preflight = impl_->bytes;
    construction_detail::charge(preflight, sizeof(typename decltype(impl_->geometries)::value_type) + 4 * sizeof(void*),
        impl_->limits.byte_budget);
    if (!impl_->geometry_roots.contains(source.get()))
        construction_detail::charge(preflight,
            sizeof(typename decltype(impl_->geometry_roots)::value_type) + 4 * sizeof(void*),
            impl_->limits.byte_budget);
    if (source->construction && !impl_->node_roots.contains(source->construction.get()))
        construction_detail::charge(preflight,
            sizeof(typename decltype(impl_->node_roots)::value_type) + 4 * sizeof(void*), impl_->limits.byte_budget);
    for (const auto& surface : stored->surfaces)
        if (!impl_->plane_roots.contains(surface.plane.get()))
            construction_detail::charge(preflight,
                sizeof(typename decltype(impl_->plane_roots)::value_type) + 4 * sizeof(void*),
                impl_->limits.byte_budget);
    construction_detail::charge(impl_->bytes,
        sizeof(typename decltype(impl_->geometries)::value_type) + 4 * sizeof(void*), impl_->limits.byte_budget);
    impl_->geometries.emplace(key, geometry);
    const auto index = [&](auto& map, const auto& identity, auto value)
    {
        const auto found = map.find(identity);
        if (found == map.end())
        {
            construction_detail::charge(impl_->bytes,
                sizeof(typename std::decay_t<decltype(map)>::value_type) + 4 * sizeof(void*),
                impl_->limits.byte_budget);
            map.emplace(identity, std::move(value));
        }
        else
        {
            if constexpr (std::is_same_v<std::decay_t<decltype(value)>, ConstructionRootKey>)
            {
                if (value < found->second)
                    found->second = std::move(value);
            }
            else if (value.key < found->second.key)
                found->second = std::move(value);
        }
    };
    index(impl_->geometry_roots, source.get(), key);
    if (source->construction)
        index(impl_->node_roots, source->construction.get(), key);
    for (std::size_t face = 0; face < stored->surfaces.size(); ++face)
        index(impl_->plane_roots, stored->surfaces[face].plane.get(),
            construction_detail::PlaneReference{key, face, {}, false});
}
template<ArithmeticMode Mode>
void ConstructionRegistry<Mode>::register_native(
    std::uint64_t gid, std::uint64_t revision, ConvexGeometryPtr<Mode> geometry)
{
    register_geometry({ConstructionRootKey::Kind::Native, gid, revision, 0, {}}, std::move(geometry));
}
template<ArithmeticMode Mode>
void ConstructionRegistry<Mode>::register_sweep(
    std::uint64_t face, std::uint64_t ordinal, ConvexGeometryPtr<Mode> geometry)
{
    register_geometry({ConstructionRootKey::Kind::Sweep, face, 0, ordinal, {}}, std::move(geometry));
}
template<ArithmeticMode Mode>
void ConstructionRegistry<Mode>::register_plane(std::uint64_t id, const GeometryHalfspace& plane)
{
    bool nonzero = false;
    for (std::size_t axis = 0; axis < 3; ++axis)
    {
        if (!std::isfinite(plane.normal[axis]) || !std::isfinite(plane.anchor[axis]))
            construction_detail::fail(GeometryErrorCode::InvalidInput, "Nonfinite registered plane.");
        nonzero |= plane.normal[axis] != 0;
    }
    if (!nonzero || !std::isfinite(plane.offset))
        construction_detail::fail(GeometryErrorCode::InvalidInput, "Invalid registered plane.");
    const ConstructionRootKey key{ConstructionRootKey::Kind::InputPlane, id, 0, 0, {}};
    if (const auto found = impl_->planes.find(key); found != impl_->planes.end())
    {
        if (!construction_detail::same_input(found->second, plane))
            construction_detail::fail(GeometryErrorCode::ProvenanceMismatch, "Conflicting authenticated plane root.");
        return;
    }
    const auto input = construction_detail::input_key(plane);
    auto preflight = impl_->bytes;
    construction_detail::charge(
        preflight, sizeof(typename decltype(impl_->planes)::value_type) + 4 * sizeof(void*), impl_->limits.byte_budget);
    if (!impl_->input_roots.contains(input))
        construction_detail::charge(preflight,
            sizeof(typename decltype(impl_->input_roots)::value_type) + 4 * sizeof(void*), impl_->limits.byte_budget);
    construction_detail::charge(impl_->bytes, sizeof(typename decltype(impl_->planes)::value_type) + 4 * sizeof(void*),
        impl_->limits.byte_budget);
    impl_->planes.emplace(key, plane);
    const auto found = impl_->input_roots.find(input);
    if (found == impl_->input_roots.end())
    {
        construction_detail::charge(impl_->bytes,
            sizeof(typename decltype(impl_->input_roots)::value_type) + 4 * sizeof(void*), impl_->limits.byte_budget);
        impl_->input_roots.emplace(input, key);
    }
    else if (key < found->second)
        found->second = key;
}
template<ArithmeticMode Mode> bool ConstructionRegistry<Mode>::contains(const ConstructionRootKey& key) const
{
    return impl_->geometries.contains(key) || impl_->planes.contains(key);
}
template<ArithmeticMode Mode>
bool ConstructionRegistry<Mode>::derived_from(
    const ConvexCellGeometry<Mode>& geometry, const ConstructionRootKey& key) const
{
    details::Operation scope(impl_->limits);
    const auto found = impl_->geometries.find(key);
    if (found == impl_->geometries.end())
        return false;
    const auto& target = DynamicGeometryAccess::implementation(*found->second->dynamic_geometry());
    const auto& source = DynamicGeometryAccess::implementation(*geometry.dynamic_geometry());
    if (source->numerical_only || target->numerical_only)
        return false;
    if (source == target)
        return true;
    for (auto node = source->construction; node;)
    {
        details::active_work->pair();
        if (target->construction && node == target->construction)
            return true;
        if (node->kind == details::GeometryConstruction::Kind::Source)
            return node->source == target;
        if (node->kind == details::GeometryConstruction::Kind::Clip)
        {
            if (!node->clip || (node->outside && !node->clip->construct_outside))
                return false;
            node = node->clip->parent;
        }
        else if (node->kind == details::GeometryConstruction::Kind::Rebase)
            node = node->parent;
        else
            return false;
    }
    return false;
}
template<ArithmeticMode Mode> const GeometryPacketContext& ConstructionRegistry<Mode>::context() const noexcept
{
    return impl_->context;
}
template<ArithmeticMode Mode> std::size_t ConstructionRegistry<Mode>::storage_bytes() const noexcept
{
    return impl_->bytes;
}
template<ArithmeticMode Mode>
std::vector<ConstructionRootKey> ConstructionRegistry<Mode>::required_roots(
    std::span<const char> packet, const GeometryPacketContext& context, GeometryLimits limits)
{
    details::Operation scope(construction_detail::limits_for<Mode>(limits));
    if (packet.size() > limits.byte_budget)
        construction_detail::fail(GeometryErrorCode::ResourceLimit, "Construction packet exceeds byte budget.");
    construction_detail::Reader<Mode> reader(packet);
    construction_detail::read_header<Mode>(reader, context);
    const auto count = reader.count(limits.byte_budget / sizeof(ConstructionRootKey), sizeof(ConstructionRootKey));
    std::vector<ConstructionRootKey> result;
    std::set<ConstructionRootKey> seen;
    result.reserve(count);
    for (std::size_t i = 0; i < count; ++i)
    {
        auto key = construction_detail::read_key<Mode>(reader, limits.byte_budget);
        if (!seen.insert(key).second)
            construction_detail::fail(
                GeometryErrorCode::ProvenanceMismatch, "Duplicate construction root declaration.");
        result.push_back(std::move(key));
    }
    return result;
}

namespace construction_detail SIMPLEFLUID_LOCAL
{
template<ArithmeticMode Mode> struct EncodePlan
{
    std::vector<ConstructionRootKey> roots;
    std::vector<Record> records;
    Reference terminal;
};
template<ArithmeticMode Mode>
EncodePlan<Mode> construction_plan(const std::shared_ptr<const DynamicConvexGeometry::Impl>& terminal,
    const typename ConstructionRegistry<Mode>::Impl& registry)
{
    using Node = details::GeometryConstruction;
    using Clip = details::GeometryClipConstruction;
    using Plane = details::StoredPlane<Mode>;
    const auto& geometry_roots = registry.geometry_roots;
    const auto& node_roots = registry.node_roots;
    const auto& plane_roots = registry.plane_roots;
    EncodePlan<Mode> result;
    std::map<ConstructionRootKey, std::size_t> root_ids;
    const auto root_id = [&](const ConstructionRootKey& key)
    {
        if (!root_ids.contains(key))
        {
            details::active_work->charge(1, sizeof(ConstructionRootKey) + sizeof(std::size_t) + 4 * sizeof(void*));
            root_ids.emplace(key, result.roots.size());
            append(result.roots, key);
        }
    };
    const auto input_root = [&](const GeometryHalfspace& input)
    {
        details::active_work->pair();
        const auto found = registry.input_roots.find(input_key(input));
        if (found != registry.input_roots.end())
            return found->second;
        fail(GeometryErrorCode::ProvenanceMismatch, "Construction uses an unregistered input plane.");
    };
    std::map<ConstructionRootKey, Reference> emitted_roots;
    const auto emit_root = [&](const ConstructionRootKey& key)
    {
        if (const auto found = emitted_roots.find(key); found != emitted_roots.end())
            return found->second;
        root_id(key);
        const Reference reference{result.records.size(), 0};
        Record record;
        record.root = key;
        append(result.records, std::move(record));
        details::active_work->charge(1, sizeof(ConstructionRootKey) + sizeof(Reference) + 4 * sizeof(void*));
        emitted_roots.emplace(key, reference);
        return reference;
    };
    if (const auto root = geometry_roots.find(terminal.get()); root != geometry_roots.end())
    {
        result.terminal = emit_root(root->second);
        return result;
    }
    if (!terminal->construction)
        fail(GeometryErrorCode::ProvenanceMismatch, "Geometry has no authenticated construction recipe.");
    std::map<const Node*, Reference> emitted;
    std::map<const Clip*, std::size_t> clips;
    std::set<const Node*> visiting;
    std::vector<std::pair<std::shared_ptr<const Node>, bool>> pending;
    append(pending, std::pair{terminal->construction, false});
    while (!pending.empty())
    {
        auto [node, finish] = std::move(pending.back());
        pending.pop_back();
        details::active_work->pair();
        if (!node)
            fail(GeometryErrorCode::ProvenanceMismatch, "Missing construction dependency.");
        if (emitted.contains(node.get()))
            continue;
        if (const auto root = node_roots.find(node.get()); root != node_roots.end())
        {
            emitted.emplace(node.get(), emit_root(root->second));
            continue;
        }
        if (node->kind == Node::Kind::Source)
        {
            const auto& source = node->source;
            const auto found = geometry_roots.find(source.get());
            if (!source || found == geometry_roots.end())
                fail(GeometryErrorCode::ProvenanceMismatch,
                    "Construction source is absent from the authenticated registry.");
            emitted.emplace(node.get(), emit_root(found->second));
            continue;
        }
        const auto parent = node->kind == Node::Kind::Clip && node->clip ? node->clip->parent : node->parent;
        if (!finish)
        {
            details::active_work->charge(1, sizeof(const Node*) + 4 * sizeof(void*));
            if (!visiting.insert(node.get()).second)
                fail(GeometryErrorCode::ProvenanceMismatch, "Cyclic local construction recipe.");
            append(pending, std::pair{node, true});
            if (!parent || !emitted.contains(parent.get()))
                append(pending, std::pair{parent, false});
            continue;
        }
        visiting.erase(node.get());
        if (!parent || !emitted.contains(parent.get()))
            fail(GeometryErrorCode::ProvenanceMismatch, "Unresolved construction predecessor.");
        Record record;
        record.parent = emitted.at(parent.get());
        Reference reference{result.records.size(), 0};
        if (node->kind == Node::Kind::Rebase)
        {
            record.kind = Record::Kind::Rebase;
            record.root = input_root(node->rebase_input);
            root_id(record.root);
        }
        else if (node->kind == Node::Kind::Clip && node->clip)
        {
            const auto& clip = *node->clip;
            details::require_mode(clip.mode, Mode);
            reference.side = node->outside ? 1 : 0;
            if (reference.side && !clip.construct_outside)
                fail(GeometryErrorCode::ProvenanceMismatch, "Construction references an unbuilt complement.");
            if (const auto cached = clips.find(node->clip.get()); cached != clips.end())
            {
                reference.node = cached->second;
                emitted.emplace(node.get(), reference);
                continue;
            }
            record.kind = Record::Kind::Clip;
            record.orientation = clip.orientation;
            record.measure_section = clip.measure_section;
            record.construct_outside = clip.construct_outside;
            record.partition_proof = clip.partition_proof;
            if (const auto root = plane_roots.find(clip.plane.get()); root != plane_roots.end())
                record.plane = root->second;
            else
            {
                const auto plane = std::static_pointer_cast<const Plane>(clip.plane);
                if (!plane || !plane->input)
                    fail(GeometryErrorCode::ProvenanceMismatch,
                        "Cut support is absent from the authenticated registry.");
                record.plane = {input_root(*plane->input), 0, plane->frame, true};
            }
            root_id(record.plane.key);
            details::active_work->charge(1, sizeof(const Clip*) + sizeof(std::size_t) + 4 * sizeof(void*));
            clips.emplace(node->clip.get(), reference.node);
        }
        else
            fail(GeometryErrorCode::ProvenanceMismatch, "Unknown local construction recipe.");
        append(result.records, std::move(record));
        details::active_work->charge(1, sizeof(const Node*) + sizeof(Reference) + 4 * sizeof(void*));
        emitted.emplace(node.get(), reference);
    }
    result.terminal = emitted.at(terminal->construction.get());
    return result;
}

template<ArithmeticMode Mode>
DynamicConvexGeometryPtr rebase_geometry(const DynamicConvexGeometryPtr& source, const GeometryHalfspace& input)
{
    const auto& original = DynamicGeometryAccess::implementation(*source);
    const auto typed = details::checked_geometry_storage<Mode>(original);
    details::active_work->charge(source->template storage_bytes<Mode>(), 1);
    auto rebased = std::make_shared<details::GeometryStorage<Mode>>(*typed);
    rebased->frame.origin = input.anchor;
    rebased->moment_cache = std::make_shared<details::StoredMomentCache<Mode>>();
    for (auto& vertex : rebased->vertices)
    {
        if constexpr (Mode == ArithmeticMode::Ordinary)
            vertex.coordinates =
                details::ordinary_kernel::transform<Mode>(vertex.coordinates, original->frame, rebased->frame);
        else
            vertex.coordinates =
                details::certification_kernel::transform<Mode>(vertex.coordinates, original->frame, rebased->frame);
    }
    if constexpr (Mode == ArithmeticMode::Ordinary)
        rebased->bounds = details::ordinary_kernel::transform<Mode>(original->bounds, original->frame, rebased->frame);
    else
        rebased->bounds =
            details::certification_kernel::transform<Mode>(original->bounds, original->frame, rebased->frame);
    details::active_work->charge(1, sizeof(details::GeometryConstruction));
    auto recipe = std::make_shared<details::GeometryConstruction>();
    recipe->kind = details::GeometryConstruction::Kind::Rebase;
    recipe->parent = details::construction_source<Mode>(original);
    recipe->rebase_input = input;
    rebased->construction = std::move(recipe);
    ++details::statistics.constructions;
    return DynamicGeometryAccess::geometry(std::move(rebased));
}

template<ArithmeticMode Mode>
std::shared_ptr<const DynamicConvexGeometry::Impl> input_plane_source(
    const GeometryHalfspace& input, details::Frame frame, GeometryLimits limits)
{
    using Storage = details::GeometryStorage<Mode>;
    using Plane = details::StoredPlane<Mode>;
    details::active_work->charge(1, sizeof(Storage) + sizeof(Plane));
    auto source = std::make_shared<Storage>();
    source->frame = frame;
    source->limits = limits_for<Mode>(limits);
    auto plane = std::make_shared<Plane>();
    plane->frame = frame;
    if constexpr (Mode == ArithmeticMode::Ordinary)
    {
        using namespace details::ordinary_kernel;
        plane->normal = point<Mode>(input.normal);
        plane->anchor = scaled<Mode>(point<Mode>(input.anchor) - point<Mode>(frame.origin), -frame.exponent);
        plane->offset = scaled<Mode>(Scalar(input.offset), -frame.exponent);
        plane->native_support =
            details::ordinary_native::input_support_packet(input.normal, input.anchor, input.offset);
    }
    else
    {
        using namespace details::certification_kernel;
        plane->normal = point<Mode>(input.normal);
        plane->anchor = scaled<Mode>(point<Mode>(input.anchor) - point<Mode>(frame.origin), -frame.exponent);
        plane->offset = scaled<Mode>(Interval(input.offset), -frame.exponent);
    }
    plane->input = input;
    source->surfaces.push_back({std::move(plane), 1});
    return source;
}
} // namespace construction_detail SIMPLEFLUID_LOCAL

template<ArithmeticMode Mode>
std::vector<char> ConvexCellGeometry<Mode>::encode(
    const ConstructionRegistry<Mode>& registry, std::size_t byte_budget) const
{
    using namespace construction_detail;
    auto limits = registry.impl_->limits;
    limits.byte_budget = std::min(limits.byte_budget, byte_budget);
    details::Operation scope(limits);
    if (geometry_->impl_->partition_proof_only)
        fail(GeometryErrorCode::ProvenanceMismatch, "Private construction witnesses cannot be serialized.");
    const auto plan = construction_plan<Mode>(geometry_->impl_, *registry.impl_);
    Writer<Mode> writer(limits.byte_budget);
    write_header<Mode>(writer, registry.context());
    writer.word(plan.roots.size());
    std::map<ConstructionRootKey, std::size_t> root_ids;
    for (std::size_t i = 0; i < plan.roots.size(); ++i)
    {
        write_key<Mode>(writer, plan.roots[i]);
        root_ids.emplace(plan.roots[i], i);
    }
    writer.word(plan.records.size());
    for (const auto& record : plan.records)
    {
        writer.word(static_cast<std::uint64_t>(record.kind));
        if (record.kind == Record::Kind::Root)
            writer.word(root_ids.at(record.root));
        else if (record.kind == Record::Kind::Rebase)
        {
            write_reference<Mode>(writer, record.parent);
            writer.word(root_ids.at(record.root));
        }
        else
        {
            write_reference<Mode>(writer, record.parent);
            writer.word(record.plane.input ? 1 : 0);
            writer.word(root_ids.at(record.plane.key));
            if (record.plane.input)
                writer.frame(record.plane.frame);
            else
                writer.word(record.plane.face);
            writer.word(record.orientation > 0 ? 1 : 0);
            writer.word((record.measure_section ? 1 : 0) | (record.construct_outside ? 2 : 0) |
                        (record.partition_proof ? 4 : 0));
        }
    }
    write_reference<Mode>(writer, plan.terminal);
    const auto shell = encode(registry.context(), limits.byte_budget);
    write_blob<Mode>(writer, shell);
    details::active_work->charge(writer.bytes.size(), 1);
    return std::move(writer.bytes);
}

template<ArithmeticMode Mode>
ConvexGeometryPtr<Mode> ConvexCellGeometry<Mode>::decode(
    std::span<const char> packet, const ConstructionRegistry<Mode>& registry, GeometryLimits requested)
{
    using namespace construction_detail;
    auto limits = limits_for<Mode>(requested);
    limits.byte_budget = std::min(limits.byte_budget, registry.impl_->limits.byte_budget);
    limits.maximum_vertices = std::min(limits.maximum_vertices, registry.impl_->limits.maximum_vertices);
    limits.maximum_faces = std::min(limits.maximum_faces, registry.impl_->limits.maximum_faces);
    limits.maximum_pair_tests = std::min(limits.maximum_pair_tests, registry.impl_->limits.maximum_pair_tests);
    limits.maximum_operations =
        std::min(details::Work(limits).operation_limit, details::Work(registry.impl_->limits).operation_limit);
    details::Operation scope(limits);
    if (packet.size() > limits.byte_budget)
        fail(GeometryErrorCode::ResourceLimit, "Construction packet exceeds byte budget.");
    Reader<Mode> reader(packet);
    read_header<Mode>(reader, registry.context());
    const auto root_count = reader.count(limits.byte_budget / sizeof(ConstructionRootKey), sizeof(ConstructionRootKey));
    std::vector<ConstructionRootKey> roots;
    std::set<ConstructionRootKey> unique;
    roots.reserve(root_count);
    for (std::size_t i = 0; i < root_count; ++i)
    {
        auto key = read_key<Mode>(reader, limits.byte_budget);
        if (!unique.insert(key).second || !registry.contains(key))
            fail(GeometryErrorCode::ProvenanceMismatch, "Missing, duplicate or unauthenticated construction root.");
        roots.push_back(std::move(key));
    }
    const auto get_root = [&](std::uint64_t index) -> const ConstructionRootKey&
    {
        if (index >= roots.size())
            fail(GeometryErrorCode::ProvenanceMismatch, "Invalid construction root reference.");
        return roots[static_cast<std::size_t>(index)];
    };
    struct Parsed
    {
        Record::Kind kind = Record::Kind::Root;
        Reference parent;
        std::size_t root = 0, face = 0;
        details::Frame frame;
        unsigned flags = 0;
        int orientation = 1;
        bool input = false, proof_only = false;
    };
    struct Outputs
    {
        DynamicConvexGeometryPtr inside, outside;
    };
    const auto count = reader.word();
    if (!count || count > limits.byte_budget / (sizeof(Parsed) + sizeof(Outputs)) ||
        count > (reader.bytes.size() - reader.cursor) / (2 * sizeof(std::uint64_t)))
        fail(GeometryErrorCode::ResourceLimit, "Construction dependency count exceeds packet or allocation limits.");
    const auto node_count = static_cast<std::size_t>(count);
    details::active_work->charge(node_count, sizeof(Parsed) + sizeof(Outputs) + sizeof(unsigned char));
    std::vector<Parsed> parsed;
    parsed.reserve(node_count);
    const auto root_reference = [&]
    {
        const auto index = reader.word();
        (void) get_root(index);
        return static_cast<std::size_t>(index);
    };
    const auto parent_reference = [&](std::size_t before)
    {
        const auto reference = read_reference<Mode>(reader, before);
        if (reference.side &&
            (parsed[reference.node].kind != Record::Kind::Clip || !(parsed[reference.node].flags & 2)))
            fail(GeometryErrorCode::ProvenanceMismatch, "Construction references an unbuilt complement.");
        return reference;
    };
    for (std::size_t i = 0; i < node_count; ++i)
    {
        details::active_work->pair();
        Parsed record;
        const auto kind = reader.word();
        if (kind < 1 || kind > 3)
            fail(GeometryErrorCode::ProvenanceMismatch, "Unknown construction dependency operation.");
        record.kind = static_cast<Record::Kind>(kind);
        if (record.kind == Record::Kind::Root)
        {
            record.root = root_reference();
            if (!registry.impl_->geometries.contains(roots[record.root]))
                fail(GeometryErrorCode::ProvenanceMismatch, "Plane root cannot stand for a geometry body.");
        }
        else if (record.kind == Record::Kind::Rebase)
        {
            record.parent = parent_reference(i);
            record.root = root_reference();
            record.proof_only = parsed[record.parent.node].proof_only;
            if (!registry.impl_->planes.contains(roots[record.root]))
                fail(GeometryErrorCode::ProvenanceMismatch, "Numerical rebase needs its registered input plane.");
        }
        else
        {
            record.parent = parent_reference(i);
            const auto input = reader.word();
            record.root = root_reference();
            if (input == 1)
            {
                record.input = true;
                record.frame = reader.frame();
                if (!registry.impl_->planes.contains(roots[record.root]))
                    fail(GeometryErrorCode::ProvenanceMismatch, "Unknown registered cutting plane.");
            }
            else if (input == 0)
            {
                const auto face = reader.word();
                const auto source = registry.impl_->geometries.find(roots[record.root]);
                if (source == registry.impl_->geometries.end() || face >= source->second->faces().size())
                    fail(GeometryErrorCode::ProvenanceMismatch, "Invalid registered source face.");
                record.face = static_cast<std::size_t>(face);
            }
            else
                fail(GeometryErrorCode::ProvenanceMismatch, "Invalid construction cutting-plane kind.");
            const auto orientation = reader.word(), flags = reader.word();
            if (orientation > 1 || flags > 7)
                fail(GeometryErrorCode::ProvenanceMismatch, "Invalid construction side or operation flags.");
            record.orientation = orientation ? 1 : -1;
            record.flags = static_cast<unsigned>(flags);
            record.proof_only = flags & 4;
            if ((parsed[record.parent.node].proof_only && !record.proof_only) ||
                (record.proof_only && ((flags & 1) || !(flags & 2))))
                fail(GeometryErrorCode::ProvenanceMismatch, "Invalid private construction witness transition.");
        }
        parsed.push_back(record);
    }
    const auto terminal_reference = parent_reference(node_count);
    const auto asserted_shell = read_blob<Mode>(reader, limits.byte_budget);
    if (reader.cursor != packet.size())
        fail(GeometryErrorCode::ProvenanceMismatch, "Trailing construction packet bytes.");
    if (parsed[terminal_reference.node].proof_only)
        fail(GeometryErrorCode::ProvenanceMismatch, "A private witness cannot be published as measured geometry.");
    std::vector<unsigned char> used(node_count, 0);
    details::active_work->charge(roots.size(), sizeof(unsigned char));
    std::vector<unsigned char> used_roots(roots.size(), 0);
    for (auto node = terminal_reference.node;;)
    {
        details::active_work->pair();
        used[node] = 1;
        used_roots[parsed[node].root] = 1;
        if (parsed[node].kind == Record::Kind::Root)
            break;
        node = parsed[node].parent.node;
    }
    if (std::find(used.begin(), used.end(), 0) != used.end() ||
        std::find(used_roots.begin(), used_roots.end(), 0) != used_roots.end())
        fail(GeometryErrorCode::ProvenanceMismatch, "Construction packet contains unused roots or dependency records.");

    // Only now replay the validated reachable DAG. Each operation, including
    // both requested complements, is evaluated once and its outputs are cached.
    std::vector<Outputs> outputs;
    outputs.reserve(node_count);
    const auto resolve = [&](Reference reference)
    {
        const auto& entry = outputs.at(reference.node);
        const auto value = reference.side ? entry.outside : entry.inside;
        if (!value)
            fail(GeometryErrorCode::ProvenanceMismatch, "Construction references an absent or boundary-only child.");
        return value;
    };
    for (const auto& record : parsed)
    {
        details::active_work->pair();
        const auto& key = roots[record.root];
        if (record.kind == Record::Kind::Root)
            outputs.push_back({registry.impl_->geometries.at(key)->dynamic_geometry(), {}});
        else if (record.kind == Record::Kind::Rebase)
            outputs.push_back({rebase_geometry<Mode>(resolve(record.parent), registry.impl_->planes.at(key)), {}});
        else
        {
            const auto parent = resolve(record.parent);
            const auto source =
                record.input
                    ? input_plane_source<Mode>(registry.impl_->planes.at(key), record.frame, limits)
                    : DynamicGeometryAccess::implementation(*registry.impl_->geometries.at(key)->dynamic_geometry());
            const auto split = Kernel<Mode>(DynamicGeometryAccess::implementation(*parent))
                                   .clip_impl(source, record.face, record.flags & 1, record.flags & 2, record.flags & 4,
                                       record.orientation);
            outputs.push_back({split.inside, split.outside});
        }
    }
    const auto terminal = resolve(terminal_reference);
    const auto restored = from_dynamic(terminal);
    const auto actual_shell = restored->encode(registry.context(), limits.byte_budget);
    if (actual_shell != asserted_shell)
        fail(GeometryErrorCode::ProvenanceMismatch, "Construction recipe does not reproduce its numerical shell.");
    return restored;
}
} // namespace SimpleFluid::PLIC
