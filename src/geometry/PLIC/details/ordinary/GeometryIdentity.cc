/** @file GeometryIdentity.cc
 * @brief Native finite support payloads and bounded canonical-body import.
 */
#include "geometry/PLIC/details/GeometryIdentityAccess.hh"
#include "geometry/PLIC/details/ordinary/NativeGeometrySupport.hh"
#include <algorithm>
#include <limits>
#include <stdexcept>

namespace SimpleFluid::PLIC::details::ordinary_identity_detail
{
using Point = OrdinaryIdentityPoint;
namespace on = ordinary_native;
using Support = PlaneSupportIdentity<ArithmeticMode::Ordinary>;
std::size_t add(std::size_t a, std::size_t b) noexcept
{
    return b > std::numeric_limits<std::size_t>::max() - a ? std::numeric_limits<std::size_t>::max() : a + b;
}
std::size_t product(std::size_t a, std::size_t b) noexcept
{
    return b && a > std::numeric_limits<std::size_t>::max() / b ? std::numeric_limits<std::size_t>::max() : a * b;
}
void admit(std::size_t amount, std::size_t& total, std::size_t budget)
{
    if (amount > budget - std::min(total, budget))
        throw std::invalid_argument("Native support import exceeds its retained payload budget.");
    total += amount;
}
long double scalar(long double value)
{
    on::finite(value);
    return value == 0 ? 0 : value;
}
struct NativeSupport : GeometryIdentityNativePayload
{
    GeometryIdentityNativeKind native_kind() const noexcept override
    {
        return GeometryIdentityNativeKind::OrdinarySupport;
    }
    [[nodiscard]] virtual OrdinarySupportKind kind() const noexcept = 0;
};
struct Axis final : NativeSupport
{
    std::uint64_t axis;
    long double location;
    Axis(std::uint64_t a, long double p) : axis(a), location(p) {}
    OrdinarySupportKind kind() const noexcept override { return OrdinarySupportKind::Axis; }
    std::size_t storage_bytes() const noexcept override { return sizeof(*this); }
};
struct Source final : NativeSupport
{
    std::vector<Point> points;
    explicit Source(std::vector<Point> p) : points(std::move(p)) {}
    OrdinarySupportKind kind() const noexcept override { return OrdinarySupportKind::Source; }
    std::size_t storage_bytes() const noexcept override
    {
        return add(sizeof(*this), product(points.capacity(), sizeof(Point)));
    }
};
struct Input final : NativeSupport
{
    Point normal, anchor;
    long double offset;
    Input(Point n, Point a, long double o) : normal(std::move(n)), anchor(std::move(a)), offset(o) {}
    OrdinarySupportKind kind() const noexcept override { return OrdinarySupportKind::Input; }
    std::size_t storage_bytes() const noexcept override { return sizeof(*this); }
};
struct Plane final : NativeSupport
{
    std::uint64_t axis;
    Point normal;
    long double location;
    Plane(std::uint64_t a, Point n, long double p) : axis(a), normal(std::move(n)), location(p) {}
    OrdinarySupportKind kind() const noexcept override { return OrdinarySupportKind::Plane; }
    std::size_t storage_bytes() const noexcept override { return sizeof(*this); }
};
struct Translation final : NativeSupport
{
    Point shift;
    explicit Translation(Point p) : shift(std::move(p)) {}
    OrdinarySupportKind kind() const noexcept override { return OrdinarySupportKind::Translation; }
    std::size_t storage_bytes() const noexcept override { return sizeof(*this); }
};
struct RetainedExact final : NativeSupport
{
    OrdinarySupportKind kind() const noexcept override { return OrdinarySupportKind::RetainedExact; }
    std::size_t storage_bytes() const noexcept override { return sizeof(*this); }
};
struct Dag final : NativeSupport
{
    std::vector<OrdinarySupportRecipe> records;
    std::size_t first, second;
    Dag(std::vector<OrdinarySupportRecipe> r, std::size_t a, std::size_t b) : records(std::move(r)), first(a), second(b)
    {
    }
    OrdinarySupportKind kind() const noexcept override { return OrdinarySupportKind::ExtrusionDag; }
    std::size_t storage_bytes() const noexcept override
    {
        auto bytes = add(sizeof(*this), product(records.capacity(), sizeof(OrdinarySupportRecipe)));
        for (const auto& r : records)
            bytes = add(bytes, product(r.parents.capacity(), sizeof(std::size_t)));
        return bytes;
    }
};
Support make(std::vector<char> key, std::unique_ptr<const GeometryIdentityNativePayload> native,
    GeometryIdentityBackingPtr parent = {})
{
    return GeometryIdentityAccess::share<Support>(std::move(key), std::move(native), std::move(parent));
}
constexpr GeometryTag axis_tag = GeometryTag::OrdinaryAxisSupport, source_tag = GeometryTag::OrdinarySourceSupport;
constexpr GeometryTag input_tag = GeometryTag::OrdinaryInputSupport, plane_tag = GeometryTag::OrdinaryPlaneSupport;
constexpr GeometryTag translation_tag = GeometryTag::OrdinaryTranslatedSupport, dag_tag = GeometryTag::OrdinaryExtrusionDagSupport;
constexpr GeometryTag exact_support_tag = GeometryTag::ExactPlaneSupport;
std::size_t node_size(std::size_t key, std::size_t native) noexcept
{
    return add(sizeof(GeometryIdentityBacking), add(key, native));
}

// Preflight uses no persistent owners or per-parent key caches. It keeps the
// original expression layers and estimates every cumulative retained cache.
std::size_t leaf_size(std::span<const char> key)
{
    on::Reader reader{key};
    const auto tag = reader.tag();
    std::size_t native = 0;
    switch (tag)
    {
        case axis_tag:
            native = sizeof(Axis);
            break;
        case input_tag:
            native = sizeof(Input);
            break;
        case plane_tag:
            native = sizeof(Plane);
            break;
        case source_tag:
            native = add(sizeof(Source), product(reader.count(4096), sizeof(Point)));
            break;
        case dag_tag: {
            const auto count = reader.count(key.size());
            native = add(sizeof(Dag), product(count, sizeof(OrdinarySupportRecipe)));
            for (std::size_t i = 0; i < count; ++i)
            {
                on::Reader record{reader.packet()};
                const auto kind = static_cast<SupportRecipeKind>(record.byte());
                if (kind == SupportRecipeKind::Derived)
                    native = add(native, product(record.count(record.bytes.size()), sizeof(std::size_t)));
            }
            break;
        }
        case exact_support_tag:
            return add(node_size(key.size(), sizeof(RetainedExact)),
                exact_identity_import_storage_bound(GeometryIdentityKind::PlaneSupport, key));
        default:
            throw std::invalid_argument("Unknown native support import kind.");
    }
    return node_size(key.size(), native);
}
struct Layer
{
    std::span<const char> key;
    Point shift;
};
GeometryIdentityBackingPtr import_leaf(std::span<const char> key, std::size_t budget)
{
    on::Reader reader{key};
    const auto tag = reader.tag();
    std::unique_ptr<const GeometryIdentityNativePayload> native;
    GeometryIdentityBackingPtr parent;
    switch (tag)
    {
        case axis_tag: {
            const auto axis = reader.word();
            const auto location = reader.real();
            native = std::make_unique<const Axis>(axis, location);
            break;
        }
        case input_tag: {
            const auto normal = reader.point(), anchor = reader.point();
            const auto offset = reader.real();
            native = std::make_unique<const Input>(normal, anchor, offset);
            break;
        }
        case plane_tag: {
            const auto axis = reader.word();
            const auto normal = reader.point();
            const auto location = reader.real();
            native = std::make_unique<const Plane>(axis, normal, location);
            break;
        }
        case source_tag: {
            const auto count = reader.count(4096);
            std::vector<Point> points;
            points.reserve(count);
            for (std::size_t i = 0; i < count; ++i)
                points.push_back(reader.point());
            native = std::make_unique<const Source>(std::move(points));
            break;
        }
        case dag_tag: {
            const auto count = reader.count(key.size());
            std::vector<OrdinarySupportRecipe> records;
            records.reserve(count);
            for (std::size_t i = 0; i < count; ++i)
            {
                on::Reader record{reader.packet()};
                OrdinarySupportRecipe value;
                value.original = static_cast<SupportRecipeKind>(record.byte()) == SupportRecipeKind::Original;
                if (value.original)
                    value.root = record.point();
                else
                {
                    const auto parents = record.count(record.bytes.size());
                    value.parents.reserve(parents);
                    for (std::size_t j = 0; j < parents; ++j)
                        value.parents.push_back(record.word());
                }
                records.push_back(std::move(value));
            }
            const auto first = reader.word(), second = reader.word();
            native = std::make_unique<const Dag>(std::move(records), first, second);
            break;
        }
        case exact_support_tag:
            parent =
                import_exact_identity_backing(GeometryIdentityKind::PlaneSupport, {key.begin(), key.end()}, budget);
            native = std::make_unique<const RetainedExact>();
            break;
        default:
            throw std::invalid_argument("Unknown native support import kind.");
    }
    return make_geometry_identity_backing({key.begin(), key.end()}, std::move(native), std::move(parent));
}
} // namespace SimpleFluid::PLIC::details::ordinary_identity_detail

namespace SimpleFluid::PLIC::details
{
using namespace ordinary_identity_detail;
using NativePoint = OrdinaryIdentityPoint;
PlaneSupportIdentity<ArithmeticMode::Ordinary> GeometryIdentityAccess::support_axis(
    std::uint64_t axis, long double location)
{
    if (axis > 2)
        on::reject("invalid native support axis.");
    location = scalar(location);
    on::Writer key;
    key.tag(axis_tag);
    key.word(axis);
    key.real(location);
    return ordinary_identity_detail::make(std::move(key.bytes), std::make_unique<const Axis>(axis, location));
}
PlaneSupportIdentity<ArithmeticMode::Ordinary> GeometryIdentityAccess::support_source(std::vector<NativePoint> points)
{
    for (auto& point : points)
        point = normalize_identity_point(std::move(point));
    if (points.size() < 3 || !std::is_sorted(points.begin(), points.end()))
        on::reject("invalid native source support points.");
    on::Writer key;
    key.tag(source_tag);
    key.word(points.size());
    for (const auto& point : points)
        key.point(point);
    return ordinary_identity_detail::make(std::move(key.bytes), std::make_unique<const Source>(std::move(points)));
}
PlaneSupportIdentity<ArithmeticMode::Ordinary> GeometryIdentityAccess::support_input(
    NativePoint normal, NativePoint anchor, long double offset)
{
    normal = normalize_identity_point(std::move(normal));
    anchor = normalize_identity_point(std::move(anchor));
    offset = scalar(offset);
    const auto first = std::find_if(normal.begin(), normal.end(), [](auto value) { return value != 0; });
    if (first == normal.end() || *first < 0)
        on::reject("native input support normal is zero or has a noncanonical sign.");
    on::Writer key;
    key.tag(input_tag);
    key.point(normal);
    key.point(anchor);
    key.real(offset);
    return ordinary_identity_detail::make(std::move(key.bytes), std::make_unique<const Input>(normal, anchor, offset));
}
PlaneSupportIdentity<ArithmeticMode::Ordinary> GeometryIdentityAccess::support_plane(
    std::uint64_t axis, NativePoint normal, long double location)
{
    if (axis > 2)
        on::reject("invalid native support axis.");
    normal = normalize_identity_point(std::move(normal));
    location = scalar(location);
    on::Writer key;
    key.tag(plane_tag);
    key.word(axis);
    key.point(normal);
    key.real(location);
    return ordinary_identity_detail::make(std::move(key.bytes), std::make_unique<const Plane>(axis, normal, location));
}
PlaneSupportIdentity<ArithmeticMode::Ordinary> GeometryIdentityAccess::translated_support(
    const Support& parent, NativePoint shift)
{
    shift = normalize_identity_point(std::move(shift));
    if (shift == NativePoint{})
        return parent;
    on::Writer key;
    key.tag(translation_tag);
    key.packet(GeometryIdentityAccess::key(parent));
    key.point(shift);
    return ordinary_identity_detail::make(
        std::move(key.bytes), std::make_unique<const Translation>(shift), backing(parent));
}
PlaneSupportIdentity<ArithmeticMode::Ordinary> GeometryIdentityAccess::retained_exact_support(
    const PlaneSupportIdentity<ArithmeticMode::Exact>& parent)
{
    if (!backing(parent))
        throw std::invalid_argument("Missing retained Exact support owner.");
    return ordinary_identity_detail::make(
        canonical_key(parent), std::make_unique<const RetainedExact>(), backing(parent));
}
PlaneSupportIdentity<ArithmeticMode::Ordinary> GeometryIdentityAccess::support_dag(
    std::vector<OrdinarySupportRecipe> records, std::size_t first, std::size_t second)
{
    if (first > second || second >= records.size())
        on::reject("invalid native support recipe endpoints.");
    on::Writer key;
    key.tag(dag_tag);
    key.word(records.size());
    for (std::size_t i = 0; i < records.size(); ++i)
    {
        auto& record = records[i];
        on::Writer fields;
        fields.byte(static_cast<std::uint8_t>(record.original ? SupportRecipeKind::Original : SupportRecipeKind::Derived));
        if (record.original)
        {
            record.root = normalize_identity_point(std::move(record.root));
            fields.point(record.root);
        }
        else
        {
            if (record.parents.empty() || !std::is_sorted(record.parents.begin(), record.parents.end()) ||
                record.parents.back() >= i)
                on::reject("invalid native support recipe parents.");
            fields.word(record.parents.size());
            for (auto parent : record.parents)
                fields.word(parent);
        }
        key.packet(fields.bytes);
    }
    key.word(first);
    key.word(second);
    return ordinary_identity_detail::make(
        std::move(key.bytes), std::make_unique<const Dag>(std::move(records), first, second));
}
OrdinarySupportKind ordinary_support_kind(const Support& support)
{
    const auto& payload = geometry_identity_native(GeometryIdentityAccess::backing(support));
    if (payload.native_kind() != GeometryIdentityNativeKind::OrdinarySupport)
        throw std::invalid_argument("Wrong native payload kind for Ordinary support.");
    return static_cast<const NativeSupport&>(payload).kind();
}
namespace
{
const NativeSupport& support_payload(const GeometryIdentityBackingPtr& owner)
{
    const auto& payload = geometry_identity_native(owner);
    if (payload.native_kind() != GeometryIdentityNativeKind::OrdinarySupport)
        throw std::invalid_argument("Wrong native payload kind for Ordinary support ordering.");
    return static_cast<const NativeSupport&>(payload);
}
std::strong_ordering compare_scalar(long double first, long double second) noexcept
{
    if (first < second)
        return std::strong_ordering::less;
    if (second < first)
        return std::strong_ordering::greater;
    return std::strong_ordering::equal;
}
std::strong_ordering compare_point(const OrdinaryIdentityPoint& first, const OrdinaryIdentityPoint& second) noexcept
{
    for (std::size_t i = 0; i < 3; ++i)
    {
        const auto order = compare_scalar(first[i], second[i]);
        if (order != 0)
            return order;
    }
    return std::strong_ordering::equal;
}
template<class Range, class Compare>
std::strong_ordering compare_sequence(const Range& first, const Range& second, Compare compare)
{
    for (std::size_t i = 0; i < std::min(first.size(), second.size()); ++i)
    {
        const auto order = compare(first[i], second[i]);
        if (order != 0)
            return order;
    }
    return first.size() <=> second.size();
}
std::strong_ordering compare_record(const OrdinarySupportRecipe& first, const OrdinarySupportRecipe& second)
{
    const unsigned first_kind = first.original ? 0 : 1, second_kind = second.original ? 0 : 1;
    if (first_kind != second_kind)
        return first_kind <=> second_kind;
    if (first.original)
        return compare_point(first.root, second.root);
    return first.parents <=> second.parents;
}
} // namespace
std::strong_ordering compare_ordinary_support_native(
    const GeometryIdentityBackingPtr& first, const GeometryIdentityBackingPtr& second)
{
    if (first == second)
        return std::strong_ordering::equal;
    const auto& a = support_payload(first);
    const auto& b = support_payload(second);
    const auto kind = a.kind();
    if (kind != b.kind())
        return static_cast<unsigned>(kind) <=> static_cast<unsigned>(b.kind());
    switch (kind)
    {
        case OrdinarySupportKind::Axis: {
            const auto& x = static_cast<const Axis&>(a);
            const auto& y = static_cast<const Axis&>(b);
            return x.axis != y.axis ? x.axis <=> y.axis : compare_scalar(x.location, y.location);
        }
        case OrdinarySupportKind::Source: {
            return compare_sequence(
                static_cast<const Source&>(a).points, static_cast<const Source&>(b).points, compare_point);
        }
        case OrdinarySupportKind::Input: {
            const auto& x = static_cast<const Input&>(a);
            const auto& y = static_cast<const Input&>(b);
            if (const auto normal = compare_point(x.normal, y.normal); normal != 0)
                return normal;
            if (const auto anchor = compare_point(x.anchor, y.anchor); anchor != 0)
                return anchor;
            return compare_scalar(x.offset, y.offset);
        }
        case OrdinarySupportKind::Plane: {
            const auto& x = static_cast<const Plane&>(a);
            const auto& y = static_cast<const Plane&>(b);
            if (x.axis != y.axis)
                return x.axis <=> y.axis;
            if (const auto normal = compare_point(x.normal, y.normal); normal != 0)
                return normal;
            return compare_scalar(x.location, y.location);
        }
        case OrdinarySupportKind::Translation: {
            const auto parent = compare_ordinary_support_native(first->parent, second->parent);
            if (parent != 0)
                return parent;
            return compare_point(static_cast<const Translation&>(a).shift, static_cast<const Translation&>(b).shift);
        }
        case OrdinarySupportKind::RetainedExact:
            return compare_exact_identity_native(GeometryIdentityKind::PlaneSupport, first->parent, second->parent);
        case OrdinarySupportKind::ExtrusionDag: {
            const auto& x = static_cast<const Dag&>(a);
            const auto& y = static_cast<const Dag&>(b);
            if (const auto records = compare_sequence(x.records, y.records, compare_record); records != 0)
                return records;
            return x.first != y.first ? x.first <=> y.first : x.second <=> y.second;
        }
    }
    throw std::invalid_argument("Unknown native support ordering variant.");
}

GeometryIdentityBackingPtr import_ordinary_support_backing(std::vector<char> checked_key, std::size_t budget)
{
    auto current = std::span<const char>(checked_key);
    std::size_t planned = 0, depth = 0;
    for (;;)
    {
        on::Reader reader{current};
        if (reader.tag() != translation_tag)
            break;
        admit(node_size(current.size(), sizeof(Translation)), planned, budget);
        const auto parent = reader.packet();
        (void) reader.point();
        current = parent;
        ++depth;
    }
    admit(leaf_size(current), planned, budget); // Before any cumulative parent key caches exist.
    std::vector<Layer> layers;
    layers.reserve(depth);
    current = checked_key;
    for (std::size_t i = 0; i < depth; ++i)
    {
        on::Reader reader{current};
        (void) reader.tag();
        const auto parent = reader.packet();
        const auto shift = reader.point();
        layers.push_back({current, shift});
        current = parent;
    }
    auto result = import_leaf(current, budget);
    std::size_t actual = 0;
    for (auto owner = result; owner; owner = geometry_identity_parent(owner))
        admit(geometry_identity_payload_bytes(owner), actual, budget);
    for (auto layer = layers.rbegin(); layer != layers.rend(); ++layer)
    {
        auto key = std::vector<char>(layer->key.begin(), layer->key.end());
        auto next =
            make_geometry_identity_backing(std::move(key), std::make_unique<const Translation>(layer->shift), result);
        admit(geometry_identity_payload_bytes(next), actual, budget);
        result = std::move(next);
    }
    return result;
}
} // namespace SimpleFluid::PLIC::details
