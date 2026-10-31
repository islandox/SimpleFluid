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
    void emit_prefix(const GeometryIdentityCanonicalWriter&, std::size_t parent_bytes) const override;
    void emit_suffix(const GeometryIdentityCanonicalWriter&) const override;
    bool encoded_fields_equal(const GeometryIdentityNativePayload&) const noexcept override;
    std::size_t next_canonical_chunk(
        bool, std::array<std::size_t, 4>&, std::size_t parent_bytes, std::span<char>) const override;
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
Support make(std::unique_ptr<const GeometryIdentityNativePayload> native, GeometryIdentityBackingPtr parent = {})
{
    return GeometryIdentityAccess::share<Support>(std::move(native), std::move(parent));
}
constexpr GeometryTag axis_tag = GeometryTag::OrdinaryAxisSupport, source_tag = GeometryTag::OrdinarySourceSupport;
constexpr GeometryTag input_tag = GeometryTag::OrdinaryInputSupport, plane_tag = GeometryTag::OrdinaryPlaneSupport;
constexpr GeometryTag translation_tag = GeometryTag::OrdinaryTranslatedSupport,
                      dag_tag = GeometryTag::OrdinaryExtrusionDagSupport;
constexpr GeometryTag exact_support_tag = GeometryTag::ExactPlaneSupport;
std::size_t point_size(const Point& point) noexcept
{
    std::size_t bytes = 0;
    for (auto value : point)
        bytes += value == 0 ? 16 : ordinary_identity_scalar_bytes;
    return bytes;
}
std::size_t record_size(const OrdinarySupportRecipe& record) noexcept
{
    return record.original ? 1 + point_size(record.root) : add(9, product(8, record.parents.size()));
}
void NativeSupport::emit_prefix(const GeometryIdentityCanonicalWriter& writer, std::size_t parent_bytes) const
{
    switch (kind())
    {
        case OrdinarySupportKind::Axis: {
            const auto& value = static_cast<const Axis&>(*this);
            writer.tag(axis_tag);
            writer.word(value.axis);
            writer.real(value.location);
            break;
        }
        case OrdinarySupportKind::Source: {
            const auto& value = static_cast<const Source&>(*this);
            writer.tag(source_tag);
            writer.word(value.points.size());
            for (const auto& point : value.points)
                writer.point(point);
            break;
        }
        case OrdinarySupportKind::Input: {
            const auto& value = static_cast<const Input&>(*this);
            writer.tag(input_tag);
            writer.point(value.normal);
            writer.point(value.anchor);
            writer.real(value.offset);
            break;
        }
        case OrdinarySupportKind::Plane: {
            const auto& value = static_cast<const Plane&>(*this);
            writer.tag(plane_tag);
            writer.word(value.axis);
            writer.point(value.normal);
            writer.real(value.location);
            break;
        }
        case OrdinarySupportKind::Translation:
            writer.tag(translation_tag);
            writer.word(parent_bytes);
            break;
        case OrdinarySupportKind::RetainedExact:
            break;
        case OrdinarySupportKind::ExtrusionDag: {
            const auto& value = static_cast<const Dag&>(*this);
            writer.tag(dag_tag);
            writer.word(value.records.size());
            for (const auto& record : value.records)
            {
                writer.word(record_size(record));
                writer.byte(static_cast<std::uint8_t>(
                    record.original ? SupportRecipeKind::Original : SupportRecipeKind::Derived));
                if (record.original)
                    writer.point(record.root);
                else
                {
                    writer.word(record.parents.size());
                    for (auto parent : record.parents)
                        writer.word(parent);
                }
            }
            writer.word(value.first);
            writer.word(value.second);
            break;
        }
    }
}
void NativeSupport::emit_suffix(const GeometryIdentityCanonicalWriter& writer) const
{
    if (kind() == OrdinarySupportKind::Translation)
        writer.point(static_cast<const Translation&>(*this).shift);
}
std::size_t NativeSupport::next_canonical_chunk(
    bool suffix, std::array<std::size_t, 4>& state, std::size_t parent_bytes, std::span<char> output) const
{
    struct Chunk
    {
        std::span<char> bytes;
        std::size_t written = 0;
    } chunk{output};
    const GeometryIdentityCanonicalWriter writer{&chunk, [](void* context, std::span<const char> bytes)
        {
            auto& destination = *static_cast<Chunk*>(context);
            if (bytes.size() > destination.bytes.size() - destination.written)
                throw std::logic_error("Canonical support chunk exceeds its fixed buffer.");
            std::copy(bytes.begin(), bytes.end(), destination.bytes.begin() + destination.written);
            destination.written += bytes.size();
        }};
    if (suffix)
    {
        if (kind() == OrdinarySupportKind::Translation && state[0] < 3)
            writer.real(static_cast<const Translation&>(*this).shift[state[0]++]);
        return chunk.written;
    }
    switch (kind())
    {
        case OrdinarySupportKind::Axis: {
            const auto& value = static_cast<const Axis&>(*this);
            switch (state[0]++)
            {
                case 0:
                    writer.tag(axis_tag);
                    break;
                case 1:
                    writer.word(value.axis);
                    break;
                case 2:
                    writer.real(value.location);
                    break;
            }
            break;
        }
        case OrdinarySupportKind::Source: {
            const auto& value = static_cast<const Source&>(*this);
            const auto field = state[0]++;
            if (field == 0)
                writer.tag(source_tag);
            else if (field == 1)
                writer.word(value.points.size());
            else if ((field - 2) / 3 < value.points.size())
                writer.real(value.points[(field - 2) / 3][(field - 2) % 3]);
            break;
        }
        case OrdinarySupportKind::Input: {
            const auto& value = static_cast<const Input&>(*this);
            const auto field = state[0]++;
            if (field == 0)
                writer.tag(input_tag);
            else if (field < 4)
                writer.real(value.normal[field - 1]);
            else if (field < 7)
                writer.real(value.anchor[field - 4]);
            else if (field == 7)
                writer.real(value.offset);
            break;
        }
        case OrdinarySupportKind::Plane: {
            const auto& value = static_cast<const Plane&>(*this);
            const auto field = state[0]++;
            if (field == 0)
                writer.tag(plane_tag);
            else if (field == 1)
                writer.word(value.axis);
            else if (field < 5)
                writer.real(value.normal[field - 2]);
            else if (field == 5)
                writer.real(value.location);
            break;
        }
        case OrdinarySupportKind::Translation:
            if (state[0] == 0)
                writer.tag(translation_tag);
            else if (state[0] == 1)
                writer.word(parent_bytes);
            ++state[0];
            break;
        case OrdinarySupportKind::RetainedExact:
            break;
        case OrdinarySupportKind::ExtrusionDag: {
            const auto& value = static_cast<const Dag&>(*this);
            for (;;)
            {
                if (state[0] == 0)
                {
                    writer.tag(dag_tag);
                    ++state[0];
                }
                else if (state[0] == 1)
                {
                    writer.word(value.records.size());
                    ++state[0];
                }
                else if (state[0] == 2)
                {
                    if (state[1] == value.records.size())
                    {
                        ++state[0];
                        continue;
                    }
                    const auto& record = value.records[state[1]];
                    if (state[2] == 0)
                    {
                        writer.word(record_size(record));
                        ++state[2];
                    }
                    else if (state[2] == 1)
                    {
                        writer.byte(static_cast<std::uint8_t>(
                            record.original ? SupportRecipeKind::Original : SupportRecipeKind::Derived));
                        ++state[2];
                    }
                    else if (record.original)
                    {
                        writer.real(record.root[state[3]++]);
                        if (state[3] == 3)
                        {
                            ++state[1];
                            state[2] = state[3] = 0;
                        }
                    }
                    else if (state[2] == 2)
                    {
                        writer.word(record.parents.size());
                        ++state[2];
                    }
                    else if (state[3] < record.parents.size())
                    {
                        writer.word(record.parents[state[3]++]);
                        if (state[3] == record.parents.size())
                        {
                            ++state[1];
                            state[2] = state[3] = 0;
                        }
                    }
                    else
                    {
                        ++state[1];
                        state[2] = state[3] = 0;
                        continue;
                    }
                }
                else if (state[0] == 3)
                {
                    writer.word(value.first);
                    ++state[0];
                }
                else if (state[0] == 4)
                {
                    writer.word(value.second);
                    ++state[0];
                }
                break;
            }
            break;
        }
    }
    return chunk.written;
}
bool NativeSupport::encoded_fields_equal(const GeometryIdentityNativePayload& other) const noexcept
{
    if (other.native_kind() != GeometryIdentityNativeKind::OrdinarySupport)
        return false;
    const auto& that = static_cast<const NativeSupport&>(other);
    if (kind() != that.kind())
        return false;
    switch (kind())
    {
        case OrdinarySupportKind::Axis: {
            const auto& a = static_cast<const Axis&>(*this);
            const auto& b = static_cast<const Axis&>(that);
            return a.axis == b.axis && a.location == b.location;
        }
        case OrdinarySupportKind::Source:
            return static_cast<const Source&>(*this).points == static_cast<const Source&>(that).points;
        case OrdinarySupportKind::Input: {
            const auto& a = static_cast<const Input&>(*this);
            const auto& b = static_cast<const Input&>(that);
            return a.normal == b.normal && a.anchor == b.anchor && a.offset == b.offset;
        }
        case OrdinarySupportKind::Plane: {
            const auto& a = static_cast<const Plane&>(*this);
            const auto& b = static_cast<const Plane&>(that);
            return a.axis == b.axis && a.normal == b.normal && a.location == b.location;
        }
        case OrdinarySupportKind::Translation:
            return static_cast<const Translation&>(*this).shift == static_cast<const Translation&>(that).shift;
        case OrdinarySupportKind::RetainedExact:
            return true;
        case OrdinarySupportKind::ExtrusionDag: {
            const auto& a = static_cast<const Dag&>(*this);
            const auto& b = static_cast<const Dag&>(that);
            if (a.first != b.first || a.second != b.second || a.records.size() != b.records.size())
                return false;
            for (std::size_t i = 0; i < a.records.size(); ++i)
            {
                const auto& x = a.records[i];
                const auto& y = b.records[i];
                if (x.original != y.original || (x.original ? x.root != y.root : x.parents != y.parents))
                    return false;
            }
            return true;
        }
    }
    return false;
}
std::size_t node_size(std::size_t native) noexcept
{
    return add(sizeof(GeometryIdentityBacking), native);
}

// Preflight uses no persistent owners or per-parent key caches. It keeps the
// original expression layers and estimates native allocations and live scratch.
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
            return add(node_size(sizeof(RetainedExact)),
                exact_identity_import_storage_bound(GeometryIdentityKind::PlaneSupport, key));
        default:
            throw std::invalid_argument("Unknown native support import kind.");
    }
    return node_size(native);
}
struct Layer
{
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
        case exact_support_tag: {
            const auto retained = node_size(sizeof(RetainedExact));
            if (retained > budget)
                throw std::invalid_argument("Retained Exact support exceeds its native import budget.");
            parent = import_exact_identity_backing(GeometryIdentityKind::PlaneSupport, key, budget - retained);
            native = std::make_unique<const RetainedExact>();
            break;
        }
        default:
            throw std::invalid_argument("Unknown native support import kind.");
    }
    std::size_t retained = node_size(native->storage_bytes());
    for (auto ancestor = parent; ancestor; ancestor = geometry_identity_parent(ancestor))
        admit(geometry_identity_payload_bytes(ancestor), retained, budget);
    if (retained > budget)
        throw std::invalid_argument("Native support import exceeds its retained payload budget.");
    return make_geometry_identity_backing(std::move(native), std::move(parent), budget - retained);
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
    return ordinary_identity_detail::make(std::make_unique<const Axis>(axis, location));
}
PlaneSupportIdentity<ArithmeticMode::Ordinary> GeometryIdentityAccess::support_source(std::vector<NativePoint> points)
{
    for (auto& point : points)
        point = normalize_identity_point(std::move(point));
    if (points.size() < 3 || !std::is_sorted(points.begin(), points.end()))
        on::reject("invalid native source support points.");
    return ordinary_identity_detail::make(std::make_unique<const Source>(std::move(points)));
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
    return ordinary_identity_detail::make(std::make_unique<const Input>(normal, anchor, offset));
}
PlaneSupportIdentity<ArithmeticMode::Ordinary> GeometryIdentityAccess::support_plane(
    std::uint64_t axis, NativePoint normal, long double location)
{
    if (axis > 2)
        on::reject("invalid native support axis.");
    normal = normalize_identity_point(std::move(normal));
    location = scalar(location);
    return ordinary_identity_detail::make(std::make_unique<const Plane>(axis, normal, location));
}
PlaneSupportIdentity<ArithmeticMode::Ordinary> GeometryIdentityAccess::translated_support(
    const Support& parent, NativePoint shift)
{
    shift = normalize_identity_point(std::move(shift));
    if (shift == NativePoint{})
        return parent;
    return ordinary_identity_detail::make(std::make_unique<const Translation>(shift), backing(parent));
}
PlaneSupportIdentity<ArithmeticMode::Ordinary> GeometryIdentityAccess::retained_exact_support(
    const PlaneSupportIdentity<ArithmeticMode::Exact>& parent)
{
    if (!backing(parent))
        throw std::invalid_argument("Missing retained Exact support owner.");
    return ordinary_identity_detail::make(std::make_unique<const RetainedExact>(), backing(parent));
}
PlaneSupportIdentity<ArithmeticMode::Ordinary> GeometryIdentityAccess::support_dag(
    std::vector<OrdinarySupportRecipe> records, std::size_t first, std::size_t second)
{
    if (first > second || second >= records.size())
        on::reject("invalid native support recipe endpoints.");
    for (std::size_t i = 0; i < records.size(); ++i)
    {
        auto& record = records[i];
        if (record.original)
            record.root = normalize_identity_point(std::move(record.root));
        else if (record.parents.empty() || !std::is_sorted(record.parents.begin(), record.parents.end()) ||
                 record.parents.back() >= i)
            on::reject("invalid native support recipe parents.");
    }
    return ordinary_identity_detail::make(std::make_unique<const Dag>(std::move(records), first, second));
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
static std::strong_ordering compare_support_node(
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
            // The iterative driver visits translation parents before shifts.
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

std::strong_ordering compare_ordinary_support_native(
    const GeometryIdentityBackingPtr& first, const GeometryIdentityBackingPtr& second)
{
    auto a = first, b = second;
    std::size_t translations = 0;
    while (a != b && support_payload(a).kind() == OrdinarySupportKind::Translation &&
           support_payload(b).kind() == OrdinarySupportKind::Translation)
    {
        a = a->parent;
        b = b->parent;
        ++translations;
    }
    if (const auto parent = compare_support_node(a, b); parent != 0)
        return parent;
    // Revisit parents rather than retaining a scratch stack. This preserves
    // parent-first native order with bounded stack and no query allocations.
    while (translations)
    {
        --translations;
        auto x = first.get(), y = second.get();
        for (std::size_t i = 0; i < translations; ++i)
        {
            x = x->parent.get();
            y = y->parent.get();
        }
        const auto order = compare_point(
            static_cast<const Translation&>(*x->native).shift, static_cast<const Translation&>(*y->native).shift);
        if (order != 0)
            return order;
    }
    return std::strong_ordering::equal;
}

GeometryIdentityBackingPtr import_ordinary_support_backing(std::span<const char> checked_key, std::size_t budget)
{
    auto current = std::span<const char>(checked_key);
    std::size_t planned = 0, depth = 0;
    for (;;)
    {
        on::Reader reader{current};
        if (reader.tag() != translation_tag)
            break;
        admit(node_size(sizeof(Translation)), planned, budget);
        const auto parent = reader.packet();
        (void) reader.point();
        current = parent;
        ++depth;
    }
    admit(leaf_size(current), planned, budget);
    const auto leaf_depth = static_cast<GeometryTag>(current.front()) == exact_support_tag ? 2u : 1u;
    const auto pointer_scratch = product(add(depth, leaf_depth), sizeof(const GeometryIdentityBacking*));
    const auto layer_scratch = product(depth, sizeof(Layer));
    admit(add(layer_scratch, pointer_scratch), planned, budget); // Wire has its own envelope limit.
    charge_geometry_identity_scratch(depth, sizeof(Layer));
    std::vector<Layer> layers;
    layers.reserve(depth);
    const auto actual_layer_scratch = product(layers.capacity(), sizeof(Layer));
    if (layers.capacity() > depth)
    {
        admit(product(layers.capacity() - depth, sizeof(Layer)), planned, budget);
        charge_geometry_identity_scratch(layers.capacity() - depth, sizeof(Layer));
    }
    current = checked_key;
    for (std::size_t i = 0; i < depth; ++i)
    {
        on::Reader reader{current};
        (void) reader.tag();
        const auto parent = reader.packet();
        const auto shift = reader.point();
        layers.push_back({shift});
        current = parent;
    }
    auto result = import_leaf(current, budget - actual_layer_scratch);
    std::size_t actual = actual_layer_scratch;
    for (auto owner = result; owner; owner = geometry_identity_parent(owner))
        admit(geometry_identity_payload_bytes(owner), actual, budget);
    for (auto layer = layers.rbegin(); layer != layers.rend(); ++layer)
    {
        const auto retained = node_size(sizeof(Translation));
        admit(retained, actual, budget);
        auto next =
            make_geometry_identity_backing(std::make_unique<const Translation>(layer->shift), result, budget - actual);
        result = std::move(next);
    }
    return result;
}
} // namespace SimpleFluid::PLIC::details
