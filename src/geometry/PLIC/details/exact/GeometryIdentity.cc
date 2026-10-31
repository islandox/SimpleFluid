/** @file GeometryIdentity.cc
 * @brief Immutable Exact tuples with streamed frozen canonical representations.
 */
#include "geometry/PLIC/details/exact/GeometryIdentity.hh"
#include "geometry/PLIC/details/GeometryIdentityBacking.hh"
#include <cstdint>
#include <memory>

namespace SimpleFluid::PLIC::exact_geometry_detail
{
namespace
{
constexpr GeometryTag point_tag = GeometryTag::ExactPoint;
constexpr GeometryTag line_tag = GeometryTag::ExactLine;
constexpr GeometryTag support_tag = GeometryTag::ExactPlaneSupport;
constexpr GeometryTag displacement_tag = GeometryTag::ExactDisplacement;
std::size_t sum(std::size_t a, std::size_t b) noexcept
{
    const auto maximum = std::numeric_limits<std::size_t>::max();
    return b > maximum - a ? maximum : a + b;
}
std::size_t product(std::size_t a, std::size_t b) noexcept
{
    const auto maximum = std::numeric_limits<std::size_t>::max();
    return b && a > maximum / b ? maximum : a * b;
}
template<class Integer> std::size_t dynamic_limbs(const Integer& value) noexcept
{
    // Inline cpp_int limbs are already contained in the native tuple object.
    const auto limbs = reinterpret_cast<std::uintptr_t>(value.limbs());
    const auto object = reinterpret_cast<std::uintptr_t>(std::addressof(value));
    if (limbs >= object && limbs - object < sizeof(value))
        return 0;
    return product(value.capacity(), sizeof(boost::multiprecision::limb_type));
}
} // namespace
// The factory's stable private kind identifies tuple layout across statically
// linked provider copies; access does not depend on cross-DSO RTTI identity.
namespace
{
template<class Integer> void emit_integer(const details::GeometryIdentityCanonicalWriter& writer, const Integer& value)
{
    writer.word(value.sign());
    using Limb = boost::multiprecision::limb_type;
    constexpr auto limb_bytes = sizeof(Limb);
    const auto* limbs = value.limbs();
    auto top = limbs[value.size() - 1];
    std::size_t top_bytes = 1;
    while (top >>= 8)
        ++top_bytes;
    writer.word((value.size() - 1) * limb_bytes + top_bytes);
    for (std::size_t limb = 0; limb < value.size(); ++limb)
    {
        const auto bytes = limb + 1 == value.size() ? top_bytes : limb_bytes;
        std::array<char, limb_bytes> encoded;
        for (std::size_t byte = 0; byte < bytes; ++byte)
            encoded[byte] = static_cast<char>(limbs[limb] >> (8 * byte));
        writer.append(writer.context, {encoded.data(), bytes});
    }
}
template<class Integer> bool equal_integer(const Integer& first, const Integer& second) noexcept
{
    if (first.sign() != second.sign() || first.size() != second.size())
        return false;
    for (std::size_t limb = 0; limb < first.size(); ++limb)
        if (first.limbs()[limb] != second.limbs()[limb])
            return false;
    return true;
}
void emit_rational(const details::GeometryIdentityCanonicalWriter& writer, const Q& value)
{
    emit_integer(writer, value.backend().num());
    emit_integer(writer, value.backend().denom());
}
I read_identity_integer(WireReader& reader)
{
    const auto sign = reader.word(), size = reader.word();
    if (sign > 1 || size == 0 || size > reader.data.size() - reader.position)
        throw std::invalid_argument("Invalid canonical identity integer.");
    const auto bytes = reader.data.subspan(reader.position, size);
    reader.position += size;
    if ((size > 1 && bytes.back() == 0) || (sign && size == 1 && bytes.front() == 0))
        throw std::invalid_argument("Noncanonical identity integer magnitude or zero sign.");
    const auto* begin = reinterpret_cast<const unsigned char*>(bytes.data());
    I value;
    import_bits(value, begin, begin + bytes.size(), 8, false);
    if (sign)
        value = -value;
    return value;
}
} // namespace
Q read_identity_rational(WireReader& reader)
{
    const auto start = reader.position;
    auto numerator = read_identity_integer(reader), denominator = read_identity_integer(reader);
    if (denominator <= 0)
        throw std::invalid_argument("Exact identity denominator must be positive.");
    Q value(std::move(numerator), std::move(denominator));
    struct Comparison
    {
        std::span<const char> bytes;
        std::size_t cursor = 0;
        bool equal = true;
    } comparison{reader.data.subspan(start, reader.position - start)};
    emit_rational({&comparison,
                      [](void* context, std::span<const char> fragment)
                      {
                          auto& state = *static_cast<Comparison*>(context);
                          if (fragment.size() > state.bytes.size() - std::min(state.cursor, state.bytes.size()) ||
                              !std::equal(fragment.begin(), fragment.end(), state.bytes.begin() + state.cursor))
                              state.equal = false;
                          state.cursor += fragment.size();
                      }},
        value);
    if (!comparison.equal || comparison.cursor != comparison.bytes.size())
        throw std::invalid_argument("Noncanonical Exact identity rational encoding.");
    return value;
}
template<std::size_t Count> struct NativeTuple final : details::GeometryIdentityNativePayload
{
    std::array<Q, Count> values;
    GeometryTag tag;
    NativeTuple(std::array<Q, Count> input, GeometryTag kind) : values(std::move(input)), tag(kind) {}
    details::GeometryIdentityNativeKind native_kind() const noexcept override
    {
        static_assert(Count == 3 || Count == 4 || Count == 6);
        if constexpr (Count == 3)
            return details::GeometryIdentityNativeKind::ExactTuple3;
        else if constexpr (Count == 4)
            return details::GeometryIdentityNativeKind::ExactTuple4;
        else
            return details::GeometryIdentityNativeKind::ExactTuple6;
    }
    std::size_t storage_bytes() const noexcept override
    {
        auto bytes = sizeof(*this);
        for (const auto& value : values)
        {
            bytes = sum(bytes, dynamic_limbs(value.backend().num()));
            bytes = sum(bytes, dynamic_limbs(value.backend().denom()));
        }
        return bytes;
    }
    void emit_prefix(const details::GeometryIdentityCanonicalWriter& writer, std::size_t) const override
    {
        writer.tag(tag);
        for (const auto& value : values)
            emit_rational(writer, value);
    }
    std::size_t next_canonical_chunk(
        bool suffix, std::array<std::size_t, 4>& state, std::size_t, std::span<char> output) const override
    {
        if (suffix)
            return 0;
        if (state[0] == 0)
        {
            output[0] = static_cast<char>(tag);
            ++state[0];
            return 1;
        }
        const auto integer_index = state[0] - 1;
        if (integer_index >= 2 * Count)
            return 0;
        const auto& integer =
            integer_index % 2 ? values[integer_index / 2].backend().denom() : values[integer_index / 2].backend().num();
        const auto word = [&](std::uint64_t value)
        {
            for (unsigned byte = 0; byte < 8; ++byte)
                output[byte] = static_cast<char>(value >> (8 * byte));
            return std::size_t{8};
        };
        if (state[1] == 0)
        {
            ++state[1];
            return word(integer.sign());
        }
        constexpr auto limb_bytes = sizeof(boost::multiprecision::limb_type);
        auto top = integer.limbs()[integer.size() - 1];
        std::size_t top_bytes = 1;
        while (top >>= 8)
            ++top_bytes;
        const auto magnitude_bytes = (integer.size() - 1) * limb_bytes + top_bytes;
        if (state[1] == 1)
        {
            ++state[1];
            return word(magnitude_bytes);
        }
        const auto bytes = std::min(output.size(), magnitude_bytes - state[2]);
        for (std::size_t byte = 0; byte < bytes; ++byte)
        {
            const auto offset = state[2] + byte;
            output[byte] = static_cast<char>(integer.limbs()[offset / limb_bytes] >> (8 * (offset % limb_bytes)));
        }
        state[2] += bytes;
        if (state[2] == magnitude_bytes)
        {
            ++state[0];
            state[1] = state[2] = 0;
        }
        return bytes;
    }
    bool encoded_fields_equal(const details::GeometryIdentityNativePayload& other) const noexcept override
    {
        if (other.native_kind() != native_kind())
            return false;
        const auto& that = static_cast<const NativeTuple&>(other);
        if (tag != that.tag)
            return false;
        for (std::size_t i = 0; i < Count; ++i)
            if (!equal_integer(values[i].backend().num(), that.values[i].backend().num()) ||
                !equal_integer(values[i].backend().denom(), that.values[i].backend().denom()))
                return false;
        return true;
    }
};
namespace
{
template<std::size_t Count> std::array<Q, Count> read_tuple(std::span<const char> key, GeometryTag tag)
{
    WireReader reader{key, 0, true};
    if (reader.tag() != tag)
        throw std::invalid_argument("Foreign Exact identity body kind.");
    std::array<Q, Count> values;
    for (auto& value : values)
        value = read_identity_rational(reader);
    reader.finish();
    return values;
}
template<std::size_t Count> std::vector<char> write_tuple(const std::array<Q, Count>& values, GeometryTag tag)
{
    std::vector<char> bytes;
    const details::GeometryIdentityCanonicalWriter writer{&bytes, [](void* context, std::span<const char> fragment)
        {
            auto& output = *static_cast<std::vector<char>*>(context);
            output.insert(output.end(), fragment.begin(), fragment.end());
        }};
    writer.tag(tag);
    for (const auto& value : values)
        emit_rational(writer, value);
    return bytes;
}
template<std::size_t Count> std::size_t tuple_copy_storage_bound(const std::array<Q, Count>& values) noexcept
{
    auto bytes = sizeof(NativeTuple<Count>);
    const Q fresh;
    const auto integer = [&](const auto& source, const auto& destination)
    {
        if (source.size() > destination.capacity())
            bytes = sum(bytes, product(std::max(product(4, destination.capacity()), source.size()),
                                   sizeof(boost::multiprecision::limb_type)));
    };
    for (const auto& value : values)
    {
        integer(value.backend().num(), fresh.backend().num());
        integer(value.backend().denom(), fresh.backend().denom());
    }
    return bytes;
}
template<std::size_t Count> std::size_t tuple_moved_storage_bound(const std::array<Q, Count>& values) noexcept
{
    auto bytes = sizeof(NativeTuple<Count>);
    for (const auto& value : values)
    {
        bytes = sum(bytes, dynamic_limbs(value.backend().num()));
        bytes = sum(bytes, dynamic_limbs(value.backend().denom()));
    }
    return bytes;
}
template<std::size_t Count> std::size_t import_storage_bound(std::span<const char> key, GeometryTag tag)
{
    const auto values = read_tuple<Count>(key, tag);
    return sum(sizeof(details::GeometryIdentityBacking), tuple_moved_storage_bound(values));
}
template<std::size_t Count>
details::GeometryIdentityBackingPtr import_tuple(std::span<const char> key, GeometryTag tag, std::size_t byte_budget)
{
    auto values = read_tuple<Count>(key, tag);
    const auto bound = sum(sizeof(details::GeometryIdentityBacking), tuple_moved_storage_bound(values));
    if (sum(bound, sizeof(const details::GeometryIdentityBacking*)) > byte_budget)
        throw std::invalid_argument("Exact identity retained native payload and scratch exceed its byte budget.");
    auto native = std::make_unique<const NativeTuple<Count>>(std::move(values), tag);
    const auto retained = sum(sizeof(details::GeometryIdentityBacking), native->storage_bytes());
    if (retained > byte_budget)
        throw std::invalid_argument("Exact identity retained capacity exceeds its byte budget.");
    return details::make_geometry_identity_backing(std::move(native), {}, byte_budget - retained);
}
template<class Identity, std::size_t Count> Identity issue(const std::array<Q, Count>& values, GeometryTag tag)
{
    auto native = std::make_unique<const NativeTuple<Count>>(values, tag);
    return details::GeometryIdentityAccess::share<Identity>(std::move(native));
}
template<std::size_t Count> const std::array<Q, Count>& backing_tuple(const details::GeometryIdentityBackingPtr& owner)
{
    const auto& payload = details::geometry_identity_native(owner);
    constexpr auto expected = Count == 3   ? details::GeometryIdentityNativeKind::ExactTuple3
                              : Count == 4 ? details::GeometryIdentityNativeKind::ExactTuple4
                                           : details::GeometryIdentityNativeKind::ExactTuple6;
    static_assert(Count == 3 || Count == 4 || Count == 6);
    if (payload.native_kind() != expected)
        throw std::invalid_argument("Exact identity has a foreign native tuple kind.");
    return static_cast<const NativeTuple<Count>&>(payload).values;
}
template<std::size_t Count, class Identity> const std::array<Q, Count>& native_tuple(const Identity& identity)
{
    return backing_tuple<Count>(details::GeometryIdentityAccess::backing(identity));
}
template<std::size_t Count>
std::strong_ordering compare_tuples(const std::array<Q, Count>& first, const std::array<Q, Count>& second)
{
    for (std::size_t coordinate = 0; coordinate < Count; ++coordinate)
    {
        if (first[coordinate] < second[coordinate])
            return std::strong_ordering::less;
        if (second[coordinate] < first[coordinate])
            return std::strong_ordering::greater;
    }
    return std::strong_ordering::equal;
}
} // namespace

PointIdentity<ArithmeticMode::Exact> make_point_identity(const P& value)
{
    return issue<PointIdentity<ArithmeticMode::Exact>>(value, point_tag);
}
DisplacementIdentity<ArithmeticMode::Exact> make_displacement_identity(const P& value)
{
    return zero(value) ? DisplacementIdentity<ArithmeticMode::Exact>{}
                       : issue<DisplacementIdentity<ArithmeticMode::Exact>>(value, displacement_tag);
}
LineIdentity<ArithmeticMode::Exact> make_line_identity(const P& direction, const P& moment)
{
    return issue<LineIdentity<ArithmeticMode::Exact>>(
        std::array<Q, 6>{direction[0], direction[1], direction[2], moment[0], moment[1], moment[2]}, line_tag);
}
PlaneSupportIdentity<ArithmeticMode::Exact> make_support_identity(const std::array<Q, 4>& value)
{
    return issue<PlaneSupportIdentity<ArithmeticMode::Exact>>(value, support_tag);
}
const P& authority_point(const PointIdentity<ArithmeticMode::Exact>& identity)
{
    return native_tuple<3>(identity);
}
const P& displacement_point(const DisplacementIdentity<ArithmeticMode::Exact>& identity)
{
    static const P zero_displacement{};
    return identity.is_zero() ? zero_displacement : native_tuple<3>(identity);
}
const std::array<Q, 6>& line_coefficients(const LineIdentity<ArithmeticMode::Exact>& identity)
{
    return native_tuple<6>(identity);
}
const std::array<Q, 4>& support_coefficients(const PlaneSupportIdentity<ArithmeticMode::Exact>& identity)
{
    return native_tuple<4>(identity);
}

// These raw readers/writers are confined to canonical-key and new-wire import boundaries.
P authority_point(std::span<const char> key)
{
    return read_tuple<3>(key, point_tag);
}
P displacement_point(std::span<const char> key)
{
    return key.empty() ? P{} : read_tuple<3>(key, displacement_tag);
}
std::vector<char> point_packet(const P& value)
{
    return write_tuple(value, point_tag);
}
std::vector<char> displacement_packet(const P& value)
{
    return zero(value) ? std::vector<char>{} : write_tuple(value, displacement_tag);
}

std::size_t point_identity_storage_bound(const P& value) noexcept
{
    return sum(sizeof(details::GeometryIdentityBacking),
        sum(tuple_copy_storage_bound(value), sizeof(const details::GeometryIdentityBacking*)));
}

} // namespace SimpleFluid::PLIC::exact_geometry_detail

namespace SimpleFluid::PLIC::details
{
std::strong_ordering compare_exact_identity_native(
    GeometryIdentityKind kind, const GeometryIdentityBackingPtr& first, const GeometryIdentityBackingPtr& second)
{
    using namespace exact_geometry_detail;
    if (first == second)
        return std::strong_ordering::equal;
    switch (kind)
    {
        case GeometryIdentityKind::Point:
            return compare_tuples(backing_tuple<3>(first), backing_tuple<3>(second));
        case GeometryIdentityKind::Line:
            return compare_tuples(backing_tuple<6>(first), backing_tuple<6>(second));
        case GeometryIdentityKind::PlaneSupport:
            return compare_tuples(backing_tuple<4>(first), backing_tuple<4>(second));
        case GeometryIdentityKind::Displacement: {
            static const P zero_displacement{};
            return compare_tuples(first ? backing_tuple<3>(first) : zero_displacement,
                second ? backing_tuple<3>(second) : zero_displacement);
        }
        default:
            throw std::invalid_argument("Unsupported Exact identity native comparison kind.");
    }
}
GeometryIdentityBackingPtr import_exact_identity_backing(
    GeometryIdentityKind kind, std::span<const char> checked_key, std::size_t byte_budget)
{
    using namespace exact_geometry_detail;
    switch (kind)
    {
        case GeometryIdentityKind::Point:
            return import_tuple<3>(checked_key, point_tag, byte_budget);
        case GeometryIdentityKind::Line:
            return import_tuple<6>(checked_key, line_tag, byte_budget);
        case GeometryIdentityKind::PlaneSupport:
            return import_tuple<4>(checked_key, support_tag, byte_budget);
        case GeometryIdentityKind::Displacement:
            return checked_key.empty() ? GeometryIdentityBackingPtr{}
                                       : import_tuple<3>(checked_key, displacement_tag, byte_budget);
        default:
            throw std::invalid_argument("Unsupported Exact identity body kind.");
    }
}
std::size_t exact_identity_import_storage_bound(GeometryIdentityKind kind, std::span<const char> checked_key)
{
    using namespace exact_geometry_detail;
    switch (kind)
    {
        case GeometryIdentityKind::Point:
            return import_storage_bound<3>(checked_key, point_tag);
        case GeometryIdentityKind::Line:
            return import_storage_bound<6>(checked_key, line_tag);
        case GeometryIdentityKind::PlaneSupport:
            return import_storage_bound<4>(checked_key, support_tag);
        case GeometryIdentityKind::Displacement:
            return checked_key.empty() ? 0 : import_storage_bound<3>(checked_key, displacement_tag);
        default:
            throw std::invalid_argument("Unsupported Exact identity body kind.");
    }
}
} // namespace SimpleFluid::PLIC::details
