/** @file GeometryIdentity.cc
 * @brief Opaque Exact tuples with eager frozen canonical key caches.
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
template<std::size_t Count> struct NativeTuple final : details::GeometryIdentityNativePayload
{
    std::array<Q, Count> values;
    explicit NativeTuple(const std::array<Q, Count>& input) : values(input) {}
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
        value = reader.rational();
    reader.finish();
    return values;
}
template<std::size_t Count> std::vector<char> write_tuple(const std::array<Q, Count>& values, GeometryTag tag)
{
    WireWriter writer;
    writer.tag(tag);
    for (const auto& value : values)
        writer.rational(value);
    return std::move(writer.data);
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
template<std::size_t Count> std::size_t import_storage_bound(std::span<const char> key, GeometryTag tag)
{
    const auto values = read_tuple<Count>(key, tag);
    return sum(sizeof(details::GeometryIdentityBacking), sum(key.size(), tuple_copy_storage_bound(values)));
}
template<std::size_t Count>
details::GeometryIdentityBackingPtr import_tuple(std::vector<char> key, GeometryTag tag, std::size_t byte_budget)
{
    const auto values = read_tuple<Count>(key, tag);
    const auto root_and_key = sum(sizeof(details::GeometryIdentityBacking), key.capacity());
    const auto bound = sum(root_and_key, tuple_copy_storage_bound(values));
    if (bound > byte_budget)
        throw std::invalid_argument("Exact identity retained native payload exceeds its byte budget.");
    auto native = std::make_unique<const NativeTuple<Count>>(values);
    if (sum(root_and_key, native->storage_bytes()) > byte_budget)
        throw std::invalid_argument("Exact identity native allocation exceeds its preflight byte budget.");
    auto result = details::make_geometry_identity_backing(std::move(key), std::move(native));
    if (details::geometry_identity_payload_bytes(result) > byte_budget)
        throw std::invalid_argument("Exact identity retained capacity exceeds its byte budget.");
    return result;
}
template<class Identity, std::size_t Count> Identity issue(const std::array<Q, Count>& values, GeometryTag tag)
{
    auto key = write_tuple(values, tag);
    auto native = std::make_unique<const NativeTuple<Count>>(values);
    return details::GeometryIdentityAccess::share<Identity>(std::move(key), std::move(native));
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
    auto bytes = sum(sizeof(details::GeometryIdentityBacking), sizeof(NativeTuple<3>));
    std::size_t encoded = sizeof(GeometryTag);
    const Q fresh;
    const auto integer = [&](const auto& source, const auto& destination)
    {
        encoded = sum(encoded, 2 * sizeof(std::uint64_t));
        encoded =
            sum(encoded, std::max<std::size_t>(1, product(source.size(), sizeof(boost::multiprecision::limb_type))));
        if (source.size() > destination.capacity())
            bytes = sum(bytes, product(std::max(product(4, destination.capacity()), source.size()),
                                   sizeof(boost::multiprecision::limb_type)));
    };
    for (const auto& coordinate : value)
    {
        integer(coordinate.backend().num(), fresh.backend().num());
        integer(coordinate.backend().denom(), fresh.backend().denom());
    }
    return sum(bytes, product(2, encoded));
}
} // namespace SimpleFluid::PLIC::exact_geometry_detail

namespace SimpleFluid::PLIC::details
{
std::strong_ordering compare_exact_identity_native(GeometryIdentityKind kind,
    const GeometryIdentityBackingPtr& first, const GeometryIdentityBackingPtr& second)
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
    GeometryIdentityKind kind, std::vector<char> checked_key, std::size_t byte_budget)
{
    using namespace exact_geometry_detail;
    switch (kind)
    {
        case GeometryIdentityKind::Point:
            return import_tuple<3>(std::move(checked_key), point_tag, byte_budget);
        case GeometryIdentityKind::Line:
            return import_tuple<6>(std::move(checked_key), line_tag, byte_budget);
        case GeometryIdentityKind::PlaneSupport:
            return import_tuple<4>(std::move(checked_key), support_tag, byte_budget);
        case GeometryIdentityKind::Displacement:
            return checked_key.empty() ? GeometryIdentityBackingPtr{}
                                       : import_tuple<3>(std::move(checked_key), displacement_tag, byte_budget);
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
