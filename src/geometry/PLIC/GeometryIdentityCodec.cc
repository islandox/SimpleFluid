#include "geometry/PLIC/GeometryIdentityCodec.hh"
#include "geometry/PLIC/details/GeometryIdentityAccess.hh"
#include "geometry/PLIC/details/OrdinaryIdentitySupport.hh"
#include "geometry/PLIC/details/exact/ConvexCellGeometryInternal.hh"
#include "geometry/PLIC/details/exact/GeometryIdentity.hh"
#include "geometry/PLIC/details/ordinary/NativeGeometrySupport.hh"

namespace SimpleFluid::PLIC
{
namespace
{
namespace on = details::ordinary_native;
namespace ex = exact_geometry_detail;
constexpr GeometryTag exact_point_tag = GeometryTag::ExactPoint;
constexpr GeometryTag exact_line_tag = GeometryTag::ExactLine;
constexpr GeometryTag exact_support_tag = GeometryTag::ExactPlaneSupport;
constexpr GeometryTag exact_displacement_tag = GeometryTag::ExactDisplacement;
void exact_packet(std::span<const char> packet, GeometryTag tag, unsigned count)
{
    ex::WireReader reader{packet, 0, true};
    if (reader.tag() != tag)
        throw std::invalid_argument("Foreign geometry identity kind or arithmetic mode.");
    std::array<ex::Q, 6> values;
    for (unsigned i = 0; i < count; ++i)
        values[i] = reader.rational();
    reader.finish();
    if (tag == exact_line_tag || tag == exact_support_tag)
    {
        unsigned first = 0;
        while (first < 3 && values[first] == 0)
            ++first;
        if (first == 3 || values[first] != 1)
            throw std::invalid_argument("Exact line/support direction is not in provider normal form.");
        if (tag == exact_line_tag && values[0] * values[3] + values[1] * values[4] + values[2] * values[5] != 0)
            throw std::invalid_argument("Exact line moment is not orthogonal to its direction.");
    }
}
/** Ordinary supports retain construction expressions. Exact support bytes can
 * occur as authenticated ancestry after the numerical-shell source bridge;
 * this is an ancestry variant, never a conversion of plane coefficients. */
void ordinary_support(std::span<const char> packet)
{
    for (;;)
    {
        on::Reader reader{packet, 0, true};
        switch (reader.tag())
        {
            case GeometryTag::OrdinaryAxisSupport:
                if (reader.word() > 2)
                    on::reject("invalid support axis.");
                (void) reader.real();
                break;
            case GeometryTag::OrdinarySourceSupport: {
                const auto count = reader.count(4096);
                if (count < 3)
                    on::reject("invalid source support count.");
                auto previous = reader.point();
                for (std::size_t i = 1; i < count; ++i)
                {
                    const auto point = reader.point();
                    if (point < previous)
                        on::reject("source support points are not sorted.");
                    previous = point;
                }
                break;
            }
            case GeometryTag::OrdinaryInputSupport: {
                const auto normal = reader.point();
                const auto first = std::find_if(normal.begin(), normal.end(), [](auto value) { return value != 0; });
                if (first == normal.end() || *first < 0)
                    on::reject("input support normal is zero or has a noncanonical sign.");
                (void) reader.point();
                (void) reader.real();
                break;
            }
            case GeometryTag::OrdinaryTranslatedSupport: {
                const auto ancestry = reader.packet();
                if (reader.point() == on::WidePoint{})
                    on::reject("zero support translation must retain its parent key.");
                reader.finish();
                packet = ancestry;
                continue;
            }
            case GeometryTag::OrdinaryPlaneSupport:
                if (reader.word() > 2)
                    on::reject("invalid support axis.");
                (void) reader.point();
                (void) reader.real();
                break;
            case GeometryTag::OrdinaryExtrusionDagSupport: {
                const auto count = reader.count(packet.size());
                if (!count)
                    on::reject("empty support construction DAG.");
                for (std::size_t i = 0; i < count; ++i)
                {
                    on::Reader record{reader.packet(), 0, true};
                    const auto kind = static_cast<SupportRecipeKind>(record.byte());
                    if (kind == SupportRecipeKind::Original)
                        (void) record.point();
                    else if (kind == SupportRecipeKind::Derived)
                    {
                        const auto parents = record.count(record.bytes.size());
                        if (!parents)
                            on::reject("empty support construction parents.");
                        std::uint64_t previous = 0;
                        for (std::size_t j = 0; j < parents; ++j)
                        {
                            const auto parent = record.word();
                            if (parent >= i)
                                on::reject("invalid support construction ancestry.");
                            if (j && parent < previous)
                                on::reject("support construction parents are not sorted.");
                            previous = parent;
                        }
                    }
                    else
                        on::reject("invalid support construction kind.");
                    record.finish();
                }
                const auto first = reader.word(), second = reader.word();
                if (first >= count || second >= count)
                    on::reject("invalid support construction endpoints.");
                if (first > second)
                    on::reject("support construction endpoints are not sorted.");
                break;
            }
            case exact_support_tag:
                exact_packet(packet, exact_support_tag, 4);
                return;
            default:
                on::reject("foreign support identity kind or arithmetic mode.");
        }
        reader.finish();
        return;
    }
}
} // namespace
std::vector<char> details::checked_identity_key(
    GeometryIdentityKind kind, ArithmeticMode mode, std::span<const char> packet)
{
    if (kind == GeometryIdentityKind::Displacement && packet.empty())
        return {};
    if (mode == ArithmeticMode::Exact)
    {
        switch (kind)
        {
            case GeometryIdentityKind::Point:
                exact_packet(packet, exact_point_tag, 3);
                break;
            case GeometryIdentityKind::Line:
                exact_packet(packet, exact_line_tag, 6);
                break;
            case GeometryIdentityKind::PlaneSupport:
                exact_packet(packet, exact_support_tag, 4);
                break;
            case GeometryIdentityKind::Displacement:
                exact_packet(packet, exact_displacement_tag, 3);
                if (ex::zero(ex::displacement_point(packet)))
                    throw std::invalid_argument("Zero displacement requires an empty canonical body.");
                break;
        }
    }
    else if (mode == ArithmeticMode::Ordinary)
    {
        switch (kind)
        {
            case GeometryIdentityKind::Point: {
                on::Reader reader{packet, 0, true};
                if (reader.tag() != on::point_tag)
                    on::reject("foreign point identity kind.");
                (void) reader.point();
                reader.finish();
                break;
            }
            case GeometryIdentityKind::Displacement: {
                on::Reader reader{packet, 0, true};
                if (reader.tag() != on::displacement_tag)
                    on::reject("foreign displacement identity kind.");
                if (reader.point() == on::WidePoint{})
                    on::reject("zero displacement requires an empty canonical body.");
                reader.finish();
                break;
            }
            case GeometryIdentityKind::PlaneSupport:
                ordinary_support(packet);
                break;
            case GeometryIdentityKind::Line: {
                on::Reader reader{packet, 0, true};
                if (reader.tag() != GeometryTag::OrdinaryLine)
                    on::reject("foreign line identity kind or arithmetic mode.");
                const auto axis = reader.word();
                if (axis > 2)
                    on::reject("invalid finite line axis.");
                const auto direction = reader.point();
                if (!(direction[axis] > 0))
                    on::reject("finite line selected component must be positive.");
                (void) reader.point();
                reader.finish();
                break;
            }
        }
    }
    else
        throw std::invalid_argument("Unsupported geometry identity arithmetic mode.");
    return {packet.begin(), packet.end()};
}
PointIdentity<ArithmeticMode::Exact> promote_identity(const PointIdentity<ArithmeticMode::Ordinary>& identity)
{
    const auto source = details::GeometryIdentityAccess::point(identity);
    return ex::make_point_identity({ex::exact_wide(source[0]), ex::exact_wide(source[1]), ex::exact_wide(source[2])});
}
DisplacementIdentity<ArithmeticMode::Exact> promote_identity(
    const DisplacementIdentity<ArithmeticMode::Ordinary>& identity)
{
    const auto source = details::GeometryIdentityAccess::displacement(identity);
    return ex::make_displacement_identity(
        {ex::exact_wide(source[0]), ex::exact_wide(source[1]), ex::exact_wide(source[2])});
}
namespace details
{
OrdinaryIdentityPoint import_ordinary_identity_point(GeometryIdentityKind kind, std::span<const char> key)
{
    if (kind == GeometryIdentityKind::Displacement && key.empty())
        return {};
    on::Reader reader{key};
    const auto tag = reader.tag();
    if (tag != (kind == GeometryIdentityKind::Point ? on::point_tag : on::displacement_tag))
        on::reject("foreign native point/displacement import kind.");
    auto value = reader.point();
    reader.finish();
    return value;
}
OrdinaryIdentityLine import_ordinary_identity_line(std::span<const char> key)
{
    on::Reader reader{key};
    if (reader.tag() != GeometryTag::OrdinaryLine)
        on::reject("foreign native line import kind.");
    const auto axis = reader.word();
    const auto direction = reader.point(), anchor = reader.point();
    reader.finish();
    return {axis, direction, anchor};
}
GeometryIdentityBackingPtr import_geometry_identity_backing(
    GeometryIdentityKind kind, ArithmeticMode mode, std::vector<char> checked_key, std::size_t budget)
{
    if (mode == ArithmeticMode::Exact)
        return import_exact_identity_backing(kind, std::move(checked_key), budget);
    if (mode == ArithmeticMode::Ordinary && kind == GeometryIdentityKind::PlaneSupport)
        return import_ordinary_support_backing(std::move(checked_key), budget);
    throw std::invalid_argument("Unsupported native identity import kind or mode.");
}
} // namespace details
} // namespace SimpleFluid::PLIC
