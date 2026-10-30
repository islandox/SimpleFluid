#include "geometry/PLIC/GeometryIdentityCodec.hh"
#include "geometry/PLIC/details/GeometryIdentityAccess.hh"
#include "geometry/PLIC/details/exact/ConvexCellGeometryInternal.hh"
#include "geometry/PLIC/details/ordinary/NativeGeometrySupport.hh"

namespace SimpleFluid::PLIC
{
namespace
{
namespace on = details::ordinary_native;
namespace ex = exact_geometry_detail;
constexpr std::uint64_t exact_point_tag = 0x5346504f494e5431ULL;
constexpr std::uint64_t exact_line_tag = 0x53464c494e453631ULL;
constexpr std::uint64_t exact_support_tag = 0x53464a504c414e36ULL;
constexpr std::uint64_t exact_displacement_tag = 0x5346444953503631ULL;
void exact_packet(std::span<const char> packet, std::uint64_t tag, unsigned count)
{
    ex::WireReader reader{packet, 0, true};
    if (reader.word() != tag)
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
        switch (reader.word())
        {
            case 0x53464e4158495331ULL:
                if (reader.word() > 2)
                    on::reject("invalid support axis.");
                (void) reader.real();
                break;
            case 0x53464e534f555231ULL: {
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
            case 0x53464e494e505431ULL: {
                const auto normal = reader.point();
                const auto first = std::find_if(normal.begin(), normal.end(), [](auto value) { return value != 0; });
                if (first == normal.end() || *first < 0)
                    on::reject("input support normal is zero or has a noncanonical sign.");
                (void) reader.point();
                (void) reader.real();
                break;
            }
            case 0x53464e5452414e31ULL: {
                const auto ancestry = reader.packet();
                if (reader.point() == on::WidePoint{})
                    on::reject("zero support translation must retain its parent key.");
                reader.finish();
                packet = ancestry;
                continue;
            }
            case 0x5346465053505431ULL:
                if (reader.word() > 2)
                    on::reject("invalid support axis.");
                (void) reader.point();
                (void) reader.real();
                break;
            case 0x5346454441475331ULL: {
                const auto count = reader.count(packet.size());
                if (!count)
                    on::reject("empty support construction DAG.");
                for (std::size_t i = 0; i < count; ++i)
                {
                    on::Reader record{reader.packet(), 0, true};
                    const auto kind = record.word();
                    if (kind == 0)
                        (void) record.point();
                    else if (kind == 1)
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
                if (reader.word() != on::point_tag)
                    on::reject("foreign point identity kind.");
                (void) reader.point();
                reader.finish();
                break;
            }
            case GeometryIdentityKind::Displacement: {
                on::Reader reader{packet, 0, true};
                if (reader.word() != on::displacement_tag)
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
                if (reader.word() != 0x534646504c494e31ULL)
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
    const auto source = on::read_point(details::GeometryIdentityAccess::key(identity));
    return details::GeometryIdentityAccess::make<PointIdentity<ArithmeticMode::Exact>>(
        ex::point_packet({ex::exact_wide(source[0]), ex::exact_wide(source[1]), ex::exact_wide(source[2])}));
}
DisplacementIdentity<ArithmeticMode::Exact> promote_identity(
    const DisplacementIdentity<ArithmeticMode::Ordinary>& identity)
{
    const auto source = on::read_displacement(details::GeometryIdentityAccess::key(identity));
    return details::GeometryIdentityAccess::make<DisplacementIdentity<ArithmeticMode::Exact>>(
        ex::displacement_packet({ex::exact_wide(source[0]), ex::exact_wide(source[1]), ex::exact_wide(source[2])}));
}
} // namespace SimpleFluid::PLIC
