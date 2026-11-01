/** @file VoFWireProtocol.hh
 * @brief Typed VoF and supporting AMR headers with deterministic native-layout padding.
 */
#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <type_traits>

namespace SimpleFluid
{
using std::uint8_t;
}

namespace SimpleFluid::packet_codec
{
enum class Family : uint8_t
{
    CapillaryPlanes = 6,
    CapillaryNodeFans = 7,
    AggregateBudgets = 8,
    OrdinaryCurvature = 9,
    ExactPhaseWorking = 10,
    ExactPhaseSource = 11,
    ExperimentalPhaseInventory = 12,
    AdaptiveStateSchema = 13,
    ConstructionControls = 14,
    InterfacePlane = 15,
    PhaseCleanup = 16,
    OperationAgreement = 17,
    BalanceForecast = 18,
    LimiterExchange = 19,
    AdaptiveRecords = 20,
    MixedJoin = 22,
    ProductRoot = 23,
    ProductBase = 24,
    ProductAxis = 25,
    NativeFragment = 26,
    ExtrudedTopologyBase = 27
};
enum class Record : uint8_t
{
    Value = 1,
    Request = 2,
    Reply = 3
};
template<class Kind> struct Header
{
    static_assert(std::is_enum_v<Kind> && sizeof(Kind) == 1);
    Family family;
    uint8_t version;
    Kind kind;
};
template<class Kind> inline constexpr std::size_t header_bytes = sizeof(Header<Kind>);

template<class Writer, class Kind> void write_header(Writer& writer, Family family, uint8_t version, Kind kind)
{
    using H = Header<Kind>;
    static_assert(std::is_standard_layout_v<H> && std::is_trivially_copyable_v<H>);
    std::array<uint8_t, sizeof(H)> bytes{};
    bytes[offsetof(H, family)] = static_cast<uint8_t>(family);
    bytes[offsetof(H, version)] = version;
    bytes[offsetof(H, kind)] = static_cast<uint8_t>(kind);
    for (const auto byte : bytes)
        writer.byte(byte);
}
template<class Reader, class Kind>
Kind read_header(Reader& reader, Family family, uint8_t version, std::initializer_list<Kind> allowed)
{
    using H = Header<Kind>;
    std::array<uint8_t, sizeof(H)> bytes{};
    for (auto& byte : bytes)
        byte = reader.byte();
    if (bytes[offsetof(H, family)] != static_cast<uint8_t>(family) || bytes[offsetof(H, version)] != version)
        throw std::invalid_argument("Unsupported packet family or version.");
    for (std::size_t i = 0; i < bytes.size(); ++i)
        if (i != offsetof(H, family) && i != offsetof(H, version) && i != offsetof(H, kind) && bytes[i])
            throw std::invalid_argument("Nonzero packet header padding.");
    for (const auto kind : allowed)
        if (bytes[offsetof(H, kind)] == static_cast<uint8_t>(kind))
            return kind;
    throw std::invalid_argument("Unsupported packet kind.");
}
template<class Reader, class Kind> void read_header(Reader& reader, Family family, uint8_t version, Kind expected)
{
    (void) read_header(reader, family, version, {expected});
}
} // namespace SimpleFluid::packet_codec
