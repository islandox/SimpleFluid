/** @file DistributedAdaptiveWire.hh
 * @brief Internal bounded owner-message primitives, independent of MPI layout.
 */
#pragma once
#include "geometry/mesh/AMR/DistributedAdaptiveTypes.hh"
#include "parallel/SparseOwnerExchange.hh"
#include <Teuchos_CommHelpers.hpp>
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <stdexcept>
#include <string>
#include <type_traits>
namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
struct Budget
{
    std::size_t limit, used = 0, peak = 0, cumulative = 0;
    void add(std::size_t bytes)
    {
        if (bytes > limit - used)
            throw std::invalid_argument("Distributed adaptive payload budget exhausted: used=" + std::to_string(used) +
                                        " request=" + std::to_string(bytes) + " limit=" + std::to_string(limit) + ".");
        used += bytes;
        peak = std::max(peak, used);
        cumulative += bytes;
    }
    void packet(std::size_t payload)
    {
        add(sizeof(OwnerPacket));
        add(payload);
    }
    /** Admit vector reallocation while its previous allocation is still live.
     * Release the old charge only after reserve has destroyed that allocation. */
    template<class T> void append_capacity(std::vector<T>& values)
    {
        if (values.size() < values.capacity())
            return;
        const auto previous = values.capacity();
        if (previous > limit / sizeof(T) / 2)
            throw std::invalid_argument("Distributed adaptive vector capacity budget exhausted.");
        const auto next = previous ? 2 * previous : 1;
        add(next * sizeof(T));
        values.reserve(next);
        if (values.capacity() != next)
            throw std::invalid_argument("Distributed adaptive vector capacity differs from admission.");
        used -= previous * sizeof(T);
    }
};
struct Writer
{
    AdaptiveBytes data;
    void byte(std::uint8_t value) { data.push_back(static_cast<char>(value)); }
    void word(std::uint64_t n)
    {
        for (unsigned i = 0; i < 8; ++i)
            data.push_back(char(n >> (8 * i)));
    }
    void bytes(std::span<const char> b)
    {
        word(b.size());
        data.insert(data.end(), b.begin(), b.end());
    }
};
struct Reader
{
    std::span<const char> data;
    std::size_t offset = 0;
    std::uint8_t byte()
    {
        if (offset == data.size())
            throw std::invalid_argument("Truncated distributed adaptive tag.");
        return static_cast<std::uint8_t>(data[offset++]);
    }
    std::uint64_t word()
    {
        if (data.size() - offset < 8)
            throw std::invalid_argument("Truncated distributed adaptive message.");
        std::uint64_t n = 0;
        for (unsigned i = 0; i < 8; ++i)
            n |= std::uint64_t(static_cast<unsigned char>(data[offset++])) << (8 * i);
        return n;
    }
    AdaptiveBytes bytes()
    {
        auto n = word();
        if (n > data.size() - offset)
            throw std::invalid_argument("Invalid distributed adaptive message length.");
        AdaptiveBytes b(data.begin() + offset, data.begin() + offset + n);
        offset += n;
        return b;
    }
    void done()
    {
        if (offset != data.size())
            throw std::invalid_argument("Trailing distributed adaptive bytes.");
    }
};
/** Wire records have independent versions; canonical keys never use this stamp. */
enum class PacketFamily : std::uint8_t
{
    GeometryRecord = 4,
    AdaptiveBatch = 5
};
enum class GeometryRecordKind : std::uint8_t
{
    Tile = 1,
    SurfaceRequest,
    EdgeRequest,
    EdgeReply,
    NodeRequest,
    NodeReply,
    ClosureContact,
    ClosureRequests,
    ClosureReply,
    GhostRequest,
    GhostReply,
    GhostNode,
    FrozenPart,
    FrozenBatch,
    ClosureSpanContact, ///< Lossless axis spans replace the original full contact shell.
    ClosureSpanRequests ///< Framed batches carrying ClosureSpanContact records.
};
struct GeometryRecordHeader
{
    PacketFamily family;
    std::uint8_t version;
    GeometryRecordKind kind;
};
struct BatchHeader
{
    PacketFamily family;
    std::uint8_t version;
    std::uint64_t dictionary_count;
};
static_assert(std::is_standard_layout_v<GeometryRecordHeader> &&
              std::is_trivially_copyable_v<GeometryRecordHeader>);
static_assert(std::is_standard_layout_v<BatchHeader> && std::is_trivially_copyable_v<BatchHeader>);
inline constexpr std::uint8_t geometry_record_version = 3;
inline constexpr std::uint8_t batch_version = 7;
inline constexpr std::size_t geometry_record_header_bytes = sizeof(GeometryRecordHeader);
template<class PacketWriter> inline void write_geometry_record_header(PacketWriter& writer, GeometryRecordKind kind)
{
    std::array<std::uint8_t, geometry_record_header_bytes> bytes{};
    bytes[offsetof(GeometryRecordHeader, family)] = static_cast<std::uint8_t>(PacketFamily::GeometryRecord);
    bytes[offsetof(GeometryRecordHeader, version)] = geometry_record_version;
    bytes[offsetof(GeometryRecordHeader, kind)] = static_cast<std::uint8_t>(kind);
    for (const auto byte : bytes)
        writer.byte(byte);
}
inline void read_geometry_record_header(Reader& reader, GeometryRecordKind kind)
{
    std::array<std::uint8_t, geometry_record_header_bytes> bytes{};
    for (auto& byte : bytes)
        byte = reader.byte();
    if (bytes[offsetof(GeometryRecordHeader, family)] != static_cast<std::uint8_t>(PacketFamily::GeometryRecord) ||
        bytes[offsetof(GeometryRecordHeader, version)] != geometry_record_version ||
        bytes[offsetof(GeometryRecordHeader, kind)] != static_cast<std::uint8_t>(kind))
        throw std::invalid_argument("Unsupported adaptive geometry record family, version or kind.");
    for (std::size_t i = 0; i < geometry_record_header_bytes; ++i)
        if (i != offsetof(GeometryRecordHeader, family) && i != offsetof(GeometryRecordHeader, version) &&
            i != offsetof(GeometryRecordHeader, kind) && bytes[i])
            throw std::invalid_argument("Nonzero adaptive geometry record header padding.");
}
template<class F> void collective(const Teuchos::Comm<int>& comm, F&& action)
{
    std::exception_ptr error;
    try
    {
        action();
    }
    catch (...)
    {
        error = std::current_exception();
    }
    int bad = bool(error), any = 0;
    Teuchos::reduceAll(comm, Teuchos::REDUCE_MAX, 1, &bad, &any);
    if (any)
    {
        if (error)
            std::rethrow_exception(error);
        throw std::invalid_argument("Distributed adaptation rejected on another rank.");
    }
}
inline int authority(std::span<const char> key, int ranks)
{
    return int(adaptive_route_hash(key) % std::uint64_t(ranks));
}
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
