/** @file GeometryWireProtocol.hh
 * @brief Versioned entry stamps for phase records carrying geometry packets.
 */
#pragma once
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace SimpleFluid::VoF::geometry_wire
{
inline constexpr std::uint64_t magic = 0x5346564f46574952ULL;
inline constexpr std::uint8_t version = 2;
inline constexpr std::size_t header_bytes = sizeof(std::uint64_t) + 2 * sizeof(std::uint8_t);
enum class Record : std::uint8_t
{
    NativeShellRequest = 1,
    NativeShellReply,
    ExactTileRequest,
    ExactTileReply,
    ExactFaceOutput,
    ExactControls,
    NodeUnion,
    NodePublication,
    ExactAtomRoute
};
template<class Writer> void write_header(Writer& writer, Record record)
{
    writer.word(magic);
    writer.byte(version);
    writer.byte(static_cast<std::uint8_t>(record));
}
template<class Reader> void read_header(Reader& reader, Record record)
{
    if (reader.word() != magic || reader.byte() != version || reader.byte() != static_cast<std::uint8_t>(record))
        throw std::invalid_argument("VoF geometry record version or kind differs.");
}
} // namespace SimpleFluid::VoF::geometry_wire
