/** @file ProtocolTags.hh
 * @brief Checked byte encoding for the existing enums used by versioned VoF packets.
 */
#pragma once
#include <cstdint>
#include <stdexcept>
#include <type_traits>

namespace SimpleFluid::VoF::protocol_tags
{
using std::uint8_t;
template<class Tag, class Writer> void write(Writer& writer, Tag value, Tag maximum)
{
    static_assert(std::is_enum_v<Tag> && sizeof(Tag) == 1);
    if (std::uint64_t(value) > std::uint64_t(maximum))
        throw std::invalid_argument("Unknown VoF packet category.");
    writer.byte(static_cast<uint8_t>(value));
}
template<class Tag, class Reader> Tag read(Reader& reader, Tag maximum)
{
    static_assert(std::is_enum_v<Tag> && sizeof(Tag) == 1);
    const auto value = reader.byte();
    if (value > static_cast<uint8_t>(maximum))
        throw std::invalid_argument("Unknown VoF packet category.");
    return static_cast<Tag>(value);
}
/** Retained word-sized fields must validate before narrowing to the enum. */
template<class Tag, class Reader> Tag read_word(Reader& reader, Tag maximum)
{
    static_assert(std::is_enum_v<Tag> && sizeof(Tag) == 1);
    const auto value = reader.word();
    if (value > static_cast<std::uint64_t>(maximum))
        throw std::invalid_argument("Unknown VoF packet category.");
    return static_cast<Tag>(value);
}
} // namespace SimpleFluid::VoF::protocol_tags
