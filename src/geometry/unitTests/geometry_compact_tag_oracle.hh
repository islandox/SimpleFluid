#pragma once
#include "geometry/PLIC/GeometryCodecTags.hh"
#include <span>
#include <stdexcept>
#include <vector>

namespace compact_tag_oracle
{
using SimpleFluid::PLIC::GeometryTag;
inline std::uint64_t read_word(std::span<const char> bytes, std::size_t& cursor)
{
    if (cursor > bytes.size() || bytes.size() - cursor < 8)
        throw std::logic_error("Truncated independent key oracle.");
    std::uint64_t result = 0;
    for (unsigned i = 0; i < 8; ++i)
        result |= std::uint64_t(static_cast<unsigned char>(bytes[cursor++])) << (8 * i);
    return result;
}
inline void word(std::vector<char>& bytes, std::uint64_t value)
{
    for (unsigned i = 0; i < 8; ++i)
        bytes.push_back(static_cast<char>(value >> (8 * i)));
}
inline std::uint64_t original_tag(GeometryTag tag)
{
    switch (tag)
    {
        case GeometryTag::OrdinaryPoint:
            return 0x53464650504e5431ULL;
        case GeometryTag::OrdinaryLine:
            return 0x534646504c494e31ULL;
        case GeometryTag::OrdinaryDisplacement:
            return 0x5346465044535031ULL;
        case GeometryTag::OrdinaryAxisSupport:
            return 0x53464e4158495331ULL;
        case GeometryTag::OrdinarySourceSupport:
            return 0x53464e534f555231ULL;
        case GeometryTag::OrdinaryInputSupport:
            return 0x53464e494e505431ULL;
        case GeometryTag::OrdinaryPlaneSupport:
            return 0x5346465053505431ULL;
        case GeometryTag::OrdinaryTranslatedSupport:
            return 0x53464e5452414e31ULL;
        case GeometryTag::OrdinaryExtrusionDagSupport:
            return 0x5346454441475331ULL;
        case GeometryTag::ExactPoint:
            return 0x5346504f494e5431ULL;
        case GeometryTag::ExactLine:
            return 0x53464c494e453631ULL;
        case GeometryTag::ExactPlaneSupport:
            return 0x53464a504c414e36ULL;
        case GeometryTag::ExactDisplacement:
            return 0x5346444953503631ULL;
        case GeometryTag::OrdinaryPlanarFace:
            return 0x5346465046414332ULL;
        case GeometryTag::ExactPlanarFace:
            return 0x5346464143454331ULL;
        default:
            throw std::logic_error("Unexpected independent key oracle category.");
    }
}
// Independent schema expansion, not a production legacy reader: only category
// widths, record-kind widths and enclosing lengths change. Numeric bytes are
// copied unchanged and checked against fingerprints captured before migration.
inline std::vector<char> expand(std::span<const char> compact)
{
    if (compact.empty())
        return {};
    const auto tag = static_cast<GeometryTag>(static_cast<std::uint8_t>(compact.front()));
    std::vector<char> original;
    word(original, original_tag(tag));
    std::size_t cursor = 1;
    const auto blob = [&]()
    {
        const auto size = read_word(compact, cursor);
        if (size > compact.size() - cursor)
            throw std::logic_error("Invalid independent key oracle blob.");
        const auto body = compact.subspan(cursor, size);
        cursor += size;
        return body;
    };
    if (tag == GeometryTag::OrdinaryPlanarFace || tag == GeometryTag::OrdinaryTranslatedSupport)
    {
        const auto parent = expand(blob());
        word(original, parent.size());
        original.insert(original.end(), parent.begin(), parent.end());
    }
    else if (tag == GeometryTag::OrdinaryExtrusionDagSupport)
    {
        const auto count = read_word(compact, cursor);
        word(original, count);
        for (std::uint64_t i = 0; i < count; ++i)
        {
            const auto record = blob();
            if (record.empty())
                throw std::logic_error("Empty independent key oracle record.");
            std::vector<char> fields;
            word(fields, static_cast<std::uint8_t>(record.front()));
            fields.insert(fields.end(), record.begin() + 1, record.end());
            word(original, fields.size());
            original.insert(original.end(), fields.begin(), fields.end());
        }
    }
    else if (tag == GeometryTag::ExactPlanarFace)
    {
        if (cursor == compact.size())
            throw std::logic_error("Missing independent key oracle version.");
        word(original, static_cast<std::uint8_t>(compact[cursor++]));
    }
    original.insert(original.end(), compact.begin() + cursor, compact.end());
    return original;
}
} // namespace compact_tag_oracle
