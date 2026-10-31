#include "geometry/PLIC/GeometryIdentity.hh"
#include "geometry/PLIC/details/GeometryIdentityBacking.hh"
#include "geometry/PLIC/details/OrdinaryIdentitySupport.hh"
#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace SimpleFluid::PLIC::details
{

namespace
{
struct OrdinaryKeyWriter
{
    OrdinaryIdentityKey result;
    void tag(GeometryTag value) noexcept { result.bytes[result.size++] = static_cast<char>(value); }
    void word(std::uint64_t value) noexcept
    {
        for (unsigned i = 0; i < 8; ++i)
            result.bytes[result.size++] = static_cast<char>(value >> (8 * i));
    }
    void real(long double value) noexcept
    {
        if (value == 0)
        {
            word(0);
            word(0);
            return;
        }
        int exponent = 0;
        auto mantissa = std::frexp(std::abs(value), &exponent);
        word((std::uint64_t(std::uint32_t(exponent)) << 1) | (std::signbit(value) ? 1 : 0));
        word(std::numeric_limits<long double>::digits);
        for (int written = 0; written < std::numeric_limits<long double>::digits; written += 32)
        {
            mantissa = std::ldexp(mantissa, 32);
            const auto limb = static_cast<std::uint32_t>(mantissa);
            word(limb);
            mantissa -= limb;
        }
    }
    void point(const OrdinaryIdentityPoint& point) noexcept
    {
        for (auto value : point)
            real(value);
    }
};
std::size_t scalar_key_size(long double value) noexcept
{
    return value == 0 ? 16 : ordinary_identity_scalar_bytes;
}
} // namespace
OrdinaryIdentityPoint normalize_identity_point(OrdinaryIdentityPoint point)
{
    for (auto& value : point)
    {
        if (!std::isfinite(value))
            throw std::invalid_argument("Ordinary FP geometry: nonfinite native geometry coordinate.");
        if (value == 0)
            value = 0;
    }
    return point;
}
OrdinaryIdentityLine normalize_identity_line(OrdinaryIdentityLine line)
{
    line.direction = normalize_identity_point(std::move(line.direction));
    line.anchor = normalize_identity_point(std::move(line.anchor));
    if (line.axis > 2 || !(line.direction[line.axis] > 0))
        throw std::invalid_argument("Ordinary FP geometry: invalid native line axis or direction.");
    return line;
}
OrdinaryIdentityKey ordinary_identity_key(GeometryIdentityKind kind, const OrdinaryIdentityPoint& point) noexcept
{
    OrdinaryKeyWriter writer;
    if (kind == GeometryIdentityKind::Displacement && point == OrdinaryIdentityPoint{})
        return writer.result;
    writer.tag(kind == GeometryIdentityKind::Point ? GeometryTag::OrdinaryPoint : GeometryTag::OrdinaryDisplacement);
    writer.point(point);
    return writer.result;
}
OrdinaryIdentityKey ordinary_identity_key(const OrdinaryIdentityLine& line) noexcept
{
    OrdinaryKeyWriter writer;
    writer.tag(GeometryTag::OrdinaryLine);
    writer.word(line.axis);
    writer.point(line.direction);
    writer.point(line.anchor);
    return writer.result;
}
std::size_t ordinary_identity_key_size(GeometryIdentityKind kind, const OrdinaryIdentityPoint& point) noexcept
{
    if (kind == GeometryIdentityKind::Displacement && point == OrdinaryIdentityPoint{})
        return 0;
    std::size_t bytes = 1;
    for (auto value : point)
        bytes += scalar_key_size(value);
    return bytes;
}
std::size_t ordinary_identity_key_size(const OrdinaryIdentityLine& line) noexcept
{
    std::size_t bytes = 9;
    for (auto value : line.direction)
        bytes += scalar_key_size(value);
    for (auto value : line.anchor)
        bytes += scalar_key_size(value);
    return bytes;
}

std::strong_ordering compare_geometry_identity_native(GeometryIdentityKind kind, ArithmeticMode mode,
    const GeometryIdentityBackingPtr& first, const GeometryIdentityBackingPtr& second)
{
    if (first == second)
        return std::strong_ordering::equal;
    if (mode == ArithmeticMode::Exact)
        return compare_exact_identity_native(kind, first, second);
    if (mode == ArithmeticMode::Ordinary && kind == GeometryIdentityKind::PlaneSupport)
        return compare_ordinary_support_native(first, second);
    throw std::invalid_argument("Unsupported native identity ordering mode or kind.");
}

namespace
{
void dispose_identity_backing(const GeometryIdentityBacking* immutable) noexcept
{
    // Only zero-strong-count owners enter this queue: weak locks can no longer
    // resurrect them. A parent released by delete appends to the active drain
    // instead of recursively deleting the rest of an immutable expression.
    static thread_local GeometryIdentityBacking* pending = nullptr;
    static thread_local bool draining = false;
    auto* owner = const_cast<GeometryIdentityBacking*>(immutable);
    owner->disposal_next = pending;
    pending = owner;
    if (draining)
        return;
    draining = true;
    while (pending)
    {
        auto* next = pending;
        pending = next->disposal_next;
        delete next;
    }
    draining = false;
}
} // namespace
void emit_geometry_identity_canonical(
    const GeometryIdentityBackingPtr& owner, const GeometryIdentityCanonicalWriter& writer, std::size_t scratch_budget)
{
    if (!owner)
        return;
    if (owner->depth > scratch_budget / sizeof(const GeometryIdentityBacking*))
        throw std::invalid_argument("Canonical identity emission exceeds its scratch budget.");
    charge_geometry_identity_scratch(owner->depth, sizeof(const GeometryIdentityBacking*));
    std::vector<const GeometryIdentityBacking*> ancestry;
    ancestry.reserve(owner->depth);
    if (ancestry.capacity() > scratch_budget / sizeof(const GeometryIdentityBacking*))
        throw std::invalid_argument("Canonical identity emission capacity exceeds its scratch budget.");
    if (ancestry.capacity() > owner->depth)
        charge_geometry_identity_scratch(ancestry.capacity() - owner->depth, sizeof(const GeometryIdentityBacking*));
    for (auto node = owner.get(); node; node = node->parent.get())
    {
        node->native->emit_prefix(writer, node->parent ? node->parent->canonical_size : 0);
        ancestry.push_back(node);
    }
    for (auto node = ancestry.rbegin(); node != ancestry.rend(); ++node)
        (*node)->native->emit_suffix(writer);
}
GeometryIdentityCanonicalCursor::GeometryIdentityCanonicalCursor(
    const GeometryIdentityBackingPtr& owner, std::size_t scratch_budget)
    : owner_(owner), remaining_(geometry_identity_key_size(owner))
{
    if (!owner_)
        return;
    if (owner_->depth > scratch_budget / sizeof(const GeometryIdentityBacking*))
        throw std::invalid_argument("Canonical identity cursor exceeds its scratch budget.");
    charge_geometry_identity_scratch(owner_->depth, sizeof(const GeometryIdentityBacking*));
    ancestry_.reserve(owner_->depth);
    if (ancestry_.capacity() > scratch_budget / sizeof(const GeometryIdentityBacking*))
        throw std::invalid_argument("Canonical identity cursor capacity exceeds its scratch budget.");
    if (ancestry_.capacity() > owner_->depth)
        charge_geometry_identity_scratch(ancestry_.capacity() - owner_->depth, sizeof(const GeometryIdentityBacking*));
    for (auto node = owner_.get(); node; node = node->parent.get())
        ancestry_.push_back(node);
    suffix_index_ = ancestry_.size();
}
void GeometryIdentityCanonicalCursor::refill()
{
    position_ = available_ = 0;
    while (!suffix_ && prefix_index_ < ancestry_.size())
    {
        const auto* node = ancestry_[prefix_index_];
        available_ =
            node->native->next_canonical_chunk(false, state_, node->parent ? node->parent->canonical_size : 0, buffer_);
        if (available_)
            return;
        ++prefix_index_;
        state_ = {};
    }
    suffix_ = true;
    while (suffix_index_)
    {
        const auto* node = ancestry_[suffix_index_ - 1];
        available_ = node->native->next_canonical_chunk(true, state_, 0, buffer_);
        if (available_)
            return;
        --suffix_index_;
        state_ = {};
    }
    if (remaining_)
        throw std::logic_error("Native canonical identity cursor ended before its cached length.");
}
std::uint64_t GeometryIdentityCanonicalCursor::next_word()
{
    std::uint64_t word = 0;
    for (unsigned byte = 0; byte < 8 && remaining_; ++byte)
    {
        if (position_ == available_)
            refill();
        word |= std::uint64_t(static_cast<unsigned char>(buffer_[position_++])) << (8 * byte);
        --remaining_;
    }
    return word;
}
GeometryIdentityBackingPtr make_geometry_identity_backing(std::unique_ptr<const GeometryIdentityNativePayload> native,
    GeometryIdentityBackingPtr parent, std::size_t scratch_budget)
{
    if (!native)
        throw std::invalid_argument("A retained identity requires a native payload.");
    if (parent && parent->depth == std::numeric_limits<std::size_t>::max())
        throw std::length_error("Canonical identity ancestry depth overflows size_t.");
    auto result = std::shared_ptr<GeometryIdentityBacking>(new GeometryIdentityBacking, dispose_identity_backing);
    result->native = std::move(native);
    result->parent = std::move(parent);
    const auto count = [](void* context, std::span<const char> bytes)
    {
        auto& size = *static_cast<std::size_t*>(context);
        if (bytes.size() > std::numeric_limits<std::size_t>::max() - size)
            throw std::length_error("Canonical identity length overflows size_t.");
        size += bytes.size();
    };
    const auto parent_size = result->parent ? result->parent->canonical_size : 0;
    result->native->emit_prefix({&result->prefix_size, count}, parent_size);
    result->native->emit_suffix({&result->suffix_size, count});
    result->canonical_size = parent_size;
    std::array<std::size_t, 2> local{result->prefix_size, result->suffix_size};
    for (auto size : local)
    {
        if (size > std::numeric_limits<std::size_t>::max() - result->canonical_size)
            throw std::length_error("Canonical identity length overflows size_t.");
        result->canonical_size += size;
    }
    result->depth = result->parent ? result->parent->depth + 1 : 1;
    const auto hash = [](void* context, std::span<const char> bytes)
    {
        auto& value = *static_cast<std::uint64_t*>(context);
        for (char byte : bytes)
        {
            value ^= static_cast<unsigned char>(byte);
            value *= 1099511628211ULL;
        }
    };
    emit_geometry_identity_canonical(result, {&result->cached_hash, hash}, scratch_budget);
    return result;
}
const GeometryIdentityNativePayload& geometry_identity_native(const GeometryIdentityBackingPtr& owner)
{
    if (!owner || !owner->native)
        throw std::invalid_argument("Missing native geometry identity payload.");
    return *owner->native;
}
std::vector<char> geometry_identity_key(const GeometryIdentityBackingPtr& owner)
{
    std::vector<char> bytes;
    bytes.reserve(geometry_identity_key_size(owner));
    emit_geometry_identity_canonical(owner, {&bytes, [](void* context, std::span<const char> fragment)
                                                {
                                                    auto& output = *static_cast<std::vector<char>*>(context);
                                                    output.insert(output.end(), fragment.begin(), fragment.end());
                                                }});
    return bytes;
}
std::size_t geometry_identity_key_size(const GeometryIdentityBackingPtr& owner) noexcept
{
    return owner ? owner->canonical_size : 0;
}
std::uint64_t geometry_identity_key_hash(const GeometryIdentityBackingPtr& owner) noexcept
{
    return owner ? owner->cached_hash : 14695981039346656037ULL;
}
bool equal_geometry_identity_native(
    const GeometryIdentityBackingPtr& first, const GeometryIdentityBackingPtr& second) noexcept
{
    for (auto a = first.get(), b = second.get(); a != b; a = a->parent.get(), b = b->parent.get())
    {
        if (!a || !b || a->canonical_size != b->canonical_size || a->cached_hash != b->cached_hash ||
            a->native->native_kind() != b->native->native_kind() || !a->native->encoded_fields_equal(*b->native))
            return false;
    }
    return true;
}
namespace
{
char identity_byte_at(const GeometryIdentityBacking* node, std::size_t offset) noexcept
{
    while (node)
    {
        const auto parent_size = node->parent ? node->parent->canonical_size : 0;
        const bool prefix = offset < node->prefix_size;
        if (!prefix)
        {
            offset -= node->prefix_size;
            if (offset < parent_size)
            {
                node = node->parent.get();
                continue;
            }
            offset -= parent_size;
            if (offset >= node->suffix_size)
                return 0;
        }
        struct Selection
        {
            std::size_t offset;
            char value = 0;
        } selected{offset};
        const GeometryIdentityCanonicalWriter writer{&selected, [](void* context, std::span<const char> bytes)
            {
                auto& selection = *static_cast<Selection*>(context);
                if (selection.offset < bytes.size())
                {
                    selection.value = bytes[selection.offset];
                    selection.offset = std::numeric_limits<std::size_t>::max();
                }
                else if (selection.offset != std::numeric_limits<std::size_t>::max())
                    selection.offset -= bytes.size();
            }};
        if (prefix)
            node->native->emit_prefix(writer, parent_size);
        else
            node->native->emit_suffix(writer);
        return selected.value;
    }
    return 0;
}
} // namespace
std::uint64_t geometry_identity_key_word(const GeometryIdentityBackingPtr& owner, std::size_t byte_offset) noexcept
{
    std::uint64_t word = 0;
    const auto size = geometry_identity_key_size(owner);
    for (unsigned i = 0; i < 8 && byte_offset < size; ++i, ++byte_offset)
        word |= std::uint64_t(static_cast<unsigned char>(identity_byte_at(owner.get(), byte_offset))) << (8 * i);
    return word;
}
std::size_t geometry_identity_backing_bytes() noexcept
{
    return sizeof(GeometryIdentityBacking);
}
std::size_t geometry_identity_payload_bytes(const GeometryIdentityBackingPtr& owner) noexcept
{
    // Match retained-construction accounting: backing structs and buffer
    // capacities, excluding allocator/control-block overhead and RSS.
    return owner ? sizeof(GeometryIdentityBacking) + (owner->native ? owner->native->storage_bytes() : 0) : 0;
}
GeometryIdentityBackingPtr geometry_identity_parent(const GeometryIdentityBackingPtr& owner) noexcept
{
    return owner ? owner->parent : GeometryIdentityBackingPtr{};
}
} // namespace SimpleFluid::PLIC::details
