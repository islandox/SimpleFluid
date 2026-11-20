#pragma once
#include "BasicContactPayloadRole.hh"
#include "GeometryContactPayloadRole.hh"
#include "ParallelSupportPayloadRole.hh"
#include <compare>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
/** Census categories have independent ordinals. ContactPayloadRole retains
 * the category alongside that ordinal, so heterogeneous visitor identities and
 * ordered containers cannot alias two roles whose underlying value is zero.
 * This local diagnostic tag is not a wire identifier or a geometry identity. */
enum class ContactPayloadCategory : std::uint8_t
{
    Basic,
    ParallelSupport,
    Geometry
};
class ContactPayloadRole
{
public:
    constexpr ContactPayloadRole() noexcept = default;
    constexpr ContactPayloadRole(BasicContactPayloadRole role) noexcept
        : category_(ContactPayloadCategory::Basic), ordinal_(static_cast<std::uint8_t>(role))
    {
    }
    constexpr ContactPayloadRole(ParallelSupportPayloadRole role) noexcept
        : category_(ContactPayloadCategory::ParallelSupport), ordinal_(static_cast<std::uint8_t>(role))
    {
    }
    constexpr ContactPayloadRole(GeometryContactPayloadRole role) noexcept
        : category_(ContactPayloadCategory::Geometry), ordinal_(static_cast<std::uint8_t>(role))
    {
    }
    [[nodiscard]] constexpr ContactPayloadCategory category() const noexcept { return category_; }
    [[nodiscard]] constexpr std::uint8_t ordinal() const noexcept { return ordinal_; }
    auto operator<=>(const ContactPayloadRole&) const = default;
    bool operator==(const ContactPayloadRole&) const = default;

private:
    ContactPayloadCategory category_ = ContactPayloadCategory::Basic;
    std::uint8_t ordinal_ = 0;
};
enum class ContactPayloadKind : std::uint8_t
{
    Object,
    Buffer,
    NodeValue
};
struct ContactPayloadIdentity
{
    const void* address;
    ContactPayloadRole role;
    bool operator==(const ContactPayloadIdentity&) const = default;
};
/** Actual, known C++ object/buffer payload only. No admission ledger or 128-byte
 * allowance enters this record. Unknown allocator, shared-control and tree-node
 * overhead is explicitly excluded. Identity is role-qualified; embedded objects
 * are omitted and accounted only by their containing object. */
struct ContactPayloadAllocation
{
    ContactPayloadIdentity identity;
    const void* owner;
    ContactPayloadKind kind;
    std::size_t object_size, element_size, live_count, capacity, payload_bytes;
    bool allocator_overhead_excluded = true;
    bool control_overhead_excluded = false;
    bool node_overhead_excluded = false;
    bool opaque_element_payload_excluded = false;
};
template<class Visitor>
void contact_payload_object(
    Visitor& visitor, const void* object, ContactPayloadRole role, std::size_t bytes, bool shared_control = false)
{
    visitor(ContactPayloadAllocation{
        {object, role}, object, ContactPayloadKind::Object, bytes, bytes, 1, 1, bytes, true, shared_control, false});
}
template<class Visitor, class T, class Allocator>
void contact_payload_buffer(Visitor& visitor, const void* owner, ContactPayloadRole role,
    const std::vector<T, Allocator>& values,
    std::size_t live, bool opaque_element_payload = false)
{
    if (values.capacity())
        visitor(ContactPayloadAllocation{{values.data(), role}, owner, ContactPayloadKind::Buffer, 0, sizeof(T), live,
            values.capacity(), sizeof(T) * values.capacity(), true, false, false, opaque_element_payload});
}
template<class Visitor, class T, class Allocator>
void contact_payload_buffer(Visitor& visitor, const void* owner, ContactPayloadRole role,
    const std::vector<T, Allocator>& values)
{
    contact_payload_buffer(visitor, owner, role, values, values.size());
}
template<class Visitor, class Pair>
void contact_payload_node_value(Visitor& visitor, const void* owner, ContactPayloadRole role, const Pair& value)
{
    visitor(ContactPayloadAllocation{{&value, role}, owner, ContactPayloadKind::NodeValue, sizeof(Pair), sizeof(Pair),
        1, 1, sizeof(Pair), true, false, true});
}
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
