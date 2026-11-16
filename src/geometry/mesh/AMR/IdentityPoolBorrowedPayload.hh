#pragma once
#include "ContactPayloadCensus.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveTypes.hh"
#include <type_traits>
#include <variant>

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
/** Typed observation seam for payload owned by a pool record. Inline records
 * need no extra visit. Shared PLIC backing belongs to its separate context
 * owner and is reached through borrowed records, not charged again here. */
template<class T> struct IdentityPoolBorrowedPayload
{
    static constexpr bool complete = std::is_arithmetic_v<T> || std::is_enum_v<T>;
    template<class Visitor> static void visit(const T&, Visitor&) {}
};
template<PLIC::GeometryIdentityKind Kind, ArithmeticMode Mode>
struct IdentityPoolBorrowedPayload<PLIC::GeometryIdentity<Kind, Mode>>
{
    static constexpr bool complete = !PLIC::GeometryIdentity<Kind, Mode>::shared_storage;
    template<class Visitor> static void visit(const PLIC::GeometryIdentity<Kind, Mode>&, Visitor&) {}
};
template<> struct IdentityPoolBorrowedPayload<AdaptiveBytes>
{
    static constexpr bool complete = true;
    template<class Visitor> static void visit(const AdaptiveBytes& value, Visitor& visitor)
    {
        contact_payload_buffer(visitor, &value, BasicContactPayloadRole::IdentityDynamicBytes, value);
    }
};
template<> struct IdentityPoolBorrowedPayload<AdaptiveCellKey>
{
    static constexpr bool complete = true;
    template<class Visitor> static void visit(const AdaptiveCellKey& value, Visitor& visitor)
    {
        std::visit(
            [&](const auto& ancestry)
            {
                using A = std::decay_t<decltype(ancestry)>;
                if constexpr (std::is_same_v<A, ConvexCellAncestry>)
                    contact_payload_buffer(visitor, &value, BasicContactPayloadRole::IdentityCellPath, ancestry.path);
                else if constexpr (std::is_same_v<A, ProductCellAncestry>)
                {
                    contact_payload_buffer(
                        visitor, &value, BasicContactPayloadRole::IdentityCellPath, ancestry.base.path);
                    contact_payload_buffer(
                        visitor, &value, BasicContactPayloadRole::IdentityCellPath, ancestry.axial.path);
                }
            },
            value.ancestry);
    }
};
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
