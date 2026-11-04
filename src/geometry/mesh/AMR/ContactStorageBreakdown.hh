#pragma once
#include <cstddef>
#include <span>
namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
/** Direct ledger categories, never differences of arena totals. Point-credit
 * admissions can also be referenced by the incidence pool: union their explicit
 * owner identities through visit_admissions() instead of summing both owners. */
struct ContactStorageBreakdown
{
    std::size_t group_metadata = 0, owner_index_metadata = 0;
    std::size_t domain_metadata = 0, domain_shared_body = 0;
    std::size_t fact_metadata = 0, fact_view_metadata = 0;
    std::size_t point_credit_metadata = 0, point_shared_body = 0;
    std::size_t binding_metadata = 0, fan_owner_metadata = 0, fan_shared_body = 0;
    [[nodiscard]] std::size_t private_metadata_bytes() const noexcept
    {
        return group_metadata + owner_index_metadata + domain_metadata + fact_metadata + fact_view_metadata +
            point_credit_metadata + binding_metadata + fan_owner_metadata;
    }
    [[nodiscard]] std::size_t shared_body_bytes() const noexcept
    {
        return domain_shared_body + point_shared_body + fan_shared_body;
    }
};
/** An explicit ledger owner: metadata identity names the admitted private
 * owner; physical body identity names independently admitted shareable storage.
 * Both byte counts come from that owner's ledger and allocation capacities. */
struct ContactStorageAdmission
{
    const void* metadata_owner;
    std::size_t private_metadata;
    const void* body_owner;
    std::size_t shared_body;
    bool immutable_body = true;
};
template<class T> struct ContactBorrowedPage
{
    const void* credit_owner;
    const void* physical_owner;
    std::size_t credit_object_size, page_object_size;
    std::span<const T> records;
    std::size_t record_capacity;
};
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
