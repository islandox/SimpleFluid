/** @file GeometryDemandPages.hh
 * @brief Private persistent geometry selectors with cumulative allocation admission.
 */
#pragma once
#include "geometry/PLIC/GeometryError.hh"
#include <cstddef>

namespace SimpleFluid::PLIC::details::geometry_demand
{
/** Borrowed only while editing. Immutable pages retain no counter or caller. */
struct Admission
{
    void* context;
    void (*retain_bytes)(void*, std::size_t);
    void charge(std::size_t count, std::size_t width) const;
};
struct PageWork
{
    std::size_t copied_pages = 0, copied_bytes = 0, copied_payload_bytes = 0;
    std::size_t visited_pages = 0, retired_page_visits = 0;
    std::size_t visited_source_rows = 0, copied_source_rows = 0;
};
/** Row copies must allocate nothing and be noexcept. Variable geometry, key and
 * certificate data belong to immutable source owners referenced by rows.
 * Include GeometryDemandPages.tcc only from a compiled implementation provider
 * or a private component test that owns its template instantiation. */
template<class Row, class Less> class OrderedPages;
} // namespace SimpleFluid::PLIC::details::geometry_demand
