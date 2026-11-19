#include "geometry/details/AllocationDomainScope.hh"

#if defined(__ELF__) && (defined(__GNUC__) || defined(__clang__))
namespace SimpleFluid::geometry_detail
{
extern "C" AllocationDomain* simplefluid_test_allocation_domain() noexcept
{
    static thread_local AllocationDomain domain = AllocationDomain::Unattributed;
    return &domain;
}
} // namespace SimpleFluid::geometry_detail
#endif
