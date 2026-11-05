#include "cartesian_incidence_test_helpers.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveIncidence.hh"
#include "utils/testing_environment.hh"
#include <Teuchos_DefaultSerialComm.hpp>
#include <algorithm>
#include <array>
#include <cstdlib>
#include <exception>
#include <gtest/gtest.h>
#include <limits>
#include <new>
#include <type_traits>
namespace
{
struct AllocationAudit
{
    std::size_t requests = 0, largest = 0;
    std::size_t request_cap = std::numeric_limits<std::size_t>::max();
    bool exceeded = false;
};
thread_local AllocationAudit* audit = nullptr;
void observe_allocation(std::size_t size)
{
    if (!audit)
        return;
    ++audit->requests;
    audit->largest = std::max(audit->largest, size);
    if (size > audit->request_cap)
    {
        audit->exceeded = true;
        throw std::bad_alloc();
    }
}
struct AuditScope
{
    explicit AuditScope(AllocationAudit& value) { audit = &value; }
    ~AuditScope() { audit = nullptr; }
};
} // namespace
void* operator new(std::size_t size)
{
    observe_allocation(size);
    if (void* result = std::malloc(size ? size : 1))
        return result;
    throw std::bad_alloc();
}
void* operator new[](std::size_t size)
{
    return ::operator new(size);
}
void operator delete(void* value) noexcept
{
    std::free(value);
}
void operator delete[](void* value) noexcept
{
    std::free(value);
}
void operator delete(void* value, std::size_t) noexcept
{
    std::free(value);
}
void operator delete[](void* value, std::size_t) noexcept
{
    std::free(value);
}
void* operator new(std::size_t size, std::align_val_t alignment)
{
    observe_allocation(size);
    void* result = nullptr;
    if (!posix_memalign(&result, static_cast<std::size_t>(alignment), size ? size : 1))
        return result;
    throw std::bad_alloc();
}
void* operator new[](std::size_t size, std::align_val_t alignment)
{
    return ::operator new(size, alignment);
}
void operator delete(void* value, std::align_val_t) noexcept
{
    std::free(value);
}
void operator delete[](void* value, std::align_val_t) noexcept
{
    std::free(value);
}
void operator delete(void* value, std::size_t, std::align_val_t) noexcept
{
    std::free(value);
}
void operator delete[](void* value, std::size_t, std::align_val_t) noexcept
{
    std::free(value);
}
namespace
{
using namespace cartesian_incidence_test;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);
static_assert(!std::is_default_constructible_v<ContactSourcePointAdmission>);
static_assert(std::is_same_v<decltype(std::declval<const IncidencePointDirectory<>&>().get(0)), ContactSourcePoint>);
TEST(CartesianIncidenceAllocations, WideBlockSingleLeafNeverRequestsOverIts8192ByteSourceCap)
{
    auto value = grid();
    value.block_cells = {8192, 2, 2};
    value.max_level = 0;
    // Original descriptor construction is outside the audited source operation.
    // The regression was source validation rebuilding a 131072-byte block array.
    const auto leaf = cartesian_adaptive_leaf(value, {value.region_id, CartesianCellAncestry{0, {0, 0, 0}}}, {'d'});
    Budget budget{8192};
    AllocationAudit actual;
    actual.request_cap = budget.limit;
    std::shared_ptr<const CartesianIncidenceSource> source;
    std::exception_ptr failure;
    {
        AuditScope scope(actual);
        try
        {
            source = CartesianIncidenceSource::make(value, std::span(&leaf, 1), {'d'}, 0, 1, {}, budget);
        }
        catch (...)
        {
            failure = std::current_exception();
        }
    }
    EXPECT_FALSE(failure);
    EXPECT_FALSE(actual.exceeded);
    EXPECT_LE(actual.largest, budget.limit);
    ASSERT_TRUE(source);
    EXPECT_EQ(source->cell_count(), 1);
    EXPECT_LE(budget.used, budget.limit);
    source.reset();
    EXPECT_EQ(budget.used, 0);
}
TEST(CartesianIncidenceAllocations, CertifiedPointValueReadsAllocateNothingAndPreserveCallerEnvironment)
{
    RestoreEnvironment restore;
    const auto value = grid();
    const auto native = leaves(value);
    const auto kernel = cartesian_adaptive_kernel(value);
    auto budget = std::make_shared<Budget>(Budget{ample});
    auto comm = Teuchos::rcp(new Teuchos::SerialComm<int>);
    IdentityBudget identities;
    DistributedAdaptiveIncidenceSession session(*kernel, comm, {}, {}, *budget, identities, budget);
    const auto lookup = [&](const AdaptiveCellKey& key) -> const AdaptiveLeaf&
    { return *std::find_if(native.begin(), native.end(), [&](const auto& leaf) { return leaf.key == key; }); };
    const auto rows = session.bootstrap(native, lookup);
    ASSERT_EQ(rows.size(), 8);
    const auto snapshot = session.point_pool().source();
    ASSERT_TRUE(snapshot);
    ASSERT_EQ(snapshot->size(), 27);
    EXPECT_EQ(snapshot->exception_count(), 0);
    EXPECT_EQ(snapshot->component_counts(), (std::array<std::size_t, 3>{3, 3, 3}));
    EXPECT_EQ(snapshot->selector_width(), 1);
    ASSERT_EQ(std::fesetround(FE_UPWARD), 0);
    ASSERT_EQ(std::feclearexcept(FE_ALL_EXCEPT), 0);
    ASSERT_EQ(std::feraiseexcept(FE_DIVBYZERO), 0);
    const auto used = budget->used, spent = budget->cumulative;
    const int flags = std::fetestexcept(FE_ALL_EXCEPT);
    bool equal = true;
    AllocationAudit actual;
    {
        AuditScope scope(actual);
        for (std::size_t i = 0; i != snapshot->size(); ++i)
            equal = equal && session.point_pool().get(i) == snapshot->value({i});
    }
    EXPECT_TRUE(equal);
    EXPECT_EQ(actual.requests, 0);
    EXPECT_EQ(std::fegetround(), FE_UPWARD);
    EXPECT_EQ(std::fetestexcept(FE_ALL_EXCEPT), flags);
    EXPECT_EQ(budget->used, used);
    EXPECT_EQ(budget->cumulative, spent);
}
} // namespace
