#include "geometry/mesh/AMR/DistributedAdaptiveHierarchy.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveNative.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveWire.hh"
#include "geometry/mesh/AMR/DistributedCartesianKernel.hh"
#include "geometry/mesh/AMR/DistributedMixedKernel.hh"
#include "geometry/mesh/AMR/DistributedProductKernel.hh"
#include "geometry/mesh/AMR/NativePublicationProof.hh"
#include "utils/testing_environment.hh"
#include <Tpetra_Core.hpp>
#include <array>
#include <cmath>
#include <gtest/gtest.h>

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
/** Read-only test observation; this class has no production proof issuer. */
class NativePublicationTestAccess
{
public:
    static bool qualified(const std::shared_ptr<const AdaptiveProviderKernel>& kernel)
    {
        return NativeKernelAuthority::qualified(kernel);
    }
    static std::weak_ptr<const NativeKernelAuthority> authority(
        const std::shared_ptr<const AdaptiveProviderKernel>& kernel)
    {
        return NativeKernelAuthority::observation(kernel);
    }
};
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail

namespace
{
using namespace SimpleFluid;
namespace A = Meshes::AMR;
namespace D = A::adaptive_detail;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);

class MutableWrapper final : public A::AdaptiveProviderKernel
{
public:
    explicit MutableWrapper(std::shared_ptr<const A::AdaptiveProviderKernel> base)
        : A::AdaptiveProviderKernel(*base), base_(std::move(base))
    {
    }
    A::AdaptiveBytes descriptor() const override { return base_->descriptor(); }
    PLIC::ConvexGeometryPtr<ArithmeticMode::Ordinary> geometry(const A::AdaptiveLeaf& leaf) const override
    {
        return base_->geometry(leaf);
    }
    std::vector<A::AdaptiveSurface> surfaces(const A::AdaptiveLeaf& leaf) const override
    {
        return base_->surfaces(leaf);
    }
    A::AdaptiveSplitResult split(const A::AdaptiveLeaf& leaf, A::AdaptiveSplit split) const override
    {
        return base_->split(leaf, split);
    }
    bool has_native_cartesian_geometry() const noexcept override { return true; }

private:
    std::shared_ptr<const A::AdaptiveProviderKernel> base_;
};

A::AdaptiveBootstrap source()
{
    A::AMRGrid grid;
    grid.block_cells = {2, 2, 2};
    A::CartesianAMR mesh(grid);
    const std::array<std::uint64_t, 1> selection{0};
    return mesh.detach_owned(selection, {'p', 'r', 'o', 'o', 'f'});
}

TEST(NativePublicationKernelAuthority, OnlyFinalFactoriesQualifyAndWeakObservationDoesNotOwnKernel)
{
    auto cartesian = source().kernel;
    EXPECT_TRUE(D::NativePublicationTestAccess::qualified(cartesian));
    EXPECT_TRUE(D::NativePublicationTestAccess::qualified(A::product_adaptive_kernel()));
    EXPECT_TRUE(D::NativePublicationTestAccess::qualified(A::convex_adaptive_kernel(4, 1e-12)));
    auto custom = std::make_shared<const MutableWrapper>(cartesian);
    EXPECT_FALSE(D::NativePublicationTestAccess::qualified(custom));
    auto foreign_control = std::shared_ptr<const A::AdaptiveProviderKernel>(cartesian.get(), [](const auto*) {});
    ASSERT_EQ(foreign_control.get(), cartesian.get());
    EXPECT_FALSE(D::NativePublicationTestAccess::qualified(foreign_control));
    foreign_control.reset();
    const auto authority = D::NativePublicationTestAccess::authority(cartesian);
    std::weak_ptr<const A::AdaptiveProviderKernel> weak = cartesian;
    custom.reset();
    cartesian.reset();
    EXPECT_TRUE(weak.expired());
    EXPECT_TRUE(authority.expired());
}

TEST(NativePublicationKernelAuthority, FixedWeakEnvelopesDoNotRetainKernelOrDynamicMixedChild)
{
    auto child = source().kernel;
    std::weak_ptr<const A::AdaptiveProviderKernel> child_observation = child;
    A::DistributedMixedRoute route;
    route.identity = 1;
    route.provider = A::AdaptiveProviderKind::Cartesian;
    route.kernel = child;
    route.compact_root_schema = {'s'};
    std::vector<A::DistributedMixedRoute> routes;
    routes.push_back(std::move(route));
    auto kernel = A::mixed_adaptive_kernel(std::move(routes), 64 * 1024 * 1024);
    child.reset();
    ASSERT_TRUE(D::NativeKernelAuthority::qualified(kernel));
    std::weak_ptr<const A::AdaptiveProviderKernel> kernel_observation = kernel;
    auto authority_observation = D::NativeKernelAuthority::observation(kernel);
    auto kernel_envelope = D::NativeKernelAuthority::kernel_allocation_observation(kernel);
    auto authority_envelope = D::NativeKernelAuthority::allocation_observation(kernel);
    auto schema = D::NativeKernelAuthority::schema_observation(kernel);
    const auto expected_schema = kernel->descriptor();
    ASSERT_TRUE(schema);
    EXPECT_TRUE(schema->matches(expected_schema));
    ASSERT_TRUE(kernel_envelope);
    ASSERT_TRUE(authority_envelope);
    std::weak_ptr<const void> counter_observation = kernel_envelope->state()->counter_owner;
    kernel.reset();
    EXPECT_TRUE(kernel_observation.expired());
    EXPECT_TRUE(authority_observation.expired());
    EXPECT_TRUE(child_observation.expired());
    EXPECT_FALSE(kernel_envelope->object_alive());
    EXPECT_FALSE(authority_envelope->object_alive());
    EXPECT_GT(kernel_envelope->fixed_allocation_bytes(), 0u);
    EXPECT_GT(authority_envelope->fixed_allocation_bytes(), 0u);
    EXPECT_FALSE(counter_observation.expired());
    kernel_observation.reset();
    authority_observation.reset();
    kernel_envelope.reset();
    authority_envelope.reset();
    EXPECT_FALSE(counter_observation.expired()); // detached schema still owns only its bytes/domain
    EXPECT_TRUE(schema->matches(expected_schema));
    schema.reset();
    EXPECT_TRUE(counter_observation.expired());
}

TEST(NativePublicationKernelAuthority, BorrowedSchemaReusesItsActualOwnerAndOutlivesSourceControllers)
{
    A::AMRGrid grid;
    grid.block_cells = {2, 2, 2};
    A::CartesianAMR mesh(grid);
    const auto comm = Tpetra::getDefaultComm();
    std::vector<std::uint64_t> owned;
    for (std::uint64_t id = 0; id < 8; ++id)
        if (id % comm->getSize() == std::uint64_t(comm->getRank()))
            owned.push_back(id);
    auto bootstrap = mesh.detach_owned(owned, {'s', 'c', 'h', 'e', 'm', 'a'});
    std::weak_ptr<const A::AdaptiveProviderKernel> kernel_observation = bootstrap.kernel;
    auto hierarchy = std::make_shared<const A::DistributedAdaptiveHierarchy>(
        std::move(bootstrap.owned), std::move(bootstrap.families), std::move(bootstrap.kernel), comm);
    std::weak_ptr<const A::DistributedAdaptiveHierarchy> source_observation = hierarchy;
    const auto expected = hierarchy->configuration_descriptor(); // explicit independent compatibility export
    auto first = A::native_publication_schema(*hierarchy);
    auto second = A::native_publication_schema(*hierarchy);
    EXPECT_EQ(first.span().data(), second.span().data());
    EXPECT_EQ(first.span().size(), expected.size());
    EXPECT_TRUE(std::ranges::equal(first.span(), expected));
    hierarchy.reset();
    EXPECT_TRUE(source_observation.expired());
    EXPECT_TRUE(kernel_observation.expired());
    EXPECT_TRUE(std::ranges::equal(first.span(), expected));
    EXPECT_TRUE(std::ranges::equal(second.span(), expected));
}

TEST(NativePublicationSourceBinding, ExactControlsAndAllSourceFieldsAreRequired)
{
    auto bootstrap = source();
    auto leaf = bootstrap.owned.front();
    auto arena = std::make_shared<D::Budget>(D::Budget{64 * 1024 * 1024});
    const auto binding = D::NativeSourceBinding::capture(leaf, arena);
    EXPECT_TRUE(binding->matches(leaf));
    auto changed = leaf;
    ++changed.revision;
    EXPECT_FALSE(binding->matches(changed));
    changed = leaf;
    changed.material ^= 1;
    EXPECT_FALSE(binding->matches(changed));
    changed = leaf;
    ++changed.native_cell;
    EXPECT_FALSE(binding->matches(changed));
    changed = leaf;
    changed.domain.push_back('x');
    EXPECT_FALSE(binding->matches(changed));
    changed = leaf;
    changed.parent_family.push_back('x');
    EXPECT_FALSE(binding->matches(changed));
    changed = leaf;
    changed.work.population = -0.0;
    ASSERT_TRUE(std::signbit(changed.work.population));
    EXPECT_FALSE(binding->matches(changed));
    changed = leaf;
    changed.native_recipe = std::make_shared<const Meshes::NativeFragmentDescriptor>(*leaf.native_recipe);
    EXPECT_FALSE(binding->matches(changed));
    changed.native_recipe =
        std::shared_ptr<const Meshes::NativeFragmentDescriptor>(leaf.native_recipe.get(), [](const auto*) {});
    ASSERT_EQ(changed.native_recipe.get(), leaf.native_recipe.get());
    EXPECT_FALSE(binding->matches(changed));
}

TEST(NativePublicationSourceBinding, DescriptorAndFactorChildControlSnapshotsDetectRebinding)
{
    auto bootstrap = source();
    auto leaf = bootstrap.owned.front();
    auto recipe = std::make_shared<Meshes::NativeFragmentDescriptor>(*leaf.native_recipe);
    leaf.native_recipe = recipe;
    auto factor = std::make_shared<A::AdaptiveFactorRecipe>();
    factor->root = std::make_shared<const A::AdaptiveBytes>(A::AdaptiveBytes{'r'});
    factor->base = std::make_shared<const A::AdaptiveBytes>(A::AdaptiveBytes{'b'});
    factor->axis = std::make_shared<const A::AdaptiveBytes>(A::AdaptiveBytes{'a'});
    leaf.factor_recipe = factor;
    auto arena = std::make_shared<D::Budget>(D::Budget{64 * 1024 * 1024});
    const auto binding = D::NativeSourceBinding::capture(leaf, arena);
    ASSERT_TRUE(binding->matches(leaf));
    const auto original = recipe->parts.front();
    recipe->parts.front() = std::make_shared<const A::AdaptiveBytes>(*original);
    EXPECT_FALSE(binding->matches(leaf));
    recipe->parts.front() = original;
    ASSERT_TRUE(binding->matches(leaf));
    recipe->view.push_back('x');
    EXPECT_FALSE(binding->matches(leaf));
    recipe->view.pop_back();
    ASSERT_TRUE(binding->matches(leaf));
    factor->base = std::make_shared<const A::AdaptiveBytes>(*factor->base);
    EXPECT_FALSE(binding->matches(leaf));
}

TEST(NativePublicationSourceBinding, OneByteShortFailsBeforeCopyAndPreservesAcceptedSelector)
{
    auto bootstrap = source();
    const auto& leaf = bootstrap.owned.front();
    auto accepted_arena = std::make_shared<D::Budget>(D::Budget{64 * 1024 * 1024});
    auto accepted = D::NativeSourceBinding::capture(leaf, accepted_arena);
    const auto exact_bound = accepted_arena->used;
    ASSERT_GT(exact_bound, 0u);
    auto short_arena = std::make_shared<D::Budget>(D::Budget{exact_bound - 1});
    EXPECT_THROW((void) D::NativeSourceBinding::capture(leaf, short_arena), std::invalid_argument);
    EXPECT_EQ(short_arena->used, 0u);
    EXPECT_EQ(accepted_arena->used, exact_bound);
    EXPECT_TRUE(accepted->matches(leaf));
    auto exact_arena = std::make_shared<D::Budget>(D::Budget{exact_bound});
    auto retry = D::NativeSourceBinding::capture(leaf, exact_arena);
    EXPECT_TRUE(retry->matches(leaf));
    retry.reset();
    EXPECT_EQ(exact_arena->used, 0u);
    accepted.reset();
    EXPECT_EQ(accepted_arena->used, 0u);
}
} // namespace
