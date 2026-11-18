/** Unsupported optimized allocation profiles preserve existing public behavior. */
#include "geometry/mesh/AMR/DistributedAdaptiveHierarchy.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveNative.hh"
#include "geometry/mesh/AMR/NativeKernelAuthority.hh"
#include "geometry/mesh/AMR/NativePublicationSession.hh"
#include "parallel/details/NativeTpetraMapAdmission.hh"
#include "utils/testing_environment.hh"
#include <cstdlib>
#include <optional>
#include <string>
#include <gtest/gtest.h>

namespace
{
using namespace SimpleFluid;
namespace A = Meshes::AMR;
namespace D = A::adaptive_detail;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);

class PublicRuntimeProfile
{
public:
    PublicRuntimeProfile()
    {
        if (const char* value = std::getenv("TPETRA_VERBOSE")) original_ = value;
        // This is a supported public Tpetra configuration, but its presence is
        // outside the finite native map allocation profile. It changes no
        // factory authority and does not install a test validation callback.
        if (setenv("TPETRA_VERBOSE", "0", 1)) throw std::runtime_error("Cannot configure public runtime profile.");
    }
    ~PublicRuntimeProfile()
    {
        if (original_) setenv("TPETRA_VERBOSE", original_->c_str(), 1);
        else unsetenv("TPETRA_VERBOSE");
    }
private:
    std::optional<std::string> original_;
};

TEST(NativePublicationProfileCompatibility, UnsupportedOutputProfileKeepsFactoryAndFullVerifierPath)
{
    PublicRuntimeProfile public_profile;
    EXPECT_FALSE(geometry_detail::NativeTpetraMapAdmission::profile_supported());
    EXPECT_THROW(geometry_detail::NativeTpetraMapAdmission::require_profile(), std::invalid_argument);
    A::AMRGrid grid;
    grid.block_cells = {2, 2, 2};
    auto kernel = A::cartesian_adaptive_kernel(grid);
    ASSERT_TRUE(D::NativeKernelAuthority::qualified(kernel));
    EXPECT_FALSE(D::NativeKernelAuthority::supports_source_validation(kernel));

    A::CartesianAMR original(grid);
    const auto comm = Tpetra::getDefaultComm();
    std::vector<std::uint64_t> selected;
    for (std::uint64_t cell = 0; cell != 8; ++cell)
        if (cell % comm->getSize() == std::uint64_t(comm->getRank())) selected.push_back(cell);
    auto detached = original.detach_owned(selected, {'f', 'a', 'l', 'l', 'b', 'a', 'c', 'k'});
    const auto source = std::make_shared<const A::DistributedAdaptiveHierarchy>(std::move(detached.owned),
        std::move(detached.families), std::move(detached.kernel), comm);
    const auto input = D::NativePublicationSession::capture(*source);
    EXPECT_FALSE(input.source_coverage);
    EXPECT_FALSE(input.cache->indexes);
    const auto schema = A::native_publication_schema(*source);
    const auto output = A::publish_adaptive_native(*source, schema);
    ASSERT_TRUE(output.partition());
    ASSERT_TRUE(output.mesh());
    EXPECT_EQ(output.partition()->plan().owned_cells().size(), selected.size());
    const auto registry = output.key_registry_storage();
    EXPECT_TRUE(registry.owns_key_maps);
    EXPECT_GE(registry.scalar_route_bytes, output.local_keys().size() * sizeof(D::NativePublicationKeyRoute));
    const auto owned = output.owned_gids();
    auto at = owned.begin();
    std::size_t ordinal = 0;
    for (const auto& leaf : source->owned_leaves())
    {
        ASSERT_NE(at, owned.end());
        EXPECT_EQ(at, owned.find(leaf.key));
        const auto entry = *at++;
        EXPECT_EQ(entry.first, leaf.key);
        EXPECT_EQ(entry.second, output.partition()->plan().owned_cells()[ordinal++]);
        EXPECT_EQ(owned.at(leaf.key), entry.second);
    }
    EXPECT_EQ(at, owned.end());
    const A::AdaptiveCellKey missing{grid.region_id, A::CartesianCellAncestry{0, {8, 0, 0}}};
    EXPECT_EQ(owned.find(missing), owned.end());
    EXPECT_FALSE(owned.contains(missing));
    EXPECT_THROW((void) owned.at(missing), std::out_of_range);
    for (const auto cell : output.partition()->plan().owned_cells())
        EXPECT_DOUBLE_EQ(output.partition()->geometry()->cell_volume(cell), .125);
    EXPECT_FALSE(D::NativePublicationSession::receipt_matches(input,
        output.partition(), output.mesh()->owned_cell_map()));
    const auto bound = source->with_partition(output.partition(), output.mesh()->owned_cell_map());
    EXPECT_EQ(bound->owned_leaves().size(), selected.size());
    EXPECT_TRUE(bound->geometry_unchanged());
    EXPECT_FALSE(D::NativePublicationSession::capture(*bound).source_coverage);
}
} // namespace
