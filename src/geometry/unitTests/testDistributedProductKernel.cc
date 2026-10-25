/** @file testDistributedProductKernel.cc
 * @brief Detached product factors, source lifetime and distributed exact inverses.
 */
#include "geometry/mesh/AMR/DistributedAdaptiveHierarchy.hh"
#include "geometry/mesh/AMR/DistributedProductKernel.hh"
#include "geometry/mesh/AMR/ExtrudedAMR.hh"
#include "utils/testing_environment.hh"
#include <Teuchos_CommHelpers.hpp>
#include <Tpetra_Core.hpp>
#include <cmath>
#include <gtest/gtest.h>
#include <numbers>
namespace
{
using namespace SimpleFluid;
using namespace SimpleFluid::Meshes;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);
::SimpleFluid::Meshes::AMR::ExtrudedAMRGrid grid()
{
    ::SimpleFluid::Meshes::AMR::ExtrudedAMRGrid g;
    for (unsigned j = 0; j < 3; ++j)
        for (unsigned i = 0; i < 3; ++i)
            g.xy_nodes.push_back({.1 * i, .1 * j, 0});
    g.base_cells = {{0, 1, 4, 3}, {1, 2, 5, 4}, {3, 4, 7, 6}, {4, 5, 8, 7}};
    g.z_edges = {0, .1, .2, .3, .4};
    g.boundary_aliases = {{"zmin", "floor"}, {"zmax", "roof"}, {"side", "wall"}};
    return g;
}
std::uint64_t global_cells(const ::SimpleFluid::Meshes::AMR::DistributedAdaptiveHierarchy& h)
{
    std::uint64_t n = h.owned().size(), sum = 0;
    Teuchos::reduceAll(*Tpetra::getDefaultComm(), Teuchos::REDUCE_SUM, 1, &n, &sum);
    return sum;
}
std::vector<::SimpleFluid::Meshes::AMR::AdaptiveMark> marks(
    const ::SimpleFluid::Meshes::AMR::DistributedAdaptiveHierarchy& h, ::SimpleFluid::Meshes::AMR::AdaptiveSplit split,
    bool refined_base, bool refined_z)
{
    std::vector<::SimpleFluid::Meshes::AMR::AdaptiveMark> result;
    for (const auto& [key, leaf] : h.owned())
    {
        const auto& p = std::get<::SimpleFluid::Meshes::AMR::ProductCellAncestry>(key.ancestry);
        if (p.base.root == 1 && p.axial.root == 0 && p.base.path.empty() != refined_base &&
            p.axial.path.empty() != refined_z)
            result.push_back({key, split});
    }
    return result;
}
std::shared_ptr<const ::SimpleFluid::Meshes::AMR::DistributedAdaptiveHierarchy> source()
{
    auto bootstrap = std::make_shared<::SimpleFluid::Meshes::AMR::ExtrudedAMR>(grid());
    std::weak_ptr<::SimpleFluid::Meshes::AMR::ExtrudedAMR> weak = bootstrap;
    const auto comm = Tpetra::getDefaultComm();
    std::vector<std::uint64_t> owned;
    for (std::uint64_t id = 0; id < 16; ++id)
        if (id % comm->getSize() == std::uint64_t(comm->getRank()))
            owned.push_back(id);
    auto detached = bootstrap->detach_owned(owned, {'p', 'r', 'o', 'd', 'u', 'c', 't'});
    bootstrap.reset();
    EXPECT_TRUE(weak.expired());
    return std::make_shared<const ::SimpleFluid::Meshes::AMR::DistributedAdaptiveHierarchy>(
        std::move(detached.owned), std::move(detached.families), detached.kernel, comm);
}
TEST(DistributedProductKernel, IndependentFactorParentsSurviveNewOtherFactorAndExactInverse)
{
    auto comm = Tpetra::getDefaultComm();
    auto root = source();
    EXPECT_EQ(global_cells(*root), 16U);
    auto xy =
        root->apply(root->plan_adaptation(marks(*root, ::SimpleFluid::Meshes::AMR::AdaptiveSplit::Base, false, false)));
    EXPECT_EQ(global_cells(*xy), 19U);
    auto both =
        xy->apply(xy->plan_adaptation(marks(*xy, ::SimpleFluid::Meshes::AMR::AdaptiveSplit::Axial, true, false)));
    EXPECT_EQ(global_cells(*both), 23U);
    std::vector<::SimpleFluid::Meshes::AMR::AdaptiveBytes> collapse;
    if (comm->getRank() == 0)
        for (unsigned z = 0; z < 2; ++z)
            collapse.push_back(::SimpleFluid::Meshes::AMR::adaptive_family_key(
                {0, ::SimpleFluid::Meshes::AMR::ProductCellAncestry{{1, 1, {}}, {0, {z}}}},
                ::SimpleFluid::Meshes::AMR::AdaptiveSplit::Base));
    auto zonly = both->apply(both->plan_adaptation({}, collapse));
    EXPECT_EQ(global_cells(*zonly), 17U);
    collapse.clear();
    if (comm->getRank() == 0)
        collapse.push_back(::SimpleFluid::Meshes::AMR::adaptive_family_key(
            {0, ::SimpleFluid::Meshes::AMR::ProductCellAncestry{{1, 1, {}}, {0, {}}}},
            ::SimpleFluid::Meshes::AMR::AdaptiveSplit::Axial));
    auto restored = zonly->apply(zonly->plan_adaptation({}, collapse));
    EXPECT_EQ(global_cells(*restored), 16U);
    for (const auto& [key, leaf] : restored->owned())
    {
        EXPECT_FALSE(leaf.geometry);
        EXPECT_TRUE(leaf.surfaces.empty());
        ASSERT_TRUE(leaf.native_recipe);
        ASSERT_TRUE(leaf.factor_recipe);
        EXPECT_EQ(leaf.native_recipe->kind, NativeFragmentDescriptor::Kind::ConstructedProduct);
        ::SimpleFluid::Meshes::AMR::product_adaptive_kernel()->validate(leaf);
    }
    auto zfirst = root->apply(
        root->plan_adaptation(marks(*root, ::SimpleFluid::Meshes::AMR::AdaptiveSplit::Axial, false, false)));
    EXPECT_EQ(global_cells(*zfirst), 17U);
    auto reversed = zfirst->apply(
        zfirst->plan_adaptation(marks(*zfirst, ::SimpleFluid::Meshes::AMR::AdaptiveSplit::Base, false, true)));
    EXPECT_EQ(global_cells(*reversed), 23U);
    double first = 0, second = 0;
    for (const auto& [key, leaf] : both->owned())
        first += ::SimpleFluid::Meshes::AMR::product_adaptive_kernel()->geometry(leaf)->moments().volume.as_double();
    for (const auto& [key, leaf] : reversed->owned())
        second += ::SimpleFluid::Meshes::AMR::product_adaptive_kernel()->geometry(leaf)->moments().volume.as_double();
    double a = 0, b = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &first, &a);
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &second, &b);
    EXPECT_NEAR(a, b, 1e-16);
    EXPECT_EQ(global_cells(*root), 16U);
}
TEST(DistributedProductKernel, RootAndAxisCapsulesAreSharedWithoutRetainedCellShells)
{
    auto state = source();
    const ::SimpleFluid::Meshes::AMR::AdaptiveBytes* root = nullptr;
    std::map<::SimpleFluid::Meshes::AMR::AxialIntervalKey, const ::SimpleFluid::Meshes::AMR::AdaptiveBytes*> axes;
    for (const auto& [key, leaf] : state->owned())
    {
        ASSERT_TRUE(leaf.factor_recipe);
        if (!root)
            root = leaf.factor_recipe->root.get();
        EXPECT_EQ(root, leaf.factor_recipe->root.get());
        const auto& p = std::get<::SimpleFluid::Meshes::AMR::ProductCellAncestry>(key.ancestry);
        auto [it, inserted] = axes.emplace(p.axial, leaf.factor_recipe->axis.get());
        if (!inserted)
            EXPECT_EQ(it->second, leaf.factor_recipe->axis.get());
        EXPECT_FALSE(leaf.geometry);
        EXPECT_TRUE(leaf.surfaces.empty());
    }
}
TEST(DistributedProductKernel, TwelveSidedFanSplitsTenChildrenAcrossOwnersAndRecoversParent)
{
    const auto comm = Tpetra::getDefaultComm();
    ::SimpleFluid::Meshes::AMR::ExtrudedAMRGrid g;
    std::vector<unsigned> polygon;
    for (unsigned i = 0; i < 12; ++i)
    {
        const double angle = 2 * std::numbers::pi * i / 12;
        g.xy_nodes.push_back({std::cos(angle), std::sin(angle), 0});
        polygon.push_back(i);
    }
    g.base_cells.push_back(polygon);
    g.z_edges = {.1, .2};
    auto bootstrap = std::make_shared<::SimpleFluid::Meshes::AMR::ExtrudedAMR>(g);
    const auto parent = bootstrap->cell_key(0);
    auto exact_parent = bootstrap->cell_geometry(0);
    std::vector<std::uint64_t> ids;
    if (comm->getRank() == 0)
        ids.push_back(0);
    auto detached = bootstrap->detach_owned(ids, {'f', 'a', 'n'});
    bootstrap.reset();
    // The radial fan has deliberately nonuniform directional widths. This
    // arity/transport gate uses an explicit admitted physical ratio, while the
    // independent one-generation trace contract remains enabled.
    ::SimpleFluid::Meshes::AMR::DistributedAdaptiveOptions options;
    options.maximum_span_ratio = 32;
    ::SimpleFluid::Meshes::AMR::DistributedAdaptiveHierarchy root(
        std::move(detached.owned), {}, detached.kernel, comm, options);
    std::vector<::SimpleFluid::Meshes::AMR::AdaptiveMark> refine;
    if (comm->getRank() == 0)
        refine.push_back({parent, ::SimpleFluid::Meshes::AMR::AdaptiveSplit::Base});
    auto fine = root.apply(root.plan_adaptation(refine));
    ASSERT_EQ(global_cells(*fine), 10U);
    std::vector<int> destinations;
    for (const auto& [key, leaf] : fine->owned())
        destinations.push_back(
            int(std::get<::SimpleFluid::Meshes::AMR::ProductCellAncestry>(key.ancestry).base.path.back() %
                comm->getSize()));
    auto distributed = fine->apply(fine->plan_migration(destinations));
    EXPECT_EQ(global_cells(*distributed), 10U);
    std::vector<::SimpleFluid::Meshes::AMR::AdaptiveBytes> families;
    if (comm->getRank() == comm->getSize() - 1)
        families.push_back(
            ::SimpleFluid::Meshes::AMR::adaptive_family_key(parent, ::SimpleFluid::Meshes::AMR::AdaptiveSplit::Base));
    auto capsules = distributed->fetch_families(families);
    for (const auto& family : capsules)
        EXPECT_EQ(family.children.size(), 10U);
    auto restored = distributed->apply(distributed->plan_adaptation({}, families));
    EXPECT_EQ(global_cells(*restored), 1U);
    for (const auto& [key, leaf] : restored->owned())
    {
        EXPECT_EQ(key, parent);
        EXPECT_TRUE(restored->geometry(leaf)->equivalent(*exact_parent));
    }
}
TEST(DistributedProductKernel, CrossedNativeXYAndAxialTracesRejectWithoutChangingSource)
{
    auto comm = Tpetra::getDefaultComm();
    auto root = source();
    std::vector<::SimpleFluid::Meshes::AMR::AdaptiveMark> axial;
    for (const auto& [key, leaf] : root->owned())
    {
        const auto& p = std::get<::SimpleFluid::Meshes::AMR::ProductCellAncestry>(key.ancestry);
        if (p.base.root == 0 && p.axial.root == 0)
            axial.push_back({key, ::SimpleFluid::Meshes::AMR::AdaptiveSplit::Axial});
    }
    auto zfine = root->apply(root->plan_adaptation(axial));
    ASSERT_EQ(global_cells(*zfine), 17u);
    const auto accepted = zfine->observation();
    std::vector<::SimpleFluid::Meshes::AMR::AdaptiveMark> crossed;
    for (const auto& [key, leaf] : zfine->owned())
    {
        const auto& p = std::get<::SimpleFluid::Meshes::AMR::ProductCellAncestry>(key.ancestry);
        if (p.base.root == 1 && p.axial.root == 0)
            crossed.push_back({key, ::SimpleFluid::Meshes::AMR::AdaptiveSplit::Base});
    }
    // At x=.1 the left native traces span full y and half z; right
    // native traces span half y and full z. Their quarters are neither
    // incident native whole face, so the procedural nesting proof rejects.
    EXPECT_THROW(zfine->plan_adaptation(crossed), std::invalid_argument);
    EXPECT_EQ(zfine->observation(), accepted);
    EXPECT_EQ(global_cells(*zfine), 17u);
    EXPECT_NO_THROW(zfine->native_draft());
}

} // namespace
