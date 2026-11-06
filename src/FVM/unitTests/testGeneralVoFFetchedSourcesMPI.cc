/** Remote source views, empty owners and collective admission recovery. */
#include "FVM/GeneralVoF.hh"
#include "geometry/mesh/AMR/DistributedCartesianKernel.hh"
#include "utils/testing_environment.hh"
#include <Tpetra_Core.hpp>
#include <array>
#include <cstddef>
#include <gtest/gtest.h>
#include <memory>
#include <vector>

namespace
{
using namespace SimpleFluid;
using namespace SimpleFluid::VoF;
namespace AMR = SimpleFluid::Meshes::AMR;
constexpr auto ordinary = ArithmeticMode::Ordinary;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);

TEST(GeneralVoFFetchedSourcesMPI, RemoteRequestedCellAndEmptyOwnersPreserveAdmissionAndLifetime)
{
    const auto comm = Tpetra::getDefaultComm();
    if (comm->getSize() != 2 && comm->getSize() != 4)
        GTEST_SKIP() << "Registered two/four-rank remote-source and empty-owner gate.";
    AMR::AMRGrid grid;
    grid.block_cells = {2, 2, 2};
    grid.extent = {2, 2, 2};
    const GeneralVoF::Key requested{grid.region_id, AMR::CartesianCellAncestry{0, {0, 0, 0}}};
    for (const auto mode : {ordinary, ArithmeticMode::BoundedCertification})
    {
        SCOPED_TRACE(static_cast<int>(mode));
        auto kernel = AMR::cartesian_adaptive_kernel(grid);
        std::vector<AMR::AdaptiveLeaf> owned;
        if (comm->getRank() == 0)
            for (std::size_t z = 0; z < 2; ++z)
                for (std::size_t y = 0; y < 2; ++y)
                    for (std::size_t x = 0; x < 2; ++x)
                        owned.push_back(AMR::cartesian_adaptive_leaf(grid,
                            {grid.region_id, AMR::CartesianCellAncestry{0, {x, y, z}}},
                            {'r', 'e', 'm', 'o', 't', 'e'}));
        owned = kernel->repack(owned);
        auto hierarchy = std::make_shared<const GeneralVoF::Hierarchy>(
            std::move(owned), std::vector<AMR::AdaptiveFamily>{}, kernel, comm);
        std::vector<GeneralVoF::Key> order;
        for (const auto& leaf : hierarchy->owned_leaves())
            order.push_back(leaf.key);
        GeneralVoFOptions options;
        options.arithmetic_mode = mode;
        options.phase_cleanup.relative_cutoff = 0;
        auto source = std::make_shared<const GeneralVoF>(
            hierarchy, order, std::vector<double>(order.size(), 0), std::vector<double>(order.size(), 1), options);
        EXPECT_EQ(source->owned_order().empty(), comm->getRank() != 0);
        auto identity = source->accepted_identity();
        std::weak_ptr<const void> accepted_lifetime = identity;
        const auto control = source->accepted_control();
        const auto gas = source->gas_volumes(), carrier = source->carrier_volumes();
        const std::array request{requested};
        PLIC::CellGeometryView<ordinary> surviving;
        {
            const auto fetched = source->fetch_native_sources(request, options.byte_budget);
            const auto& draft = fetched.draft();
            const auto views = fetched.geometry();
            EXPECT_EQ(views.size(), draft.owned.size() + draft.ghosts.size());
            EXPECT_EQ(fetched.fetch_byte_budget(), hierarchy->options().byte_budget);
            EXPECT_LE(fetched.admitted_bytes(), options.byte_budget);
            std::size_t slot = 0, occurrences = 0;
            for (const auto* rows : {&draft.owned, &draft.ghosts})
                for (const auto& leaf : *rows)
                {
                    const auto view = views.at(slot++);
                    EXPECT_EQ(view.volume().value, 1.L);
                    EXPECT_EQ(view.is_procedural(), mode == ordinary);
                    if (leaf.key == requested)
                    {
                        surviving = view;
                        ++occurrences;
                        EXPECT_EQ(view.centroid(), (PLIC::GeometryPoint{.5, .5, .5}));
                    }
                }
            EXPECT_EQ(occurrences, 1u);
            std::size_t corners = 0;
            for (const auto& node : draft.nodes)
                if ((node.coordinate[0] == 0 || node.coordinate[0] == 1) &&
                    (node.coordinate[1] == 0 || node.coordinate[1] == 1) &&
                    (node.coordinate[2] == 0 || node.coordinate[2] == 1))
                    ++corners;
            EXPECT_EQ(corners, 8u);
        }
        // A single empty owner runs out of operation allowance before fetch.
        // Every rank must reject, then enter the next valid fetch together.
        const auto allowance = comm->getRank() == comm->getSize() - 1 ? 1u : options.byte_budget;
        EXPECT_THROW(static_cast<void>(source->fetch_native_sources(request, allowance)), std::invalid_argument);
        EXPECT_EQ(source->accepted_identity().get(), identity.get());
        EXPECT_EQ(source->accepted_control(), control);
        EXPECT_EQ(source->gas_volumes(), gas);
        EXPECT_EQ(source->carrier_volumes(), carrier);
        EXPECT_NO_THROW(static_cast<void>(source->fetch_native_sources(request, options.byte_budget)));
        std::weak_ptr<const GeneralVoF> lifetime = source;
        identity.reset();
        source.reset();
        hierarchy.reset();
        EXPECT_TRUE(lifetime.expired());
        EXPECT_TRUE(accepted_lifetime.expired());
        EXPECT_TRUE(surviving);
        if (surviving)
        {
            EXPECT_EQ(surviving.volume().value, 1.L);
            EXPECT_EQ(surviving.centroid(), (PLIC::GeometryPoint{.5, .5, .5}));
        }
    }
}
} // namespace
