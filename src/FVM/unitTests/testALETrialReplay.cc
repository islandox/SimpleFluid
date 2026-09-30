/** @file testALETrialReplay.cc
 * @brief Retained ALE views cannot survive identical Planar trial replay.
 */
#include <gtest/gtest.h>

#include "FVM/ALEControlVolumeState.hh"
#include "geometry/PlanarALEMeshMotion.hh"
#include "geometry/unitTests/region_mesh_helpers.hh"
#include "utils/testing_environment.hh"

#include <string>
#include <vector>

namespace
{
using namespace SimpleFluid;
using Handle = MeshHandle<>;
testing::Environment* const environment =
    testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);
}

TEST(ALETrialReplayTest, SameTargetAndTimeStepRejectRetainedViewOnTwoRanks)
{
    const auto comm = Tpetra::getDefaultComm();
    if (comm->getSize() != 2) GTEST_SKIP() << "Requires exactly two MPI ranks.";

    const auto expect_rejection = [&](const FVM::ALEControlVolumeState& state,
                                      bool expect_epoch, const Handle& current_mesh)
    {
        std::string message;
        bool category_matches = false;
        try { state.validate(current_mesh); }
        catch (const std::invalid_argument& error)
        {
            message = error.what();
            category_matches = expect_epoch;
        }
        catch (const std::logic_error& error)
        {
            message = error.what();
            category_matches = !expect_epoch;
        }
        catch (const std::exception& error) { message = error.what(); }
        catch (...) { message = "Unexpected non-standard exception"; }
        const char* expected = expect_epoch
            ? "ALE control-volume state does not represent the mesh's current trial geometry epoch."
            : "ALE control-volume state requires its originating motion trial to remain active.";
        EXPECT_TRUE(category_matches);
        EXPECT_EQ(message, expected);
        const int local_match = category_matches && message == expected;
        int matching_ranks = 0;
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local_match, &matching_ranks);
        EXPECT_EQ(matching_ranks, comm->getSize());
    };

    auto mesh = std::make_shared<Handle>(test::two_regions());
    PlanarALEMeshMotion<> motion(mesh);
    constexpr real_t target = 1.25;
    constexpr real_t time_step = 0.2;
    const auto accepted_epoch = mesh->geometry_epoch();
    motion.begin_trial(target, time_step);
    const auto retained = FVM::make_ale_control_volume_state(*mesh, motion);
    EXPECT_NO_THROW(retained.validate(*mesh));
    const auto first_epoch = mesh->geometry_epoch();
    EXPECT_GT(first_epoch, accepted_epoch);
    const std::vector<real_t> old_volumes(retained.old_cell_volumes().begin(), retained.old_cell_volumes().end());
    const std::vector<real_t> new_volumes(retained.new_cell_volumes().begin(), retained.new_cell_volumes().end());
    const std::vector<real_t> face_fluxes(retained.face_mesh_fluxes().begin(), retained.face_mesh_fluxes().end());

    motion.rollback_trial();
    const auto rollback_epoch = mesh->geometry_epoch();
    EXPECT_GT(rollback_epoch, first_epoch);
    EXPECT_FALSE(motion.has_active_trial());
    expect_rejection(retained, false, *mesh);

    motion.begin_trial(target, time_step);
    EXPECT_GT(mesh->geometry_epoch(), rollback_epoch);
    EXPECT_TRUE(motion.has_active_trial());
    EXPECT_TRUE(motion.diagnostics().trial_active);
    EXPECT_EQ(motion.diagnostics().new_surface_elevation, target);
    EXPECT_EQ(motion.diagnostics().time_step, time_step);
    const auto fresh = FVM::make_ale_control_volume_state(*mesh, motion);
    EXPECT_EQ(std::vector<real_t>(fresh.old_cell_volumes().begin(), fresh.old_cell_volumes().end()), old_volumes);
    EXPECT_EQ(std::vector<real_t>(fresh.new_cell_volumes().begin(), fresh.new_cell_volumes().end()), new_volumes);
    EXPECT_EQ(std::vector<real_t>(fresh.face_mesh_fluxes().begin(), fresh.face_mesh_fluxes().end()), face_fluxes);

    // Do not dereference retained spans after replay replaces their backing
    // storage: validate must reject this epoch before looking at span values.
    expect_rejection(retained, true, *mesh);
    EXPECT_NO_THROW(fresh.validate(*mesh));

    const auto replay_epoch = mesh->geometry_epoch();
    motion.accept_trial();
    EXPECT_EQ(mesh->geometry_epoch(), replay_epoch);
    EXPECT_FALSE(motion.has_active_trial());
    EXPECT_FALSE(motion.diagnostics().trial_active);
    expect_rejection(fresh, false, *mesh);
}
