/** @file testVolumeContinuityModel.cc
 * @brief Empty-owner optional-span selection and transactional continuity-preview regressions.
 */
#include "FVM/ALEControlVolumeState.hh"
#include "geometry/PlanarALEMeshMotion.hh"
#include "solvers/VolumeContinuityModel.hh"
#include "utils/testing_environment.hh"

#include <gtest/gtest.h>

#include <algorithm>
#include <limits>
#include <memory>
#include <span>
#include <vector>

namespace
{
using Pack = SimpleFluid::DefaultTpetraTypes;
using Mesh = SimpleFluid::MeshHandle<Pack>;
using Motion = SimpleFluid::PlanarALEMeshMotion<Pack>;
using Model = SimpleFluid::VolumeContinuityModel<Pack>;
using Flux = SimpleFluid::ScalarFaceFieldStored<Pack>;

testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);

std::shared_ptr<Mesh> make_two_cell_mesh()
{
    auto geometry = std::make_shared<Mesh::SemiStructured>(
        SimpleFluid::Arr<Mesh::SemiStructured::Vec3>{{0., 0., 0.}, {1., 0., 0.}, {1., 1., 0.}, {0., 1., 0.}},
        SimpleFluid::Arr<SimpleFluid::Arr<unsigned>>{{0, 1, 2, 3}}, SimpleFluid::ArrReal{0., .25, 1.});
    return std::make_shared<Mesh>(geometry, Mesh::DistributionOptions{.allow_empty_partitions = true});
}

SimpleFluid::FVM::ALEControlVolumeState begin_trial(const std::shared_ptr<Mesh>& mesh, Motion& motion)
{
    motion.begin_trial(2., .5);
    return SimpleFluid::FVM::make_ale_control_volume_state(*mesh, motion);
}

struct Fixture
{
    std::shared_ptr<Mesh> mesh = make_two_cell_mesh();
    Motion motion{mesh};
    SimpleFluid::FVM::ALEControlVolumeState ale = begin_trial(mesh, motion);
    Flux relative{mesh, 0., "continuity_zero_relative_flux"};
    Model model{mesh};
    std::vector<double> old_material{ale.old_cell_volumes().begin(), ale.old_cell_volumes().end()};
    std::vector<double> new_material{ale.new_cell_volumes().begin(), ale.new_cell_volumes().end()};
    std::vector<double> carrier_fraction = std::vector<double>(mesh->num_local_cells(), 1.);
    std::vector<double> previous = std::vector<double>(mesh->num_owned_cells());

    Fixture()
    {
        for (std::size_t owned = 0; owned < previous.size(); ++owned)
            previous[owned] = expected_target(owned) - .125;
    }

    double expected_target(std::size_t owned) const
    {
        return (new_material[owned] - old_material[owned]) / ale.time_step();
    }

    Model::Inputs inputs() const
    {
        return {.ale = ale,
            .old_material_volume = old_material,
            .new_material_volume = new_material,
            .carrier_material_fraction = carrier_fraction,
            .carrier_relative_flux = relative,
            .old_pool_volume = 1.,
            .new_pool_volume = 2.,
            .previous_target = previous};
    }

    void expect_owner_counts() const
    {
        const auto comm = mesh->owned_cell_map()->getComm();
        const int empty = mesh->num_owned_cells() == 0 ? 1 : 0;
        int global_empty = 0;
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &empty, &global_empty);
        EXPECT_EQ(global_empty, std::max(0, comm->getSize() - 2));
        EXPECT_EQ(mesh->owned_cell_map()->getGlobalNumElements(), 2U);
    }

    void expect_accepted(std::uint64_t generation, bool populated) const
    {
        EXPECT_EQ(model.generation(), generation);
        EXPECT_DOUBLE_EQ(model.diagnostics().global_material_source, populated ? 2. : 0.);
        for (std::size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
        {
            const auto cell = static_cast<Pack::local_ordinal_type>(owned);
            const auto expected = populated ? expected_target(owned) : 0.;
            EXPECT_DOUBLE_EQ(model.continuity_target_field().value(cell), expected);
            EXPECT_DOUBLE_EQ(model.material_source_field().value(cell), expected);
            EXPECT_DOUBLE_EQ(model.slip_contribution_field().value(cell), 0.);
        }
    }
};
} // namespace

TEST(VolumeContinuityEmptyOwnerTest, OptionalSpansAcceptActualEmptyOwnersAndKeepPreviewImmutable)
{
    Fixture fixture;
    fixture.expect_owner_counts();
    const auto accepted = fixture.model.snapshot();
    const auto trial = fixture.model.preview(fixture.inputs(), 7);
    EXPECT_DOUBLE_EQ(trial.diagnostics().global_material_source, 2.);
    EXPECT_DOUBLE_EQ(trial.diagnostics().maximum_target_change, .125);
    EXPECT_DOUBLE_EQ(trial.diagnostics().source_pool_closure_residual, 0.);
    fixture.expect_accepted(0, false);
    EXPECT_EQ(trial.material_source().size(), fixture.mesh->num_owned_cells());
    for (std::size_t owned = 0; owned < fixture.mesh->num_owned_cells(); ++owned)
        EXPECT_DOUBLE_EQ(trial.target().integrated_rate(static_cast<Pack::local_ordinal_type>(owned)),
            fixture.expected_target(owned));
    fixture.model.commit(trial);
    fixture.expect_accepted(7, true);
    fixture.model.restore(accepted);
    fixture.expect_accepted(0, false);
    EXPECT_THROW(fixture.model.commit(trial), std::logic_error);
    const auto replay = fixture.model.preview(fixture.inputs(), 7);
    EXPECT_DOUBLE_EQ(replay.diagnostics().maximum_target_change, .125);
    fixture.model.commit(replay);
    fixture.expect_accepted(7, true);
    fixture.motion.rollback_trial();
}

TEST(VolumeContinuityEmptyOwnerTest,
    RejectsNonemptyOwnerDisagreementAndBadSpansWithoutInvalidatingAcceptedOrPendingState)
{
    Fixture fixture;
    fixture.expect_owner_counts();
    fixture.model.commit(fixture.model.preview(fixture.inputs(), 1));
    const auto pending = fixture.model.preview(fixture.inputs(), 2);
    const auto comm = fixture.mesh->owned_cell_map()->getComm();
    const auto rank = comm->getRank();
    std::vector<int> local_owners(comm->getSize()), global_owners(comm->getSize());
    local_owners[rank] = fixture.mesh->num_owned_cells() != 0 ? 1 : 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, comm->getSize(), local_owners.data(), global_owners.data());
    std::vector<int> nonempty_ranks, empty_ranks;
    for (int candidate = 0; candidate < comm->getSize(); ++candidate)
        (global_owners[candidate] ? nonempty_ranks : empty_ranks).push_back(candidate);
    const auto invalid_rank = nonempty_ranks.back();
    auto every_rank_rejects = [&](const Model::Inputs& inputs)
    {
        int rejected = 0;
        try
        {
            static_cast<void>(fixture.model.preview(inputs, 3));
        }
        catch (const std::exception&)
        {
            // Collective validation preserves a local exception and reports
            // remote failures through std::runtime_error on the other ranks.
            rejected = 1;
        }
        int all_rejected = 0;
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_MIN, 1, &rejected, &all_rejected);
        EXPECT_EQ(all_rejected, 1);
        fixture.expect_accepted(1, true);
    };
    if (nonempty_ranks.size() > 1)
    {
        auto inputs = fixture.inputs();
        if (rank == invalid_rank)
            inputs.previous_target = {};
        every_rank_rejects(inputs);
        inputs.previous_target = fixture.previous;
        inputs.carrier_material_fraction = fixture.carrier_fraction;
        if (rank == invalid_rank)
            inputs.carrier_material_fraction = {};
        every_rank_rejects(inputs);
    }

    auto previous = fixture.previous;
    if (rank == nonempty_ranks.front())
        previous.push_back(0.);
    auto inputs = fixture.inputs();
    inputs.previous_target = previous;
    every_rank_rejects(inputs);

    previous = fixture.previous;
    if (rank == invalid_rank)
        previous[0] = std::numeric_limits<double>::quiet_NaN();
    inputs.previous_target = fixture.previous;
    inputs.carrier_material_fraction = fixture.carrier_fraction;
    inputs.previous_target = previous;
    every_rank_rejects(inputs);

    previous = fixture.previous;
    if (rank == invalid_rank)
        previous[0] = std::numeric_limits<double>::infinity();
    inputs.previous_target = fixture.previous;
    inputs.carrier_material_fraction = fixture.carrier_fraction;
    inputs.previous_target = previous;
    every_rank_rejects(inputs);

    auto fractions = fixture.carrier_fraction;
    if (rank == invalid_rank)
        fractions[0] = std::numeric_limits<double>::quiet_NaN();
    inputs.previous_target = fixture.previous;
    inputs.carrier_material_fraction = fixture.carrier_fraction;
    inputs.carrier_material_fraction = fractions;
    every_rank_rejects(inputs);

    if (!empty_ranks.empty())
    {
        previous = fixture.previous;
        if (rank == empty_ranks.front())
            previous.push_back(0.); // Empty owners cannot supply an invented target entry.
        inputs.previous_target = fixture.previous;
        inputs.carrier_material_fraction = fixture.carrier_fraction;
        inputs.previous_target = previous;
        every_rank_rejects(inputs);
    }
    EXPECT_NO_THROW(fixture.model.commit(pending));
    fixture.expect_accepted(2, true);
    fixture.motion.rollback_trial();
}

TEST(VolumeContinuityEmptyOwnerTest, EmptyOwnersAlsoAcceptUniformlyOmittedOptionalSpans)
{
    Fixture fixture;
    fixture.expect_owner_counts();
    auto inputs = fixture.inputs();
    inputs.previous_target = {};
    inputs.carrier_material_fraction = {};
    const auto trial = fixture.model.preview(inputs, 1);
    EXPECT_DOUBLE_EQ(trial.diagnostics().maximum_target_change, 0.);
    EXPECT_DOUBLE_EQ(trial.diagnostics().source_pool_closure_residual, 0.);
    for (std::size_t owned = 0; owned < fixture.mesh->num_owned_cells(); ++owned)
        EXPECT_DOUBLE_EQ(trial.target().integrated_rate(static_cast<Pack::local_ordinal_type>(owned)),
            fixture.expected_target(owned));
    fixture.expect_accepted(0, false);
    fixture.model.commit(trial);
    fixture.expect_accepted(1, true);
    fixture.motion.rollback_trial();
}
