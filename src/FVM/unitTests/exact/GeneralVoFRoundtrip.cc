/** @file GeneralVoFRoundtrip.cc
 * @brief Explicit exact capsule handoff back to bounded geometry.
 */
#include "../general_vof_finite_test_helpers.hh"
namespace
{
TEST(ExactGeneralVoF, ExplicitExactRoundtripKeepsHistoryAndRebuildsFiniteGeometryAuthority)
{
    auto input = fixture();
    for (std::size_t i = 0; i < input.order.size(); ++i)
    {
        const auto center = input.hierarchy->geometry(input.hierarchy->owned().at(input.order[i]))->centroid();
        input.gas[i] = center[2] < 1 ? .9 : 0;
        input.carrier[i] = 1 - input.gas[i];
    }
    GeneralVoF source(input.hierarchy, input.order, input.gas, input.carrier);
    auto control = circulation(source);
    for (auto& face : control)
        face.integrated_volume = face.integrated_volume > 0 ? .0125 : face.integrated_volume < 0 ? -.0125 : 0;
    const auto moved = source.apply(source.plan_transport(stationary(source), control, 1));
    const auto comm = Tpetra::getDefaultComm();
    double local_history = 0, global_history = 0;
    for (const auto& error : moved->accumulated_phase_errors()) local_history += error.gas + error.carrier;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local_history, &global_history);
    ASSERT_GT(global_history, 0);
    // This call is an explicitly selected exact reference operation. It may
    // prepare exact geometry; the subsequent numerical route cannot inherit
    // its coordinate/measure authority or reset its finite error history.
    const auto exported = moved->export_transfer_source(true);
    for (const auto& capsule : exported.owned())
        if (capsule.geometry_ready) EXPECT_FALSE(capsule.geometry.finite_geometry);
    std::vector<int> destination(moved->hierarchy()->owned().size(),
        (comm->getRank() + 1) % std::min(2, comm->getSize()));
    const auto plan = moved->hierarchy()->plan_migration(destination);
    const auto raw = moved->hierarchy()->apply(plan);
    const auto native = assemble_adaptive_native(*raw, raw->configuration_descriptor());
    const auto target = moved->hierarchy()->apply(plan,
        raw->with_partition(native.partition, native.mesh->owned_cell_map()));
    const auto transfer = ExactPhaseTransfer::prepare(exported, plan, target);
    const auto adopted = moved->adopt_transfer(transfer);
    EXPECT_NO_THROW(adopted->validate_transfer_target(transfer));
    std::map<AdaptiveCellKey, const ExactPhaseTransferCapsule*> capsules;
    for (const auto& capsule : transfer.owned()) capsules.emplace(capsule.key, &capsule);
    for (std::size_t i = 0; i < adopted->owned_order().size(); ++i)
    {
        const auto& capsule = *capsules.at(adopted->owned_order()[i]);
        EXPECT_EQ(adopted->phase_volumes()[i].gas, capsule.phases.gas);
        EXPECT_EQ(adopted->phase_volumes()[i].carrier, capsule.phases.carrier);
        EXPECT_EQ(adopted->accumulated_phase_errors()[i].gas, capsule.accumulated_roundoff.gas);
        EXPECT_EQ(adopted->accumulated_phase_errors()[i].carrier, capsule.accumulated_roundoff.carrier);
    }
    GeneralVoF::enable_geometry_statistics(true);
    GeneralVoF::reset_geometry_statistics();
    const auto rebuilt = adopted->reconstruct();
    for (const auto& interface : rebuilt) EXPECT_TRUE(interface.finite_geometry);
    const auto observed = GeneralVoF::geometry_statistics();
    GeneralVoF::enable_geometry_statistics(false);
    EXPECT_EQ(observed.exact.volume_evaluations, 0);
    EXPECT_EQ(observed.exact.moment_evaluations, 0);
    EXPECT_EQ(observed.exact.clip_constructions, 0);
    EXPECT_EQ(observed.exact.measure_arithmetic, 0);
}
} // namespace
