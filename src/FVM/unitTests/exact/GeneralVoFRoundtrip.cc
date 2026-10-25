/** @file GeneralVoFRoundtrip.cc
 * @brief Explicit exact geometry inspection stays separate from finite migration history.
 */
#include "../general_vof_finite_test_helpers.hh"
#include <array>
namespace
{
TEST(ExactGeneralVoF, ExplicitExactGeometryInspectionPreservesFiniteMigrationHistory)
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
    for (const auto& error : moved->accumulated_phase_errors())
        local_history += error.gas + error.carrier;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local_history, &global_history);
    ASSERT_GT(global_history, 0);
    const auto identity = moved->accepted_identity();
    const auto accepted_control = moved->accepted_control();

    // Gather each physical cell's represented state before changing ownership.
    // The fixed 2x2x2 fixture gives every cell a unique integer-center address;
    // one owner contributes each entry, so the sum copies its doubles exactly.
    constexpr std::size_t components = 7;
    const auto address = [](const PLIC::GeometryPoint& center)
    { return std::size_t(center[0]) + 2 * std::size_t(center[1]) + 4 * std::size_t(center[2]); };
    std::array<double, 8 * components> local_values{}, original_values{};
    std::array<int, 8> local_owners{}, original_owners{};
    const auto native_geometry = moved->owned_geometry();
    for (std::size_t i = 0; i < moved->owned_order().size(); ++i)
    {
        const auto cell = address(native_geometry[i]->centroid());
        EXPECT_LT(cell, local_owners.size());
        if (cell >= local_owners.size())
            continue;
        ++local_owners[cell];
        const auto& phase = moved->phase_volumes()[i];
        const auto& roundoff = moved->phase_roundoff_bounds()[i];
        const auto& history = moved->accumulated_phase_errors()[i];
        const std::array values{
            phase.bulk, phase.gas, phase.carrier, roundoff.gas, roundoff.carrier, history.gas, history.carrier};
        std::copy(values.begin(), values.end(), local_values.begin() + components * cell);
    }
    Teuchos::reduceAll(
        *comm, Teuchos::REDUCE_SUM, int(local_values.size()), local_values.data(), original_values.data());
    Teuchos::reduceAll(
        *comm, Teuchos::REDUCE_SUM, int(local_owners.size()), local_owners.data(), original_owners.data());
    for (const auto owners : original_owners)
        EXPECT_EQ(owners, 1);

    struct StatisticsScope
    {
        StatisticsScope()
        {
            GeneralVoF::enable_geometry_statistics(true);
            PLIC::enable_backend_statistics<ArithmeticMode::Exact>(true);
        }
        ~StatisticsScope()
        {
            GeneralVoF::enable_geometry_statistics(false);
            PLIC::enable_backend_statistics<ArithmeticMode::Exact>(false);
        }
    } statistics;
    PLIC::reset_backend_statistics<ArithmeticMode::Exact>();
    using ExactGeometry = PLIC::ConvexCellGeometry<ArithmeticMode::Exact>;
    using ExactMeasure = PLIC::GeometryMeasure<ArithmeticMode::Exact>;
    for (const auto& native : native_geometry)
    {
        const auto exact = ExactGeometry::from_geometry(*native);
        const auto split = exact->clip({{1, 0, 0}, native->centroid(), 0});
        EXPECT_TRUE(split.inside);
        EXPECT_TRUE(split.outside);
        if (!split.inside || !split.outside)
            continue;
        const auto inside = ExactMeasure::volume(*split.inside);
        const auto outside = ExactMeasure::volume(*split.outside);
        EXPECT_GT(inside.sign(), 0);
        EXPECT_GT(outside.sign(), 0);
        EXPECT_EQ((inside + outside).compare(ExactMeasure::volume(*exact)), 0);
    }
    const auto reference = PLIC::backend_statistics<ArithmeticMode::Exact>();
    std::uint64_t local_clips = reference.clip_constructions, global_clips = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local_clips, &global_clips);
    EXPECT_GT(global_clips, 0u);
    EXPECT_THROW((void) moved->export_transfer_source(false), std::invalid_argument);
    EXPECT_THROW((void) moved->export_transfer_source(true), std::invalid_argument);
    EXPECT_EQ(moved->accepted_identity(), identity);
    EXPECT_EQ(moved->accepted_control(), accepted_control);

    // Ownership migration uses the accepted finite state directly. Exact
    // inspection cannot become its authority or replace its accumulated history.
    std::vector<int> destination(
        moved->hierarchy()->owned().size(), (comm->getRank() + 1) % std::min(2, comm->getSize()));
    const auto plan = moved->hierarchy()->plan_migration(destination);
    const auto raw = moved->hierarchy()->apply(plan);
    const auto native = assemble_adaptive_native(*raw, raw->configuration_descriptor());
    const auto target =
        moved->hierarchy()->apply(plan, raw->with_partition(native.partition, native.mesh->owned_cell_map()));
    const auto transfer = moved->plan_remap(plan, target);
    EXPECT_TRUE(transfer.ownership_only());
    const auto adopted = transfer.candidate();
    EXPECT_NO_THROW(transfer.validate_observations());
    EXPECT_EQ(adopted->arithmetic_mode(), ArithmeticMode::Ordinary);
    EXPECT_EQ(adopted->hierarchy()->global_cells(), 8u);
    const auto migrated_geometry = adopted->owned_geometry();
    for (std::size_t i = 0; i < adopted->owned_order().size(); ++i)
    {
        const auto cell = address(migrated_geometry[i]->centroid());
        EXPECT_LT(cell, original_owners.size());
        if (cell >= original_owners.size())
            continue;
        const auto& phase = adopted->phase_volumes()[i];
        const auto& roundoff = adopted->phase_roundoff_bounds()[i];
        const auto& history = adopted->accumulated_phase_errors()[i];
        const std::array values{
            phase.bulk, phase.gas, phase.carrier, roundoff.gas, roundoff.carrier, history.gas, history.carrier};
        for (std::size_t component = 0; component < components; ++component)
            EXPECT_EQ(values[component], original_values[components * cell + component]);
    }
    EXPECT_EQ(moved->accepted_identity(), identity);
    EXPECT_EQ(moved->accepted_control(), accepted_control);

    GeneralVoF::reset_geometry_statistics();
    PLIC::reset_backend_statistics<ArithmeticMode::Exact>();
    PLIC::reset_geometry_statistics();
    const auto rebuilt = adopted->reconstruct();
    for (const auto& interface : rebuilt)
        EXPECT_TRUE(interface.finite_geometry);
    const auto observed = GeneralVoF::geometry_statistics();
    for (const auto& exact : {observed.exact, PLIC::backend_statistics<ArithmeticMode::Exact>()})
    {
        EXPECT_EQ(exact.source_coordinate_exports, 0u);
        EXPECT_EQ(exact.source_support_exports, 0u);
        EXPECT_EQ(exact.volume_evaluations, 0u);
        EXPECT_EQ(exact.moment_evaluations, 0u);
        EXPECT_EQ(exact.clip_constructions, 0u);
        EXPECT_EQ(exact.intersection_constructions, 0u);
        EXPECT_EQ(exact.measure_values, 0u);
        EXPECT_EQ(exact.measure_arithmetic, 0u);
        EXPECT_EQ(exact.cover_certifications, 0u);
    }
    EXPECT_EQ(observed.bounded.runtime_dispatches, 0u);
    EXPECT_EQ(PLIC::geometry_statistics().runtime_dispatches, 0u);
    const std::uint64_t local_finite_work = observed.bounded.volume_evaluations;
    std::uint64_t global_finite_work = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local_finite_work, &global_finite_work);
    EXPECT_GT(global_finite_work, 0u);
}
} // namespace
