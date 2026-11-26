/** Forced NativeImplicit planar ALE on closed hydrostatic columns and an analytic multiphase piston. */
#include "QbmmPlanarGravityTestSupport.hh"
#include "QbmmRestartTestSupport.hh"
#include "utils/testing_environment.hh"

#include <tuple>

namespace
{
using namespace SimpleFluid;
using namespace SimpleFluid::Testing::QbmmPlanarGravity;
const auto environment = ::testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);

class QbmmPlanarGravityColumn : public testing::TestWithParam<std::tuple<Geometry, Velocity>>
{
};

TEST_P(QbmmPlanarGravityColumn, StationaryTargetMatchesFixedHydrostatics)
{
    const auto [kind, velocity] = GetParam();
    for (double g : {-1., 1., 0.})
        hydrostatic(kind, velocity, g);
}

TEST_P(QbmmPlanarGravityColumn, CompressionAndExpansionCloseGasWorkGclAndMomentum)
{
    const auto [kind, velocity] = GetParam();
    piston(kind, velocity);
}

std::string column_case_name(const testing::TestParamInfo<std::tuple<Geometry, Velocity>>& info)
{
    const auto [kind, velocity] = info.param;
    constexpr std::array names{"Cartesian", "Sector", "Extruded"};
    return std::string(names[static_cast<size_t>(kind)]) + (velocity == Velocity::SharedCarrier ? "Shared" : "Slip");
}

INSTANTIATE_TEST_SUITE_P(NativeImplicit, QbmmPlanarGravityColumn,
    testing::Combine(testing::Values(Geometry::Cartesian, Geometry::Sector, Geometry::Extruded),
        testing::Values(Velocity::SharedCarrier, Velocity::SuppliedPhaseSlip)),
    column_case_name);

void compare_count_only_endpoints(const Column& uninterrupted, const Column& resumed)
{
    const auto& expected = uninterrupted.flow->state();
    const auto& actual = resumed.flow->state();
    EXPECT_EQ(actual.time, expected.time);
    EXPECT_EQ(actual.revision, expected.revision);
    EXPECT_EQ(actual.thermodynamic_pressure, expected.thermodynamic_pressure);
    EXPECT_TRUE(actual.qbmm_families.empty());
    const auto& first = uninterrupted.controller->diagnostics().flow;
    const auto& second = resumed.controller->diagnostics().flow;
    EXPECT_EQ(second.body_acceleration_m_per_s2, first.body_acceleration_m_per_s2);
    EXPECT_EQ(second.impulse_parts, first.impulse_parts);
    EXPECT_EQ(second.pressure_impulse, first.pressure_impulse);
    EXPECT_EQ(second.pressure_work, first.pressure_work);
    for (size_t i = 0; i < actual.cells.size(); ++i)
    {
        const auto& a = expected.cells.at(i);
        const auto& b = actual.cells[i];
        EXPECT_EQ(b.velocity, a.velocity);
        EXPECT_EQ(b.drift_velocity, a.drift_velocity);
        EXPECT_EQ(b.mechanical_pressure, a.mechanical_pressure);
        EXPECT_EQ(b.phase_numbers, a.phase_numbers);
        EXPECT_TRUE(b.qbmm_shape.empty());
        for (size_t p = 0; p < b.phases.size(); ++p)
        {
            EXPECT_EQ(b.phases[p].moles, a.phases.at(p).moles);
            EXPECT_EQ(b.phases[p].temperature, a.phases[p].temperature);
            EXPECT_EQ(b.phases[p].absolute_pressure, a.phases[p].absolute_pressure);
            EXPECT_EQ(b.phases[p].internal_energy, a.phases[p].internal_energy);
        }
        const auto id = static_cast<LO>(i);
        EXPECT_EQ(resumed.mesh->cell_volume(id), uninterrupted.mesh->cell_volume(id));
        EXPECT_EQ(resumed.flow->density().owned_value(id), uninterrupted.flow->density().owned_value(id));
        EXPECT_EQ(resumed.flow->velocity().owned_value(id), uninterrupted.flow->velocity().owned_value(id));
    }
    for (const auto face : resumed.flow->volume_flux().owned_face_ids())
        EXPECT_EQ(resumed.flow->volume_flux().owned_value(face), uninterrupted.flow->volume_flux().owned_value(face));
}

TEST(QbmmPlanarGravity, CountOnlyMovingControllerRetainsDispersedNumber)
{
    for (const double gravity : {0., .5})
        for (const auto velocity : {Velocity::SharedCarrier, Velocity::SuppliedPhaseSlip})
        {
            SCOPED_TRACE(gravity);
            SCOPED_TRACE(static_cast<int>(velocity));
            Column column(Geometry::Cartesian, velocity, 1, gravity, true, true);
            const auto initial = column.flow->state();
            const auto advance = [&](Column& c, double target)
            {
                c.target = target;
                const auto candidate = c.controller->preview(dt);
                check_impulse(c, candidate, gravity, false);
                EXPECT_TRUE(candidate.state().qbmm_families.empty());
                for (size_t i = 0; i < initial.cells.size(); ++i)
                {
                    const auto& old = c.flow->state().cells[i];
                    const auto& cell = candidate.state().cells[i];
                    const double liquid_v = old.phases[0].moles[0] / rho_liquid;
                    const double gas_n = old.phases[1].moles[0], capacity = cv * mass(old);
                    const double dv = target - c.mesh->cell_volume(static_cast<LO>(i));
                    const double expected_t =
                        old.phases[1].temperature / (1. + gas_n * dv / (capacity * (target - liquid_v)));
                    const double expected_p = gas_n * expected_t / (target - liquid_v);
                    check_stocks(cell, initial.cells[i]);
                    for (const auto& phase : cell.phases)
                    {
                        EXPECT_NEAR(phase.absolute_pressure, expected_p, 1e-6);
                        EXPECT_NEAR(phase.temperature, expected_t, 2e-7);
                    }
                    EXPECT_NEAR(candidate.diagnostics().flow.pressure_work, -expected_p * dv, 2e-7);
                }
                c.controller->commit(candidate);
            };
            advance(column, 1. + 1. / 512.);
            const auto archive = column.controller->serialize_qbmm_restart();
            std::vector<std::byte> header;
            for (auto word : std::array<std::uint64_t, 3>{0x5453524d4d425153ULL, 9, 12})
                Testing::QbmmRestart::append_restart_word(header, word);
            EXPECT_TRUE(std::ranges::starts_with(archive, header));
            if (gravity != 0. && velocity == Velocity::SharedCarrier)
            {
                Column mismatch(Geometry::Cartesian, velocity, 1, gravity, true, true, 1e-14);
                const auto before = mismatch.controller->serialize_qbmm_restart();
                const auto epoch = mismatch.mesh->geometry_epoch();
                EXPECT_THROW(mismatch.controller->restore_qbmm_restart(archive), std::exception);
                EXPECT_EQ(mismatch.controller->serialize_qbmm_restart(), before);
                EXPECT_EQ(mismatch.mesh->geometry_epoch(), epoch);
            }
            Column resumed(Geometry::Cartesian, velocity, 1, gravity, true, true);
            resumed.controller->restore_qbmm_restart(archive);
            compare_count_only_endpoints(column, resumed);
            advance(column, 1. - 1. / 512.);
            advance(resumed, 1. - 1. / 512.);
            EXPECT_EQ(resumed.flow->state().time, 2. * dt);
            compare_count_only_endpoints(column, resumed);
        }
}
} // namespace
