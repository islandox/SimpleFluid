/** Forced shared-carrier SSPRK2 on closed planar hydrostatic columns and a moving piston. */
#include "QbmmPlanarGravityTestSupport.hh"
#include "utils/testing_environment.hh"

#include <tuple>

namespace
{
using namespace SimpleFluid::Testing::QbmmPlanarGravity;
const auto environment = ::testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);

class QbmmPlanarLinearGravity : public testing::TestWithParam<std::tuple<Geometry, Limiter>>
{
};

TEST_P(QbmmPlanarLinearGravity, StationaryTargetMatchesFixedHydrostatics)
{
    const auto [kind, limiter] = GetParam();
    for (const double gravity : {-1., 1., 0.})
        hydrostatic(kind, Velocity::SharedCarrier, gravity, limiter);
}

TEST_P(QbmmPlanarLinearGravity, CompressionAndExpansionCloseGasWorkGclAndMomentum)
{
    const auto [kind, limiter] = GetParam();
    piston(kind, Velocity::SharedCarrier, limiter);
}

TEST(QbmmPlanarLinearGravityFourier, MovingTvdMcLiquidContactClosesHeatAndCaloricLedgers)
{
    constexpr double gravity = 1., conductance = .25;
    Column column(
        Geometry::Cartesian, Velocity::SharedCarrier, 2, gravity, true, false, 0., Limiter::TvdMc, conductance);
    column.target = 1. + 1. / 8192.;
    const auto trial = column.controller->preview(dt);
    const auto& state = trial.state();
    // Two closed axial cells with one liquid contact and no external heating. Recover its two heat
    // packets independently from total thermal energy, the face transport snapshot and EOS work.
    std::array<double, 2> heat{}, temperature{};
    double delta_u = 0., work = 0.;
    for (size_t i = 0; i < state.cells.size(); ++i)
    {
        const auto layer = column.mesh->cell_centroid(static_cast<LO>(i)).z < .5 ? 0u : 1u;
        const auto& phases = state.cells[i].phases;
        const auto& transport = trial.transport_snapshot().cells[i].phases;
        temperature[layer] = phases[0].temperature;
        for (size_t p = 0; p < phases.size(); ++p)
        {
            const auto& phase = phases[p];
            const double volume =
                p == 0 ? phase.moles[0] / rho_liquid : phase.moles[0] * phase.temperature / phase.absolute_pressure;
            const double pdv = -phase.absolute_pressure * (volume - transport[p].advected_material_volume);
            heat[layer] += phase.internal_energy - transport[p].internal_energy - pdv;
            work += pdv;
            delta_u += phase.internal_energy - column.flow->state().cells[i].phases[p].internal_energy;
        }
    }
    for (size_t i = 0; i < heat.size(); ++i)
    {
        heat[i] = column.sum(heat[i]);
        temperature[i] = column.sum(temperature[i]);
    }
    const double expected = dt * conductance / column.target * (temperature[1] - temperature[0]);
    EXPECT_GT(expected, 1e-4);
    EXPECT_NEAR(heat[0], expected, 2e-7);
    EXPECT_NEAR(heat[1], -expected, 2e-7);
    EXPECT_NEAR(heat[0] + heat[1], 0., 2e-7);
    work = column.sum(work);
    delta_u = column.sum(delta_u);
    metric("lower_contact_heat_J", heat[0]);
    metric("upper_contact_heat_J", heat[1]);
    metric("caloric_defect_J", std::abs(delta_u - work));
    EXPECT_GT(std::abs(work), 1e-4);
    EXPECT_NEAR(trial.diagnostics().flow.pressure_work, work, 2e-7);
    EXPECT_NEAR(delta_u, work, 2e-7);
    check_impulse(column, trial, gravity, false);
    column.controller->commit(trial);
}

std::string case_name(const testing::TestParamInfo<std::tuple<Geometry, Limiter>>& info)
{
    const auto [kind, limiter] = info.param;
    constexpr std::array names{"CartesianZ", "NativeSector", "Extruded", "CartesianX", "CartesianY", "Disk", "Annulus"};
    return std::string(names[static_cast<size_t>(kind)]) + (limiter == Limiter::TvdMc ? "TvdMc" : "Realizability");
}

INSTANTIATE_TEST_SUITE_P(SharedCarrier, QbmmPlanarLinearGravity,
    testing::Combine(testing::Values(Geometry::CartesianX, Geometry::CartesianY, Geometry::Cartesian, Geometry::Sector,
                         Geometry::Disk, Geometry::Annulus),
        testing::Values(Limiter::RealizabilityOnly, Limiter::TvdMc)),
    case_name);
} // namespace
