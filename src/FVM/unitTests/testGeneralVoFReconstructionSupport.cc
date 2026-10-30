/** @file testGeneralVoFReconstructionSupport.cc
 * @brief Public reconstruction separates retained phase availability from the local fitting ring.
 */
#include "FVM/GeneralVoF.hh"
#include "geometry/mesh/AMR/CartesianAMR.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveHierarchy.hh"
#include "utils/testing_environment.hh"
#include <Teuchos_CommHelpers.hpp>
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <gtest/gtest.h>
#include <iomanip>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace
{
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);
using namespace SimpleFluid;
using namespace SimpleFluid::VoF;
using namespace SimpleFluid::Meshes::AMR;

// Independent analytic unit-cube CDF of X+Y+Z. The two physical phases are
// integrated separately as F(t) and F(3-t), rather than deriving a complement.
long double cube_fraction(long double t)
{
    if (t <= 0)
        return 0;
    if (t >= 3)
        return 1;
    long double volume = 0;
    for (unsigned corner = 0; corner < 8; ++corner)
    {
        const auto count = std::popcount(corner);
        const auto distance = std::max(0.L, t - count);
        volume += (count % 2 ? -1 : 1) * distance * distance * distance;
    }
    return volume / 6;
}

TEST(GeneralVoFReconstructionSupport, PublicCallerUsesAvailableMajorityAndRetainsWeakAndTinyGradientPolicies)
{
    const auto comm = Tpetra::getDefaultComm();
    AMRGrid grid;
    grid.root_blocks = {3, 3, 3};
    grid.block_cells = {2, 2, 2};
    grid.origin = {-2, -2, -2};
    grid.extent = {6, 6, 6};
    const CartesianAMR native(grid);
    std::vector<std::uint64_t> owned;
    for (std::uint64_t gid = 0; gid < 216; ++gid)
        if (int(gid % std::min(3, comm->getSize())) == comm->getRank())
            owned.push_back(gid);
    auto detached = native.detach_owned(owned, {'r', 'e', 'c', 'o', 'n', '_', 's', 'u', 'p', 'p', 'o', 'r', 't'});
    const auto hierarchy = std::make_shared<const DistributedAdaptiveHierarchy>(
        std::move(detached.owned), std::move(detached.families), detached.kernel, comm);
    EXPECT_EQ(hierarchy->global_cells(), 216u);
    std::vector<AdaptiveCellKey> order;
    std::map<AdaptiveCellKey, PLIC::GeometryPoint> centers;
    for (const auto& [key, leaf] : hierarchy->owned())
    {
        const auto metrics = hierarchy->metrics(leaf);
        EXPECT_EQ(metrics.volume, 1);
        order.push_back(key);
        centers.emplace(key, metrics.centroid);
    }
    const int local_empty = order.empty();
    int empty = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local_empty, &empty);
    EXPECT_EQ(empty, std::max(0, comm->getSize() - 3)); // MPI4 has one actual empty owner.
    RecordProperty("actual_ranks", std::to_string(comm->getSize()));
    RecordProperty("actual_rank", std::to_string(comm->getRank()));
    RecordProperty("actual_empty_owners", std::to_string(empty));

    GeneralVoFOptions options;
    options.arithmetic_mode = ArithmeticMode::Ordinary;
    options.reconstruction.measure_only = true;
    options.phase_cleanup.relative_cutoff = 0;
    EXPECT_EQ(options.support_rings, 2u);
    EXPECT_EQ(options.reconstruction.minimum_phase_support_fraction, .5);
    EXPECT_EQ(options.reconstruction.maximum_fit_iterations, 24u);
    EXPECT_EQ(options.reconstruction.maximum_clip_evaluations, 32768u);
    EXPECT_EQ(options.reconstruction.volume_tolerance, 8e-15);
    EXPECT_EQ(options.reconstruction.fit_tolerance, 2e-9);
    EXPECT_EQ(options.reconstruction.maximum_model_residual, .2);
    EXPECT_EQ(options.reconstruction.minimum_optimized_phase_length_ratio, .05);
    // Geometry independently fixes the availability premise. At cutoff .4,
    // the local six neighbors contain at most F(1.4)<.5, while retained ring2
    // reaches F(2.4)>.5. This availability must not enlarge the local objective.
    // At cutoff .5, F(1.5)==.5 already reaches equality in the first ring.
    EXPECT_LT(cube_fraction(1.4L), .5L);
    EXPECT_GT(cube_fraction(2.4L), .5L);
    EXPECT_EQ(cube_fraction(1.5L), .5L);

    enum class Case
    {
        ObliqueModerate,
        ObliqueEquality,
        WallQuarter,
        WallTiny
    };
    for (const auto scenario : {Case::ObliqueModerate, Case::ObliqueEquality, Case::WallQuarter, Case::WallTiny})
    {
        SCOPED_TRACE(int(scenario));
        const std::string scenario_name = scenario == Case::ObliqueModerate   ? "oblique_04"
                                          : scenario == Case::ObliqueEquality ? "oblique_05_equality"
                                          : scenario == Case::WallQuarter     ? "wall_quarter"
                                                                              : "wall_tiny";
        const bool oblique = scenario == Case::ObliqueModerate || scenario == Case::ObliqueEquality;
        const auto cutoff = scenario == Case::ObliqueEquality ? .5L : .4L;
        std::vector<double> gas, carrier;
        std::map<AdaptiveCellKey, std::array<double, 2>> physical;
        for (const auto& key : order)
        {
            const auto& center = centers.at(key);
            double g = 0, c = 1;
            if (oblique)
            {
                const auto t = cutoff - ((center[0] - .5L) + (center[1] - .5L) + (center[2] - .5L));
                g = static_cast<double>(cube_fraction(t));
                c = static_cast<double>(cube_fraction(3 - t));
            }
            else if (center[0] == -1.5)
            {
                g = scenario == Case::WallQuarter ? .25 : 1e-6;
                c = scenario == Case::WallQuarter ? .75 : static_cast<double>(1.L - 1e-6L);
            }
            gas.push_back(g);
            carrier.push_back(c);
            physical.emplace(key, std::array<double, 2>{g, c});
        }
        const GeneralVoF state(hierarchy, order, gas, carrier, options);
        EXPECT_EQ(state.phase_cleanup_state().cleaned_cells, 0u);
        std::vector<PLIC::PolyhedralInterface> planes;
        EXPECT_NO_THROW(planes = state.reconstruct());
        EXPECT_EQ(planes.size(), state.owned_order().size());
        const auto geometry = state.owned_finite_geometry<ArithmeticMode::Ordinary>();
        const auto declarations = state.phase_geometry_declarations();
        const PLIC::GeometryPoint target_center =
            oblique ? PLIC::GeometryPoint{.5, .5, .5} : PLIC::GeometryPoint{-1.5, .5, .5};
        int local_target = 0;
        for (std::size_t i = 0; i < state.owned_order().size(); ++i)
        {
            const auto& key = state.owned_order()[i];
            const auto expected = physical.at(key);
            EXPECT_EQ(state.gas_volumes()[i], expected[0]);
            EXPECT_EQ(state.carrier_volumes()[i], expected[1]);
            if (expected[0] > 0 && expected[1] > 0)
                EXPECT_EQ(declarations[i], GeneralVoFPhaseGeometry::Infer);
            if (centers.at(key) != target_center || i >= planes.size())
                continue;
            ++local_target;
            const auto& fit = planes[i];
            EXPECT_FALSE(fit.pure_gas);
            EXPECT_FALSE(fit.pure_carrier);
            EXPECT_TRUE(fit.resolution.admitted());
            EXPECT_EQ(geometry[i]->centroid(), target_center);
            EXPECT_LE(std::abs(fit.finite_target_volume_difference.value) + fit.target_volume_numerical_error_estimate,
                static_cast<long double>(std::min(expected[0], expected[1])) * options.reconstruction.volume_tolerance);
            if (oblique)
            {
                EXPECT_EQ(fit.resolution.stencil_cells, 6u);
                EXPECT_TRUE(fit.resolution.angular_fit_eligible);
                EXPECT_GE(fit.resolution.gas_support_fraction, options.reconstruction.minimum_phase_support_fraction);
                EXPECT_GE(
                    fit.resolution.carrier_support_fraction, options.reconstruction.minimum_phase_support_fraction);
                if (scenario == Case::ObliqueModerate)
                    EXPECT_LT(cube_fraction(cutoff + 1), options.reconstruction.minimum_phase_support_fraction);
                EXPECT_EQ(fit.orientation_method, PLIC::PolyhedralOrientationMethod::AngularVolumeFit);
                EXPECT_TRUE(fit.fit_converged);
                EXPECT_TRUE(fit.stencil_volume_residual_measured);
                EXPECT_LE(fit.stencil_volume_residual, options.reconstruction.fit_tolerance);
                EXPECT_LE(fit.stencil_volume_residual, options.reconstruction.maximum_model_residual);
                EXPECT_LE(fit.fit_iterations, options.reconstruction.maximum_fit_iterations);
            }
            else
            {
                EXPECT_EQ(fit.orientation_method, PLIC::PolyhedralOrientationMethod::Gradient);
                EXPECT_FALSE(fit.fit_converged);
                EXPECT_EQ(fit.resolution.stencil_cells, 5u);
                EXPECT_LT(fit.resolution.gas_support_fraction, options.reconstruction.minimum_phase_support_fraction);
                EXPECT_EQ(fit.stencil_volume_residual_measured, scenario == Case::WallQuarter);
                if (scenario == Case::WallQuarter)
                    // The independent plane has zero model defect; measured
                    // finite volume integration retains its arithmetic error.
                    EXPECT_LE(fit.stencil_volume_residual, options.reconstruction.volume_tolerance);
                else
                    EXPECT_EQ(fit.stencil_volume_residual, 0); // Unmeasured sentinel, not an accuracy claim.
            }
            const auto method =
                fit.orientation_method == PLIC::PolyhedralOrientationMethod::AngularVolumeFit ? "AngularVolumeFit"
                : fit.orientation_method == PLIC::PolyhedralOrientationMethod::Gradient       ? "Gradient"
                                                                                              : "SuppliedNormal";
            std::ostringstream metrics;
            metrics << std::setprecision(17) << "owner=" << comm->getRank()
                    << " stencil_cells=" << fit.resolution.stencil_cells << " method=" << method
                    << " rms=" << fit.stencil_volume_residual
                    << " available_gas_support=" << fit.resolution.gas_support_fraction
                    << " available_carrier_support=" << fit.resolution.carrier_support_fraction
                    << " rms_measured=" << fit.stencil_volume_residual_measured << std::setprecision(21)
                    << " target_residual=" << fit.finite_target_volume_difference.value
                    << " numerical_indicator=" << fit.target_volume_numerical_error_estimate
                    << " target_residual_with_indicator="
                    << std::abs(fit.finite_target_volume_difference.value) + fit.target_volume_numerical_error_estimate;
            RecordProperty(scenario_name + "_target_metrics", metrics.str());
        }
        int global_target = 0;
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local_target, &global_target);
        EXPECT_EQ(global_target, 1);
        RecordProperty(scenario_name + "_actual_global_targets", std::to_string(global_target));
    }
}
} // namespace
