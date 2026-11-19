#pragma once

#include "geometry/mesh/OrthogonalCartesian3D.hh"
#include "solvers/LowMachMixtureSolver.hh"

#include <Teuchos_CommHelpers.hpp>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <functional>
#include <iomanip>
#include <memory>
#include <sstream>
#include <utility>
#include <vector>

namespace SimpleFluid::Testing::QbmmKinetic
{
namespace R = Reacting;
using Flow = R::LowMachMixtureSolver;
using LO = Flow::Pack::local_ordinal_type;
using Model = R::QbmmModelOptions;
using Law = std::function<std::array<double, 3>(const R::QbmmNodeSlipInput&)>;

inline SP<Flow::Mesh> mesh(ArrReal x, ArrReal y = {0., 1.})
{
    return std::make_shared<Flow::Mesh>(
        std::make_shared<Flow::Mesh::Cartesian>(Vec3D<ArrReal>{std::move(x), std::move(y), {0., 1.}}),
        Flow::Mesh::DistributionOptions{.allow_empty_partitions = true});
}

inline std::shared_ptr<const R::SpeciesCatalog> catalog()
{
    return std::make_shared<const R::SpeciesCatalog>(
        std::vector<R::SpeciesDefinition>{{47, "kinetic gas", 1., {{"G", 1}}, {R::SpeciesPhase::Dispersed}}});
}

inline Flow::Options options(const std::shared_ptr<const R::SpeciesCatalog>& species, Law evaluate,
    R::ThermodynamicBounds bounds, double cfl = 1., int cap = 1024,
    R::QbmmApplicabilityGate gate = R::QbmmApplicabilityGate::ReportOnly, std::vector<R::QbmmSiRange> ranges = {})
{
    Flow::Options result;
    result.minimum_pressure = bounds.minimum_pressure;
    result.maximum_pressure = bounds.maximum_pressure;
    result.linear.preconditioner = LinearPreconditioner::None;
    result.linear.tolerance = 1e-12;
    result.phase_roles = {{Flow::PhaseRole::DispersedGas, "kinetic bubbles"}};
    result.qbmm_model.family_transport = Model::FamilyTransport::KineticEuler;
    result.qbmm_model.velocity = Model::Velocity::NodeDependentSlip;
    result.qbmm_model.kinetic_cfl = cfl;
    result.qbmm_model.maximum_kinetic_substeps = cap;
    R::QbmmLawMetadata meta{"kinetic fixture velocity", 3, R::QbmmLawKind::NodeSlip,
        "active atom mass and owned position", "m/s", gate, true, std::move(ranges)};
    R::QbmmClosureVersions versions;
    versions.laws.push_back(meta);
    result.qbmm_families = {std::make_shared<const R::QbmmPopulationSchema>(species, species->fingerprint(),
        "kinetic bubbles", result.phase_roles[0], 1., std::vector<R::QbmmMassFraction>{{47, 1.}}, std::move(versions))};
    result.qbmm_node_slip_laws = {
        std::make_shared<const R::QbmmNodeSlipLaw>(R::QbmmNodeSlipLaw{std::move(meta), std::move(evaluate)})};
    return result;
}

inline R::QbmmMomentBlock moments(std::span<const R::QbmmAtom> atoms)
{
    R::QbmmMomentBlock h{};
    for (const auto& [count, mass] : atoms)
    {
        double weighted = count;
        for (double& order : h)
        {
            order += weighted;
            weighted *= mass;
        }
    }
    return h;
}

inline void set_stock(Flow::CellState& cell, const std::shared_ptr<const R::MixtureThermodynamics>& eos,
    const R::QbmmMomentBlock& h, double pressure, double temperature, double specific_energy)
{
    cell.phases = {{eos, {h[1]}, pressure, h[1] * specific_energy, h[1] > 0 ? temperature : 0.}};
    cell.phase_numbers = {h[0]};
    cell.qbmm_shape = {R::QbmmShapeState{h[2], h[3]}};
}

inline R::QbmmMomentBlock stock(const Flow::CellState& cell)
{
    return {*cell.phase_numbers[0], cell.phases[0].moles[0], cell.qbmm_shape[0]->H2, cell.qbmm_shape[0]->H3};
}

template<class T> T reduce(const Flow::Mesh& grid, Teuchos::EReductionType type, T local)
{
    T global{};
    Teuchos::reduceAll(*grid.owned_cell_map()->getComm(), type, 1, &local, &global);
    return global;
}

inline void record_ownership(const Flow::Mesh& grid, int cells)
{
    const auto comm = grid.owned_cell_map()->getComm();
    const int empty = reduce(grid, Teuchos::REDUCE_SUM, int(grid.num_owned_cells() == 0));
    EXPECT_EQ(reduce(grid, Teuchos::REDUCE_SUM, int(grid.num_owned_cells())), cells);
    EXPECT_EQ(empty, std::max(0, comm->getSize() - cells));
    ::testing::Test::RecordProperty("mpi_ranks", comm->getSize());
    ::testing::Test::RecordProperty("empty_owners", empty);
}

inline void record_maximum(const Flow::Mesh& grid, const char* key, double local)
{
    std::ostringstream value;
    value << std::setprecision(17) << reduce(grid, Teuchos::REDUCE_MAX, local);
    ::testing::Test::RecordProperty(key, value.str());
}
} // namespace SimpleFluid::Testing::QbmmKinetic
