/** Explicit mesh and coupled-operator choices for matched verification runs. */
#pragma once

#include "VerificationMesh.hh"
#include "equations/TimeStepperOptions.hh"
#include "geometry/MeshHandle.hh"
#include "geometry/mesh/MultiRegionMesh.hh"
#include "solvers/BelosLinearSolver.hh"

#include <cctype>
#include <optional>
#include <string_view>

namespace SimpleFluid::Verification
{
struct BackendControls
{
    std::string mesh_backend = "native";
    size_t regions = 1;
    std::optional<PressureVelocityCoupling> coupling;
    std::optional<CoupledOperatorBackend> coupled_operator;
    std::optional<CoupledWorkspacePolicy> workspace;
    std::optional<CoupledLinearization> nonlinear_method;
    std::optional<LinearSolverBackend> nonlinear_linear_solver;
    std::optional<int> nonlinear_iterations;
    std::optional<int> nonlinear_restart;

    bool parse(std::string_view argument, std::string_view value)
    {
        if (argument == "--mesh-backend")
        {
            if (value != "native" && value != "isoregion")
                throw std::invalid_argument("--mesh-backend requires native or isoregion");
            mesh_backend = value;
        }
        else if (argument == "--regions")
        {
            if (value.empty() || !std::ranges::all_of(value, [](unsigned char c) { return std::isdigit(c); }))
                throw std::invalid_argument("--regions requires a positive integer");
            regions = std::stoull(std::string(value));
            if (!regions)
                throw std::invalid_argument("--regions requires a positive integer");
        }
        else if (argument == "--coupling")
        {
            if (value == "piso")
                coupling = PressureVelocityCoupling::PISO;
            else if (value == "coupled")
                coupling = PressureVelocityCoupling::CoupledKrylov;
            else if (value == "nox")
                coupling = PressureVelocityCoupling::CoupledNonlinear;
            else
                throw std::invalid_argument("--coupling requires piso, coupled, or nox");
        }
        else if (argument == "--coupled-operator")
            coupled_operator = coupled_operator_backend_from_string(value);
        else if (argument == "--coupled-workspace")
            workspace = coupled_workspace_policy_from_string(value);
        else if (argument == "--nonlinear-method")
        {
            if (value == "newton")
                nonlinear_method = CoupledLinearization::AnalyticNewton;
            else if (value == "picard")
                nonlinear_method = CoupledLinearization::Picard;
            else
                throw std::invalid_argument("--nonlinear-method requires newton or picard");
        }
        else if (argument == "--nonlinear-linear-solver")
        {
            if (value == "gmres")
                nonlinear_linear_solver = LinearSolverBackend::Gmres;
            else if (value == "bicgstab")
                nonlinear_linear_solver = LinearSolverBackend::BiCGStab;
            else
                throw std::invalid_argument("--nonlinear-linear-solver requires gmres or bicgstab");
        }
        else if (argument == "--nonlinear-iterations" || argument == "--nonlinear-restart")
        {
            if (value.empty() || !std::ranges::all_of(value, [](unsigned char c) { return std::isdigit(c); }))
                throw std::invalid_argument(std::string(argument) + " requires a positive integer");
            const int number = std::stoi(std::string(value));
            if (number <= 0)
                throw std::invalid_argument(std::string(argument) + " requires a positive integer");
            if (argument == "--nonlinear-iterations")
                nonlinear_iterations = number;
            else
                nonlinear_restart = number;
        }
        else
            return false;
        return true;
    }

    void apply(TimeStepperOptions& options) const
    {
        if (coupling)
            options.pressure_velocity_coupling = *coupling;
        if ((coupled_operator || workspace) &&
            options.pressure_velocity_coupling != PressureVelocityCoupling::CoupledKrylov &&
            options.pressure_velocity_coupling != PressureVelocityCoupling::CoupledNonlinear)
            throw std::invalid_argument("Coupled operator/workspace options require --coupling coupled or nox");
        if ((nonlinear_method || nonlinear_linear_solver || nonlinear_iterations || nonlinear_restart) &&
            options.pressure_velocity_coupling != PressureVelocityCoupling::CoupledNonlinear)
            throw std::invalid_argument("Nonlinear options require --coupling nox");
        if (coupled_operator)
            options.coupled_operator_backend = *coupled_operator;
        if (workspace)
            options.coupled_workspace_policy = *workspace;
        if (nonlinear_method)
            options.nonlinear.linearization = *nonlinear_method;
        if (nonlinear_linear_solver)
            options.nonlinear.linear_backend = *nonlinear_linear_solver;
        if (nonlinear_iterations)
            options.nonlinear.maximum_iterations = *nonlinear_iterations;
        if (nonlinear_restart)
            options.nonlinear.krylov_restart = *nonlinear_restart;
    }
};

/** Split only along Z, retaining all cell coordinates and physical patch names. */
template<class Pack>
std::shared_ptr<MeshHandle<Pack>> make_backend_mesh(const VerificationMesh& grid, const BackendControls& controls)
{
    using Handle = MeshHandle<Pack>;
    if (controls.mesh_backend == "native")
    {
        if (controls.regions != 1)
            throw std::invalid_argument("Multiple regions require --mesh-backend isoregion");
        return std::make_shared<Handle>(std::make_shared<Meshes::OrthogonalCartesian3D>(grid.coordinates()));
    }
    if (controls.mesh_backend != "isoregion" || !controls.regions || controls.regions > grid.nz())
        throw std::invalid_argument("Invalid IsoRegion selection or region count");
    using Composite = Meshes::MultiRegionMesh;
    std::vector<Composite::Region> regions;
    std::vector<Composite::Interface> interfaces;
    for (size_t r = 0; r < controls.regions; ++r)
    {
        const auto begin = grid.nz() * r / controls.regions;
        const auto end = grid.nz() * (r + 1) / controls.regions;
        ArrReal z(grid.z.begin() + begin, grid.z.begin() + end + 1);
        regions.emplace_back(Meshes::cartesian_region("region" + std::to_string(r), {{grid.x, grid.y, z}}));
        if (r)
            interfaces.emplace_back(Meshes::StructuredPatchInterface{{r - 1, 5}, {r, 4}});
    }
    return std::make_shared<Handle>(std::make_shared<Composite>(std::move(regions), std::move(interfaces),
        Meshes::InterfaceTolerance{}, Composite::BoundaryNamePolicy::MergeMatchingNames));
}
} // namespace SimpleFluid::Verification
