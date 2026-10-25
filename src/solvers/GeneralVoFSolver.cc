/** @file GeneralVoFSolver.cc
 * @brief Public general VoF solver forwarding and statistics entry points.
 */
#include "solvers/GeneralVoFSolver.hh"
#include "geometry/PLIC/GeometryInstrumentation.hh"
#include "solvers/details/GeneralVoF/SolverImpl.hh"
#include <stdexcept>
#include <utility>

namespace SimpleFluid
{
GeneralVoFSolver::GeneralVoFSolver(
    std::shared_ptr<const VoF::GeneralVoF> phases, VoFMaterial material, GeneralVoFSolverOptions options)
{
    if (!phases)
        throw std::invalid_argument("General VoF flow requires a phase authority.");
    impl_ = std::make_unique<Impl>(std::move(phases), material, options);
    impl_->initialize({}, {});
}
GeneralVoFSolver::~GeneralVoFSolver() = default;
GeneralVoFSolver::GeneralVoFSolver(GeneralVoFSolver&&) noexcept = default;
GeneralVoFSolver& GeneralVoFSolver::operator=(GeneralVoFSolver&&) noexcept = default;
void GeneralVoFSolver::initialize(const std::vector<VoF::Vector>& velocity, const std::vector<double>& face_velocity)
{
    impl_->initialize(velocity, face_velocity);
}
VoFStepDiagnostics GeneralVoFSolver::step(double dt)
{
    return impl_->step(dt);
}
double GeneralVoFSolver::stable_timestep() const
{
    return impl_->stable_timestep();
}
void GeneralVoFSolver::regrid(
    std::span<const Meshes::AMR::AdaptiveMark> refine, std::span<const Meshes::AMR::AdaptiveBytes> coarsen)
{
    impl_->remap(false, {}, refine, coarsen);
}
void GeneralVoFSolver::repartition(std::span<const int> destinations)
{
    impl_->remap(true, destinations, {}, {});
}
double GeneralVoFSolver::time() const noexcept
{
    return impl_->accepted->diagnostics.time;
}
const std::shared_ptr<const VoF::GeneralVoF>& GeneralVoFSolver::phases() const noexcept
{
    return impl_->accepted->phases;
}
const std::shared_ptr<const MeshHandle<>>& GeneralVoFSolver::mesh() const noexcept
{
    return impl_->resources->mesh;
}
const std::vector<double>& GeneralVoFSolver::gas_fraction() const noexcept
{
    return impl_->accepted->fraction;
}
const std::array<GeneralVoFMomentumDiagnostics, 3>& GeneralVoFSolver::momentum_transport_diagnostics() const noexcept
{
    return impl_->accepted->momentum_transport;
}
const std::vector<VoF::Vector>& GeneralVoFSolver::velocity() const noexcept
{
    return impl_->accepted->velocity;
}
const std::vector<double>& GeneralVoFSolver::face_velocity() const noexcept
{
    return impl_->accepted->face_velocity;
}
const std::vector<double>& GeneralVoFSolver::pressure() const noexcept
{
    return impl_->accepted->pressure;
}
const std::vector<double>& GeneralVoFSolver::curvature() const noexcept
{
    return impl_->accepted->curvature;
}
const VoFStepDiagnostics& GeneralVoFSolver::diagnostics() const noexcept
{
    return impl_->accepted->diagnostics;
}
} // namespace SimpleFluid

namespace SimpleFluid
{
void GeneralVoFSolver::enable_geometry_statistics(bool enabled)
{
    SimpleFluid::PLIC::enable_backend_statistics<SimpleFluid::ArithmeticMode::Exact>(enabled);
}
void GeneralVoFSolver::reset_geometry_statistics()
{
    SimpleFluid::PLIC::reset_backend_statistics<SimpleFluid::ArithmeticMode::Exact>();
    PLIC::reset_geometry_statistics();
}
Meshes::GeometryStatisticsSnapshot GeneralVoFSolver::geometry_statistics()
{
    return {SimpleFluid::PLIC::backend_statistics_domain<SimpleFluid::ArithmeticMode::Exact>(),
        PLIC::geometry_statistics_domain(), SimpleFluid::PLIC::backend_statistics<SimpleFluid::ArithmeticMode::Exact>(),
        PLIC::geometry_statistics()};
}
} // namespace SimpleFluid
