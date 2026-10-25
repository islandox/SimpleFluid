/** @file GeneralVoFSolver.cc
 * @brief Transactional native phase/implicit-momentum/physical-flow composition.
 */
#include "solvers/GeneralVoFSolver.hh"
#include "FVM/ExactFaceTransfer.hh"
#include "FVM/ExactPhaseTransfer.hh"
#include "geometry/PLIC/PlanarFace.hh"
#include "geometry/PLIC/PlanarGeometry.hh"
#include "geometry/PLIC/details/PhaseArithmetic.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveNative.hh"
#include "parallel/SparseOwnerExchange.hh"
#include "solvers/details/ConservativeVoFExactArithmetic.hh"
#include "solvers/details/ConservativeVoFTransport.hh"
#include "solvers/details/GeneralVoFCurvature.hh"
#include "solvers/details/GeneralVoFExactStorage.hh"
#include "solvers/details/GeneralVoFFlowWorkspace.hh"
#include "solvers/details/GeneralVoFPhiClosure.hh"
#include "utils/CompensatedSum.hh"
#include <Teuchos_CommHelpers.hpp>
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <exception>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>
namespace SimpleFluid
{
namespace
{
using Pack = DefaultTpetraTypes;
using LO = Pack::local_ordinal_type;
using GO = Pack::global_ordinal_type;
using Vector = VoF::Vector;
double dot(const Vector& a, const Vector& b)
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
void append_word(std::vector<char>& bytes, std::uint64_t value)
{
    for (unsigned a = 0; a < 8; ++a)
        bytes.push_back(char(value >> (8 * a)));
}
std::uint64_t read_word(const std::vector<char>& bytes, std::size_t offset)
{
    std::uint64_t value = 0;
    for (unsigned a = 0; a < 8; ++a)
        value |= std::uint64_t(static_cast<unsigned char>(bytes.at(offset + a))) << (8 * a);
    return value;
}
std::shared_ptr<const MeshHandle<>> validated_phase_mesh(const std::shared_ptr<const VoF::GeneralVoF>& phases)
{
    const auto mesh = phases->mesh();
    int local = phases->hierarchy()->geometry_unchanged(), global = 0;
    Teuchos::reduceAll(*mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_MIN, 1, &local, &global);
    if (!global)
        throw std::runtime_error("General VoF source geometry is stale before workspace construction.");
    return mesh;
}
std::vector<Vector> finite_spans(const std::shared_ptr<const VoF::GeneralVoF>& phases)
{
    std::vector<Vector> result;
    if (phases->arithmetic_mode() != ArithmeticMode::Exact)
        for (const auto& geometry : phases->owned_finite_geometry())
        {
            const auto spans = geometry->axis_spans();
            result.push_back({spans[0].as_double(), spans[1].as_double(), spans[2].as_double()});
        }
    return result;
}
} // namespace
#include "solvers/details/GeneralVoFExactSolver.tcc"
#include "solvers/details/GeneralVoFSolverFinite.tcc"
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
    return impl_->stable_timestep(*impl_->accepted, impl_->node_velocity(*impl_->accepted));
}
void GeneralVoFSolver::regrid(
    std::span<const Meshes::AMR::AdaptiveMark> refine, std::span<const Meshes::AMR::AdaptiveBytes> coarsen)
{
    impl_ = impl_->remap(false, {}, refine, coarsen);
}
void GeneralVoFSolver::repartition(std::span<const int> destinations)
{
    impl_ = impl_->remap(true, destinations, {}, {});
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
    return impl_->mesh;
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
}
