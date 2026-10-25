/** @file SolverImpl.hh
 * @brief Private solver ownership and operation declarations.
 */
#pragma once
#include "solvers/GeneralVoFSolver.hh"
#include "solvers/details/ConservativeVoFTransport.hh"
#include "solvers/details/GeneralVoF/State.hh"
#include "solvers/details/GeneralVoFFlowWorkspace.hh"
#include "solvers/details/GeneralVoFPhiClosure.hh"
#include <memory>
#include <span>
#include <vector>

namespace SimpleFluid
{
struct GeneralVoFSolver::Impl
{
    using Pack = DefaultTpetraTypes;
    using Vector = VoF::Vector;
    using State = detail::general_vof::State;
    struct PreparedRemap
    {
        std::unique_ptr<Impl> resources;
        std::shared_ptr<State> state;
    };
    std::shared_ptr<const MeshHandle<>> mesh;
    ArithmeticMode arithmetic_mode_ = ArithmeticMode::Ordinary;
    VoFMaterial material;
    GeneralVoFSolverOptions options;
    detail::GeneralVoFFlowWorkspace flow;
    detail::ConservativeVoFTransport transport;
    std::vector<double> volume, thickness;
    std::shared_ptr<const State> accepted;

    void require(bool condition, const char* message) const;

    template<class Work> void collective(Work&& work) const;

    double maximum(double value) const;

    Impl(std::shared_ptr<const VoF::GeneralVoF> phases, VoFMaterial m, GeneralVoFSolverOptions o);

    template<ArithmeticMode Mode>
    std::shared_ptr<State> construct_mode(
        std::shared_ptr<const VoF::GeneralVoF> phases, VoFMaterial m, GeneralVoFSolverOptions o);
    bool finite(const State& state) const { return state.phases->arithmetic_mode() != ArithmeticMode::Exact; }

    std::vector<double> import(
        const std::vector<double>& owned, const Teuchos::RCP<const Pack::map_type>& target) const;

    void surface_curvature(State& state) const;

    template<ArithmeticMode Mode> void surface_curvature_mode(State& state) const;

    std::vector<VoF::GeneralVoFNodeVelocity> node_velocity(const State& state) const;

    double stable_timestep(const State& state, const std::vector<VoF::GeneralVoFNodeVelocity>& nodes) const;

    template<ArithmeticMode Mode>
    double stable_timestep_mode(const State& state, const std::vector<VoF::GeneralVoFNodeVelocity>& nodes) const;

    void initialize(const std::vector<Vector>& owned_velocity, const std::vector<double>& face_velocity);

    template<ArithmeticMode Mode>
    std::shared_ptr<State> prepare_initialize_mode(
        const std::vector<Vector>& owned_velocity, const std::vector<double>& face_velocity);

    template<class Topology = detail::Exact::GeneralVoFProjectionClosureTopology>
    Topology projection_topology(const VoF::GeneralVoF& phases) const;

    void phase_storage(State& state) const;

    template<ArithmeticMode Mode> void phase_storage_mode(State& state) const;

    template<ArithmeticMode Mode> std::vector<PLIC::DynamicGeometryValue> finite_rates_mode(const State& state) const;

    template<ArithmeticMode Mode>
    std::vector<VoF::GeneralVoFFaceVolume> finite_face_volumes_mode(const State& state, double dt) const;

    template<ArithmeticMode Mode>
    void verify_phase_publication_mode(const VoF::GeneralVoF::Trial& trial, const VoF::GeneralVoF& candidate) const;

    void budget(State& state) const;

    template<ArithmeticMode Mode> void budget_mode(State& state) const;

    VoFStepDiagnostics step(double dt);

    template<ArithmeticMode Mode> std::shared_ptr<State> prepare_step_mode(double dt);

    template<ArithmeticMode Mode>
    double handoff_finite_adapted_faces_mode(
        const State& old, Impl& target, State& state, const VoF::GeneralVoF::FiniteRemapTrial& trial) const;

    template<ArithmeticMode Mode>
    double handoff_finite_owned_faces_mode(const State& old, Impl& target, State& state) const;

    std::unique_ptr<Impl> remap(bool migration, std::span<const int> destinations,
        std::span<const Meshes::AMR::AdaptiveMark> refine, std::span<const Meshes::AMR::AdaptiveBytes> coarsen) const;

    template<ArithmeticMode Mode>
    PreparedRemap prepare_remap_mode(bool migration, std::span<const int> destinations,
        std::span<const Meshes::AMR::AdaptiveMark> refine, std::span<const Meshes::AMR::AdaptiveBytes> coarsen) const;
};
} // namespace SimpleFluid

#include "solvers/details/GeneralVoF/SolverImpl.ipp"
