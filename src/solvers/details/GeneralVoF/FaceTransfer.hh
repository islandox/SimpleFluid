/** @file FaceTransfer.hh
 * @brief Borrowed transfer inputs and detached canonical face results.
 */
#pragma once
#include "FVM/GeneralVoF.hh"
#include "SimpleFluidExport.hh"
#include "solvers/details/GeneralVoFFlowWorkspace.hh"
#include <span>
namespace SimpleFluid::detail::general_vof
{
/** Inputs borrow publications pinned by the remap coordinator for the whole call. */
struct FaceTransferSource
{
    const MeshHandle<>& mesh;
    const VoF::GeneralVoF& phases;
    std::span<const VoF::Vector> velocity;
    const GeneralVoFFiniteProjectionProvenance& projection;
    const GeneralVoFFlowWorkspace& flow;
    ArithmeticMode arithmetic_mode;
    /** Ownership migration copies accepted face diagnostics without division. */
    bool preserve_accepted_fields = false;
    std::span<const double> accepted_face_velocity;
};
struct FaceTransferTarget
{
    const MeshHandle<>& mesh;
    const VoF::GeneralVoF& phases;
    std::span<const double> volume;
    const GeneralVoFFlowWorkspace& flow;
    ArithmeticMode arithmetic_mode;
};
struct FaceTransferControls
{
    std::size_t byte_budget;
    double divergence_tolerance;
};
/** Detached output with no authority to publish accepted solver state. */
struct FaceTransferResult
{
    std::vector<double> face_velocity;
    GeneralVoFFiniteProjectionProvenance projection;
    double divergence = 0;
};
SIMPLEFLUID_SOLVERS_LOCAL void append_face_word(std::vector<char>&, std::uint64_t);
SIMPLEFLUID_SOLVERS_LOCAL std::uint64_t read_face_word(const std::vector<char>&, std::size_t offset);
} // namespace SimpleFluid::detail::general_vof
