/** @file LowMachFaceTransportAdapter.hh
 * @brief distributed diffusion/Fourier packets after pure chemistry.
 */
#pragma once
#include "solvers/LowMachMixtureSolver.hh"
#include "solvers/MixtureFaceTransport.hh"
#include <Tpetra_Import.hpp>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <span>

namespace SimpleFluid::Reacting::low_mach_faces
{
using Flow = LowMachMixtureSolver;
using Pack = Flow::Pack;
using Mesh = Flow::Mesh;
using LO = Pack::local_ordinal_type;
using GO = Pack::global_ordinal_type;

using SideSample = MixtureFacePhaseSample;
using FaceContext = MixtureFaceContext;
using DiffusionLaw = MixtureDiffusionLaw;
using FourierLaw = MixtureFourierLaw;
using Options = MixtureFaceTransportOptions;
using PhaseBindings = std::vector<std::vector<DiffusionCompartment>>;

struct StageStamp
{
    const Mesh* mesh = nullptr;
    std::uint64_t accepted_revision = 0, accepted_generation = 0;
    std::uint64_t old_geometry_epoch = 0, new_geometry_epoch = 0, schema_fingerprint = 0;
    double accepted_time = 0, time_step = 0;
    Flow::SourceStageMetadata provider;
    bool operator==(const StageStamp&) const = default;
};
struct MolarTerm
{
    SpeciesId species = 0;
    DiffusionCompartment compartment;
    double moles = 0;
};
struct MolarEnthalpyTerm
{
    double moles = 0, molar_enthalpy = 0; ///< Root accumulates long-double products before final narrowing.
};
struct RawCellReceipt
{
    std::vector<MolarTerm> material;                 ///< Individual signed face terms, not endpoint differences.
    std::vector<MolarEnthalpyTerm> diffusion_energy; ///< Same canonical donor-h term on both sides, opposite mol sign.
    std::vector<double> fourier_energy;              ///< Individual opposite-signed integrated J packets.
};
struct FourierFaceLedger
{
    GO geometry_identity = 0;
    std::optional<std::size_t> owner, neighbor;
    double energy = 0; ///< Canonical integrated J owner -> neighbor; count global diagnostics only if owner is owned.
    double decrease = 0, increase = 0;    ///< Canonical signed packet's binary64 temperature operand bounds, J.
    double conductance = 0;               ///< W/K, as evaluated by the canonical face owner.
    GO owner_cell = 0, neighbor_cell = 0; ///< Algebraic cell map IDs for the thermal block.
};
struct FourierCellLinearization
{
    double capacity = 0, temperature = 0; ///< Fixed-pressure enthalpy derivative (J/K) and actual face operand (K).
    double minimum_temperature = 0, maximum_temperature = std::numeric_limits<double>::infinity();
};
class SIMPLEFLUID_SOLVERS_LOCAL LowMachFaceTransportAdapter;

struct Receipt
{
    StageStamp stamp;
    std::size_t physical_phase = 0;
    SpeciesDiffusionPreview diffusion;
    std::vector<CanonicalDiffusionFace> raw_diffusion_faces;
    std::vector<FourierFaceLedger> fourier_faces;
    std::optional<std::vector<FourierCellLinearization>> fourier_thermal; ///< Owned cells, including empty owners.
    std::vector<RawCellReceipt> cells;
    // Later interphase stages consume diffusion.state as actual donors, never the pre-face base.
    Flow::SourceStageResult rounded;
    /** Controller must accumulate raw cells together with raw chemistry channels in
     * compensated accumulators, then emit one CellSources packet. Do not add raw
     * and rounded views twice. Raw channels retain small terms before cancellation
     * against a large prior-stage heat/source term.
     */
private:
    friend class LowMachFaceTransportAdapter;
    // Controller retains each immutable producer through SourceEngine ownership.
    // Distinct ordered option instances must not accept one another's receipts.
    const LowMachFaceTransportAdapter* producer_ = nullptr;
};

enum class FaceTopology
{
    NonPeriodic,
    ClosedAzimuthalFourier ///< Fixed native cylindrical kinetic carrier, with a canonical azimuthal seam.
};

/** Same physical carrier, zero-net-mixture-mass barycentric diffusion: momentum
 * packets are exactly zero by declaration, not donor-velocity cancellation.
 * Physical exterior faces carry no flux. The reacting controller can admit a
 * native cylindrical azimuthal seam as a canonical interior Fourier contact.
 * Every pure local stage is fenced before the following import/reduction.
 * Each canonical owned face is evaluated once and imported to incident ranks.
 * All owned primary/counter donors over all faces are jointly reserved before credits.
 * The caller owns final caloric/pressure-work recovery and its strict T/energy gates.
 */
class SIMPLEFLUID_SOLVERS_LOCAL LowMachFaceTransportAdapter
{
public:
    LowMachFaceTransportAdapter(std::shared_ptr<const Mesh> mesh, std::shared_ptr<const SpeciesCatalog> catalog,
        Options options, PhaseBindings compartments, std::optional<PhaseThermalMode> thermal_mode = std::nullopt,
        FaceTopology topology = FaceTopology::NonPeriodic);
    Receipt preview(const Flow::SourceStageContext& context, std::span<const CellSpeciesState> post_chemistry) const;
    /** Explicit prepared T/p/EOS guesses, e.g. arriving-phase T recovered from transported U by SourceEngine.
     * Donor composition STILL comes only from post_chemistry, never these cells' moles.
     */
    Receipt preview(const Flow::SourceStageContext& context, std::span<const CellSpeciesState> post_chemistry,
        std::span<const Flow::CellState> prepared_caloric_guesses) const;
    /** Actual owner-local predecessor parts; never included in halo sampling/packets. */
    Receipt preview(const Flow::SourceStageContext& context, std::span<const CellSpeciesState> post_chemistry,
        std::span<const Flow::CellState> prepared_caloric_guesses,
        std::span<const std::vector<std::array<long double, 2>>> owner_parts) const;
    void validate_receipt(const Receipt& receipt, const Flow::SourceStageContext& context) const;

private:
    Receipt preview_impl(const Flow::SourceStageContext& context, std::span<const CellSpeciesState> post_chemistry,
        std::span<const Flow::CellState> prepared_caloric_guesses,
        std::span<const std::vector<std::array<long double, 2>>> owner_parts, bool raw_authority) const;
    std::shared_ptr<const Mesh> mesh_;
    std::shared_ptr<const SpeciesCatalog> catalog_;
    Options options_;
    PhaseBindings compartments_;
    std::optional<PhaseThermalMode> thermal_mode_;
    FaceTopology topology_;
    Teuchos::RCP<const Pack::map_type> owned_cells_, overlap_cells_, owned_faces_, overlap_faces_;
    std::unique_ptr<Pack::import_type> cell_import_, face_import_;
};
} // namespace SimpleFluid::Reacting::low_mach_faces
