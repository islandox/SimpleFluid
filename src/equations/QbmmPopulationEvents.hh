/** @file QbmmPopulationEvents.hh
 * @brief Constructive local events for a fixed-composition bubble family.
 */
#pragma once

#include "equations/QbmmMoments.hh"
#include "equations/QbmmPopulationSchema.hh"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <variant>
#include <vector>

namespace SimpleFluid::Reacting
{

class QbmmBulkGasPoolBinding;
enum class QbmmPoolConversionDirection;

struct QbmmResolvedPoolConversion
{
    std::uint64_t pool_fingerprint;
    QbmmPoolConversionDirection direction;
};

/** An unnamed cell compartment. The host owns partial-molar enthalpy and momentum transfer.
 * The bubble kernel supplies paired material only and creates no heat or pressure work.
 */
struct QbmmPhaseExchange
{
    SpeciesPhase phase = SpeciesPhase::Dissolved;
    std::optional<QbmmResolvedPoolConversion> resolved_pool{};
};

/** Net external energy and momentum to this cell for this allocation, independent of pressure work. */
struct QbmmReservoirExchange
{
    std::string reservoir;
    double energy_j = 0;
    std::array<double, 3> momentum{}; ///< kg m/s, signed components in the cell frame.
};

using QbmmTransferCounterpart = std::variant<QbmmPhaseExchange, QbmmReservoirExchange>;

struct QbmmMaterialTransfer
{
    QbmmTransferCounterpart counterpart;
    std::vector<double> moles; ///< Nonnegative integrated mol per cell, full catalog order.
};

struct QbmmBirthAtom
{
    QbmmAtom atom;
    QbmmMaterialTransfer transfer;
};

struct QbmmBirth
{
    std::vector<QbmmBirthAtom> atoms; ///< One paired material allocation per positive birth atom.
};

struct QbmmResize
{
    std::size_t donor = 0;
    double count = 0, mass_over_scale = 0; ///< Existing count and its strictly positive destination size.
    QbmmMaterialTransfer transfer;         ///< Direction inferred from the change in size.
};

struct QbmmDisappearance
{
    std::size_t donor = 0;
    double count = 0;
    QbmmMaterialTransfer transfer; ///< Removes all material carried by this count of the donor atom.
};

struct QbmmCoalescence
{
    std::size_t first = 0, second = 0;
    double count = 0; ///< Number of pairs. Equal indices consume twice this count at that node.
};

struct QbmmBreakup
{
    std::size_t donor = 0;
    double count = 0;
    std::vector<QbmmAtom> daughters; ///< Positive daughter multiplicities and scaled sizes per parent.
};

using QbmmEventOperation = std::variant<QbmmBirth, QbmmResize, QbmmDisappearance, QbmmCoalescence, QbmmBreakup>;

/** The caller supplies already integrated law results on the pre-material quadrature.
 * All donor indices refer to that same quadrature, even when other events create products.
 */
struct QbmmPopulationEvent
{
    std::string provider;
    std::uint64_t provider_revision = 0;
    QbmmEventOperation operation;
};

struct QbmmEventBatch
{
    std::uint64_t family_fingerprint = 0;
    std::uint64_t substep = 0;
    double duration = 0; ///< Provenance in seconds. The kernel does not rescale integrated counts or moles.
    std::vector<QbmmPopulationEvent> events;
};

enum class QbmmEventKind
{
    Birth,
    Growth,
    Shrinkage,
    Disappearance,
    Coalescence,
    Breakup
};

struct QbmmDonorAtom
{
    std::size_t index = 0;
    QbmmAtom atom; ///< Original size and the count reserved from this index.
};

struct QbmmMaterialTransferReceipt
{
    QbmmMaterialTransfer transfer; ///< The original paired material packet and thermal/momentum handoff.
    bool into_family = false;
};

using QbmmEventMoments = std::array<long double, 4>;

struct QbmmEventReceipt
{
    std::string population, provider;
    std::uint64_t family_fingerprint = 0, catalog_fingerprint = 0, provider_revision = 0, substep = 0;
    double duration = 0;
    QbmmEventKind kind = QbmmEventKind::Birth;
    std::vector<QbmmDonorAtom> donors;
    std::vector<QbmmAtom> products;
    std::vector<QbmmMaterialTransferReceipt> transfers;
    QbmmEventMoments gross_consumed{}, gross_created{}; ///< Atomic removals/insertions, including resized counts.
    QbmmEventMoments moment_change{};                   ///< Signed event report. Replay uses the original atoms.
    long double material_mass_change_over_scale = 0;    ///< Signed mass evaluated from the raw moles.
};

struct QbmmPopulationEndpoint
{
    CellSpeciesState species;
    double count = 0;
    QbmmShapeState shape{};
    QbmmQuadrature quadrature; ///< Derived from the accepted species/count/shape block.
    std::vector<QbmmEventReceipt> receipts;
    QbmmEventMoments moment_change{}, gross_consumed{}, gross_created{};
    std::uint64_t family_fingerprint = 0;
};

enum class QbmmEventStatus
{
    Accepted,
    InvalidState,
    InvalidProvenance,
    InvalidEvent,
    InvalidTransfer,
    SpeciesDonorOverdraw,
    CountDonorOverdraw,
    UnrepresentableEndpoint,
    InconsistentEndpoint
};

struct QbmmSpeciesDonor
{
    SpeciesId species;
    SpeciesPhase phase;
    std::string population;
};

struct QbmmDonorShortfall
{
    std::variant<std::size_t, QbmmSpeciesDonor> stock; ///< An original atom index or catalogued species compartment.
    long double available; ///< Remaining original stock before this withdrawal, in counts or mol/cell.
    double requested;
    long double shortfall; ///< Positive deficit evaluated before rounding available and requested.
};

struct QbmmRejectedEventReceipt
{
    std::string population;
    std::uint64_t family_fingerprint, catalog_fingerprint, substep;
    double duration;
    QbmmPopulationEvent attempt;
    QbmmEventStatus reason;
    std::optional<QbmmDonorShortfall> donor_shortfall;
};

struct QbmmPopulationAdvance
{
    QbmmEventStatus status = QbmmEventStatus::InvalidState;
    std::optional<QbmmPopulationEndpoint> endpoint;
    std::optional<std::size_t> failed_event;
    std::string diagnostic;
    std::optional<QbmmRejectedEventReceipt> rejected_event;
};

/** Pure atomic local advance. The initial species state already contains the named family,
 * including when empty. The family view must alias that exact species row and carry the
 * actual state provenance. All gross donors are reserved against the same predecessor.
 * A failed result carries diagnostics and any attributable rejected-event provenance.
 */
[[nodiscard]] SIMPLEFLUID_EQUATIONS_EXPORT QbmmPopulationAdvance advance_qbmm_population(
    const QbmmPopulationSchema& schema, const CellSpeciesState& initial, QbmmCellView pre_cell,
    const QbmmEventBatch& batch);

struct QbmmFamilyEventInput
{
    const QbmmPopulationSchema& schema;
    QbmmCellView pre_cell;
    const QbmmEventBatch& batch;
};

struct QbmmFamilyEventEndpoint
{
    double count = 0;
    QbmmShapeState shape{};
    QbmmQuadrature quadrature;
    std::vector<QbmmEventReceipt> receipts;
    std::uint64_t family_fingerprint = 0;
};

struct QbmmJointPopulationEndpoint
{
    CellSpeciesState species;
    std::vector<QbmmFamilyEventEndpoint> families; ///< Same order as the request.
};

struct QbmmJointPopulationAdvance
{
    QbmmEventStatus status = QbmmEventStatus::InvalidState;
    std::optional<QbmmJointPopulationEndpoint> endpoint;
    std::optional<std::size_t> failed_family; ///< Index in the request.
    std::optional<std::size_t> failed_event;
    std::string diagnostic;
    std::optional<QbmmRejectedEventReceipt> rejected_event;
};

/** Pure cell transaction for one or more distinct named families sharing one authoritative predecessor and substep.
 * Gross withdrawals reserve original atom and species stocks before any credits can fund them.
 * Families are processed by identity, with accepted results and failed indices in request order.
 * Resolved exchanges use Birth and Disappearance with the matching pool identity and direction.
 * A birth requires the original occupied pool to match the fixed family composition.
 */
[[nodiscard]] SIMPLEFLUID_EQUATIONS_EXPORT QbmmJointPopulationAdvance advance_qbmm_populations(
    const CellSpeciesState& initial, std::span<const QbmmFamilyEventInput> families,
    const QbmmBulkGasPoolBinding* resolved_pool = nullptr);

} // namespace SimpleFluid::Reacting
