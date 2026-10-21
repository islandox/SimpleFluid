/** @file BubbleRepresentation.hh
 * @brief Explicit birth identities and conservative resolved/dispersed conversion policy.
 */
#pragma once

#include "FVM/AdaptiveVoF.hh"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace SimpleFluid
{

/** Opt-in numerical thresholds; their application sensitivity requires qualification. */
struct BubbleRepresentationOptions
{
    bool enabled = false;
    double promotion_diameter = 0.01; ///< m; promotion also requires finest-level resolution.
    double demotion_diameter = 0.004; ///< m; strictly below promotion_diameter.
    unsigned promotion_cells_per_diameter = 24;
    unsigned demotion_cells_per_diameter = 16;
    unsigned interface_guard_cells = 4;
    double demotion_neighborhood_factor = 3.0; ///< Host box extent relative to component bounds.
    double maximum_dispersed_fraction = 0.1;   ///< Maximum conditional dispersed volume in the host.
    double maximum_demoted_extent_ratio = 2.0; ///< Longest support extent / equivalent diameter.
    std::size_t maximum_event_cells = 1048576; ///< Bound on the accumulated sparse stencil union in one preview.
};

/**
 * Caller-supplied spatial identity: this is a localization closure, not a
 * random draw or a decision inferred from void fraction. One successful seed
 * consumes exactly one whole representative bubble from its named fixed bin.
 * IDs and payloads are collective metadata and cannot be reused with new data.
 */
struct BubbleBirthSeed
{
    std::uint64_t id = 0;
    VoF::Vector center{};
    SimpleFluid::Meshes::AMR::AMRBox donor_region{};
    std::string population;
};

enum class BubbleBirthStatus
{
    Pending,
    Reserved,
    Promoted
};

enum class BubbleConversionReason
{
    Ready,
    AlreadyPromoted,
    BelowPromotionDiameter,
    AboveDemotionDiameter,
    WellResolved,
    NeedsRefinement,
    FinestResolutionLimited,
    InsufficientDonors,
    InsufficientCarrier,
    OverlappingResolvedGas,
    OverlappingReservation,
    AttachedToBoundary,
    HeadspaceConnected,
    ElongatedComponent,
    ContainsInert,
    OutsideMolarGrid,
    StencilTooLarge,
    PeriodicStencilUnsupported,
    PendingRelaxation
};

/** Compact replicated birth metadata; cellwise donor reservations remain sparse. */
struct BubbleBirthRecord
{
    BubbleBirthSeed seed;
    BubbleBirthStatus status = BubbleBirthStatus::Pending;
    BubbleConversionReason reason = BubbleConversionReason::InsufficientDonors;
    std::uint64_t source_revision = 0;
    std::uint64_t source_topology = 0;
    double reserved_number = 0;
    double hydrogen_moles = 0;
    double gas_volume = 0;
    double diameter = 0;
};

/**
 * Complete connected gas component, assembled through shared geometric gas
 * apertures. label is a topology-dependent minimum cell key, not a persistent
 * bubble identity. Proven origin IDs come from a committed promotion and are
 * invalidated after physical advection or regrid; an empty list means unknown.
 * Caller birth records and conversion event IDs persist independently.
 */
struct ResolvedBubbleComponent
{
    std::uint64_t label = 0;
    std::uint64_t topology_generation = 0;
    std::size_t cells = 0;
    VoF::Vector centroid{};
    SimpleFluid::Meshes::AMR::AMRBox bounds{};
    double gas_volume = 0;
    double hydrogen_moles = 0;
    double inert_moles = 0;
    double diameter = 0;
    double surface_area = 0;
    double maximum_cell_width = 0; ///< Actual largest cell edge supporting this component.
    bool touches_wall = false;
    bool touches_headspace = false;
    bool crosses_periodic_seam = false;
    BubbleConversionReason reason = BubbleConversionReason::Ready;
    std::vector<std::uint64_t> originating_birth_ids;
};

enum class BubbleConversionKind
{
    Promotion,
    Demotion
};

/** A representation event creates neither physical hydrogen nor vent escape. */
struct BubbleConversionEvent
{
    std::uint64_t event_id = 0;
    BubbleConversionKind kind = BubbleConversionKind::Promotion;
    std::optional<std::uint64_t> birth_seed_id;
    std::uint64_t component_label = 0;
    VoF::Vector center{};
    double gas_volume = 0;
    double hydrogen_moles = 0;
    double dispersed_number_change = 0;
    double resolved_number_change = 0;
    double surface_area_change = 0;   ///< Actual area change on this event's disjoint material support.
    double surface_energy_change = 0; ///< sigma*area_change; reported, never silently added as heat.
    std::vector<std::uint64_t> originating_birth_ids;
};

/** Private previews can expose these records without publishing registry state. */
struct BubbleRepresentationInspection
{
    std::vector<BubbleBirthRecord> births;
    std::vector<ResolvedBubbleComponent> components;
    std::vector<BubbleConversionEvent> events;
    VoF::RefinementPlan refinement;
};

} // namespace SimpleFluid
