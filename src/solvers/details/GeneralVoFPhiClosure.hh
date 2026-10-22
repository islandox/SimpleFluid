/** @file GeneralVoFPhiClosure.hh
 * @brief Bounded canonical arithmetic closure of source-free integrated volume.
 */
#pragma once
#include "SimpleFluidExport.hh"
#include "geometry/ExactSweptGeometry.hh"
#include "geometry/MeshHandle.hh"
#include "solvers/details/GeneralVoFProjectionArithmetic.hh"
#include <optional>
#include <stdexcept>
namespace SimpleFluid::detail
{
/** Persistent identities are authenticated geometry-provider certificates,
 * independent of observed global numbering and ownership. */
struct GeneralVoFPhiCell
{
    DefaultTpetraTypes::global_ordinal_type cell = 0;
    std::vector<char> identity;
    Meshes::ExactGeometryMeasure arithmetic_radius;
};
struct GeneralVoFPhiFace
{
    DefaultTpetraTypes::global_ordinal_type face = 0, first = 0, second = 0;
    std::vector<char> identity;
    Meshes::ExactGeometryMeasure volume, arithmetic_radius;
};
struct GeneralVoFPhiClosure
{
    std::vector<Meshes::ExactGeometryMeasure> volume, correction; ///< Input face order.
    std::size_t forest_rounds = 0;
    bool bounded_resource_fallback = false; ///< Diagnostic only; original gates were rerun unchanged.
};
/** Bounded correction is infeasible; metadata/range/resource errors are distinct. */
class GeneralVoFPhiInfeasible : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};
struct GeneralVoFProjectionRateCertificate
{
    bool arithmetic_admitted = false;
    std::vector<Meshes::ExactGeometryMeasure> owned_rate, owned_face_radius, owned_cell_radius, owned_net_rate;
    // Local buffer accounting diagnostics; these never determine physics.
    std::size_t packet_input_cap = 0, packet_exchange_cap = 0, packet_arena_cap = 0;
    std::size_t packet_sent_bytes = 0, packet_received_bytes = 0, packet_exchange_peak_bytes = 0;
    std::size_t packet_resident_input_bytes = 0, packet_resident_output_bytes = 0;
    std::size_t packet_decode_cap = 0, packet_codec_peak_bytes = 0;
};
/** Persistent topology input. Scalar fields must be zero; the pressure
 * operation certificate supplies rates/radii independently each attempt. */
struct GeneralVoFProjectionClosureTopology
{
    std::vector<GeneralVoFPhiCell> cells;
    std::vector<GeneralVoFPhiFace> faces;
};
/** Inputs contain owned cells and canonical-owned INTERNAL faces only; closed
 * exterior controls must be identically zero before entry. Independently derived
 * arithmetic radii are absolute operation/projection provenance bounds, never
 * selected from the observed divergence or physical solver tolerance.
 *
 * A persistent-identity BFS forest routes exact subtree defects. Every correction
 * must lie in its face radius and the sum of incident absolute corrections must
 * lie in the cell radius. Closed roots must have exactly zero aggregate defect.
 * Failure leaves all inputs untouched; empty owners participate collectively.
 */
[[nodiscard]] SIMPLEFLUID_SOLVERS_LOCAL GeneralVoFPhiClosure close_general_vof_phi(const SP<const MeshHandle<>>&,
    const std::vector<GeneralVoFPhiCell>&, const std::vector<GeneralVoFPhiFace>&, std::size_t byte_budget);
/** Propose fixed-precision dyadic rates before forest closure. All discarded
 * differences consume the ORIGINAL face/cell arithmetic capacities. Exact
 * nonzero signs are preserved; infeasible proposals or typed proposal/packet resource
 * failures retry the original closure. Invalid metadata still rejects.
 * No physical or arithmetic tolerance is changed. */
[[nodiscard]] SIMPLEFLUID_SOLVERS_LOCAL GeneralVoFPhiClosure close_general_vof_bounded_rate(
    const SP<const MeshHandle<>>&, const std::vector<GeneralVoFPhiCell>&,
    const std::vector<GeneralVoFPhiFace>&, std::size_t byte_budget);
[[nodiscard]] SIMPLEFLUID_SOLVERS_LOCAL std::optional<GeneralVoFPhiClosure> try_close_general_vof_phi(
    const SP<const MeshHandle<>>&, const std::vector<GeneralVoFPhiCell>&, const std::vector<GeneralVoFPhiFace>&,
    std::size_t byte_budget);
/** Optional primary rate/radius are authoritative exact pressure iterates;
 * represented velocity values must match their diagnostic rate/area views.
 * Returned cell net rates support exact original-equation defect normalization. */
[[nodiscard]] SIMPLEFLUID_SOLVERS_LOCAL GeneralVoFProjectionRateCertificate certify_general_vof_projection_rate(
    const SP<const MeshHandle<>>&, const std::vector<GeneralVoFRoundedValue>&, std::size_t byte_budget,
    const std::vector<Meshes::ExactGeometryMeasure>* primary_rate = nullptr,
    const std::vector<Meshes::ExactGeometryMeasure>* primary_radius = nullptr);
/** Exact source-free rate admission before any arithmetic face repair. */
[[nodiscard]] SIMPLEFLUID_SOLVERS_LOCAL bool general_vof_projection_arithmetic_admitted(
    const SP<const MeshHandle<>>&, const std::vector<GeneralVoFRoundedValue>&, std::size_t byte_budget);
} // namespace SimpleFluid::detail
