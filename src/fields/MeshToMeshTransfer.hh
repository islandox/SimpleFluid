/**
 * @file MeshToMeshTransfer.hh
 * @brief Reusable cell-field interpolation and conservative mesh projection.
 */
#pragma once

#include "fields/FieldStored.hh"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace SimpleFluid
{

enum class MeshToMeshTransferMethod
{
    NearestCell,
    InverseDistance,
    ConservativeCellAverage
};

enum class MeshToMeshCoverageMode
{
    RequireFull,
    AllowPartial
};

/// The same representation is used by the source and target of project().
enum class MeshToMeshQuantity
{
    Intensive, ///< Cell average or density: its integral is value * cell volume.
    Extensive  ///< Cell inventory: the value is already integrated over the cell.
};

/** @brief Conservative geometry snapshot; vectors follow each owned cell map.
 *
 * Global totals count owned cells only. Uncovered volumes are the signed
 * residual cell_volume - covered_volume; tiny negative roundoff is retained,
 * not clipped or used to renormalize weights. Significant overcoverage fails.
 */
struct MeshToMeshCoverage
{
    std::vector<double> source_cell_volumes, source_covered_volumes,
                        source_uncovered_volumes;
    std::vector<double> target_cell_volumes, target_covered_volumes,
                        target_uncovered_volumes;
    double source_volume = 0, target_volume = 0, overlap_volume = 0;
    double uncovered_source_volume = 0, uncovered_target_volume = 0;
};

/** @brief Globally reduced inventories/integrals, one entry per field component.
 *
 * All entries use integral units (e.g. joules or moles), irrespective of the
 * source/target representation. Signed values and roundoff remain visible.
 */
struct MeshToMeshConservationReport
{
    std::vector<double> source_integral;
    std::vector<double> transferred_integral;
    std::vector<double> uncovered_source_integral;
    std::vector<double> target_integral;
    std::vector<double> transfer_error;     ///< target - transferred
    std::vector<double> conservation_error; ///< target + uncovered - source
};

struct MeshToMeshTransferOptions
{
    MeshToMeshTransferMethod method = MeshToMeshTransferMethod::InverseDistance;
    size_t neighbors = 8;
    /// Maximum distance to the nearest donor centroid for interpolation.
    double max_distance = std::numeric_limits<double>::infinity();
    /// Relative uncovered-volume tolerance for conservative projection.
    double coverage_tolerance = 1e-10;
    MeshToMeshCoverageMode coverage_mode = MeshToMeshCoverageMode::RequireFull;
};

/**
 * @brief Cached linear transfer between two distributed cell maps.
 *
 * Construction uses one stable collective context: Tpetra::getDefaultComm()
 * by default, or the explicit construction_comm overload. Each endpoint must
 * use one consistent communicator context across its participating ranks;
 * source, target and construction contexts may be distinct MPI_CONGRUENT
 * communicators. Applications are collective on the captured source context.
 * Options and component counts must agree on every rank.
 * Interpolation uses Cartesian centroid distances and extrapolates unless
 * max_distance limits the nearest donor distance. InverseDistance uses the
 * nearest k cells with inverse-square weights and reproduces coincident data.
 * Neither interpolator promises integral conservation or linear exactness.
 * ConservativeCellAverage builds exact Cartesian or coaxial cylindrical
 * annular-sector overlaps. Full coverage is required by default. project()
 * distinguishes cell averages/densities from cell inventories and returns a
 * conservation report. Partial coverage must be explicitly enabled; no
 * extrapolation, inventory clipping or covered-volume renormalization occurs.
 * Vector/tensor components retain the same Cartesian frame without rotation.
 *
 * Meshes are retained by shared ownership. Reconstruct after geometry motion,
 * repartitioning or topology changes; changed geometry epochs are rejected.
 * Setup replicates source geometry and searches every source cell for each
 * owned target cell. Applications use a sparse distributed Tpetra matrix.
 * See docs/mesh_to_mesh_transfer.md for usage and limits.
 */
template<TpetraTypePack Pack = DefaultTpetraTypes>
class MeshToMeshTransfer
{
public:
    using mesh_type = MeshHandle<Pack>;
    using multi_vector_type = typename Pack::multi_vector_type;

    /** @brief Construct collectively on Tpetra::getDefaultComm(), even for null inputs.
     * @note Call the explicit-context overload for a subgroup of that communicator.
     */
    MeshToMeshTransfer(SP<const mesh_type> source,
                       SP<const mesh_type> target,
                       MeshToMeshTransferOptions options = {});

    /** @brief Construct on a caller-supplied stable validation communicator.
     * @pre All participants supply the same communicator context; choosing
     *      different congruent duplicates on different ranks is invalid.
     *      Each non-null endpoint is a collectively consistent distributed mesh.
     * @note The reference need only live through construction. Both endpoint
     *       communicators must be identical or congruent to this context.
     *       A missing source or target is rejected collectively on this context.
     */
    MeshToMeshTransfer(const typename Pack::comm_type& construction_comm,
                       SP<const mesh_type> source,
                       SP<const mesh_type> target,
                       MeshToMeshTransferOptions options = {});

    /**
     * @brief Transfer authoritative owned components on the original cell maps.
     * @note Does not update ghosts. Source and target may alias. Invalid input
     *       and non-finite output are rejected before writing target values.
     *       Conservative apply uses intensive values and requires full coverage;
     *       partial coverage and inventories require project().
     */
    void apply(const multi_vector_type& source, multi_vector_type& target) const;

    /**
     * @brief Conservative transfer with an explicit input/output representation.
     *
     * Intensive: target = sum(overlap * source) / entire target cell volume.
     * Extensive: target = sum(overlap * source / source cell volume).
     * Uncovered target portions contribute zero; uncovered source inventory is
     * returned in the report for the caller to retain/account for. Neither form
     * divides by covered volume. Requires ConservativeCellAverage. Collective;
     * raw multivectors do not refresh ghosts. Input and output may alias.
     */
    [[nodiscard]] MeshToMeshConservationReport project(
        const multi_vector_type& source, multi_vector_type& target,
        MeshToMeshQuantity quantity) const;

    /** @brief Collective geometry/map validation and borrowed coverage snapshot.
     * @note Requires conservative construction. The reference belongs to this
     *       transfer and represents its construction geometry, not live geometry.
     */
    const MeshToMeshCoverage& coverage() const;

    /** @brief Transfer scalar/vector/tensor cell fields and refresh target ghosts. */
    template<class Value>
    void apply(const FieldStored<Field<Value, CellLocation>, Pack>& source,
               FieldStored<Field<Value, CellLocation>, Pack>& target) const
    {
        validate_meshes(source.mesh(), target.mesh());
        apply(source.owned_data(), target.owned_data());
        target.sync_ghosts();
    }

    /** @brief Conservative field transfer, diagnostics and refreshed target ghosts. */
    template<class Value>
    [[nodiscard]] MeshToMeshConservationReport project(
        const FieldStored<Field<Value, CellLocation>, Pack>& source,
        FieldStored<Field<Value, CellLocation>, Pack>& target,
        MeshToMeshQuantity quantity) const
    {
        validate_meshes(source.mesh(), target.mesh());
        auto report = project(source.owned_data(), target.owned_data(), quantity);
        target.sync_ghosts();
        return report;
    }

private:
    void validate_meshes(const mesh_type& source, const mesh_type& target) const;
    void validate_geometry() const;
    void validate_data(const multi_vector_type& source,
                       const multi_vector_type& target) const;

    SP<const mesh_type> d_source;
    SP<const mesh_type> d_target;
    // Retain original geometry ownership as well as its identity across handle assignment.
    SP<const void> d_source_geometry, d_target_geometry;
    std::uint64_t d_source_epoch;
    std::uint64_t d_target_epoch;
    Teuchos::RCP<const typename Pack::map_type> d_source_map, d_target_map;
    MeshToMeshTransferOptions d_options;
    MeshToMeshCoverage d_coverage;
    // Conservative entries are overlap volumes; interpolation entries are weights.
    Teuchos::RCP<typename Pack::matrix_type> d_weights;
};

extern template class MeshToMeshTransfer<DefaultTpetraTypes>;

} // namespace SimpleFluid
