/**
 * @file FVM/TransportSystem.hh
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Semi-implicit finite-volume transport-system assembly.
 * @version 0.1
 * @date 2026-05-30
 *
 * @copyright Copyright (c) 2026
 *
 */
#pragma once

#include "FVM/ALEControlVolumeState.hh"
#include "FVM/AssemblyCallbacks.hh"
#include "FVM/BoundaryCache.hh"
#include "FVM/NonOrthogonalTreatment.hh"
#include "FVM/ScalarTransportDiscretization.hh"
#include "FVM/details/FieldStoredTransportSystem.hh"
#include "FVM/details/TransportAssemblyGeometry.hh"
#include "FVM/details/TransportValidation.hh"
#include "FVM/details/OperatorDetails.hh"
#include "SimpleFluidExport.hh"
#include "fields/CellField.hh"
#include "fields/FaceField.hh"
#include "fields/MeshFieldTraits.hh"
#include "fields/VectorCellField.hh"
#include "geometry/GeometryEpoch.hh"
#include "geometry/Mesh.hh"
#include "geometry/MeshTraverse.hh"

#include <Teuchos_Array.hpp>
#include <Teuchos_RCP.hpp>

#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace SimpleFluid::FVM
{
namespace detail
{

/**
 * @brief Matrix prepared for fresh insertion or cached-graph reuse.
 * @tparam Pack Tpetra type pack providing the matrix type.
 */
template<TpetraTypePack Pack> struct PreparedTransportMatrix
{
    Teuchos::RCP<typename Pack::matrix_type> matrix;
    bool reused = false;
};

/**
 * @brief Validated matrix/RHS fractions for a non-orthogonal treatment.
 * @tparam Scalar Scalar type used by the transport system.
 */
template<class Scalar> using NonOrthogonalTransportWeights = NonOrthogonalWeights<Scalar>;

/**
 * @brief Collectively validate the lagged-field and treatment selection.
 *
 * Partition-gradient reconstruction synchronizes a gradient field. Every
 * rank must therefore differentiate the same category of input field and
 * enter the same explicit, implicit, or hybrid branch. The field-state code
 * also folds mesh validation into the same packed reduction so a foreign
 * field on one rank cannot make only that rank throw before another enters
 * a gradient synchronization.
 *
 * @tparam Pack Tpetra type pack defining the communicator and scalar type.
 * @tparam Field Transported cell-field type.
 * @param mesh Mesh shared by the transported and correction fields.
 * @param treatment Requested non-orthogonal treatment.
 * @param correction_field Optional lagged correction field.
 * @param context Assembly routine name used in diagnostics.
 * @return Validated implicit and explicit treatment fractions.
 * @throws std::invalid_argument collectively if the treatment is invalid,
 *         ranks disagree on the treatment or correction-field selection, or
 *         a supplied correction field belongs to another mesh.
 */
template<TpetraTypePack Pack, class Field>
NonOrthogonalTransportWeights<typename Pack::scalar_type> validate_non_orthogonal_transport_selection(
    const Mesh<Pack>& mesh, NonOrthogonalTreatment treatment, const Field* correction_field, std::string_view context);

/**
 * @brief Allocate a transport matrix or reset a compatible cached matrix.
 * @tparam Pack Tpetra type pack providing the matrix type.
 * @tparam MeshType Mesh interface type.
 * @param mesh Mesh defining row, column, and domain maps.
 * @param cached_matrix Optional fill-complete matrix to reuse.
 * @param entries_per_row Initial graph allocation estimate.
 * @param required_rows Optional operator rows to check before mutating a cached graph.
 * @return Prepared matrix and whether its existing graph is being reused.
 * @throws std::invalid_argument if @p cached_matrix is incompatible.
 */
template<TpetraTypePack Pack, class MeshType>
PreparedTransportMatrix<Pack> prepare_transport_matrix(
    const MeshType& mesh, Teuchos::RCP<typename Pack::matrix_type> cached_matrix, size_t entries_per_row,
    const std::vector<TransportMatrixRow<Pack>>* required_rows = nullptr);

/**
 * @brief Insert a row on a fresh graph or sum values into a reused graph.
 * @tparam Pack Tpetra type pack providing matrix scalar and ordinal types.
 * @param prepared Matrix and graph-reuse state.
 * @param row Local matrix row.
 * @param columns Local column identifiers.
 * @param values Coefficients corresponding to @p columns.
 * @throws std::invalid_argument if a reused graph lacks an entry.
 */
template<TpetraTypePack Pack>
void add_transport_values(const PreparedTransportMatrix<Pack>& prepared, typename Pack::local_ordinal_type row,
    const Teuchos::ArrayView<const typename Pack::local_ordinal_type>& columns,
    const Teuchos::ArrayView<const typename Pack::scalar_type>& values);

/**
 * @brief Insert or update one accumulated flat row.
 * @tparam Pack Tpetra type pack providing matrix scalar and ordinal types.
 * @param prepared Matrix and graph-reuse state.
 * @param row Local matrix row.
 * @param row_values Unique columns and accumulated coefficients.
 */
template<TpetraTypePack Pack>
void add_transport_values(const PreparedTransportMatrix<Pack>& prepared, typename Pack::local_ordinal_type row,
    const FlatMatrixRow<typename Pack::local_ordinal_type, typename Pack::scalar_type>& row_values);

} // namespace detail

/**
 * @brief Holds the assembled left-hand-side matrix and right-hand-side
 *        vector for a semi-implicit transport step.
 *
 * @tparam Pack The Tpetra type pack.
 */
template<TpetraTypePack Pack> struct TransportSystem
{
    Teuchos::RCP<typename Pack::matrix_type> matrix;
    Teuchos::RCP<typename Pack::vector_type> rhs;
};

/**
 * @brief Holds the assembled left-hand-side matrix and three-component
 *        right-hand side for a vector-valued semi-implicit transport step.
 *
 * @tparam Pack The Tpetra type pack.
 */
template<TpetraTypePack Pack> struct VectorTransportSystem
{
    Teuchos::RCP<typename Pack::matrix_type> matrix;
    Teuchos::RCP<typename Pack::multi_vector_type> rhs;
};

/**
 * @brief Mesh-bound cache for transport reconstruction geometry.
 *
 * The cache owns only topology- and geometry-dependent data. Boundary
 * condition types and values remain callback-driven and are materialized for
 * each assembly, so changing boundary data does not require rebuilding this
 * cache. Fixed-topology geometry motion invalidates the cache until refresh()
 * rebuilds its numeric data while retaining the gradient traversal plan;
 * topology changes require a new cache.
 *
 * @tparam MeshType Mesh interface used by the transport fields.
 */
template<class MeshType> class SIMPLEFLUID_FVM_EXPORT TransportGeometryCache
{
public:
    using interior_stencils_type = std::vector<detail::LeastSquaresGradientStencil<MeshType>>;
    using boundary_locations_type = std::vector<detail::BoundaryFaceLocation<MeshType>>;
    using boundary_geometry_type = std::vector<detail::BoundaryAwareGradientCellGeometry<MeshType>>;
    using assembly_geometry_type = detail::TransportAssemblyGeometry<MeshType>;
    using traverse_plan_type = MeshTraversePlan<MeshType>;
    /** Immutable mesh metrics; field and boundary values are never retained. */
    struct SharedGeometry
    {
        explicit SharedGeometry(const MeshType& mesh);
        SharedGeometry(const MeshType& mesh, std::shared_ptr<const traverse_plan_type> traverse);
        SharedGeometry(const MeshType& mesh, std::shared_ptr<const traverse_plan_type> traverse,
            const assembly_geometry_type* previous_assembly);
        const MeshType* const mesh;
        const void* const identity;
        const std::uint64_t epoch;
        /** Ordered gradient topology retained across geometry-only refreshes. */
        const std::shared_ptr<const traverse_plan_type> gradient_traverse;
        const interior_stencils_type interior;
        const boundary_locations_type locations;
        const boundary_geometry_type boundary;
        const assembly_geometry_type assembly;
        const bool has_non_orthogonal_faces;
    };
    using shared_geometry_type = std::shared_ptr<const SharedGeometry>;

    explicit TransportGeometryCache(const MeshType& mesh);

    /** @brief Throw if this cache was built for another mesh instance. */
    void require_mesh(const MeshType& mesh) const;

    /** Refresh once per epoch, retaining the compiled no-argument entry point. */
    void refresh();
    /** Refresh once per epoch, or adopt a validated snapshot of the same mesh. */
    void refresh(shared_geometry_type geometry);
    const shared_geometry_type& shared_geometry() const noexcept { return d_geometry; }
    const traverse_plan_type& traverse_plan() const noexcept { return *d_geometry->gradient_traverse; }

    /** @brief Geometry epoch represented by this cache. */
    std::uint64_t geometry_epoch() const noexcept { return d_geometry->epoch; }

    const interior_stencils_type& interior_stencils() const;

    const boundary_locations_type& boundary_locations() const;

    const boundary_geometry_type& boundary_geometry() const;

    /** @brief Ordered face topology, owned volumes and physical boundary mask. */
    const assembly_geometry_type& assembly_geometry() const;

    /** @brief Whether any locally owned cell has a non-orthogonal transport face. */
    bool has_non_orthogonal_faces() const;

    std::vector<detail::AffineLeastSquaresGradientStencil<MeshType>> scalar_affine_stencils(
        std::function<BoundaryCondition(int, size_t)> boundary_condition,
        std::function<typename MeshType::scalar_type(int, size_t)> boundary_value) const;

    std::vector<detail::VectorAffineLeastSquaresGradientStencil<MeshType>> vector_affine_stencils(
        std::function<typename MeshType::Vec3(int, size_t)> boundary_value) const;

private:
    const MeshType* d_mesh;
    shared_geometry_type d_geometry;
};

/**
 * @brief Explicit instantiations for common transport geometry caches.
 *
 * Generic definitions are header-visible so static mapped meshes can use the
 * cache directly. These declarations retain compiled instantiations for the
 * default legacy and runtime mesh types.
 */
extern template class TransportGeometryCache<Mesh<DefaultTpetraTypes>>;
extern template class TransportGeometryCache<MeshHandle<DefaultTpetraTypes>>;

} // namespace SimpleFluid::FVM

#include "FVM/details/TransportGeometryCache.hh"
