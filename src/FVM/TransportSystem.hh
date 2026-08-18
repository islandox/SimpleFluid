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

#include "FVM/AssemblyCallbacks.hh"
#include "FVM/BoundaryCache.hh"
#include "FVM/NonOrthogonalTreatment.hh"
#include "FVM/details/FieldStoredTransportSystem.hh"
#include "FVM/details/OperatorDetails.hh"
#include "fields/CellField.hh"
#include "fields/FaceField.hh"
#include "fields/VectorCellField.hh"
#include "geometry/Mesh.hh"

#include <Teuchos_Array.hpp>
#include <Teuchos_RCP.hpp>

#include <cstddef>
#include <functional>
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
template<class Scalar> struct NonOrthogonalTransportWeights
{
    Scalar implicit{};
    Scalar explicit_{};
};

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
 * @return Prepared matrix and whether its existing graph is being reused.
 * @throws std::invalid_argument if @p cached_matrix is incompatible.
 */
template<TpetraTypePack Pack, class MeshType>
PreparedTransportMatrix<Pack> prepare_transport_matrix(
    const MeshType& mesh, Teuchos::RCP<typename Pack::matrix_type> cached_matrix, size_t entries_per_row);

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
 * cache. The referenced mesh topology and geometry must remain unchanged for
 * the cache lifetime; construct a new cache after any mesh revision.
 *
 * @tparam MeshType Mesh interface used by the transport fields.
 */
template<class MeshType> class TransportGeometryCache
{
public:
    using interior_stencils_type = std::vector<detail::LeastSquaresGradientStencil<MeshType>>;
    using boundary_locations_type = std::vector<detail::BoundaryFaceLocation<MeshType>>;
    using boundary_geometry_type = std::vector<detail::BoundaryAwareGradientCellGeometry<MeshType>>;
    explicit TransportGeometryCache(const MeshType& mesh);

    /** @brief Throw if this cache was built for another mesh instance. */
    void require_mesh(const MeshType& mesh) const;

    const interior_stencils_type& interior_stencils() const noexcept;

    const boundary_locations_type& boundary_locations() const noexcept;

    const boundary_geometry_type& boundary_geometry() const noexcept;

    std::vector<detail::AffineLeastSquaresGradientStencil<MeshType>> scalar_affine_stencils(
        std::function<BoundaryCondition(int, size_t)> boundary_condition,
        std::function<typename MeshType::scalar_type(int, size_t)> boundary_value) const;

    std::vector<detail::VectorAffineLeastSquaresGradientStencil<MeshType>> vector_affine_stencils(
        std::function<typename MeshType::Vec3(int, size_t)> boundary_value) const;

private:
    const MeshType* d_mesh;
    interior_stencils_type d_interior_stencils;
    boundary_locations_type d_boundary_locations;
    boundary_geometry_type d_boundary_geometry;
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

/**
 * @brief Assemble scalar transport directly on a mapped FieldStored mesh.
 *
 * This compatibility overload uses the orthogonal two-point diffusion
 * contribution. Use non_orthogonal_transport_system() to select explicit,
 * implicit, or hybrid correction on a semi-structured mesh.
 */
template<TpetraTypePack Pack, class MeshType>
TransportSystem<Pack> transport_system(const ScalarCellFieldStored<Pack, MeshType>& old_values,
    const ScalarFaceFieldStored<Pack, MeshType>& face_fluxes, typename Pack::scalar_type time_step,
    typename Pack::scalar_type diffusivity, ScalarBoundaryConditionProvider<Pack> boundary_condition,
    ScalarBoundaryValueProvider<Pack> boundary_value, ScalarCellValueProvider<Pack> right_hand_source,
    Teuchos::RCP<typename Pack::matrix_type> cached_matrix = Teuchos::null)
{
    return detail::stored_scalar_transport_system<Pack>(old_values, face_fluxes, time_step, diffusivity,
        std::move(boundary_condition), std::move(boundary_value), std::move(right_hand_source),
        std::move(cached_matrix));
}

/** @brief Assemble mapped scalar transport with a zero source. */
template<TpetraTypePack Pack, class MeshType>
TransportSystem<Pack> transport_system(const ScalarCellFieldStored<Pack, MeshType>& old_values,
    const ScalarFaceFieldStored<Pack, MeshType>& face_fluxes, typename Pack::scalar_type time_step,
    typename Pack::scalar_type diffusivity, ScalarBoundaryConditionProvider<Pack> boundary_condition,
    ScalarBoundaryValueProvider<Pack> boundary_value,
    Teuchos::RCP<typename Pack::matrix_type> cached_matrix = Teuchos::null)
{
    auto zero_source = [](typename Pack::local_ordinal_type) { return typename Pack::scalar_type{}; };
    return transport_system<Pack>(old_values, face_fluxes, time_step, diffusivity, std::move(boundary_condition),
        std::move(boundary_value), ScalarCellValueProvider<Pack>{zero_source}, cached_matrix);
}

/**
 * @brief Assemble mapped scalar transport with selectable non-orthogonal diffusion.
 *
 * The transported and optional lagged correction fields remain on their
 * original mapped mesh. Geometry and matrix caches may be reused across
 * steps when they were constructed for the same mesh and operator graph.
 */
template<TpetraTypePack Pack, class MeshType>
TransportSystem<Pack> non_orthogonal_transport_system(const ScalarCellFieldStored<Pack, MeshType>& old_values,
    const ScalarFaceFieldStored<Pack, MeshType>& face_fluxes, typename Pack::scalar_type time_step,
    typename Pack::scalar_type diffusivity, ScalarBoundaryConditionProvider<Pack> boundary_condition,
    ScalarBoundaryValueProvider<Pack> boundary_value, ScalarCellValueProvider<Pack> right_hand_source,
    NonOrthogonalTreatment treatment,
    const std::type_identity_t<ScalarCellFieldStored<Pack, MeshType>>* correction_field = nullptr,
    Teuchos::RCP<typename Pack::matrix_type> cached_matrix = Teuchos::null,
    const std::type_identity_t<TransportGeometryCache<MeshType>>* geometry_cache = nullptr)
{
    return detail::stored_scalar_non_orthogonal_transport_system<Pack>(old_values, face_fluxes, time_step, diffusivity,
        std::move(boundary_condition), std::move(boundary_value), std::move(right_hand_source), treatment,
        correction_field, std::move(cached_matrix), geometry_cache);
}

/**
 * @brief Assemble weighted scalar transport directly on mapped fields.
 *
 * Storage, advection, and diffusion coefficients are cell fields. The
 * overload mirrors the legacy weighted system while retaining the original
 * mapped mesh and FieldStored ownership.
 */
template<TpetraTypePack Pack, class MeshType>
TransportSystem<Pack> weighted_scalar_transport_system(const ScalarCellFieldStored<Pack, MeshType>& old_values,
    const ScalarFaceFieldStored<Pack, MeshType>& face_fluxes, typename Pack::scalar_type time_step,
    const ScalarCellFieldStored<Pack, MeshType>& storage_weight,
    const ScalarCellFieldStored<Pack, MeshType>& advection_weight,
    const ScalarCellFieldStored<Pack, MeshType>& diffusivity, ScalarBoundaryConditionProvider<Pack> boundary_condition,
    ScalarBoundaryValueProvider<Pack> boundary_value, ScalarCellValueProvider<Pack> source,
    NonOrthogonalTreatment treatment,
    const std::type_identity_t<ScalarCellFieldStored<Pack, MeshType>>* correction_field = nullptr,
    Teuchos::RCP<typename Pack::matrix_type> cached_matrix = Teuchos::null,
    std::function<typename Pack::scalar_type(typename Pack::local_ordinal_type)> implicit_sink = {},
    std::function<std::optional<typename Pack::scalar_type>(typename Pack::local_ordinal_type)> fixed_cell_value = {},
    const std::type_identity_t<FieldStoredBoundaryCache<Pack, MeshType>>* boundary_diffusivity = nullptr,
    const std::type_identity_t<TransportGeometryCache<MeshType>>* geometry_cache = nullptr,
    FaceCoefficientInterpolation coefficient_interpolation = FaceCoefficientInterpolation::Harmonic)
{
    return detail::stored_weighted_scalar_transport_system<Pack>(old_values, face_fluxes, time_step, storage_weight,
        advection_weight, diffusivity, std::move(boundary_condition), std::move(boundary_value), std::move(source),
        treatment, correction_field, std::move(cached_matrix), std::move(implicit_sink), std::move(fixed_cell_value),
        boundary_diffusivity, geometry_cache, coefficient_interpolation);
}

/**
 * @brief Assemble conservative physical temperature transport on mapped fields.
 *
 * The transient and advective coefficient is rho*cp, diffusion uses thermal
 * conductivity, and the source is volumetric power density.
 */
template<TpetraTypePack Pack, class MeshType>
TransportSystem<Pack> physical_temperature_transport_system(
    const ScalarCellFieldStored<Pack, MeshType>& old_temperature,
    const ScalarFaceFieldStored<Pack, MeshType>& face_fluxes, typename Pack::scalar_type time_step,
    const ScalarCellFieldStored<Pack, MeshType>& density,
    const ScalarCellFieldStored<Pack, MeshType>& specific_heat_capacity,
    const ScalarCellFieldStored<Pack, MeshType>& thermal_conductivity,
    ScalarBoundaryConditionProvider<Pack> boundary_condition, ScalarBoundaryValueProvider<Pack> boundary_value,
    ScalarCellValueProvider<Pack> power_density, NonOrthogonalTreatment treatment,
    const std::type_identity_t<ScalarCellFieldStored<Pack, MeshType>>* correction_field = nullptr,
    Teuchos::RCP<typename Pack::matrix_type> cached_matrix = Teuchos::null,
    const std::type_identity_t<FieldStoredBoundaryCache<Pack, MeshType>>* boundary_thermal_conductivity = nullptr,
    const std::type_identity_t<TransportGeometryCache<MeshType>>* geometry_cache = nullptr,
    FaceCoefficientInterpolation coefficient_interpolation = FaceCoefficientInterpolation::Harmonic)
{
    return detail::stored_physical_temperature_transport_system<Pack>(old_temperature, face_fluxes, time_step, density,
        specific_heat_capacity, thermal_conductivity, std::move(boundary_condition), std::move(boundary_value),
        std::move(power_density), treatment, correction_field, std::move(cached_matrix), boundary_thermal_conductivity,
        geometry_cache, coefficient_interpolation);
}

/**
 * @brief Assemble mapped vector transport with selectable treatment.
 *
 * Explicit, implicit, and hybrid non-orthogonal treatments use the same
 * mapped reconstruction stencils as the legacy mesh path.
 */
template<TpetraTypePack Pack, class MeshType>
VectorTransportSystem<Pack> non_orthogonal_transport_system(const VectorCellFieldStored<Pack, MeshType>& old_values,
    const ScalarFaceFieldStored<Pack, MeshType>& face_fluxes, typename Pack::scalar_type time_step,
    typename Pack::scalar_type diffusivity, VectorBoundaryValueProvider<Pack> boundary_value,
    VectorCellValueProvider<Pack> right_hand_source, NonOrthogonalTreatment treatment,
    const std::type_identity_t<VectorCellFieldStored<Pack, MeshType>>* correction_field = nullptr,
    Teuchos::RCP<typename Pack::matrix_type> cached_matrix = Teuchos::null,
    BoundaryFaceSelector boundary_diffusion = detail::AlwaysDiffuseBoundary{},
    const std::type_identity_t<TransportGeometryCache<MeshType>>* geometry_cache = nullptr)
{
    return detail::stored_vector_transport_system<Pack>(old_values, face_fluxes, time_step, diffusivity,
        std::move(boundary_value), std::move(right_hand_source), treatment, correction_field, std::move(cached_matrix),
        std::move(boundary_diffusion), geometry_cache);
}

/**
 * @brief Assemble mapped momentum transport with variable viscosity.
 *
 * The mapped path includes variable-coefficient orthogonal diffusion,
 * explicit/implicit non-orthogonal corrections, and the deviatoric
 * transpose-gradient stress used by physical momentum transport.
 */
template<TpetraTypePack Pack, class MeshType>
VectorTransportSystem<Pack> physical_momentum_transport_system(
    const VectorCellFieldStored<Pack, MeshType>& old_velocity, const ScalarFaceFieldStored<Pack, MeshType>& face_fluxes,
    typename Pack::scalar_type time_step, const ScalarCellFieldStored<Pack, MeshType>& dynamic_viscosity,
    typename Pack::scalar_type reference_density, VectorBoundaryValueProvider<Pack> boundary_value,
    VectorCellValueProvider<Pack> acceleration_source, NonOrthogonalTreatment treatment,
    const std::type_identity_t<VectorCellFieldStored<Pack, MeshType>>* correction_field = nullptr,
    Teuchos::RCP<typename Pack::matrix_type> cached_matrix = Teuchos::null,
    BoundaryFaceSelector boundary_diffusion = detail::AlwaysDiffuseBoundary{},
    const std::type_identity_t<FieldStoredBoundaryCache<Pack, MeshType>>* boundary_dynamic_viscosity = nullptr,
    const std::type_identity_t<TransportGeometryCache<MeshType>>* geometry_cache = nullptr,
    FaceCoefficientInterpolation coefficient_interpolation = FaceCoefficientInterpolation::Harmonic)
{
    return detail::stored_physical_momentum_transport_system<Pack>(old_velocity, face_fluxes, time_step,
        dynamic_viscosity, reference_density, std::move(boundary_value), std::move(acceleration_source), treatment,
        correction_field, std::move(cached_matrix), std::move(boundary_diffusion), boundary_dynamic_viscosity,
        geometry_cache, coefficient_interpolation);
}

/**
 * @brief Assemble the semi-implicit transport system (unsteady
 *        advection-diffusion) for a scalar field.
 *
 * The assembly uses first-order upwinding for advection and a two-point
 * flux approximation for diffusion.  Boundary values and right-hand source
 * values are supplied lazily via callables.  The source term is interpreted
 * as a per-unit-volume quantity and contributes @f$V_i s_i@f$ to the RHS.
 *
 * @tparam Pack The Tpetra type pack.
 * @param old_values Previous time-step scalar field. Its overlap values
 *        must be synchronized before assembly.
 * @param face_fluxes Pre-computed face volumetric fluxes.
 * @param time_step Time-step size (must be positive).
 * @param diffusivity Constant scalar diffusivity (non-negative).
 * @param boundary_condition Callable that returns the boundary-condition type
 *        and flux value for a face.
 * @param boundary_value Callable that returns the prescribed boundary value
 *        used by Dirichlet diffusion and inflow advection.
 * @param right_hand_source Callable that returns the per-unit-volume source
 *        term for an owned cell.
 * @param[in,out] cached_matrix Optional fill-complete matrix whose graph and
 *        storage are reused when its maps match the mesh.
 * @return TransportSystem containing the assembled matrix and RHS vector.
 * @throws std::invalid_argument if @p face_fluxes is on a different mesh,
 *         @p time_step <= 0, or @p diffusivity < 0.
 * @throws std::runtime_error if any interior face connects coincident
 *         cell centers.
 */
template<TpetraTypePack Pack>
TransportSystem<Pack> transport_system(const CellField<Pack>& old_values, const FaceField<Pack>& face_fluxes,
    typename Pack::scalar_type time_step, typename Pack::scalar_type diffusivity,
    ScalarBoundaryConditionProvider<Pack> boundary_condition, ScalarBoundaryValueProvider<Pack> boundary_value,
    ScalarCellValueProvider<Pack> right_hand_source,
    Teuchos::RCP<typename Pack::matrix_type> cached_matrix = Teuchos::null);

/**
 * @brief Assemble scalar transport using value-only boundary data.
 *
 * This compatibility overload treats every boundary value as Dirichlet,
 * matching the historical transport_system() behavior.
 */
template<TpetraTypePack Pack>
TransportSystem<Pack> transport_system(const CellField<Pack>& old_values, const FaceField<Pack>& face_fluxes,
    typename Pack::scalar_type time_step, typename Pack::scalar_type diffusivity,
    ScalarBoundaryValueProvider<Pack> boundary_value, ScalarCellValueProvider<Pack> right_hand_source,
    Teuchos::RCP<typename Pack::matrix_type> cached_matrix = Teuchos::null);

/**
 * @brief Assemble scalar transport with boundary-condition-aware diffusion
 *        and zero explicit source.
 */
template<TpetraTypePack Pack>
TransportSystem<Pack> transport_system(const CellField<Pack>& old_values, const FaceField<Pack>& face_fluxes,
    typename Pack::scalar_type time_step, typename Pack::scalar_type diffusivity,
    ScalarBoundaryConditionProvider<Pack> boundary_condition, ScalarBoundaryValueProvider<Pack> boundary_value,
    Teuchos::RCP<typename Pack::matrix_type> cached_matrix = Teuchos::null);

/**
 * @brief Assemble the scalar semi-implicit transport system with no
 *        explicit right-hand source term.
 *
 * This overload preserves the existing call pattern and delegates to the
 * source-aware transport assembly with a zero source.
 *
 * @tparam Pack Tpetra type pack.
 * @param old_values Previous time-step scalar field.
 * @param face_fluxes Pre-computed face volumetric fluxes.
 * @param time_step Time-step size (must be positive).
 * @param diffusivity Constant scalar diffusivity (non-negative).
 * @param boundary_value Callable that returns the prescribed boundary
 *        value for a face.
 * @param cached_matrix Optional fill-complete matrix cache.
 * @return TransportSystem containing the assembled matrix and RHS vector.
 */
template<TpetraTypePack Pack>
TransportSystem<Pack> transport_system(const CellField<Pack>& old_values, const FaceField<Pack>& face_fluxes,
    typename Pack::scalar_type time_step, typename Pack::scalar_type diffusivity,
    ScalarBoundaryValueProvider<Pack> boundary_value,
    Teuchos::RCP<typename Pack::matrix_type> cached_matrix = Teuchos::null);

/**
 * @brief Assemble legacy scalar transport with selectable non-orthogonal diffusion.
 *
 * This constant-coefficient overload preserves the scalar transport contract
 * exposed by the mapped-field specialization while retaining legacy field
 * storage.  It delegates to the weighted scalar kernel with unit storage and
 * advection weights.
 */
template<TpetraTypePack Pack>
TransportSystem<Pack> non_orthogonal_transport_system(const CellField<Pack>& old_values,
    const FaceField<Pack>& face_fluxes, typename Pack::scalar_type time_step, typename Pack::scalar_type diffusivity,
    ScalarBoundaryConditionProvider<Pack> boundary_condition, ScalarBoundaryValueProvider<Pack> boundary_value,
    ScalarCellValueProvider<Pack> right_hand_source, NonOrthogonalTreatment treatment,
    const CellField<Pack>* correction_field = nullptr,
    Teuchos::RCP<typename Pack::matrix_type> cached_matrix = Teuchos::null,
    const TransportGeometryCache<Mesh<Pack>>* geometry_cache = nullptr);

/**
 * @brief Assemble the semi-implicit transport system for a 3-component
 *        vector field using one shared transport matrix.
 *
 * The transport operator is identical for all components, while transient,
 * boundary, and source terms live in separate RHS columns.  The source term
 * is interpreted as a per-unit-volume vector and contributes
 * @f$V_i \mathbf{s}_i@f$ to the RHS.
 *
 * @tparam Pack The Tpetra type pack.
 * @param old_values Previous time-step vector field. Its overlap values
 *        must be synchronized before assembly.
 * @param face_fluxes Pre-computed face volumetric fluxes.
 * @param time_step Time-step size (must be positive).
 * @param diffusivity Constant scalar diffusivity (non-negative).
 * @param boundary_condition Callable that returns the boundary-condition type
 *        and vector flux value for a face.
 * @param boundary_value Callable that returns the prescribed boundary vector
 *        used by Dirichlet diffusion and inflow advection.
 * @param right_hand_source Callable that returns the per-unit-volume source
 *        vector for an owned cell.
 * @param[in,out] cached_matrix Optional fill-complete matrix whose graph and
 *        storage are reused when its maps match the mesh.
 * @return VectorTransportSystem containing the assembled matrix and
 *         three-column RHS.
 * @throws std::invalid_argument if @p face_fluxes is on a different mesh,
 *         @p time_step <= 0, or @p diffusivity < 0.
 * @throws std::runtime_error if any interior face connects coincident
 *         cell centers.
 */
template<TpetraTypePack Pack>
VectorTransportSystem<Pack> transport_system(const VectorCellField<Pack>& old_values,
    const FaceField<Pack>& face_fluxes, typename Pack::scalar_type time_step, typename Pack::scalar_type diffusivity,
    VectorBoundaryConditionProvider<Pack> boundary_condition, VectorBoundaryValueProvider<Pack> boundary_value,
    VectorCellValueProvider<Pack> right_hand_source,
    Teuchos::RCP<typename Pack::matrix_type> cached_matrix = Teuchos::null);

/**
 * @brief Assemble vector transport using value-only boundary data.
 *
 * This compatibility overload treats every boundary value as Dirichlet,
 * matching the historical transport_system() behavior.
 */
template<TpetraTypePack Pack>
VectorTransportSystem<Pack> transport_system(const VectorCellField<Pack>& old_values,
    const FaceField<Pack>& face_fluxes, typename Pack::scalar_type time_step, typename Pack::scalar_type diffusivity,
    VectorBoundaryValueProvider<Pack> boundary_value, VectorCellValueProvider<Pack> right_hand_source,
    Teuchos::RCP<typename Pack::matrix_type> cached_matrix = Teuchos::null);

/**
 * @brief Assemble vector transport with boundary-condition-aware diffusion
 *        and zero explicit source.
 */
template<TpetraTypePack Pack>
VectorTransportSystem<Pack> transport_system(const VectorCellField<Pack>& old_values,
    const FaceField<Pack>& face_fluxes, typename Pack::scalar_type time_step, typename Pack::scalar_type diffusivity,
    VectorBoundaryConditionProvider<Pack> boundary_condition, VectorBoundaryValueProvider<Pack> boundary_value,
    Teuchos::RCP<typename Pack::matrix_type> cached_matrix = Teuchos::null);

/**
 * @brief Assemble vector transport with selectable non-orthogonal viscous
 *        diffusion treatment.
 *
 * The transient and first-order upwind advection terms are assembled as in
 * transport_system(). Viscous diffusion uses the Phase 1 orthogonal
 * two-point stencil plus Phase 2/3 non-orthogonal treatment selected by
 * @p treatment.
 *
 * @param face_fluxes Oriented volumetric fluxes on the @p old_values mesh.
 * @param time_step Must be positive.
 * @param diffusivity Must be non-negative.
 * @param right_hand_source Volumetric vector-source provider.
 * @param correction_field Optional lagged field for explicit correction terms.
 * @param[in,out] cached_matrix Optional compatible matrix to reuse.
 * @param boundary_diffusion Selects boundary faces with viscous diffusion.
 * @param geometry_cache Optional mesh-bound reconstruction geometry cache.
 * @throws std::invalid_argument If fields use different meshes, the time step
 *         is not positive, diffusivity is negative, or ranks disagree on the
 *         correction-field or non-orthogonal-treatment selection.
 */
template<TpetraTypePack Pack>
VectorTransportSystem<Pack> non_orthogonal_transport_system(const VectorCellField<Pack>& old_values,
    const FaceField<Pack>& face_fluxes, typename Pack::scalar_type time_step, typename Pack::scalar_type diffusivity,
    VectorBoundaryValueProvider<Pack> boundary_value, VectorCellValueProvider<Pack> right_hand_source,
    NonOrthogonalTreatment treatment, const VectorCellField<Pack>* correction_field = nullptr,
    Teuchos::RCP<typename Pack::matrix_type> cached_matrix = Teuchos::null,
    BoundaryFaceSelector boundary_diffusion = detail::AlwaysDiffuseBoundary{},
    const TransportGeometryCache<Mesh<Pack>>* geometry_cache = nullptr);

/**
 * @brief Assemble conservative scalar transport with independent storage,
 *        advection, and diffusion weights.
 *
 * The solved variable is phi:
 *
 *   d(storage_weight * phi)/dt
 * + div(face_flux * advection_weight * phi)
 * = div(diffusivity * grad(phi)) + source - implicit_sink * phi.
 *
 * Boundary diffusion honors the supplied boundary-condition type, while
 * advection remains first-order upwind and outflow conservative.
 *
 * @param face_fluxes Oriented volumetric fluxes on the @p old_values mesh.
 * @param time_step Must be positive.
 * @param storage_weight Must be positive and share the transported-field mesh.
 * @param advection_weight Must be non-negative and share that mesh.
 * @param correction_field Optional lagged field for explicit correction terms.
 * @param[in,out] cached_matrix Optional compatible matrix to reuse.
 * @param implicit_sink Optional finite, non-negative cell-centered sink.
 * @param fixed_cell_value Optional finite values imposed as exact identity rows.
 * @param boundary_diffusivity Optional compatible boundary-face coefficients.
 * @param geometry_cache Optional mesh-bound reconstruction geometry cache.
 * @throws std::invalid_argument If field/cache meshes are incompatible,
 *         ranks disagree on correction-field or treatment selection, or a
 *         validated time step, coefficient, sink, or fixed value is invalid.
 * @throws std::runtime_error For a Robin boundary condition, which is not yet
 *         implemented by this assembly path.
 */
template<TpetraTypePack Pack>
TransportSystem<Pack> weighted_scalar_transport_system(const CellField<Pack>& old_values,
    const FaceField<Pack>& face_fluxes, typename Pack::scalar_type time_step, const CellField<Pack>& storage_weight,
    const CellField<Pack>& advection_weight, const CellField<Pack>& diffusivity,
    ScalarBoundaryConditionProvider<Pack> boundary_condition, ScalarBoundaryValueProvider<Pack> boundary_value,
    ScalarCellValueProvider<Pack> source, NonOrthogonalTreatment treatment,
    const CellField<Pack>* correction_field = nullptr,
    Teuchos::RCP<typename Pack::matrix_type> cached_matrix = Teuchos::null,
    std::function<typename Pack::scalar_type(typename Pack::local_ordinal_type)> implicit_sink = {},
    std::function<std::optional<typename Pack::scalar_type>(typename Pack::local_ordinal_type)> fixed_cell_value = {},
    const BoundaryCache<Pack>* boundary_diffusivity = nullptr,
    const TransportGeometryCache<Mesh<Pack>>* geometry_cache = nullptr,
    FaceCoefficientInterpolation coefficient_interpolation = FaceCoefficientInterpolation::Harmonic);

/**
 * @brief Assemble conservative temperature transport with physical material
 *        fields and a volumetric heat source.
 *
 * Storage and advection use volumetric heat capacity rho*cp, diffusion uses
 * distance-weighted harmonic conductivity, and the source is power density.
 *
 * @param face_fluxes Oriented volumetric fluxes on the temperature mesh.
 * @param time_step Must be positive.
 * @param correction_field Optional lagged temperature for explicit correction.
 * @param[in,out] cached_matrix Optional compatible matrix to reuse.
 * @param boundary_thermal_conductivity Optional compatible boundary-face values.
 * @param geometry_cache Optional mesh-bound reconstruction geometry cache.
 * @throws std::invalid_argument If field/cache meshes are incompatible,
 *         ranks disagree on correction-field or treatment selection, or the
 *         time step is not positive.
 * @throws std::runtime_error For a Robin boundary condition, which is not yet
 *         implemented by this assembly path.
 */
template<TpetraTypePack Pack>
TransportSystem<Pack> physical_temperature_transport_system(const CellField<Pack>& old_temperature,
    const FaceField<Pack>& face_fluxes, typename Pack::scalar_type time_step, const CellField<Pack>& density,
    const CellField<Pack>& specific_heat_capacity, const CellField<Pack>& thermal_conductivity,
    ScalarBoundaryConditionProvider<Pack> boundary_condition, ScalarBoundaryValueProvider<Pack> boundary_value,
    ScalarCellValueProvider<Pack> power_density, NonOrthogonalTreatment treatment,
    const CellField<Pack>* correction_field = nullptr,
    Teuchos::RCP<typename Pack::matrix_type> cached_matrix = Teuchos::null,
    const BoundaryCache<Pack>* boundary_thermal_conductivity = nullptr,
    const TransportGeometryCache<Mesh<Pack>>* geometry_cache = nullptr,
    FaceCoefficientInterpolation coefficient_interpolation = FaceCoefficientInterpolation::Harmonic);

/**
 * @brief Assemble incompressible momentum transport with a variable dynamic
 *        viscosity field and constant reference density.
 * @param face_fluxes Oriented volumetric fluxes on the velocity mesh.
 * @param time_step Must be positive.
 * @param reference_density Must be positive.
 * @param acceleration_source Volumetric acceleration provider.
 * @param correction_field Optional lagged velocity for explicit correction.
 * @param[in,out] cached_matrix Optional compatible matrix to reuse.
 * @param boundary_diffusion Selects boundary faces with viscous diffusion.
 * @param boundary_dynamic_viscosity Optional compatible boundary-face values.
 * @param geometry_cache Optional mesh-bound reconstruction geometry cache.
 * @throws std::invalid_argument If field/cache meshes are incompatible,
 *         ranks disagree on correction-field or treatment selection, or the
 *         time step or reference density is not positive.
 */
template<TpetraTypePack Pack>
VectorTransportSystem<Pack> physical_momentum_transport_system(const VectorCellField<Pack>& old_velocity,
    const FaceField<Pack>& face_fluxes, typename Pack::scalar_type time_step, const CellField<Pack>& dynamic_viscosity,
    typename Pack::scalar_type reference_density, VectorBoundaryValueProvider<Pack> boundary_value,
    VectorCellValueProvider<Pack> acceleration_source, NonOrthogonalTreatment treatment,
    const VectorCellField<Pack>* correction_field = nullptr,
    Teuchos::RCP<typename Pack::matrix_type> cached_matrix = Teuchos::null,
    BoundaryFaceSelector boundary_diffusion = detail::AlwaysDiffuseBoundary{},
    const BoundaryCache<Pack>* boundary_dynamic_viscosity = nullptr,
    const TransportGeometryCache<Mesh<Pack>>* geometry_cache = nullptr,
    FaceCoefficientInterpolation coefficient_interpolation = FaceCoefficientInterpolation::Harmonic);

/**
 * @brief Assemble the vector semi-implicit transport system with no
 *        explicit right-hand source term.
 *
 * This overload preserves the existing call pattern and delegates to the
 * source-aware transport assembly with a zero vector source.
 */
template<TpetraTypePack Pack>
VectorTransportSystem<Pack> transport_system(const VectorCellField<Pack>& old_values,
    const FaceField<Pack>& face_fluxes, typename Pack::scalar_type time_step, typename Pack::scalar_type diffusivity,
    VectorBoundaryValueProvider<Pack> boundary_value,
    Teuchos::RCP<typename Pack::matrix_type> cached_matrix = Teuchos::null);

} // namespace SimpleFluid::FVM

#include "FVM/details/TransportGeometryCache.hh"
