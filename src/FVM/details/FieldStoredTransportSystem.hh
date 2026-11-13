/**
 * @file FVM/details/FieldStoredTransportSystem.hh
 * @brief Transport assembly for mapped meshes and FieldStored.
 */

#pragma once

#include "SimpleFluidExport.hh"

#include "FVM/ALEControlVolumeState.hh"
#include "FVM/AssemblyCallbacks.hh"
#include "FVM/BoundaryCache.hh"
#include "FVM/NonOrthogonalTreatment.hh"
#include "FVM/ScalarTransportDiscretization.hh"
#include "FVM/details/NonOrthogonalSelection.hh"
#include "FVM/details/OperatorDetails.hh"
#include "FVM/details/TransportCoefficientAccess.hh"
#include "FVM/details/TransportValidation.hh"
#include "fields/FieldStored.hh"

#include <Teuchos_Array.hpp>
#include <Teuchos_CommHelpers.hpp>
#include <Teuchos_RCP.hpp>

#include <cstddef>
#include <optional>
#include <string_view>
#include <utility>
#include <vector>

namespace SimpleFluid::FVM
{

template<TpetraTypePack Pack> struct TransportSystem;

template<TpetraTypePack Pack> struct VectorTransportSystem;

namespace detail
{

template<class MeshType> struct TransportAssemblyGeometry;

template<TpetraTypePack Pack> struct StoredPreparedTransportMatrix
{
    Teuchos::RCP<typename Pack::matrix_type> matrix;
    bool reused = false;
    bool symbolic_reuse = false;
};

template<TpetraTypePack Pack> using StoredTransportMatrixRow = TransportMatrixRow<Pack>;

/**
 * @brief Row slots for a matrix whose graph was created and frozen by assembly.
 *
 * Only newly assembled matrices enter this plan. Their Tpetra static graph
 * forbids insertion/removal through the matrix API. External matrices always
 * retain the full collective validation path. Required operator columns and
 * map identities are checked on every reuse; numeric values are never cached.
 */
template<TpetraTypePack Pack> class StoredTransportSymbolicPlan
{
public:
    using matrix_type = typename Pack::matrix_type;
    using row_type = StoredTransportMatrixRow<Pack>;

    template<class MeshType>
    SIMPLEFLUID_LOCAL bool matches(
        const MeshType& mesh, const Teuchos::RCP<matrix_type>& matrix, const std::vector<row_type>& rows) const;

    SIMPLEFLUID_LOCAL bool owns(const Teuchos::RCP<matrix_type>& matrix) const noexcept
    {
        return !matrix.is_null() && matrix.get() == d_matrix.get() && matrix->isStaticGraph();
    }

private:
    template<TpetraTypePack OtherPack, class MeshType>
    friend Teuchos::RCP<typename OtherPack::matrix_type> finish_stored_transport_matrix(const MeshType&,
        Teuchos::RCP<typename OtherPack::matrix_type>, size_t, const std::vector<StoredTransportMatrixRow<OtherPack>>&,
        StoredTransportSymbolicPlan<OtherPack>*);

    template<class MeshType>
    SIMPLEFLUID_LOCAL void record(const MeshType& mesh, Teuchos::RCP<matrix_type> matrix, const std::vector<row_type>& rows);

    SIMPLEFLUID_LOCAL void refresh_values(const std::vector<row_type>& rows) const;

    Teuchos::RCP<matrix_type> d_matrix;
    Teuchos::RCP<const typename matrix_type::crs_graph_type> d_graph;
    const void* d_mesh = nullptr;
    std::vector<std::vector<typename Pack::local_ordinal_type>> d_columns;
    std::vector<std::vector<size_t>> d_slots;
};

template<class Scalar> using StoredNonOrthogonalWeights = NonOrthogonalWeights<Scalar>;

/** Collectively validate mapped treatment and correction-field selection. */
template<TpetraTypePack Pack, class MeshType, class Field>
SIMPLEFLUID_LOCAL StoredNonOrthogonalWeights<typename Pack::scalar_type> validate_stored_non_orthogonal_selection(
    const MeshType& mesh, NonOrthogonalTreatment treatment, const Field* correction_field, std::string_view context);

/** Allocate a fresh mapped matrix or reset a collectively validated cached graph. */
template<TpetraTypePack Pack, class MeshType>
SIMPLEFLUID_LOCAL StoredPreparedTransportMatrix<Pack> prepare_stored_transport_matrix(const MeshType& mesh,
    Teuchos::RCP<typename Pack::matrix_type> cached_matrix, size_t entries_per_row,
    const std::vector<StoredTransportMatrixRow<Pack>>& rows,
    StoredTransportSymbolicPlan<Pack>* symbolic_plan = nullptr);

/** Insert an accumulated row, validating every slot of a reused graph. */
template<TpetraTypePack Pack>
SIMPLEFLUID_LOCAL void add_stored_transport_values(const StoredPreparedTransportMatrix<Pack>& prepared,
    typename Pack::local_ordinal_type row, const StoredTransportMatrixRow<Pack>& row_values);

/** Complete assembly, reusing validated immutable row slots when available. */
template<TpetraTypePack Pack>
SIMPLEFLUID_LOCAL void complete_stored_transport_matrix(const StoredPreparedTransportMatrix<Pack>& prepared);

template<TpetraTypePack Pack, class MeshType>
SIMPLEFLUID_LOCAL Teuchos::RCP<typename Pack::matrix_type> finish_stored_transport_matrix(const MeshType& mesh,
    Teuchos::RCP<typename Pack::matrix_type> cached_matrix, size_t entries_per_row,
    const std::vector<StoredTransportMatrixRow<Pack>>& rows, StoredTransportSymbolicPlan<Pack>* symbolic_plan);

/** Materialize a stored vector/tensor gradient from generic stencils. */
template<TpetraTypePack Pack, class MeshType, class Stencils>
SIMPLEFLUID_LOCAL void evaluate_stored_vector_gradients(const VectorCellFieldStored<Pack, MeshType>& field, const Stencils& stencils,
    TensorCellFieldStored<Pack, MeshType>& gradients);

/** Materialize a stored scalar gradient from affine reconstruction stencils. */
template<TpetraTypePack Pack, class MeshType, class Stencils>
SIMPLEFLUID_LOCAL void evaluate_stored_scalar_gradients(const ScalarCellFieldStored<Pack, MeshType>& field, const Stencils& stencils,
    VectorCellFieldStored<Pack, MeshType>& gradients);

template<TpetraTypePack Pack, class MeshType, class BoundaryType, class BoundaryValue, class Source>
SIMPLEFLUID_LOCAL TransportSystem<Pack> stored_scalar_transport_system(const ScalarCellFieldStored<Pack, MeshType>& old_values,
    const ScalarFaceFieldStored<Pack, MeshType>& face_fluxes, typename Pack::scalar_type time_step,
    typename Pack::scalar_type diffusivity, BoundaryType boundary_type, BoundaryValue boundary_value, Source source,
    Teuchos::RCP<typename Pack::matrix_type> cached_matrix);

template<class MeshType, class GeometryCache>
SIMPLEFLUID_LOCAL void select_stored_interior_geometry(const MeshType& mesh, const GeometryCache* geometry_cache,
    std::vector<LeastSquaresGradientStencil<MeshType>>& local_stencils,
    std::vector<BoundaryFaceLocation<MeshType>>& local_locations,
    const std::vector<LeastSquaresGradientStencil<MeshType>>*& stencils,
    const std::vector<BoundaryFaceLocation<MeshType>>*& locations);

template<class MeshType, class BoundaryValue, class GeometryCache>
SIMPLEFLUID_LOCAL void select_stored_affine_geometry(const MeshType& mesh, BoundaryValue boundary_value,
    const GeometryCache* geometry_cache, std::vector<VectorAffineLeastSquaresGradientStencil<MeshType>>& stencils,
    std::vector<BoundaryFaceLocation<MeshType>>& local_locations,
    const std::vector<BoundaryFaceLocation<MeshType>>*& locations);

template<class MeshType, class BoundaryCondition, class BoundaryValue, class GeometryCache>
SIMPLEFLUID_LOCAL void select_stored_scalar_affine_geometry(const MeshType& mesh, BoundaryCondition boundary_condition,
    BoundaryValue boundary_value, const GeometryCache* geometry_cache,
    std::vector<AffineLeastSquaresGradientStencil<MeshType>>& stencils,
    std::vector<BoundaryFaceLocation<MeshType>>& local_locations,
    const std::vector<BoundaryFaceLocation<MeshType>>*& locations);

template<TpetraTypePack Pack, class MeshType, class BoundaryCondition, class Stencils, class BoundaryLocations>
SIMPLEFLUID_LOCAL void add_stored_scalar_explicit_non_orthogonal_correction(const ScalarCellFieldStored<Pack, MeshType>& correction_field,
    typename Pack::scalar_type diffusivity, BoundaryCondition boundary_condition, typename Pack::vector_type& rhs,
    typename Pack::scalar_type correction_weight, const Stencils& stencils, const BoundaryLocations& boundary_locations,
    const TransportAssemblyGeometry<MeshType>& assembly_geometry);

template<TpetraTypePack Pack, class MeshType, class BoundaryCondition, class BoundaryValue, class Source,
    class GeometryCache>
SIMPLEFLUID_LOCAL TransportSystem<Pack> stored_scalar_non_orthogonal_transport_system(
    const ScalarCellFieldStored<Pack, MeshType>& old_values, const ScalarFaceFieldStored<Pack, MeshType>& face_fluxes,
    typename Pack::scalar_type time_step, typename Pack::scalar_type diffusivity, BoundaryCondition boundary_condition,
    BoundaryValue boundary_value, Source source, NonOrthogonalTreatment treatment,
    const ScalarCellFieldStored<Pack, MeshType>* correction_field,
    Teuchos::RCP<typename Pack::matrix_type> cached_matrix, const GeometryCache* geometry_cache);

/**
 * @brief Detect transport faces whose area is not parallel to the center direction.
 *
 * Exact cross products avoid imposing a numerical near-orthogonality cutoff.
 * Degenerate geometry retains the normal assembly/validation path. The result
 * is local; callers must agree collectively before skipping gradient imports.
 */
template<class MeshType> bool stored_transport_has_non_orthogonal_faces(const MeshType& mesh);

/** Apply a variable-coefficient explicit scalar non-orthogonal correction. */
template<TpetraTypePack Pack, class MeshType, class BoundaryCondition, class DiffusivityValue,
    class BoundaryCoefficient, class Stencils, class BoundaryLocations>
SIMPLEFLUID_LOCAL void add_stored_variable_scalar_explicit_non_orthogonal_correction(
    const ScalarCellFieldStored<Pack, MeshType>& correction_field, BoundaryCondition boundary_condition,
    DiffusivityValue diffusivity_value, BoundaryCoefficient boundary_diffusivity, typename Pack::vector_type& rhs,
    typename Pack::scalar_type correction_weight, const Stencils& stencils, const BoundaryLocations& boundary_locations,
    FaceCoefficientInterpolation coefficient_interpolation,
    const TransportAssemblyGeometry<MeshType>& assembly_geometry);

/** Shared mapped weighted-scalar assembly kernel. */
template<TpetraTypePack Pack, class MeshType, class BoundaryCondition, class BoundaryValue, class Source,
    class BoundaryCache, class GeometryCache>
SIMPLEFLUID_LOCAL TransportSystem<Pack> stored_weighted_scalar_transport_system_impl(
    const ScalarCellFieldStored<Pack, MeshType>& old_values, const ScalarFaceFieldStored<Pack, MeshType>& face_fluxes,
    typename Pack::scalar_type time_step, const ScalarTransportCoefficientAccess<Pack>& coefficients,
    bool has_distinct_old_storage, BoundaryCondition boundary_condition, BoundaryValue boundary_value, Source source,
    NonOrthogonalTreatment treatment, const ScalarCellFieldStored<Pack, MeshType>* correction_field,
    Teuchos::RCP<typename Pack::matrix_type> cached_matrix,
    std::function<typename Pack::scalar_type(typename Pack::local_ordinal_type)> implicit_sink,
    std::function<std::optional<typename Pack::scalar_type>(typename Pack::local_ordinal_type)> fixed_cell_value,
    const BoundaryCache* boundary_diffusivity, const GeometryCache* geometry_cache,
    FaceCoefficientInterpolation coefficient_interpolation, int incompatible_fields, std::string_view context,
    ScalarTransportDiscretization discretization, const ScalarCellFieldStored<Pack, MeshType>* older_values,
    const ALEControlVolumeState* ale, StoredTransportSymbolicPlan<Pack>* symbolic_plan = nullptr,
    Teuchos::RCP<typename Pack::vector_type> cached_rhs = Teuchos::null, bool allow_robin = false,
    bool rhs_only = false);

/** Assemble weighted scalar transport from mapped coefficient fields. */
template<TpetraTypePack Pack, class MeshType, class BoundaryCondition, class BoundaryValue, class Source,
    class BoundaryCache, class GeometryCache>
SIMPLEFLUID_LOCAL TransportSystem<Pack> stored_weighted_scalar_transport_system(const ScalarCellFieldStored<Pack, MeshType>& old_values,
    const ScalarFaceFieldStored<Pack, MeshType>& face_fluxes, typename Pack::scalar_type time_step,
    const ScalarCellFieldStored<Pack, MeshType>& storage_weight,
    const ScalarCellFieldStored<Pack, MeshType>& advection_weight,
    const ScalarCellFieldStored<Pack, MeshType>& diffusivity, BoundaryCondition boundary_condition,
    BoundaryValue boundary_value, Source source, NonOrthogonalTreatment treatment,
    const ScalarCellFieldStored<Pack, MeshType>* correction_field,
    Teuchos::RCP<typename Pack::matrix_type> cached_matrix,
    std::function<typename Pack::scalar_type(typename Pack::local_ordinal_type)> implicit_sink,
    std::function<std::optional<typename Pack::scalar_type>(typename Pack::local_ordinal_type)> fixed_cell_value,
    const BoundaryCache* boundary_diffusivity, const GeometryCache* geometry_cache,
    FaceCoefficientInterpolation coefficient_interpolation, ScalarTransportDiscretization discretization,
    const ScalarCellFieldStored<Pack, MeshType>* older_values,
    const ScalarCellFieldStored<Pack, MeshType>* old_storage_weight, const ALEControlVolumeState* ale,
    StoredTransportSymbolicPlan<Pack>* symbolic_plan = nullptr,
    Teuchos::RCP<typename Pack::vector_type> cached_rhs = Teuchos::null);

/** Assemble mapped conservative physical temperature transport. */
template<TpetraTypePack Pack, class MeshType, class BoundaryCondition, class BoundaryValue, class Source,
    class BoundaryCache, class GeometryCache>
SIMPLEFLUID_LOCAL TransportSystem<Pack> stored_physical_temperature_transport_system(
    const ScalarCellFieldStored<Pack, MeshType>& old_temperature,
    const ScalarFaceFieldStored<Pack, MeshType>& face_fluxes, typename Pack::scalar_type time_step,
    const ScalarCellFieldStored<Pack, MeshType>& density,
    const ScalarCellFieldStored<Pack, MeshType>& specific_heat_capacity,
    const ScalarCellFieldStored<Pack, MeshType>& thermal_conductivity, BoundaryCondition boundary_condition,
    BoundaryValue boundary_value, Source power_density, NonOrthogonalTreatment treatment,
    const ScalarCellFieldStored<Pack, MeshType>* correction_field,
    Teuchos::RCP<typename Pack::matrix_type> cached_matrix, const BoundaryCache* boundary_thermal_conductivity,
    const GeometryCache* geometry_cache, FaceCoefficientInterpolation coefficient_interpolation,
    ScalarTransportDiscretization discretization, const ScalarCellFieldStored<Pack, MeshType>* older_temperature,
    const ScalarCellFieldStored<Pack, MeshType>* old_density,
    const ScalarCellFieldStored<Pack, MeshType>* old_specific_heat_capacity, const ALEControlVolumeState* ale,
    bool rhs_only = false);

template<TpetraTypePack Pack, class MeshType, class Stencils, class BoundaryLocations, class BoundaryDiffusion>
SIMPLEFLUID_LOCAL void add_stored_explicit_non_orthogonal_correction(const VectorCellFieldStored<Pack, MeshType>& correction_field,
    typename Pack::scalar_type diffusivity, typename Pack::multi_vector_type& rhs,
    typename Pack::scalar_type correction_weight, BoundaryDiffusion boundary_diffusion, const Stencils& stencils,
    const BoundaryLocations& boundary_locations, const TransportAssemblyGeometry<MeshType>& assembly_geometry);

template<TpetraTypePack Pack, class MeshType, class BoundaryValue, class Source, class BoundaryDiffusion,
    class GeometryCache>
SIMPLEFLUID_LOCAL VectorTransportSystem<Pack> stored_vector_transport_system(const VectorCellFieldStored<Pack, MeshType>& old_values,
    const ScalarFaceFieldStored<Pack, MeshType>& face_fluxes, typename Pack::scalar_type time_step,
    typename Pack::scalar_type diffusivity, BoundaryValue boundary_value, Source source,
    NonOrthogonalTreatment treatment, const VectorCellFieldStored<Pack, MeshType>* correction_field,
    Teuchos::RCP<typename Pack::matrix_type> cached_matrix, BoundaryDiffusion boundary_diffusion,
    const GeometryCache* geometry_cache, const ALEControlVolumeState* ale);

template<TpetraTypePack Pack, class MeshType, class Stencils, class BoundaryLocations, class BoundaryDiffusion,
    class BoundaryCoefficient>
SIMPLEFLUID_LOCAL void add_stored_variable_explicit_non_orthogonal_correction(
    const VectorCellFieldStored<Pack, MeshType>& correction_field,
    const ScalarCellFieldStored<Pack, MeshType>& dynamic_viscosity, typename Pack::multi_vector_type& rhs,
    typename Pack::scalar_type correction_weight, BoundaryDiffusion boundary_diffusion,
    BoundaryCoefficient boundary_viscosity, const Stencils& stencils, const BoundaryLocations& boundary_locations,
    FaceCoefficientInterpolation coefficient_interpolation,
    const TransportAssemblyGeometry<MeshType>& assembly_geometry);

template<TpetraTypePack Pack, class MeshType, class Stencils, class BoundaryLocations, class BoundaryDiffusion,
    class BoundaryCoefficient>
SIMPLEFLUID_LOCAL void add_stored_deviatoric_transpose_gradient_stress(const VectorCellFieldStored<Pack, MeshType>& old_velocity,
    const ScalarCellFieldStored<Pack, MeshType>& dynamic_viscosity, typename Pack::scalar_type reference_density,
    typename Pack::multi_vector_type& rhs, BoundaryDiffusion boundary_stress, BoundaryCoefficient boundary_viscosity,
    const Stencils& stencils, const BoundaryLocations& boundary_locations,
    FaceCoefficientInterpolation coefficient_interpolation,
    const TransportAssemblyGeometry<MeshType>& assembly_geometry,
    TensorCellFieldStored<Pack, MeshType>* gradient_workspace = nullptr, BoundaryFaceSelector zero_gradient_stress = {},
    const char* dump_prefix = nullptr);

/**
 * @brief Assemble mapped momentum transport with cell viscosity storage.
 *
 * This is the mapped-mesh counterpart of physical_momentum_transport_system.
 * It retains transient, upwind-advection, and orthogonal viscous terms while
 * avoiding a legacy Mesh materialization.
 */
template<TpetraTypePack Pack, class MeshType, class BoundaryValue, class Source, class BoundaryDiffusion,
    class BoundaryCache, class GeometryCache>
SIMPLEFLUID_LOCAL VectorTransportSystem<Pack> stored_physical_momentum_transport_system(
    const VectorCellFieldStored<Pack, MeshType>& old_velocity, const ScalarFaceFieldStored<Pack, MeshType>& face_fluxes,
    typename Pack::scalar_type time_step, const ScalarCellFieldStored<Pack, MeshType>& dynamic_viscosity,
    typename Pack::scalar_type reference_density, BoundaryValue boundary_value, Source source,
    NonOrthogonalTreatment treatment, const VectorCellFieldStored<Pack, MeshType>* correction_field,
    Teuchos::RCP<typename Pack::matrix_type> cached_matrix, BoundaryDiffusion boundary_diffusion,
    const BoundaryCache* boundary_dynamic_viscosity, const GeometryCache* geometry_cache,
    FaceCoefficientInterpolation coefficient_interpolation, const ALEControlVolumeState* ale,
    TensorCellFieldStored<Pack, MeshType>* gradient_workspace = nullptr, BoundaryFaceSelector zero_gradient_stress = {},
    AssemblyMode mode = AssemblyMode::Transient);

} // namespace detail
} // namespace SimpleFluid::FVM

#include "FVM/details/FieldStoredTransportSystem.ipp"
