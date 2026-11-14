/** @file FVM/TransportAssemblyPacks.hh
 * @brief Borrowed, synchronous inputs to transport assembly.
 */
#pragma once

#include "FVM/AssemblyCallbacks.hh"
#include "FVM/BoundaryCache.hh"
#include "FVM/FaceCoefficientInterpolation.hh"
#include "FVM/NonOrthogonalTreatment.hh"
#include "FVM/ScalarTransportDiscretization.hh"
#include "fields/MeshFieldTraits.hh"

#include <Teuchos_RCP.hpp>
#include <optional>
#include <type_traits>

namespace SimpleFluid::FVM
{
class ALEControlVolumeState;
template<class MeshType> class TransportGeometryCache;
namespace detail
{
template<TpetraTypePack Pack> class StoredTransportSymbolicPlan;
}

template<class Field, class Scalar> struct TransportHistory
{
    const Field& old_values;
    Scalar time_step;
};
template<class Field, class Scalar> struct ScalarTransportHistory
{
    const Field& old_values;
    Scalar time_step;
    const Field* older_values = nullptr;
};
template<class Field> struct TransportCorrection
{
    NonOrthogonalTreatment treatment = NonOrthogonalTreatment::Explicit;
    const Field* lagged_field = nullptr;
};

template<TpetraTypePack Pack> struct ScalarTransportBoundary
{
    ScalarBoundaryConditionProvider<Pack> condition;
    ScalarBoundaryValueProvider<Pack> value;
};
template<TpetraTypePack Pack> struct BasicVectorTransportBoundary
{
    VectorBoundaryConditionProvider<Pack> condition;
    VectorBoundaryValueProvider<Pack> value;
};
template<TpetraTypePack Pack> struct VelocityTransportBoundary
{
    VectorBoundaryValueProvider<Pack> value;
    BoundaryFaceSelector diffusion = [](int, size_t) { return true; };
};
template<TpetraTypePack Pack, class MeshType> struct CoefficientTransportBoundary
{
    ScalarBoundaryConditionProvider<Pack> condition;
    ScalarBoundaryValueProvider<Pack> value;
    const MeshBoundaryCache<Pack, MeshType>* coefficients = nullptr;
};
template<TpetraTypePack Pack, class MeshType, bool Legacy = std::is_same_v<MeshType, Mesh<Pack>>>
struct MomentumTransportBoundary
{
    VectorBoundaryValueProvider<Pack> value;
    BoundaryFaceSelector diffusion = [](int, size_t) { return true; };
    const MeshBoundaryCache<Pack, MeshType>* coefficients = nullptr;
    BoundaryFaceSelector zero_gradient_stress = {};
};
template<TpetraTypePack Pack, class MeshType> struct MomentumTransportBoundary<Pack, MeshType, true>
{
    VectorBoundaryValueProvider<Pack> value;
    BoundaryFaceSelector diffusion = [](int, size_t) { return true; };
    const MeshBoundaryCache<Pack, MeshType>* coefficients = nullptr;
};

template<TpetraTypePack Pack, class MeshType> struct ScalarTerms
{
    using mesh_type = MeshType;
    using Field = typename MeshFieldTraits<Pack, MeshType>::scalar_cell_type;
    TransportHistory<Field, typename Pack::scalar_type> history;
    const typename MeshFieldTraits<Pack, MeshType>::scalar_face_type& face_fluxes;
    struct Coefficients
    {
        typename Pack::scalar_type diffusivity;
    } coefficients;
    ScalarCellValueProvider<Pack> source = {};
};
template<TpetraTypePack Pack, class MeshType> struct VectorTerms
{
    using mesh_type = MeshType;
    using Field = typename MeshFieldTraits<Pack, MeshType>::vector_cell_type;
    TransportHistory<Field, typename Pack::scalar_type> history;
    const typename MeshFieldTraits<Pack, MeshType>::scalar_face_type& face_fluxes;
    struct Coefficients
    {
        typename Pack::scalar_type diffusivity;
    } coefficients;
    VectorCellValueProvider<Pack> source = {};
};
template<TpetraTypePack Pack, class MeshType> struct WeightedScalarTerms
{
    using mesh_type = MeshType;
    using Field = typename MeshFieldTraits<Pack, MeshType>::scalar_cell_type;
    ScalarTransportHistory<Field, typename Pack::scalar_type> history;
    const typename MeshFieldTraits<Pack, MeshType>::scalar_face_type& face_fluxes;
    struct Coefficients
    {
        const Field& storage_weight;
        const Field& advection_weight;
        const Field& diffusivity;
        const Field* old_storage_weight = nullptr;
    } coefficients;
    struct Sources
    {
        ScalarCellValueProvider<Pack> source = {};
        ScalarCellValueProvider<Pack> implicit_sink = {};
    } sources = {};
    struct Constraints
    {
        std::function<std::optional<typename Pack::scalar_type>(typename Pack::local_ordinal_type)> fixed_cell_value =
            {};
    } constraints = {};
};
template<TpetraTypePack Pack, class MeshType, bool Legacy = std::is_same_v<MeshType, Mesh<Pack>>>
struct TemperatureTerms
{
    using mesh_type = MeshType;
    using Field = typename MeshFieldTraits<Pack, MeshType>::scalar_cell_type;
    ScalarTransportHistory<Field, typename Pack::scalar_type> history;
    const typename MeshFieldTraits<Pack, MeshType>::scalar_face_type& face_fluxes;
    struct Material
    {
        const Field& density;
        const Field& specific_heat_capacity;
        const Field& thermal_conductivity;
        const Field* old_density = nullptr;
        const Field* old_specific_heat_capacity = nullptr;
    } material;
    ScalarCellValueProvider<Pack> power_density = {};
};
template<TpetraTypePack Pack, class MeshType> struct TemperatureTerms<Pack, MeshType, true>
{
    using mesh_type = MeshType;
    using Field = typename MeshFieldTraits<Pack, MeshType>::scalar_cell_type;
    TransportHistory<Field, typename Pack::scalar_type> history;
    const typename MeshFieldTraits<Pack, MeshType>::scalar_face_type& face_fluxes;
    struct Material
    {
        const Field& density;
        const Field& specific_heat_capacity;
        const Field& thermal_conductivity;
    } material;
    ScalarCellValueProvider<Pack> power_density = {};
};
template<TpetraTypePack Pack, class MeshType> struct MomentumTerms
{
    using mesh_type = MeshType;
    using Field = typename MeshFieldTraits<Pack, MeshType>::vector_cell_type;
    TransportHistory<Field, typename Pack::scalar_type> history;
    const typename MeshFieldTraits<Pack, MeshType>::scalar_face_type& face_fluxes;
    struct Material
    {
        const typename MeshFieldTraits<Pack, MeshType>::scalar_cell_type& dynamic_viscosity;
        typename Pack::scalar_type reference_density;
    } material;
    VectorCellValueProvider<Pack> acceleration_source = {};
};

template<TpetraTypePack Pack, class MeshType> struct ScalarAssemblyOptions
{
    std::optional<TransportCorrection<typename MeshFieldTraits<Pack, MeshType>::scalar_cell_type>> correction = {};
};
struct BasicVectorAssemblyOptions
{
};
template<TpetraTypePack Pack, class MeshType> struct VectorAssemblyOptions
{
    TransportCorrection<typename MeshFieldTraits<Pack, MeshType>::vector_cell_type> correction = {};
};
template<TpetraTypePack Pack, class MeshType> struct WeightedScalarAssemblyOptions
{
    ScalarTransportDiscretization discretization = {};
    TransportCorrection<typename MeshFieldTraits<Pack, MeshType>::scalar_cell_type> correction = {};
    FaceCoefficientInterpolation coefficient_interpolation = FaceCoefficientInterpolation::Harmonic;
};
template<TpetraTypePack Pack, class MeshType, bool Legacy = std::is_same_v<MeshType, Mesh<Pack>>>
struct TemperatureAssemblyOptions
{
    ScalarTransportDiscretization discretization = {};
    TransportCorrection<typename MeshFieldTraits<Pack, MeshType>::scalar_cell_type> correction = {};
    FaceCoefficientInterpolation coefficient_interpolation = FaceCoefficientInterpolation::Harmonic;
};
template<TpetraTypePack Pack, class MeshType> struct TemperatureAssemblyOptions<Pack, MeshType, true>
{
    TransportCorrection<typename MeshFieldTraits<Pack, MeshType>::scalar_cell_type> correction = {};
    FaceCoefficientInterpolation coefficient_interpolation = FaceCoefficientInterpolation::Harmonic;
    AssemblyMode mode = AssemblyMode::Transient;
};
template<TpetraTypePack Pack, class MeshType> struct MomentumAssemblyOptions
{
    TransportCorrection<typename MeshFieldTraits<Pack, MeshType>::vector_cell_type> correction = {};
    FaceCoefficientInterpolation coefficient_interpolation = FaceCoefficientInterpolation::Harmonic;
    AssemblyMode mode = AssemblyMode::Transient;
};

template<TpetraTypePack Pack> struct BasicVectorAssemblyContext
{
    Teuchos::RCP<typename Pack::matrix_type> cached_matrix = Teuchos::null;
};
template<TpetraTypePack Pack, class MeshType> struct ScalarAssemblyContext
{
    Teuchos::RCP<typename Pack::matrix_type> cached_matrix = Teuchos::null;
    const TransportGeometryCache<MeshType>* geometry_cache = nullptr;
};
template<TpetraTypePack Pack, class MeshType, bool Legacy = std::is_same_v<MeshType, Mesh<Pack>>>
struct VectorAssemblyContext
{
    Teuchos::RCP<typename Pack::matrix_type> cached_matrix = Teuchos::null;
    const TransportGeometryCache<MeshType>* geometry_cache = nullptr;
    const ALEControlVolumeState* ale = nullptr;
};
template<TpetraTypePack Pack, class MeshType> struct VectorAssemblyContext<Pack, MeshType, true>
{
    Teuchos::RCP<typename Pack::matrix_type> cached_matrix = Teuchos::null;
    const TransportGeometryCache<MeshType>* geometry_cache = nullptr;
};
template<TpetraTypePack Pack, class MeshType, bool Legacy = std::is_same_v<MeshType, Mesh<Pack>>>
struct WeightedScalarAssemblyContext
{
    Teuchos::RCP<typename Pack::matrix_type> cached_matrix = Teuchos::null;
    const TransportGeometryCache<MeshType>* geometry_cache = nullptr;
    const ALEControlVolumeState* ale = nullptr;
    detail::StoredTransportSymbolicPlan<Pack>* symbolic_plan = nullptr;
    Teuchos::RCP<typename Pack::vector_type> cached_rhs = Teuchos::null;
};
template<TpetraTypePack Pack, class MeshType> struct WeightedScalarAssemblyContext<Pack, MeshType, true>
{
    Teuchos::RCP<typename Pack::matrix_type> cached_matrix = Teuchos::null;
    const TransportGeometryCache<MeshType>* geometry_cache = nullptr;
};
template<TpetraTypePack Pack, class MeshType, bool Legacy = std::is_same_v<MeshType, Mesh<Pack>>>
struct TemperatureAssemblyContext
{
    Teuchos::RCP<typename Pack::matrix_type> cached_matrix = Teuchos::null;
    const TransportGeometryCache<MeshType>* geometry_cache = nullptr;
    const ALEControlVolumeState* ale = nullptr;
};
template<TpetraTypePack Pack, class MeshType> struct TemperatureAssemblyContext<Pack, MeshType, true>
{
    Teuchos::RCP<typename Pack::matrix_type> cached_matrix = Teuchos::null;
    const TransportGeometryCache<MeshType>* geometry_cache = nullptr;
};
template<TpetraTypePack Pack, class MeshType, bool Legacy = std::is_same_v<MeshType, Mesh<Pack>>>
struct MomentumAssemblyContext
{
    Teuchos::RCP<typename Pack::matrix_type> cached_matrix = Teuchos::null;
    const TransportGeometryCache<MeshType>* geometry_cache = nullptr;
    const ALEControlVolumeState* ale = nullptr;
    typename MeshFieldTraits<Pack, MeshType>::tensor_cell_type* gradient_workspace = nullptr;
};
template<TpetraTypePack Pack, class MeshType> struct MomentumAssemblyContext<Pack, MeshType, true>
{
    Teuchos::RCP<typename Pack::matrix_type> cached_matrix = Teuchos::null;
    const TransportGeometryCache<MeshType>* geometry_cache = nullptr;
};
} // namespace SimpleFluid::FVM
