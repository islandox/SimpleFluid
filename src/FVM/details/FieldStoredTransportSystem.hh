/**
 * @file FVM/details/FieldStoredTransportSystem.hh
 * @brief Transport assembly for mapped meshes and FieldStored.
 */

#pragma once

#include "FVM/AssemblyCallbacks.hh"
#include "FVM/BoundaryCache.hh"
#include "FVM/NonOrthogonalTreatment.hh"
#include "FVM/details/OperatorDetails.hh"
#include "fields/FieldStored.hh"

#include <Teuchos_Array.hpp>
#include <Teuchos_CommHelpers.hpp>
#include <Teuchos_RCP.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <exception>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace SimpleFluid::FVM
{

template<TpetraTypePack Pack> struct TransportSystem;

template<TpetraTypePack Pack> struct VectorTransportSystem;

namespace detail
{

template<TpetraTypePack Pack> struct StoredPreparedTransportMatrix
{
    Teuchos::RCP<typename Pack::matrix_type> matrix;
    bool reused = false;
};

template<TpetraTypePack Pack> struct StoredTransportMatrixRow
{
    std::vector<typename Pack::local_ordinal_type> columns;
    std::vector<typename Pack::scalar_type> values;
};

template<class Scalar> struct StoredNonOrthogonalWeights
{
    Scalar implicit{};
    Scalar explicit_{};
};

template<TpetraTypePack Pack, class MeshType, size_t StateCount>
std::array<int, StateCount> reduce_stored_validation_state(
    const MeshType& mesh, const std::array<int, StateCount>& local_state)
{
    auto global_state = local_state;
    const auto communicator = mesh.owned_cell_map()->getComm();
    if (communicator->getSize() > 1)
    {
        Teuchos::reduceAll(
            *communicator, Teuchos::REDUCE_MAX, static_cast<int>(StateCount), local_state.data(), global_state.data());
    }
    return global_state;
}

/** Header-visible mapped equivalent of the compiled legacy validator. */
template<TpetraTypePack Pack, class MeshType, class Field>
StoredNonOrthogonalWeights<typename Pack::scalar_type> validate_stored_non_orthogonal_selection(
    const MeshType& mesh, NonOrthogonalTreatment treatment, const Field* correction_field, std::string_view context)
{
    using scalar_type = typename Pack::scalar_type;

    int treatment_state = 3;
    switch (treatment)
    {
        case NonOrthogonalTreatment::Explicit:
            treatment_state = 0;
            break;
        case NonOrthogonalTreatment::Implicit:
            treatment_state = 1;
            break;
        case NonOrthogonalTreatment::Hybrid:
            treatment_state = 2;
            break;
    }

    const int correction_state = correction_field == nullptr ? 0 : correction_field->mesh_ptr().get() == &mesh ? 1 : 2;
    const int local_state[] = {treatment_state, -treatment_state, correction_state, -correction_state};
    int maximum_state[] = {local_state[0], local_state[1], local_state[2], local_state[3]};
    const auto communicator = mesh.owned_cell_map()->getComm();
    if (communicator->getSize() > 1)
    {
        Teuchos::reduceAll(*communicator, Teuchos::REDUCE_MAX, 4, local_state, maximum_state);
    }

    const auto prefix = std::string(context);
    if (maximum_state[0] == 3)
    {
        throw std::invalid_argument(prefix + " received an unknown non-orthogonal treatment.");
    }
    if (-maximum_state[1] != maximum_state[0])
    {
        throw std::invalid_argument(prefix + " requires every rank to use the same "
                                             "non-orthogonal treatment.");
    }
    if (-maximum_state[3] != maximum_state[2])
    {
        throw std::invalid_argument(prefix + " requires every rank to select the same category "
                                             "of correction field.");
    }
    if (maximum_state[2] == 2)
    {
        throw std::invalid_argument(prefix + " requires the correction field on the "
                                             "transported-field mesh.");
    }

    switch (treatment)
    {
        case NonOrthogonalTreatment::Explicit:
            return {scalar_type{}, scalar_type{1}};
        case NonOrthogonalTreatment::Implicit:
            return {scalar_type{1}, scalar_type{}};
        case NonOrthogonalTreatment::Hybrid:
            return {scalar_type{0.5}, scalar_type{0.5}};
    }
    throw std::invalid_argument(prefix + " received an unknown non-orthogonal treatment.");
}

template<TpetraTypePack Pack>
StoredTransportMatrixRow<Pack> capture_stored_transport_row(
    const FlatMatrixRow<typename Pack::local_ordinal_type, typename Pack::scalar_type>& row_values)
{
    StoredTransportMatrixRow<Pack> row;
    row.columns.assign(row_values.column_data(), row_values.column_data() + row_values.size());
    row.values.assign(row_values.value_data(), row_values.value_data() + row_values.size());
    return row;
}

/** Allocate a fresh mapped matrix or reset a collectively validated cached graph. */
template<TpetraTypePack Pack, class MeshType>
StoredPreparedTransportMatrix<Pack> prepare_stored_transport_matrix(const MeshType& mesh,
    Teuchos::RCP<typename Pack::matrix_type> cached_matrix, size_t entries_per_row,
    const std::vector<StoredTransportMatrixRow<Pack>>& rows)
{
    using matrix_type = typename Pack::matrix_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;

    const int cache_state = cached_matrix.is_null() ? 0 : 1;
    const auto cache_states = reduce_stored_validation_state<Pack>(mesh, std::array<int, 2>{cache_state, -cache_state});
    if (cache_states[0] != -cache_states[1])
    {
        throw std::invalid_argument("transport_system requires every rank to use the same cached-matrix category.");
    }
    if (cache_states[0] == 0)
    {
        return {Teuchos::rcp(new matrix_type(mesh.owned_cell_map(), mesh.overlap_cell_map(), entries_per_row)), false};
    }

    const auto row_map = cached_matrix->getRowMap();
    const auto column_map = cached_matrix->getColMap();
    const auto domain_map = cached_matrix->getDomainMap();
    const auto range_map = cached_matrix->getRangeMap();
    const auto structural_state = reduce_stored_validation_state<Pack>(mesh,
        std::array<int, 1>{!cached_matrix->isFillComplete() || row_map.is_null() || column_map.is_null() ||
                                   domain_map.is_null() || range_map.is_null() || rows.size() != mesh.num_owned_cells()
                               ? 1
                               : 0});
    if (structural_state[0] != 0)
    {
        throw std::invalid_argument("transport_system cached matrix is incompatible with the mesh.");
    }

    const int incompatible_row_map = !row_map->isSameAs(*mesh.owned_cell_map()) ? 1 : 0;
    const int incompatible_column_map = !column_map->isSameAs(*mesh.overlap_cell_map()) ? 1 : 0;
    const int incompatible_domain_map = !domain_map->isSameAs(*mesh.owned_cell_map()) ? 1 : 0;
    const int incompatible_range_map = !range_map->isSameAs(*mesh.owned_cell_map()) ? 1 : 0;
    const int incompatible_maps =
        incompatible_row_map || incompatible_column_map || incompatible_domain_map || incompatible_range_map;
    const auto map_state = reduce_stored_validation_state<Pack>(mesh, std::array<int, 1>{incompatible_maps});
    if (map_state[0] != 0)
    {
        throw std::invalid_argument("transport_system cached matrix is incompatible with the mesh.");
    }

    int incompatible_graph = 0;
    try
    {
        for (size_t row = 0; row < rows.size() && incompatible_graph == 0; ++row)
        {
            typename matrix_type::local_inds_host_view_type cached_columns;
            typename matrix_type::values_host_view_type cached_values;
            cached_matrix->getLocalRowView(static_cast<local_ordinal_type>(row), cached_columns, cached_values);
            for (const auto required_column : rows[row].columns)
            {
                bool found = false;
                for (size_t entry = 0; entry < cached_columns.extent(0); ++entry)
                {
                    found = found || cached_columns[entry] == required_column;
                }
                incompatible_graph = incompatible_graph || !found;
            }
        }
    }
    catch (const std::exception&)
    {
        incompatible_graph = 1;
    }
    const auto graph_state = reduce_stored_validation_state<Pack>(mesh, std::array<int, 1>{incompatible_graph});
    if (graph_state[0] != 0)
    {
        throw std::invalid_argument("transport_system cached matrix graph is incompatible with the operator.");
    }

    cached_matrix->resumeFill();
    cached_matrix->setAllToScalar(typename Pack::scalar_type{});
    return {std::move(cached_matrix), true};
}

/** Insert an accumulated row, validating every slot of a reused graph. */
template<TpetraTypePack Pack>
void add_stored_transport_values(const StoredPreparedTransportMatrix<Pack>& prepared,
    typename Pack::local_ordinal_type row, const StoredTransportMatrixRow<Pack>& row_values)
{
    const auto row_size = SimpleFluid::detail::checked_size_to_ordinal<Teuchos::Ordinal>(
        row_values.columns.size(), "stored transport row entry count");
    const auto columns = Teuchos::arrayView(row_values.columns.data(), row_size);
    const auto values = Teuchos::arrayView(row_values.values.data(), row_size);
    if (!prepared.reused)
    {
        prepared.matrix->insertLocalValues(row, columns, values);
        return;
    }

    const auto updated = prepared.matrix->sumIntoLocalValues(row, columns, values);
    if (updated != static_cast<typename Pack::local_ordinal_type>(columns.size()))
    {
        throw std::invalid_argument("transport_system cached matrix graph is incompatible with the "
                                    "operator.");
    }
}

/** Materialize a stored vector/tensor gradient from generic stencils. */
template<TpetraTypePack Pack, class MeshType, class Stencils>
void evaluate_stored_vector_gradients(const VectorCellFieldStored<Pack, MeshType>& field, const Stencils& stencils,
    TensorCellFieldStored<Pack, MeshType>& gradients)
{
    using local_ordinal_type = typename Pack::local_ordinal_type;
    using tensor_type = typename TensorCellFieldStored<Pack, MeshType>::tensor_type;

    if (field.mesh_ptr().get() != gradients.mesh_ptr().get() || stencils.size() != field.mesh().num_owned_cells())
    {
        throw std::invalid_argument("Stored vector gradient stencils are incompatible with the mesh.");
    }

    for (size_t owned = 0; owned < stencils.size(); ++owned)
    {
        const auto cell_lid = static_cast<local_ordinal_type>(owned);
        const auto& stencil = stencils[owned];
        tensor_type gradient{};
        if constexpr (requires { stencil.constants; })
        {
            gradient = stencil.constants;
        }

        const auto& entries = [&]() -> const auto&
        {
            if constexpr (requires { stencil.entries; })
            {
                return stencil.entries;
            }
            else
            {
                return stencil;
            }
        }();
        for (const auto& entry : entries)
        {
            const auto value = field.local_value(entry.cell_lid);
            for (size_t component = 0; component < 3; ++component)
            {
                const auto component_value = value.component(component);
                gradient[component].x += entry.coefficient.x * component_value;
                gradient[component].y += entry.coefficient.y * component_value;
                gradient[component].z += entry.coefficient.z * component_value;
            }
        }
        gradients.set_owned_value(cell_lid, gradient);
    }
    gradients.sync_ghosts();
}

/** Materialize a stored scalar gradient from affine reconstruction stencils. */
template<TpetraTypePack Pack, class MeshType, class Stencils>
void evaluate_stored_scalar_gradients(const ScalarCellFieldStored<Pack, MeshType>& field, const Stencils& stencils,
    VectorCellFieldStored<Pack, MeshType>& gradients)
{
    using local_ordinal_type = typename Pack::local_ordinal_type;

    if (field.mesh_ptr().get() != gradients.mesh_ptr().get() || stencils.size() != field.mesh().num_owned_cells())
    {
        throw std::invalid_argument("Stored scalar gradient stencils are incompatible with the mesh.");
    }

    for (size_t owned = 0; owned < stencils.size(); ++owned)
    {
        const auto cell_lid = static_cast<local_ordinal_type>(owned);
        const auto& stencil = stencils[owned];
        auto gradient = stencil.constant;
        for (const auto& entry : stencil.entries)
        {
            gradient = gradient + entry.coefficient * field.local_value(entry.cell_lid);
        }
        gradients.set_owned_value(cell_lid, gradient);
    }
    gradients.sync_ghosts();
}

template<TpetraTypePack Pack, class MeshType, class BoundaryType, class BoundaryValue, class Source>
TransportSystem<Pack> stored_scalar_transport_system(const ScalarCellFieldStored<Pack, MeshType>& old_values,
    const ScalarFaceFieldStored<Pack, MeshType>& face_fluxes, typename Pack::scalar_type time_step,
    typename Pack::scalar_type diffusivity, BoundaryType boundary_type, BoundaryValue boundary_value, Source source,
    Teuchos::RCP<typename Pack::matrix_type> cached_matrix)
{
    using scalar_type = typename Pack::scalar_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;

    const auto& mesh = old_values.mesh();
    const auto validation_state = reduce_stored_validation_state<Pack>(
        mesh, std::array<int, 3>{old_values.mesh_ptr().get() != face_fluxes.mesh_ptr().get() ? 1 : 0,
                  !std::isfinite(time_step) || time_step <= scalar_type{} ? 1 : 0,
                  !std::isfinite(diffusivity) || diffusivity < scalar_type{} ? 1 : 0});
    if (validation_state[0] != 0)
    {
        throw std::invalid_argument("transport_system requires fields on the same mesh.");
    }
    if (validation_state[1] != 0)
    {
        throw std::invalid_argument("transport_system requires a finite positive time step.");
    }
    if (validation_state[2] != 0)
    {
        throw std::invalid_argument("transport_system requires finite non-negative diffusivity.");
    }

    auto rhs = Teuchos::rcp(new typename Pack::vector_type(mesh.owned_cell_map(), true));
    const auto boundary_locations = boundary_face_locations(mesh);
    FlatMatrixRow<local_ordinal_type, scalar_type> row_values(mesh.num_local_cells(), 32);
    std::vector<StoredTransportMatrixRow<Pack>> rows;
    rows.reserve(mesh.num_owned_cells());

    for (size_t owned = 0; owned < mesh.num_owned_cells(); ++owned)
    {
        const auto cell_lid = static_cast<local_ordinal_type>(owned);
        const auto volume = static_cast<scalar_type>(mesh.cell_volume(cell_lid));
        const auto transient = volume / time_step;
        auto rhs_value = transient * old_values.local_value(cell_lid) + volume * source(cell_lid);
        row_values.clear();
        add_matrix_entry(row_values, cell_lid, transient);

        for (const auto face_lid : mesh.faces(cell_lid))
        {
            const auto is_interior = mesh.is_interior_face(face_lid);
            local_ordinal_type other{};
            if (is_interior)
            {
                other = mesh.opposite_or_periodic_neighbor_cell(face_lid, cell_lid);
                row_values.ensure(other);
            }

            const auto owner_flux = face_fluxes.local_value(face_lid);
            const auto outward_flux = mesh.owner_cell(face_lid) == cell_lid ? owner_flux : -owner_flux;
            if (outward_flux >= scalar_type{})
            {
                add_matrix_entry(row_values, cell_lid, outward_flux);
            }
            else if (is_interior)
            {
                add_matrix_entry(row_values, other, outward_flux);
            }
            else if (mesh.is_boundary_face(face_lid))
            {
                const auto index = packed_face_local_id(mesh, face_lid);
                if (index < boundary_locations.size() && boundary_locations[index].active)
                {
                    const auto location = boundary_locations[index];
                    rhs_value -= outward_flux * boundary_value(location.batch_id, location.in_batch_id);
                }
            }

            if (is_interior && diffusivity > scalar_type{})
            {
                const auto coefficient = interior_diffusion_coefficient(mesh, face_lid, cell_lid, other, diffusivity);
                add_matrix_entry(row_values, cell_lid, coefficient);
                add_matrix_entry(row_values, other, -coefficient);
            }
            if (!mesh.is_boundary_face(face_lid))
            {
                continue;
            }
            const auto index = packed_face_local_id(mesh, face_lid);
            if (index >= boundary_locations.size() || !boundary_locations[index].active)
            {
                continue;
            }
            const auto location = boundary_locations[index];
            const auto condition = boundary_type(location.batch_id, location.in_batch_id);
            const auto value = boundary_value(location.batch_id, location.in_batch_id);
            if (condition.type == BoundaryConditionType::Dirichlet)
            {
                const auto coefficient = boundary_diffusion_coefficient(mesh, face_lid, cell_lid, diffusivity);
                add_matrix_entry(row_values, cell_lid, coefficient);
                rhs_value += coefficient * value;
            }
            else if (condition.type == BoundaryConditionType::Neumann)
            {
                rhs_value += diffusivity * condition.value * static_cast<scalar_type>(mesh.face_area(face_lid));
            }
            else if (condition.type == BoundaryConditionType::Robin)
            {
                throw std::runtime_error("transport_system does not implement Robin boundaries.");
            }
        }

        rows.push_back(capture_stored_transport_row<Pack>(row_values));
        rhs->replaceLocalValue(cell_lid, rhs_value);
    }

    const auto prepared = prepare_stored_transport_matrix<Pack>(mesh, std::move(cached_matrix), 12, rows);
    const auto& matrix = prepared.matrix;
    for (size_t row = 0; row < rows.size(); ++row)
    {
        add_stored_transport_values<Pack>(prepared, static_cast<local_ordinal_type>(row), rows[row]);
    }
    matrix->fillComplete();
    return {matrix, rhs};
}

template<class MeshType, class GeometryCache>
void select_stored_interior_geometry(const MeshType& mesh, const GeometryCache* geometry_cache,
    std::vector<LeastSquaresGradientStencil<MeshType>>& local_stencils,
    std::vector<BoundaryFaceLocation<MeshType>>& local_locations,
    const std::vector<LeastSquaresGradientStencil<MeshType>>*& stencils,
    const std::vector<BoundaryFaceLocation<MeshType>>*& locations)
{
    if (geometry_cache != nullptr)
    {
        geometry_cache->require_mesh(mesh);
        stencils = &geometry_cache->interior_stencils();
        locations = &geometry_cache->boundary_locations();
        return;
    }

    local_stencils = least_squares_gradient_stencils(mesh);
    local_locations = boundary_face_locations(mesh);
    stencils = &local_stencils;
    locations = &local_locations;
}

template<class MeshType, class BoundaryValue, class GeometryCache>
void select_stored_affine_geometry(const MeshType& mesh, BoundaryValue boundary_value,
    const GeometryCache* geometry_cache, std::vector<VectorAffineLeastSquaresGradientStencil<MeshType>>& stencils,
    std::vector<BoundaryFaceLocation<MeshType>>& local_locations,
    const std::vector<BoundaryFaceLocation<MeshType>>*& locations)
{
    if (geometry_cache != nullptr)
    {
        geometry_cache->require_mesh(mesh);
        stencils = geometry_cache->vector_affine_stencils(boundary_value);
        locations = &geometry_cache->boundary_locations();
        return;
    }

    stencils = vector_affine_gradient_stencils(mesh, boundary_value);
    local_locations = boundary_face_locations(mesh);
    locations = &local_locations;
}

template<class MeshType, class BoundaryCondition, class BoundaryValue, class GeometryCache>
void select_stored_scalar_affine_geometry(const MeshType& mesh, BoundaryCondition boundary_condition,
    BoundaryValue boundary_value, const GeometryCache* geometry_cache,
    std::vector<AffineLeastSquaresGradientStencil<MeshType>>& stencils,
    std::vector<BoundaryFaceLocation<MeshType>>& local_locations,
    const std::vector<BoundaryFaceLocation<MeshType>>*& locations)
{
    if (geometry_cache != nullptr)
    {
        geometry_cache->require_mesh(mesh);
        stencils = geometry_cache->scalar_affine_stencils(boundary_condition, boundary_value);
        locations = &geometry_cache->boundary_locations();
        return;
    }

    stencils = scalar_affine_gradient_stencils(mesh, boundary_condition, boundary_value);
    local_locations = boundary_face_locations(mesh);
    locations = &local_locations;
}

template<TpetraTypePack Pack, class MeshType, class BoundaryCondition, class Stencils, class BoundaryLocations>
void add_stored_scalar_explicit_non_orthogonal_correction(const ScalarCellFieldStored<Pack, MeshType>& correction_field,
    typename Pack::scalar_type diffusivity, BoundaryCondition boundary_condition, typename Pack::vector_type& rhs,
    typename Pack::scalar_type correction_weight, const Stencils& stencils, const BoundaryLocations& boundary_locations)
{
    using scalar_type = typename Pack::scalar_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;
    if (diffusivity <= scalar_type{} || correction_weight == scalar_type{})
    {
        return;
    }

    VectorCellFieldStored<Pack, MeshType> gradients(
        correction_field.mesh_ptr(), "stored_scalar_non_orthogonal_gradient");
    evaluate_stored_scalar_gradients(correction_field, stencils, gradients);
    const auto& mesh = correction_field.mesh();

    for (size_t owned = 0; owned < mesh.num_owned_cells(); ++owned)
    {
        const auto cell_lid = static_cast<local_ordinal_type>(owned);
        for (const auto face_lid : mesh.faces(cell_lid))
        {
            auto gradient = gradients.local_value(cell_lid);
            typename MeshType::Vec3 direction{};
            if (mesh.is_interior_face(face_lid))
            {
                const auto other = mesh.opposite_or_periodic_neighbor_cell(face_lid, cell_lid);
                gradient = (gradient + gradients.local_value(other)) / scalar_type{2};
                direction = mesh.cell_center_vector(face_lid, cell_lid);
            }
            else if (mesh.is_boundary_face(face_lid))
            {
                const auto index = packed_face_local_id(mesh, face_lid);
                if (index >= boundary_locations.size() || !boundary_locations[index].active)
                {
                    continue;
                }
                const auto location = boundary_locations[index];
                if (boundary_condition(location.batch_id, location.in_batch_id).type !=
                    BoundaryConditionType::Dirichlet)
                {
                    continue;
                }
                direction = mesh.face_centroid(face_lid) - mesh.cell_centroid(cell_lid);
            }
            else
            {
                continue;
            }

            const auto tangential_area =
                non_orthogonal_area_vector(mesh.face_area_vector_outward(face_lid, cell_lid), direction);
            rhs.sumIntoLocalValue(cell_lid, correction_weight * diffusivity * gradient.dot(tangential_area));
        }
    }
}

template<TpetraTypePack Pack, class MeshType, class BoundaryCondition, class BoundaryValue, class Source,
    class GeometryCache>
TransportSystem<Pack> stored_scalar_non_orthogonal_transport_system(
    const ScalarCellFieldStored<Pack, MeshType>& old_values, const ScalarFaceFieldStored<Pack, MeshType>& face_fluxes,
    typename Pack::scalar_type time_step, typename Pack::scalar_type diffusivity, BoundaryCondition boundary_condition,
    BoundaryValue boundary_value, Source source, NonOrthogonalTreatment treatment,
    const ScalarCellFieldStored<Pack, MeshType>* correction_field,
    Teuchos::RCP<typename Pack::matrix_type> cached_matrix, const GeometryCache* geometry_cache)
{
    using scalar_type = typename Pack::scalar_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;

    const auto& mesh = old_values.mesh();
    const auto weights = validate_stored_non_orthogonal_selection<Pack>(
        mesh, treatment, correction_field, "non_orthogonal_transport_system");
    int invalid_geometry_cache = 0;
    if (geometry_cache != nullptr)
    {
        try
        {
            geometry_cache->require_mesh(mesh);
        }
        catch (const std::invalid_argument&)
        {
            invalid_geometry_cache = 1;
        }
    }
    const auto validation_state = reduce_stored_validation_state<Pack>(
        mesh, std::array<int, 4>{old_values.mesh_ptr().get() != face_fluxes.mesh_ptr().get() ? 1 : 0,
                  !std::isfinite(time_step) || time_step <= scalar_type{} ? 1 : 0,
                  !std::isfinite(diffusivity) || diffusivity < scalar_type{} ? 1 : 0, invalid_geometry_cache});
    if (validation_state[0] != 0)
    {
        throw std::invalid_argument("non_orthogonal_transport_system requires fields on the same mesh.");
    }
    if (validation_state[1] != 0)
    {
        throw std::invalid_argument("non_orthogonal_transport_system requires a finite positive time step.");
    }
    if (validation_state[2] != 0)
    {
        throw std::invalid_argument("non_orthogonal_transport_system requires finite non-negative diffusivity.");
    }
    if (validation_state[3] != 0)
    {
        throw std::invalid_argument("non_orthogonal_transport_system received a geometry cache on the wrong mesh.");
    }

    std::vector<AffineLeastSquaresGradientStencil<MeshType>> gradient_stencils;
    std::vector<BoundaryFaceLocation<MeshType>> local_locations;
    const std::vector<BoundaryFaceLocation<MeshType>>* locations{};
    select_stored_scalar_affine_geometry(
        mesh, boundary_condition, boundary_value, geometry_cache, gradient_stencils, local_locations, locations);

    std::unique_ptr<VectorCellFieldStored<Pack, MeshType>> partition_gradients;
    if (weights.implicit > scalar_type{})
    {
        partition_gradients = std::make_unique<VectorCellFieldStored<Pack, MeshType>>(
            old_values.mesh_ptr(), "stored_partition_scalar_non_orthogonal_gradient");
        const auto& lagged_field = correction_field == nullptr ? old_values : *correction_field;
        evaluate_stored_scalar_gradients(lagged_field, gradient_stencils, *partition_gradients);
    }

    auto rhs = Teuchos::rcp(new typename Pack::vector_type(mesh.owned_cell_map(), true));
    FlatMatrixRow<local_ordinal_type, scalar_type> row_values(mesh.num_local_cells(), 64);
    std::vector<StoredTransportMatrixRow<Pack>> rows;
    rows.reserve(mesh.num_owned_cells());

    for (size_t owned = 0; owned < mesh.num_owned_cells(); ++owned)
    {
        const auto cell_lid = static_cast<local_ordinal_type>(owned);
        const auto volume = static_cast<scalar_type>(mesh.cell_volume(cell_lid));
        const auto transient = volume / time_step;
        auto rhs_value = transient * old_values.local_value(cell_lid) + volume * source(cell_lid);
        row_values.clear();
        add_matrix_entry(row_values, cell_lid, transient);

        auto add_non_orthogonal_stencil = [&](local_ordinal_type gradient_cell_lid, scalar_type gradient_weight,
                                              const typename MeshType::Vec3& tangential_area)
        {
            if (weights.implicit == scalar_type{} || gradient_weight == scalar_type{} ||
                !mesh.is_owned_cell(gradient_cell_lid) ||
                static_cast<size_t>(gradient_cell_lid) >= gradient_stencils.size())
            {
                return;
            }
            const auto scale = -weights.implicit * diffusivity * gradient_weight;
            const auto& stencil = gradient_stencils[static_cast<size_t>(gradient_cell_lid)];
            rhs_value -= scale * stencil.constant.dot(tangential_area);
            for (const auto& entry : stencil.entries)
            {
                add_matrix_entry(row_values, entry.cell_lid, scale * entry.coefficient.dot(tangential_area));
            }
        };

        for (const auto face_lid : mesh.faces(cell_lid))
        {
            const auto is_interior = mesh.is_interior_face(face_lid);
            local_ordinal_type other{};
            if (is_interior)
            {
                other = mesh.opposite_or_periodic_neighbor_cell(face_lid, cell_lid);
                row_values.ensure(other);
            }

            const auto owner_flux = face_fluxes.local_value(face_lid);
            const auto outward_flux = mesh.owner_cell(face_lid) == cell_lid ? owner_flux : -owner_flux;
            if (outward_flux >= scalar_type{})
            {
                add_matrix_entry(row_values, cell_lid, outward_flux);
            }
            else if (is_interior)
            {
                add_matrix_entry(row_values, other, outward_flux);
            }
            else if (mesh.is_boundary_face(face_lid))
            {
                const auto index = packed_face_local_id(mesh, face_lid);
                if (index < locations->size() && (*locations)[index].active)
                {
                    const auto location = (*locations)[index];
                    rhs_value -= outward_flux * boundary_value(location.batch_id, location.in_batch_id);
                }
            }

            if (diffusivity <= scalar_type{})
            {
                continue;
            }
            if (is_interior)
            {
                const auto coefficient = interior_diffusion_coefficient(mesh, face_lid, cell_lid, other, diffusivity);
                add_matrix_entry(row_values, cell_lid, coefficient);
                add_matrix_entry(row_values, other, -coefficient);
                const auto tangential_area = non_orthogonal_area_vector(
                    mesh.face_area_vector_outward(face_lid, cell_lid), mesh.cell_center_vector(face_lid, cell_lid));
                if (mesh.is_owned_cell(other) && static_cast<size_t>(other) < gradient_stencils.size())
                {
                    add_non_orthogonal_stencil(cell_lid, scalar_type{0.5}, tangential_area);
                    add_non_orthogonal_stencil(other, scalar_type{0.5}, tangential_area);
                }
                else if (weights.implicit > scalar_type{})
                {
                    add_non_orthogonal_stencil(cell_lid, scalar_type{0.5}, tangential_area);
                    if (partition_gradients == nullptr)
                    {
                        throw std::logic_error("Partition-face implicit scalar reconstruction requires "
                                               "synchronized remote gradients.");
                    }
                    rhs_value += weights.implicit * diffusivity * scalar_type{0.5} *
                                 partition_gradients->local_value(other).dot(tangential_area);
                }
                continue;
            }

            if (!mesh.is_boundary_face(face_lid))
            {
                continue;
            }
            const auto index = packed_face_local_id(mesh, face_lid);
            if (index >= locations->size() || !(*locations)[index].active)
            {
                continue;
            }
            const auto location = (*locations)[index];
            const auto condition = boundary_condition(location.batch_id, location.in_batch_id);
            if (condition.type == BoundaryConditionType::Dirichlet)
            {
                const auto coefficient = boundary_diffusion_coefficient(mesh, face_lid, cell_lid, diffusivity);
                add_matrix_entry(row_values, cell_lid, coefficient);
                rhs_value += coefficient * boundary_value(location.batch_id, location.in_batch_id);
                const auto tangential_area =
                    non_orthogonal_area_vector(mesh.face_area_vector_outward(face_lid, cell_lid),
                        mesh.face_centroid(face_lid) - mesh.cell_centroid(cell_lid));
                add_non_orthogonal_stencil(cell_lid, scalar_type{1}, tangential_area);
            }
            else if (condition.type == BoundaryConditionType::Neumann)
            {
                rhs_value += diffusivity * condition.value * static_cast<scalar_type>(mesh.face_area(face_lid));
            }
            else if (condition.type == BoundaryConditionType::Robin)
            {
                throw std::runtime_error("non_orthogonal_transport_system does not implement Robin boundaries.");
            }
        }

        rows.push_back(capture_stored_transport_row<Pack>(row_values));
        rhs->replaceLocalValue(cell_lid, rhs_value);
    }

    if (correction_field != nullptr && weights.explicit_ > scalar_type{})
    {
        add_stored_scalar_explicit_non_orthogonal_correction<Pack>(
            *correction_field, diffusivity, boundary_condition, *rhs, weights.explicit_, gradient_stencils, *locations);
    }

    const auto prepared = prepare_stored_transport_matrix<Pack>(mesh, std::move(cached_matrix), 32, rows);
    const auto& matrix = prepared.matrix;
    for (size_t row = 0; row < rows.size(); ++row)
    {
        add_stored_transport_values<Pack>(prepared, static_cast<local_ordinal_type>(row), rows[row]);
    }
    matrix->fillComplete();
    return {matrix, rhs};
}

/** Apply a variable-coefficient explicit scalar non-orthogonal correction. */
template<TpetraTypePack Pack, class MeshType, class BoundaryCondition, class DiffusivityValue,
    class BoundaryCoefficient, class Stencils, class BoundaryLocations>
void add_stored_variable_scalar_explicit_non_orthogonal_correction(
    const ScalarCellFieldStored<Pack, MeshType>& correction_field, BoundaryCondition boundary_condition,
    DiffusivityValue diffusivity_value, BoundaryCoefficient boundary_diffusivity, typename Pack::vector_type& rhs,
    typename Pack::scalar_type correction_weight, const Stencils& stencils, const BoundaryLocations& boundary_locations,
    FaceCoefficientInterpolation coefficient_interpolation)
{
    using scalar_type = typename Pack::scalar_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;
    if (correction_weight == scalar_type{})
    {
        return;
    }

    VectorCellFieldStored<Pack, MeshType> gradients(
        correction_field.mesh_ptr(), "stored_variable_scalar_non_orthogonal_gradient");
    evaluate_stored_scalar_gradients(correction_field, stencils, gradients);
    const auto& mesh = correction_field.mesh();

    for (size_t owned = 0; owned < mesh.num_owned_cells(); ++owned)
    {
        const auto cell_lid = static_cast<local_ordinal_type>(owned);
        for (const auto face_lid : mesh.faces(cell_lid))
        {
            auto gradient = gradients.local_value(cell_lid);
            auto face_diffusivity = diffusivity_value(cell_lid);
            typename MeshType::Vec3 direction{};
            if (mesh.is_interior_face(face_lid))
            {
                const auto other = mesh.opposite_or_periodic_neighbor_cell(face_lid, cell_lid);
                gradient = (gradient + gradients.local_value(other)) / scalar_type{2};
                face_diffusivity = face_coefficient_value(mesh, face_lid, cell_lid, other, diffusivity_value(cell_lid),
                    diffusivity_value(other), coefficient_interpolation);
                direction = mesh.cell_center_vector(face_lid, cell_lid);
            }
            else if (mesh.is_boundary_face(face_lid))
            {
                const auto index = packed_face_local_id(mesh, face_lid);
                if (index >= boundary_locations.size() || !boundary_locations[index].active)
                {
                    continue;
                }
                const auto location = boundary_locations[index];
                if (boundary_condition(location.batch_id, location.in_batch_id).type !=
                    BoundaryConditionType::Dirichlet)
                {
                    continue;
                }
                face_diffusivity = boundary_diffusivity(location.batch_id, location.in_batch_id, face_diffusivity);
                direction = mesh.face_centroid(face_lid) - mesh.cell_centroid(cell_lid);
            }
            else
            {
                continue;
            }

            const auto tangential_area =
                non_orthogonal_area_vector(mesh.face_area_vector_outward(face_lid, cell_lid), direction);
            rhs.sumIntoLocalValue(cell_lid, correction_weight * face_diffusivity * gradient.dot(tangential_area));
        }
    }
}

/** Shared mapped weighted-scalar assembly kernel. */
template<TpetraTypePack Pack, class MeshType, class StorageValue, class AdvectionValue, class DiffusivityValue,
    class BoundaryCondition, class BoundaryValue, class Source, class BoundaryCache, class GeometryCache>
TransportSystem<Pack> stored_weighted_scalar_transport_system_impl(
    const ScalarCellFieldStored<Pack, MeshType>& old_values, const ScalarFaceFieldStored<Pack, MeshType>& face_fluxes,
    typename Pack::scalar_type time_step, StorageValue storage_value, AdvectionValue advection_value,
    DiffusivityValue diffusivity_value, BoundaryCondition boundary_condition, BoundaryValue boundary_value,
    Source source, NonOrthogonalTreatment treatment, const ScalarCellFieldStored<Pack, MeshType>* correction_field,
    Teuchos::RCP<typename Pack::matrix_type> cached_matrix,
    std::function<typename Pack::scalar_type(typename Pack::local_ordinal_type)> implicit_sink,
    std::function<std::optional<typename Pack::scalar_type>(typename Pack::local_ordinal_type)> fixed_cell_value,
    const BoundaryCache* boundary_diffusivity, const GeometryCache* geometry_cache,
    FaceCoefficientInterpolation coefficient_interpolation, int incompatible_fields, std::string_view context)
{
    using scalar_type = typename Pack::scalar_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;

    const auto& mesh = old_values.mesh();
    const auto weights = validate_stored_non_orthogonal_selection<Pack>(mesh, treatment, correction_field, context);

    int invalid_boundary_cache = 0;
    try
    {
        validate_boundary_coefficient_cache<Pack>(mesh, boundary_diffusivity, std::string(context));
    }
    catch (const std::invalid_argument&)
    {
        invalid_boundary_cache = 1;
    }
    int invalid_geometry_cache = 0;
    if (geometry_cache != nullptr)
    {
        try
        {
            geometry_cache->require_mesh(mesh);
        }
        catch (const std::invalid_argument&)
        {
            invalid_geometry_cache = 1;
        }
    }

    int invalid_coefficients = 0;
    if (incompatible_fields == 0)
    {
        for (size_t local = 0; local < mesh.num_local_cells(); ++local)
        {
            const auto cell_lid = static_cast<local_ordinal_type>(local);
            const auto storage = storage_value(cell_lid);
            const auto advection = advection_value(cell_lid);
            const auto diffusivity = diffusivity_value(cell_lid);
            invalid_coefficients = invalid_coefficients || !std::isfinite(storage) || !std::isfinite(advection) ||
                                   !std::isfinite(diffusivity) || storage <= scalar_type{} ||
                                   advection < scalar_type{} || diffusivity < scalar_type{};
        }
    }
    int interpolation_state = 2;
    switch (coefficient_interpolation)
    {
        case FaceCoefficientInterpolation::Harmonic:
            interpolation_state = 0;
            break;
        case FaceCoefficientInterpolation::Linear:
            interpolation_state = 1;
            break;
    }
    const auto validation_state = reduce_stored_validation_state<Pack>(
        mesh, std::array<int, 8>{incompatible_fields, !std::isfinite(time_step) || time_step <= scalar_type{} ? 1 : 0,
                  invalid_boundary_cache, invalid_geometry_cache, invalid_coefficients, interpolation_state,
                  -interpolation_state, source ? 0 : 1});
    const auto prefix = std::string(context);
    if (validation_state[0] != 0)
    {
        throw std::invalid_argument(prefix + " requires all fields on the transported-field mesh.");
    }
    if (validation_state[1] != 0)
    {
        throw std::invalid_argument(prefix + " requires a finite positive time step.");
    }
    if (validation_state[2] != 0)
    {
        throw std::invalid_argument(prefix + " received an invalid boundary-coefficient cache.");
    }
    if (validation_state[3] != 0)
    {
        throw std::invalid_argument(prefix + " received a geometry cache on the wrong mesh.");
    }
    if (validation_state[4] != 0)
    {
        throw std::invalid_argument(prefix + " requires finite positive storage and finite "
                                             "non-negative advection and diffusion coefficients.");
    }
    if (validation_state[5] == 2 || -validation_state[6] != validation_state[5])
    {
        throw std::invalid_argument(prefix + " requires every rank to use the same valid "
                                             "face-coefficient interpolation.");
    }
    if (validation_state[7] != 0)
    {
        throw std::invalid_argument(prefix + " requires a source provider.");
    }

    auto boundary_face_diffusivity = [&](int batch_id, size_t in_batch_id, scalar_type owner_value)
    { return boundary_coefficient<Pack>(boundary_diffusivity, batch_id, in_batch_id, owner_value); };

    std::vector<std::optional<scalar_type>> fixed_values(mesh.num_owned_cells());
    for (size_t owned = 0; owned < mesh.num_owned_cells(); ++owned)
    {
        const auto cell_lid = static_cast<local_ordinal_type>(owned);
        const auto fixed = fixed_cell_value ? fixed_cell_value(cell_lid) : std::optional<scalar_type>{};
        if (fixed && !std::isfinite(*fixed))
        {
            throw std::invalid_argument(prefix + " requires finite fixed-cell values.");
        }
        fixed_values[owned] = fixed;
    }

    std::vector<AffineLeastSquaresGradientStencil<MeshType>> gradient_stencils;
    std::vector<BoundaryFaceLocation<MeshType>> local_locations;
    const std::vector<BoundaryFaceLocation<MeshType>>* locations{};
    select_stored_scalar_affine_geometry(
        mesh, boundary_condition, boundary_value, geometry_cache, gradient_stencils, local_locations, locations);

    std::unique_ptr<VectorCellFieldStored<Pack, MeshType>> partition_gradients;
    if (weights.implicit > scalar_type{})
    {
        partition_gradients = std::make_unique<VectorCellFieldStored<Pack, MeshType>>(
            old_values.mesh_ptr(), "stored_partition_weighted_scalar_gradient");
        const auto& lagged = correction_field == nullptr ? old_values : *correction_field;
        evaluate_stored_scalar_gradients(lagged, gradient_stencils, *partition_gradients);
    }

    auto rhs = Teuchos::rcp(new typename Pack::vector_type(mesh.owned_cell_map(), true));
    FlatMatrixRow<local_ordinal_type, scalar_type> row_values(mesh.num_local_cells(), 64);
    std::vector<StoredTransportMatrixRow<Pack>> rows;
    rows.reserve(mesh.num_owned_cells());

    for (size_t owned = 0; owned < mesh.num_owned_cells(); ++owned)
    {
        const auto cell_lid = static_cast<local_ordinal_type>(owned);
        const auto volume = static_cast<scalar_type>(mesh.cell_volume(cell_lid));
        const auto cell_storage = storage_value(cell_lid);
        const auto cell_advection = advection_value(cell_lid);
        const auto transient = cell_storage * volume / time_step;
        const auto sink = implicit_sink ? implicit_sink(cell_lid) : scalar_type{};
        if (!std::isfinite(sink) || sink < scalar_type{})
        {
            throw std::invalid_argument(prefix + " requires a finite non-negative implicit sink.");
        }

        row_values.clear();
        add_matrix_entry(row_values, cell_lid, transient + volume * sink);
        auto rhs_value = transient * old_values.local_value(cell_lid) + volume * source(cell_lid);

        auto add_non_orthogonal_stencil = [&](local_ordinal_type gradient_cell_lid, scalar_type gradient_weight,
                                              scalar_type face_diffusivity,
                                              const typename MeshType::Vec3& tangential_area)
        {
            if (weights.implicit == scalar_type{} || gradient_weight == scalar_type{} ||
                !mesh.is_owned_cell(gradient_cell_lid) ||
                static_cast<size_t>(gradient_cell_lid) >= gradient_stencils.size())
            {
                return;
            }
            const auto scale = -weights.implicit * face_diffusivity * gradient_weight;
            const auto& stencil = gradient_stencils[static_cast<size_t>(gradient_cell_lid)];
            rhs_value -= scale * stencil.constant.dot(tangential_area);
            for (const auto& entry : stencil.entries)
            {
                add_matrix_entry(row_values, entry.cell_lid, scale * entry.coefficient.dot(tangential_area));
            }
        };

        for (const auto face_lid : mesh.faces(cell_lid))
        {
            const auto is_interior = mesh.is_interior_face(face_lid);
            local_ordinal_type other{};
            if (is_interior)
            {
                other = mesh.opposite_or_periodic_neighbor_cell(face_lid, cell_lid);
                row_values.ensure(other);
            }

            const auto owner_flux = face_fluxes.local_value(face_lid);
            const auto outward_flux = mesh.owner_cell(face_lid) == cell_lid ? owner_flux : -owner_flux;
            if (outward_flux >= scalar_type{})
            {
                add_matrix_entry(row_values, cell_lid, outward_flux * cell_advection);
            }
            else if (is_interior)
            {
                add_matrix_entry(row_values, other, outward_flux * advection_value(other));
            }
            else if (mesh.is_boundary_face(face_lid))
            {
                const auto index = packed_face_local_id(mesh, face_lid);
                if (index < locations->size() && (*locations)[index].active)
                {
                    const auto location = (*locations)[index];
                    rhs_value -=
                        outward_flux * cell_advection * boundary_value(location.batch_id, location.in_batch_id);
                }
            }

            if (is_interior)
            {
                const auto face_diffusivity = face_coefficient_value(mesh, face_lid, cell_lid, other,
                    diffusivity_value(cell_lid), diffusivity_value(other), coefficient_interpolation);
                if (face_diffusivity <= scalar_type{})
                {
                    continue;
                }
                const auto coefficient =
                    interior_diffusion_coefficient(mesh, face_lid, cell_lid, other, face_diffusivity);
                add_matrix_entry(row_values, cell_lid, coefficient);
                add_matrix_entry(row_values, other, -coefficient);

                const auto tangential_area = non_orthogonal_area_vector(
                    mesh.face_area_vector_outward(face_lid, cell_lid), mesh.cell_center_vector(face_lid, cell_lid));
                if (mesh.is_owned_cell(other) && static_cast<size_t>(other) < gradient_stencils.size())
                {
                    add_non_orthogonal_stencil(cell_lid, scalar_type{0.5}, face_diffusivity, tangential_area);
                    add_non_orthogonal_stencil(other, scalar_type{0.5}, face_diffusivity, tangential_area);
                }
                else if (weights.implicit > scalar_type{})
                {
                    add_non_orthogonal_stencil(cell_lid, scalar_type{0.5}, face_diffusivity, tangential_area);
                    if (partition_gradients == nullptr)
                    {
                        throw std::logic_error("Partition-face implicit weighted scalar "
                                               "reconstruction requires synchronized gradients.");
                    }
                    rhs_value += weights.implicit * face_diffusivity * scalar_type{0.5} *
                                 partition_gradients->local_value(other).dot(tangential_area);
                }
                continue;
            }

            if (!mesh.is_boundary_face(face_lid))
            {
                continue;
            }
            const auto index = packed_face_local_id(mesh, face_lid);
            if (index >= locations->size() || !(*locations)[index].active)
            {
                continue;
            }
            const auto location = (*locations)[index];
            const auto condition = boundary_condition(location.batch_id, location.in_batch_id);
            const auto face_diffusivity =
                boundary_face_diffusivity(location.batch_id, location.in_batch_id, diffusivity_value(cell_lid));
            if (condition.type == BoundaryConditionType::Dirichlet)
            {
                const auto coefficient = boundary_diffusion_coefficient(mesh, face_lid, cell_lid, face_diffusivity);
                if (coefficient > scalar_type{})
                {
                    add_matrix_entry(row_values, cell_lid, coefficient);
                    rhs_value += coefficient * boundary_value(location.batch_id, location.in_batch_id);
                }
                const auto tangential_area =
                    non_orthogonal_area_vector(mesh.face_area_vector_outward(face_lid, cell_lid),
                        mesh.face_centroid(face_lid) - mesh.cell_centroid(cell_lid));
                add_non_orthogonal_stencil(cell_lid, scalar_type{1}, face_diffusivity, tangential_area);
            }
            else if (condition.type == BoundaryConditionType::Neumann)
            {
                rhs_value += face_diffusivity * condition.value * static_cast<scalar_type>(mesh.face_area(face_lid));
            }
            else if (condition.type == BoundaryConditionType::Robin)
            {
                throw std::runtime_error(prefix + " does not implement Robin boundaries.");
            }
        }

        if (fixed_values[owned])
        {
            row_values.fill(scalar_type{});
            row_values.set(cell_lid, scalar_type{1});
            rhs_value = *fixed_values[owned];
        }
        rows.push_back(capture_stored_transport_row<Pack>(row_values));
        rhs->replaceLocalValue(cell_lid, rhs_value);
    }

    if (correction_field != nullptr && weights.explicit_ > scalar_type{})
    {
        add_stored_variable_scalar_explicit_non_orthogonal_correction<Pack>(*correction_field, boundary_condition,
            diffusivity_value, boundary_face_diffusivity, *rhs, weights.explicit_, gradient_stencils, *locations,
            coefficient_interpolation);
    }
    for (size_t owned = 0; owned < fixed_values.size(); ++owned)
    {
        if (fixed_values[owned])
        {
            rhs->replaceLocalValue(static_cast<local_ordinal_type>(owned), *fixed_values[owned]);
        }
    }

    const auto prepared = prepare_stored_transport_matrix<Pack>(mesh, std::move(cached_matrix), 32, rows);
    const auto& matrix = prepared.matrix;
    for (size_t row = 0; row < rows.size(); ++row)
    {
        add_stored_transport_values<Pack>(prepared, static_cast<local_ordinal_type>(row), rows[row]);
    }
    matrix->fillComplete();
    return {matrix, rhs};
}

/** Assemble weighted scalar transport from mapped coefficient fields. */
template<TpetraTypePack Pack, class MeshType, class BoundaryCondition, class BoundaryValue, class Source,
    class BoundaryCache, class GeometryCache>
TransportSystem<Pack> stored_weighted_scalar_transport_system(const ScalarCellFieldStored<Pack, MeshType>& old_values,
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
    FaceCoefficientInterpolation coefficient_interpolation)
{
    const auto incompatible_fields = old_values.mesh_ptr().get() != face_fluxes.mesh_ptr().get() ||
                                             old_values.mesh_ptr().get() != storage_weight.mesh_ptr().get() ||
                                             old_values.mesh_ptr().get() != advection_weight.mesh_ptr().get() ||
                                             old_values.mesh_ptr().get() != diffusivity.mesh_ptr().get()
                                         ? 1
                                         : 0;
    auto storage = [&](typename Pack::local_ordinal_type cell_lid) { return storage_weight.local_value(cell_lid); };
    auto advection = [&](typename Pack::local_ordinal_type cell_lid) { return advection_weight.local_value(cell_lid); };
    auto diffusion = [&](typename Pack::local_ordinal_type cell_lid) { return diffusivity.local_value(cell_lid); };
    return stored_weighted_scalar_transport_system_impl<Pack>(old_values, face_fluxes, time_step, storage, advection,
        diffusion, std::move(boundary_condition), std::move(boundary_value), std::move(source), treatment,
        correction_field, std::move(cached_matrix), std::move(implicit_sink), std::move(fixed_cell_value),
        boundary_diffusivity, geometry_cache, coefficient_interpolation, incompatible_fields,
        "weighted_scalar_transport_system");
}

/** Assemble mapped conservative physical temperature transport. */
template<TpetraTypePack Pack, class MeshType, class BoundaryCondition, class BoundaryValue, class Source,
    class BoundaryCache, class GeometryCache>
TransportSystem<Pack> stored_physical_temperature_transport_system(
    const ScalarCellFieldStored<Pack, MeshType>& old_temperature,
    const ScalarFaceFieldStored<Pack, MeshType>& face_fluxes, typename Pack::scalar_type time_step,
    const ScalarCellFieldStored<Pack, MeshType>& density,
    const ScalarCellFieldStored<Pack, MeshType>& specific_heat_capacity,
    const ScalarCellFieldStored<Pack, MeshType>& thermal_conductivity, BoundaryCondition boundary_condition,
    BoundaryValue boundary_value, Source power_density, NonOrthogonalTreatment treatment,
    const ScalarCellFieldStored<Pack, MeshType>* correction_field,
    Teuchos::RCP<typename Pack::matrix_type> cached_matrix, const BoundaryCache* boundary_thermal_conductivity,
    const GeometryCache* geometry_cache, FaceCoefficientInterpolation coefficient_interpolation)
{
    const auto incompatible_fields =
        old_temperature.mesh_ptr().get() != face_fluxes.mesh_ptr().get() ||
                old_temperature.mesh_ptr().get() != density.mesh_ptr().get() ||
                old_temperature.mesh_ptr().get() != specific_heat_capacity.mesh_ptr().get() ||
                old_temperature.mesh_ptr().get() != thermal_conductivity.mesh_ptr().get()
            ? 1
            : 0;
    auto capacity = [&](typename Pack::local_ordinal_type cell_lid)
    { return density.local_value(cell_lid) * specific_heat_capacity.local_value(cell_lid); };
    auto conductivity = [&](typename Pack::local_ordinal_type cell_lid)
    { return thermal_conductivity.local_value(cell_lid); };
    return stored_weighted_scalar_transport_system_impl<Pack>(old_temperature, face_fluxes, time_step, capacity,
        capacity, conductivity, std::move(boundary_condition), std::move(boundary_value), std::move(power_density),
        treatment, correction_field, std::move(cached_matrix), {}, {}, boundary_thermal_conductivity, geometry_cache,
        coefficient_interpolation, incompatible_fields, "physical_temperature_transport_system");
}

template<TpetraTypePack Pack, class MeshType, class Stencils, class BoundaryLocations, class BoundaryDiffusion>
void add_stored_explicit_non_orthogonal_correction(const VectorCellFieldStored<Pack, MeshType>& correction_field,
    typename Pack::scalar_type diffusivity, typename Pack::multi_vector_type& rhs,
    typename Pack::scalar_type correction_weight, BoundaryDiffusion boundary_diffusion, const Stencils& stencils,
    const BoundaryLocations& boundary_locations)
{
    using scalar_type = typename Pack::scalar_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;
    if (diffusivity <= scalar_type{} || correction_weight == scalar_type{})
    {
        return;
    }

    TensorCellFieldStored<Pack, MeshType> gradients(
        correction_field.mesh_ptr(), "stored_vector_non_orthogonal_gradient");
    evaluate_stored_vector_gradients(correction_field, stencils, gradients);
    const auto& mesh = correction_field.mesh();

    for (size_t owned = 0; owned < mesh.num_owned_cells(); ++owned)
    {
        const auto cell_lid = static_cast<local_ordinal_type>(owned);
        for (const auto face_lid : mesh.faces(cell_lid))
        {
            auto gradient = gradients.local_value(cell_lid);
            typename MeshType::Vec3 direction{};
            if (mesh.is_interior_face(face_lid))
            {
                const auto other = mesh.opposite_or_periodic_neighbor_cell(face_lid, cell_lid);
                const auto other_gradient = gradients.local_value(other);
                for (size_t component = 0; component < 3; ++component)
                {
                    gradient[component] = (gradient[component] + other_gradient[component]) / scalar_type{2};
                }
                direction = mesh.cell_center_vector(face_lid, cell_lid);
            }
            else if (mesh.is_boundary_face(face_lid))
            {
                const auto index = packed_face_local_id(mesh, face_lid);
                if (index >= boundary_locations.size() || !boundary_locations[index].active)
                {
                    continue;
                }
                const auto location = boundary_locations[index];
                if (!boundary_diffusion(location.batch_id, location.in_batch_id))
                {
                    continue;
                }
                direction = mesh.face_centroid(face_lid) - mesh.cell_centroid(cell_lid);
            }
            else
            {
                continue;
            }

            const auto tangential_area =
                non_orthogonal_area_vector(mesh.face_area_vector_outward(face_lid, cell_lid), direction);
            for (size_t component = 0; component < 3; ++component)
            {
                rhs.sumIntoLocalValue(
                    cell_lid, component, correction_weight * diffusivity * gradient[component].dot(tangential_area));
            }
        }
    }
}

template<TpetraTypePack Pack, class MeshType, class BoundaryValue, class Source, class BoundaryDiffusion,
    class GeometryCache>
VectorTransportSystem<Pack> stored_vector_transport_system(const VectorCellFieldStored<Pack, MeshType>& old_values,
    const ScalarFaceFieldStored<Pack, MeshType>& face_fluxes, typename Pack::scalar_type time_step,
    typename Pack::scalar_type diffusivity, BoundaryValue boundary_value, Source source,
    NonOrthogonalTreatment treatment, const VectorCellFieldStored<Pack, MeshType>* correction_field,
    Teuchos::RCP<typename Pack::matrix_type> cached_matrix, BoundaryDiffusion boundary_diffusion,
    const GeometryCache* geometry_cache)
{
    using scalar_type = typename Pack::scalar_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;
    constexpr size_t components = 3;

    const auto& mesh = old_values.mesh();
    const auto weights = validate_stored_non_orthogonal_selection<Pack>(
        mesh, treatment, correction_field, "non_orthogonal_transport_system");
    int invalid_geometry_cache = 0;
    if (geometry_cache != nullptr)
    {
        try
        {
            geometry_cache->require_mesh(mesh);
        }
        catch (const std::invalid_argument&)
        {
            invalid_geometry_cache = 1;
        }
    }
    const auto validation_state = reduce_stored_validation_state<Pack>(
        mesh, std::array<int, 4>{old_values.mesh_ptr().get() != face_fluxes.mesh_ptr().get() ? 1 : 0,
                  !std::isfinite(time_step) || time_step <= scalar_type{} ? 1 : 0,
                  !std::isfinite(diffusivity) || diffusivity < scalar_type{} ? 1 : 0, invalid_geometry_cache});
    if (validation_state[0] != 0)
    {
        throw std::invalid_argument("non_orthogonal_transport_system requires fields on the same "
                                    "mesh.");
    }
    if (validation_state[1] != 0)
    {
        throw std::invalid_argument("non_orthogonal_transport_system requires a finite positive time step.");
    }
    if (validation_state[2] != 0)
    {
        throw std::invalid_argument("non_orthogonal_transport_system requires finite non-negative "
                                    "diffusivity.");
    }
    if (validation_state[3] != 0)
    {
        throw std::invalid_argument("non_orthogonal_transport_system received a geometry cache on the wrong mesh.");
    }

    std::vector<LeastSquaresGradientStencil<MeshType>> local_stencils;
    std::vector<BoundaryFaceLocation<MeshType>> local_locations;
    const std::vector<LeastSquaresGradientStencil<MeshType>>* stencils{};
    const std::vector<BoundaryFaceLocation<MeshType>>* locations{};
    select_stored_interior_geometry(mesh, geometry_cache, local_stencils, local_locations, stencils, locations);

    std::unique_ptr<TensorCellFieldStored<Pack, MeshType>> partition_gradients;
    if (weights.implicit > scalar_type{})
    {
        partition_gradients = std::make_unique<TensorCellFieldStored<Pack, MeshType>>(
            old_values.mesh_ptr(), "stored_partition_vector_non_orthogonal_gradient");
        const auto& lagged_field = correction_field == nullptr ? old_values : *correction_field;
        evaluate_stored_vector_gradients(lagged_field, *stencils, *partition_gradients);
    }

    auto rhs = Teuchos::rcp(new typename Pack::multi_vector_type(mesh.owned_cell_map(), components, true));
    FlatMatrixRow<local_ordinal_type, scalar_type> row_values(mesh.num_local_cells(), 64);
    std::vector<StoredTransportMatrixRow<Pack>> rows;
    rows.reserve(mesh.num_owned_cells());

    for (size_t owned = 0; owned < mesh.num_owned_cells(); ++owned)
    {
        const auto cell_lid = static_cast<local_ordinal_type>(owned);
        const auto volume = static_cast<scalar_type>(mesh.cell_volume(cell_lid));
        const auto transient = volume / time_step;
        const auto old_value = old_values.local_value(cell_lid);
        const auto source_value = source(cell_lid);
        row_values.clear();
        add_matrix_entry(row_values, cell_lid, transient);
        for (size_t component = 0; component < components; ++component)
        {
            rhs->replaceLocalValue(cell_lid, component,
                transient * old_value.component(component) + volume * source_value.component(component));
        }

        auto add_non_orthogonal_stencil = [&](local_ordinal_type gradient_cell_lid, scalar_type gradient_weight,
                                              const typename MeshType::Vec3& tangential_area)
        {
            if (gradient_weight == scalar_type{} || !mesh.is_owned_cell(gradient_cell_lid) ||
                static_cast<size_t>(gradient_cell_lid) >= stencils->size())
            {
                return;
            }
            const auto scale = -weights.implicit * diffusivity * gradient_weight;
            for (const auto& entry : (*stencils)[static_cast<size_t>(gradient_cell_lid)])
            {
                add_matrix_entry(row_values, entry.cell_lid, scale * entry.coefficient.dot(tangential_area));
            }
        };

        for (const auto face_lid : mesh.faces(cell_lid))
        {
            const auto is_interior = mesh.is_interior_face(face_lid);
            local_ordinal_type other{};
            if (is_interior)
            {
                other = mesh.opposite_or_periodic_neighbor_cell(face_lid, cell_lid);
                row_values.ensure(other);
            }

            const auto owner_flux = face_fluxes.local_value(face_lid);
            const auto outward_flux = mesh.owner_cell(face_lid) == cell_lid ? owner_flux : -owner_flux;
            if (outward_flux >= scalar_type{})
            {
                add_matrix_entry(row_values, cell_lid, outward_flux);
            }
            else if (is_interior)
            {
                add_matrix_entry(row_values, other, outward_flux);
            }
            else if (mesh.is_boundary_face(face_lid))
            {
                const auto index = packed_face_local_id(mesh, face_lid);
                if (index < locations->size() && (*locations)[index].active)
                {
                    const auto location = (*locations)[index];
                    const auto value = boundary_value(location.batch_id, location.in_batch_id);
                    for (size_t component = 0; component < components; ++component)
                    {
                        rhs->sumIntoLocalValue(cell_lid, component, -outward_flux * value.component(component));
                    }
                }
            }

            if (diffusivity <= scalar_type{})
            {
                continue;
            }
            if (is_interior)
            {
                const auto coefficient = interior_diffusion_coefficient(mesh, face_lid, cell_lid, other, diffusivity);
                add_matrix_entry(row_values, cell_lid, coefficient);
                add_matrix_entry(row_values, other, -coefficient);
                const auto tangential_area = non_orthogonal_area_vector(
                    mesh.face_area_vector_outward(face_lid, cell_lid), mesh.cell_center_vector(face_lid, cell_lid));
                if (mesh.is_owned_cell(other) && static_cast<size_t>(other) < stencils->size())
                {
                    add_non_orthogonal_stencil(cell_lid, scalar_type{0.5}, tangential_area);
                    add_non_orthogonal_stencil(other, scalar_type{0.5}, tangential_area);
                }
                else if (weights.implicit > scalar_type{})
                {
                    add_non_orthogonal_stencil(cell_lid, scalar_type{0.5}, tangential_area);
                    if (partition_gradients == nullptr)
                    {
                        throw std::logic_error("Partition-face implicit vector reconstruction "
                                               "requires synchronized remote gradients.");
                    }
                    const auto remote_gradient = partition_gradients->local_value(other);
                    for (size_t component = 0; component < components; ++component)
                    {
                        rhs->sumIntoLocalValue(cell_lid, component,
                            weights.implicit * diffusivity * scalar_type{0.5} *
                                remote_gradient[component].dot(tangential_area));
                    }
                }
                continue;
            }

            if (!mesh.is_boundary_face(face_lid))
            {
                continue;
            }
            const auto index = packed_face_local_id(mesh, face_lid);
            if (index >= locations->size() || !(*locations)[index].active)
            {
                continue;
            }
            const auto location = (*locations)[index];
            if (!boundary_diffusion(location.batch_id, location.in_batch_id))
            {
                continue;
            }
            const auto value = boundary_value(location.batch_id, location.in_batch_id);
            const auto coefficient = boundary_diffusion_coefficient(mesh, face_lid, cell_lid, diffusivity);
            add_matrix_entry(row_values, cell_lid, coefficient);
            for (size_t component = 0; component < components; ++component)
            {
                rhs->sumIntoLocalValue(cell_lid, component, coefficient * value.component(component));
            }
            const auto tangential_area = non_orthogonal_area_vector(mesh.face_area_vector_outward(face_lid, cell_lid),
                mesh.face_centroid(face_lid) - mesh.cell_centroid(cell_lid));
            add_non_orthogonal_stencil(cell_lid, scalar_type{1}, tangential_area);
        }

        rows.push_back(capture_stored_transport_row<Pack>(row_values));
    }

    if (correction_field != nullptr && weights.explicit_ > scalar_type{})
    {
        add_stored_explicit_non_orthogonal_correction<Pack>(
            *correction_field, diffusivity, *rhs, weights.explicit_, boundary_diffusion, *stencils, *locations);
    }

    const auto prepared = prepare_stored_transport_matrix<Pack>(mesh, std::move(cached_matrix), 32, rows);
    const auto& matrix = prepared.matrix;
    for (size_t row = 0; row < rows.size(); ++row)
    {
        add_stored_transport_values<Pack>(prepared, static_cast<local_ordinal_type>(row), rows[row]);
    }
    matrix->fillComplete();
    return {matrix, rhs};
}

template<TpetraTypePack Pack, class MeshType, class Stencils, class BoundaryLocations, class BoundaryDiffusion,
    class BoundaryCoefficient>
void add_stored_variable_explicit_non_orthogonal_correction(
    const VectorCellFieldStored<Pack, MeshType>& correction_field,
    const ScalarCellFieldStored<Pack, MeshType>& dynamic_viscosity, typename Pack::multi_vector_type& rhs,
    typename Pack::scalar_type correction_weight, BoundaryDiffusion boundary_diffusion,
    BoundaryCoefficient boundary_viscosity, const Stencils& stencils, const BoundaryLocations& boundary_locations,
    FaceCoefficientInterpolation coefficient_interpolation)
{
    using scalar_type = typename Pack::scalar_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;
    if (correction_weight == scalar_type{})
    {
        return;
    }

    TensorCellFieldStored<Pack, MeshType> gradients(
        correction_field.mesh_ptr(), "stored_variable_vector_non_orthogonal_gradient");
    evaluate_stored_vector_gradients(correction_field, stencils, gradients);
    const auto& mesh = correction_field.mesh();

    for (size_t owned = 0; owned < mesh.num_owned_cells(); ++owned)
    {
        const auto cell_lid = static_cast<local_ordinal_type>(owned);
        for (const auto face_lid : mesh.faces(cell_lid))
        {
            auto gradient = gradients.local_value(cell_lid);
            auto face_viscosity = dynamic_viscosity.local_value(cell_lid);
            typename MeshType::Vec3 direction{};
            if (mesh.is_interior_face(face_lid))
            {
                const auto other = mesh.opposite_or_periodic_neighbor_cell(face_lid, cell_lid);
                const auto other_gradient = gradients.local_value(other);
                for (size_t component = 0; component < 3; ++component)
                {
                    gradient[component] = (gradient[component] + other_gradient[component]) / scalar_type{2};
                }
                face_viscosity =
                    face_coefficient_value(mesh, face_lid, cell_lid, other, dynamic_viscosity.local_value(cell_lid),
                        dynamic_viscosity.local_value(other), coefficient_interpolation);
                direction = mesh.cell_center_vector(face_lid, cell_lid);
            }
            else if (mesh.is_boundary_face(face_lid))
            {
                const auto index = packed_face_local_id(mesh, face_lid);
                if (index >= boundary_locations.size() || !boundary_locations[index].active)
                {
                    continue;
                }
                const auto location = boundary_locations[index];
                if (!boundary_diffusion(location.batch_id, location.in_batch_id))
                {
                    continue;
                }
                face_viscosity = boundary_viscosity(location.batch_id, location.in_batch_id, face_viscosity);
                direction = mesh.face_centroid(face_lid) - mesh.cell_centroid(cell_lid);
            }
            else
            {
                continue;
            }

            const auto tangential_area =
                non_orthogonal_area_vector(mesh.face_area_vector_outward(face_lid, cell_lid), direction);
            for (size_t component = 0; component < 3; ++component)
            {
                rhs.sumIntoLocalValue(
                    cell_lid, component, correction_weight * face_viscosity * gradient[component].dot(tangential_area));
            }
        }
    }
}

template<TpetraTypePack Pack, class MeshType, class Stencils, class BoundaryLocations, class BoundaryDiffusion,
    class BoundaryCoefficient>
void add_stored_deviatoric_transpose_gradient_stress(const VectorCellFieldStored<Pack, MeshType>& old_velocity,
    const ScalarCellFieldStored<Pack, MeshType>& dynamic_viscosity, typename Pack::scalar_type reference_density,
    typename Pack::multi_vector_type& rhs, BoundaryDiffusion boundary_stress, BoundaryCoefficient boundary_viscosity,
    const Stencils& stencils, const BoundaryLocations& boundary_locations,
    FaceCoefficientInterpolation coefficient_interpolation)
{
    using scalar_type = typename Pack::scalar_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;
    TensorCellFieldStored<Pack, MeshType> gradients(
        old_velocity.mesh_ptr(), "stored_transpose_gradient_stress_velocity_gradient");
    evaluate_stored_vector_gradients(old_velocity, stencils, gradients);
    const auto& mesh = old_velocity.mesh();

    for (size_t owned = 0; owned < mesh.num_owned_cells(); ++owned)
    {
        const auto cell_lid = static_cast<local_ordinal_type>(owned);
        for (const auto face_lid : mesh.faces(cell_lid))
        {
            auto face_gradient = gradients.local_value(cell_lid);
            auto face_viscosity = dynamic_viscosity.local_value(cell_lid);
            if (mesh.is_interior_face(face_lid))
            {
                const auto other = mesh.opposite_or_periodic_neighbor_cell(face_lid, cell_lid);
                const auto other_gradient = gradients.local_value(other);
                for (size_t component = 0; component < 3; ++component)
                {
                    face_gradient[component] = (face_gradient[component] + other_gradient[component]) / scalar_type{2};
                }
                face_viscosity =
                    face_coefficient_value(mesh, face_lid, cell_lid, other, dynamic_viscosity.local_value(cell_lid),
                        dynamic_viscosity.local_value(other), coefficient_interpolation);
            }
            else
            {
                if (!mesh.is_boundary_face(face_lid))
                {
                    continue;
                }
                const auto index = packed_face_local_id(mesh, face_lid);
                if (index >= boundary_locations.size() || !boundary_locations[index].active)
                {
                    continue;
                }
                const auto location = boundary_locations[index];
                if (!boundary_stress(location.batch_id, location.in_batch_id))
                {
                    continue;
                }
                face_viscosity = boundary_viscosity(location.batch_id, location.in_batch_id, face_viscosity);
            }

            if (!std::isfinite(face_viscosity) || face_viscosity < scalar_type{})
            {
                throw std::invalid_argument("physical_momentum_transport_system requires finite "
                                            "non-negative face viscosity.");
            }
            const auto area = mesh.face_area_vector_outward(face_lid, cell_lid);
            const auto divergence = face_gradient[0].x + face_gradient[1].y + face_gradient[2].z;
            const auto isotropic_scale = scalar_type{2.0 / 3.0} * divergence;
            const typename MeshType::Vec3 traction{face_gradient[0].x * area.x + face_gradient[1].x * area.y +
                                                       face_gradient[2].x * area.z - isotropic_scale * area.x,
                face_gradient[0].y * area.x + face_gradient[1].y * area.y + face_gradient[2].y * area.z -
                    isotropic_scale * area.y,
                face_gradient[0].z * area.x + face_gradient[1].z * area.y + face_gradient[2].z * area.z -
                    isotropic_scale * area.z};
            const auto scale = face_viscosity / reference_density;
            for (size_t component = 0; component < 3; ++component)
            {
                rhs.sumIntoLocalValue(cell_lid, component, scale * traction.component(component));
            }
        }
    }
}

/**
 * @brief Assemble mapped momentum transport with cell viscosity storage.
 *
 * This is the mapped-mesh counterpart of physical_momentum_transport_system.
 * It retains transient, upwind-advection, and orthogonal viscous terms while
 * avoiding a legacy Mesh materialization.
 */
template<TpetraTypePack Pack, class MeshType, class BoundaryValue, class Source, class BoundaryDiffusion,
    class BoundaryCache, class GeometryCache>
VectorTransportSystem<Pack> stored_physical_momentum_transport_system(
    const VectorCellFieldStored<Pack, MeshType>& old_velocity, const ScalarFaceFieldStored<Pack, MeshType>& face_fluxes,
    typename Pack::scalar_type time_step, const ScalarCellFieldStored<Pack, MeshType>& dynamic_viscosity,
    typename Pack::scalar_type reference_density, BoundaryValue boundary_value, Source source,
    NonOrthogonalTreatment treatment, const VectorCellFieldStored<Pack, MeshType>* correction_field,
    Teuchos::RCP<typename Pack::matrix_type> cached_matrix, BoundaryDiffusion boundary_diffusion,
    const BoundaryCache* boundary_dynamic_viscosity, const GeometryCache* geometry_cache,
    FaceCoefficientInterpolation coefficient_interpolation)
{
    using scalar_type = typename Pack::scalar_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;
    constexpr size_t components = 3;

    const auto& mesh = old_velocity.mesh();
    const auto weights = validate_stored_non_orthogonal_selection<Pack>(
        mesh, treatment, correction_field, "physical_momentum_transport_system");

    const int incompatible_fields = old_velocity.mesh_ptr().get() != face_fluxes.mesh_ptr().get() ||
                                            old_velocity.mesh_ptr().get() != dynamic_viscosity.mesh_ptr().get()
                                        ? 1
                                        : 0;
    int invalid_boundary_cache = 0;
    try
    {
        validate_boundary_coefficient_cache<Pack>(
            mesh, boundary_dynamic_viscosity, "physical_momentum_transport_system");
    }
    catch (const std::invalid_argument&)
    {
        invalid_boundary_cache = 1;
    }
    int invalid_geometry_cache = 0;
    if (geometry_cache != nullptr)
    {
        try
        {
            geometry_cache->require_mesh(mesh);
        }
        catch (const std::invalid_argument&)
        {
            invalid_geometry_cache = 1;
        }
    }
    int invalid_viscosity = 0;
    if (incompatible_fields == 0)
    {
        for (size_t local = 0; local < mesh.num_local_cells(); ++local)
        {
            const auto cell_lid = static_cast<local_ordinal_type>(local);
            const auto viscosity = dynamic_viscosity.local_value(cell_lid);
            invalid_viscosity = invalid_viscosity || !std::isfinite(viscosity) || viscosity < scalar_type{};
        }
    }
    int interpolation_state = 2;
    switch (coefficient_interpolation)
    {
        case FaceCoefficientInterpolation::Harmonic:
            interpolation_state = 0;
            break;
        case FaceCoefficientInterpolation::Linear:
            interpolation_state = 1;
            break;
    }
    const auto validation_state = reduce_stored_validation_state<Pack>(
        mesh, std::array<int, 7>{incompatible_fields,
                  !std::isfinite(time_step) || time_step <= scalar_type{} || !std::isfinite(reference_density) ||
                          reference_density <= scalar_type{}
                      ? 1
                      : 0,
                  invalid_boundary_cache, invalid_geometry_cache, invalid_viscosity, interpolation_state,
                  -interpolation_state});
    if (validation_state[0] != 0)
    {
        throw std::invalid_argument("physical_momentum_transport_system requires fields on the "
                                    "same mesh.");
    }
    if (validation_state[1] != 0)
    {
        throw std::invalid_argument("physical_momentum_transport_system requires a finite positive "
                                    "time step and reference density.");
    }
    if (validation_state[2] != 0)
    {
        throw std::invalid_argument(
            "physical_momentum_transport_system received an invalid boundary-coefficient cache.");
    }
    if (validation_state[3] != 0)
    {
        throw std::invalid_argument("physical_momentum_transport_system received a geometry cache on the wrong mesh.");
    }
    if (validation_state[4] != 0)
    {
        throw std::invalid_argument("physical_momentum_transport_system requires finite "
                                    "non-negative dynamic viscosity.");
    }
    if (validation_state[5] == 2 || -validation_state[6] != validation_state[5])
    {
        throw std::invalid_argument("physical_momentum_transport_system requires every rank to use the same valid "
                                    "face-coefficient interpolation.");
    }

    auto boundary_viscosity = [&](int batch_id, size_t in_batch_id, scalar_type owner_value)
    { return boundary_coefficient<Pack>(boundary_dynamic_viscosity, batch_id, in_batch_id, owner_value); };

    std::vector<VectorAffineLeastSquaresGradientStencil<MeshType>> gradient_stencils;
    std::vector<BoundaryFaceLocation<MeshType>> local_locations;
    const std::vector<BoundaryFaceLocation<MeshType>>* locations{};
    select_stored_affine_geometry(mesh, boundary_value, geometry_cache, gradient_stencils, local_locations, locations);

    std::unique_ptr<TensorCellFieldStored<Pack, MeshType>> partition_gradients;
    if (weights.implicit > scalar_type{})
    {
        partition_gradients = std::make_unique<TensorCellFieldStored<Pack, MeshType>>(
            old_velocity.mesh_ptr(), "stored_partition_momentum_non_orthogonal_gradient");
        const auto& lagged_field = correction_field == nullptr ? old_velocity : *correction_field;
        evaluate_stored_vector_gradients(lagged_field, gradient_stencils, *partition_gradients);
    }

    auto rhs = Teuchos::rcp(new typename Pack::multi_vector_type(mesh.owned_cell_map(), components, true));
    FlatMatrixRow<local_ordinal_type, scalar_type> row_values(mesh.num_local_cells(), 64);
    std::vector<StoredTransportMatrixRow<Pack>> rows;
    rows.reserve(mesh.num_owned_cells());

    for (size_t owned = 0; owned < mesh.num_owned_cells(); ++owned)
    {
        const auto cell_lid = static_cast<local_ordinal_type>(owned);
        const auto volume = static_cast<scalar_type>(mesh.cell_volume(cell_lid));
        const auto transient = volume / time_step;
        const auto old_value = old_velocity.local_value(cell_lid);
        const auto source_value = source(cell_lid);
        row_values.clear();
        add_matrix_entry(row_values, cell_lid, transient);
        for (size_t component = 0; component < components; ++component)
        {
            rhs->replaceLocalValue(cell_lid, component,
                transient * old_value.component(component) + volume * source_value.component(component));
        }

        auto add_non_orthogonal_stencil = [&](local_ordinal_type gradient_cell_lid, scalar_type gradient_weight,
                                              scalar_type face_kinematic_viscosity,
                                              const typename MeshType::Vec3& tangential_area)
        {
            if (weights.implicit == scalar_type{} || gradient_weight == scalar_type{} ||
                !mesh.is_owned_cell(gradient_cell_lid) ||
                static_cast<size_t>(gradient_cell_lid) >= gradient_stencils.size())
            {
                return;
            }
            const auto scale = -weights.implicit * face_kinematic_viscosity * gradient_weight;
            const auto& stencil = gradient_stencils[static_cast<size_t>(gradient_cell_lid)];
            for (size_t component = 0; component < components; ++component)
            {
                rhs->sumIntoLocalValue(cell_lid, component, -scale * stencil.constants[component].dot(tangential_area));
            }
            for (const auto& entry : stencil.entries)
            {
                add_matrix_entry(row_values, entry.cell_lid, scale * entry.coefficient.dot(tangential_area));
            }
        };

        for (const auto face_lid : mesh.faces(cell_lid))
        {
            const auto is_interior = mesh.is_interior_face(face_lid);
            local_ordinal_type other{};
            if (is_interior)
            {
                other = mesh.opposite_or_periodic_neighbor_cell(face_lid, cell_lid);
                row_values.ensure(other);
            }

            const auto owner_flux = face_fluxes.local_value(face_lid);
            const auto outward_flux = mesh.owner_cell(face_lid) == cell_lid ? owner_flux : -owner_flux;
            if (outward_flux >= scalar_type{})
            {
                add_matrix_entry(row_values, cell_lid, outward_flux);
            }
            else if (is_interior)
            {
                add_matrix_entry(row_values, other, outward_flux);
            }
            else if (mesh.is_boundary_face(face_lid))
            {
                const auto index = packed_face_local_id(mesh, face_lid);
                if (index < locations->size() && (*locations)[index].active)
                {
                    const auto location = (*locations)[index];
                    const auto value = boundary_value(location.batch_id, location.in_batch_id);
                    for (size_t component = 0; component < components; ++component)
                    {
                        rhs->sumIntoLocalValue(cell_lid, component, -outward_flux * value.component(component));
                    }
                }
            }

            if (is_interior)
            {
                const auto face_kinematic_viscosity =
                    face_coefficient_value(mesh, face_lid, cell_lid, other, dynamic_viscosity.local_value(cell_lid),
                        dynamic_viscosity.local_value(other), coefficient_interpolation) /
                    reference_density;
                if (face_kinematic_viscosity <= scalar_type{})
                {
                    continue;
                }
                const auto coefficient =
                    interior_diffusion_coefficient(mesh, face_lid, cell_lid, other, face_kinematic_viscosity);
                add_matrix_entry(row_values, cell_lid, coefficient);
                add_matrix_entry(row_values, other, -coefficient);
                const auto tangential_area = non_orthogonal_area_vector(
                    mesh.face_area_vector_outward(face_lid, cell_lid), mesh.cell_center_vector(face_lid, cell_lid));
                if (mesh.is_owned_cell(other) && static_cast<size_t>(other) < gradient_stencils.size())
                {
                    add_non_orthogonal_stencil(cell_lid, scalar_type{0.5}, face_kinematic_viscosity, tangential_area);
                    add_non_orthogonal_stencil(other, scalar_type{0.5}, face_kinematic_viscosity, tangential_area);
                }
                else if (weights.implicit > scalar_type{})
                {
                    add_non_orthogonal_stencil(cell_lid, scalar_type{0.5}, face_kinematic_viscosity, tangential_area);
                    if (partition_gradients == nullptr)
                    {
                        throw std::logic_error("Partition-face implicit momentum reconstruction "
                                               "requires synchronized remote gradients.");
                    }
                    const auto remote_gradient = partition_gradients->local_value(other);
                    for (size_t component = 0; component < components; ++component)
                    {
                        rhs->sumIntoLocalValue(cell_lid, component,
                            weights.implicit * face_kinematic_viscosity * scalar_type{0.5} *
                                remote_gradient[component].dot(tangential_area));
                    }
                }
                continue;
            }

            if (!mesh.is_boundary_face(face_lid))
            {
                continue;
            }
            const auto index = packed_face_local_id(mesh, face_lid);
            if (index >= locations->size() || !(*locations)[index].active)
            {
                continue;
            }
            const auto location = (*locations)[index];
            if (!boundary_diffusion(location.batch_id, location.in_batch_id))
            {
                continue;
            }
            const auto face_kinematic_viscosity =
                boundary_viscosity(location.batch_id, location.in_batch_id, dynamic_viscosity.local_value(cell_lid)) /
                reference_density;
            const auto coefficient = boundary_diffusion_coefficient(mesh, face_lid, cell_lid, face_kinematic_viscosity);
            if (coefficient > scalar_type{})
            {
                add_matrix_entry(row_values, cell_lid, coefficient);
                const auto value = boundary_value(location.batch_id, location.in_batch_id);
                for (size_t component = 0; component < components; ++component)
                {
                    rhs->sumIntoLocalValue(cell_lid, component, coefficient * value.component(component));
                }
            }
            const auto tangential_area = non_orthogonal_area_vector(mesh.face_area_vector_outward(face_lid, cell_lid),
                mesh.face_centroid(face_lid) - mesh.cell_centroid(cell_lid));
            add_non_orthogonal_stencil(cell_lid, scalar_type{1}, face_kinematic_viscosity, tangential_area);
        }

        rows.push_back(capture_stored_transport_row<Pack>(row_values));
    }

    if (correction_field != nullptr && weights.explicit_ > scalar_type{})
    {
        add_stored_variable_explicit_non_orthogonal_correction<Pack>(*correction_field, dynamic_viscosity, *rhs,
            weights.explicit_ / reference_density, boundary_diffusion, boundary_viscosity, gradient_stencils,
            *locations, coefficient_interpolation);
    }

    add_stored_deviatoric_transpose_gradient_stress<Pack>(old_velocity, dynamic_viscosity, reference_density, *rhs,
        boundary_diffusion, boundary_viscosity, gradient_stencils, *locations, coefficient_interpolation);

    const auto prepared = prepare_stored_transport_matrix<Pack>(mesh, std::move(cached_matrix), 32, rows);
    const auto& matrix = prepared.matrix;
    for (size_t row = 0; row < rows.size(); ++row)
    {
        add_stored_transport_values<Pack>(prepared, static_cast<local_ordinal_type>(row), rows[row]);
    }
    matrix->fillComplete();
    return {matrix, rhs};
}

} // namespace detail
} // namespace SimpleFluid::FVM
