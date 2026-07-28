/**
 * @file FVM/TransportSystem.tcc
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Out-of-line template implementations for compiled FVM assembly.
 * @version 0.1
 * @date 2026-07-29
 *
 * @copyright Copyright (c) 2026
 *
 */
#include "FVM/FieldViewAccess.hh"
#include "FVM/NonOrthogonalCorrection.hh"
#include "FVM/TransportSystem.hh"

#include <Teuchos_CommHelpers.hpp>

#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace SimpleFluid::FVM
{
namespace detail
{

template<TpetraTypePack Pack, class Field>
NonOrthogonalTransportWeights<typename Pack::scalar_type> validate_non_orthogonal_transport_selection(
    const Mesh<Pack>& mesh, NonOrthogonalTreatment treatment, const Field* correction_field, std::string_view context)
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

    const int correction_state = correction_field == nullptr ? 0 : (&correction_field->mesh() == &mesh ? 1 : 2);
    const int local_state[] = {treatment_state, -treatment_state, correction_state, -correction_state};
    int maximum_state[4] = {local_state[0], local_state[1], local_state[2], local_state[3]};
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

template<TpetraTypePack Pack, class MeshType>
PreparedTransportMatrix<Pack> prepare_transport_matrix(
    const MeshType& mesh, Teuchos::RCP<typename Pack::matrix_type> cached_matrix, size_t entries_per_row)
{
    using matrix_type = typename Pack::matrix_type;

    if (cached_matrix.is_null())
    {
        return {Teuchos::rcp(new matrix_type(mesh.owned_cell_map(), mesh.overlap_cell_map(), entries_per_row)), false};
    }
    if (!cached_matrix->isFillComplete() || !cached_matrix->getRowMap()->isSameAs(*mesh.owned_cell_map()) ||
        cached_matrix->getColMap().is_null() || !cached_matrix->getColMap()->isSameAs(*mesh.overlap_cell_map()) ||
        !cached_matrix->getDomainMap()->isSameAs(*mesh.owned_cell_map()))
    {
        throw std::invalid_argument("transport_system cached matrix is incompatible with the mesh.");
    }

    cached_matrix->resumeFill();
    cached_matrix->setAllToScalar(typename Pack::scalar_type{});
    return {cached_matrix, true};
}

template<TpetraTypePack Pack>
void add_transport_values(const PreparedTransportMatrix<Pack>& prepared, typename Pack::local_ordinal_type row,
    const Teuchos::ArrayView<const typename Pack::local_ordinal_type>& columns,
    const Teuchos::ArrayView<const typename Pack::scalar_type>& values)
{
    if (!prepared.reused)
    {
        prepared.matrix->insertLocalValues(row, columns, values);
        return;
    }

    const auto updated = prepared.matrix->sumIntoLocalValues(row, columns, values);
    if (updated != static_cast<typename Pack::local_ordinal_type>(columns.size()))
    {
        throw std::invalid_argument("transport_system cached matrix graph is incompatible with the operator.");
    }
}

template<TpetraTypePack Pack>
void add_transport_values(const PreparedTransportMatrix<Pack>& prepared, typename Pack::local_ordinal_type row,
    const FlatMatrixRow<typename Pack::local_ordinal_type, typename Pack::scalar_type>& row_values)
{
    const auto row_size =
        SimpleFluid::detail::checked_size_to_ordinal<Teuchos::Ordinal>(row_values.size(), "transport row entry count");
    add_transport_values<Pack>(prepared, row, Teuchos::arrayView(row_values.column_data(), row_size),
        Teuchos::arrayView(row_values.value_data(), row_size));
}

} // namespace detail

template<class MeshType>
TransportGeometryCache<MeshType>::TransportGeometryCache(const MeshType& mesh)
    : d_mesh(&mesh), d_interior_stencils(detail::least_squares_gradient_stencils(mesh)),
      d_boundary_locations(detail::boundary_face_locations(mesh)),
      d_boundary_geometry(detail::boundary_aware_gradient_geometry(mesh, d_boundary_locations))
{
}

template<class MeshType> void TransportGeometryCache<MeshType>::require_mesh(const MeshType& mesh) const
{
    if (&mesh != d_mesh)
    {
        throw std::invalid_argument("transport geometry cache belongs to another mesh.");
    }
}

template<class MeshType>
const typename TransportGeometryCache<MeshType>::interior_stencils_type&
TransportGeometryCache<MeshType>::interior_stencils() const noexcept
{
    return d_interior_stencils;
}

template<class MeshType>
const typename TransportGeometryCache<MeshType>::boundary_locations_type&
TransportGeometryCache<MeshType>::boundary_locations() const noexcept
{
    return d_boundary_locations;
}

template<class MeshType>
const typename TransportGeometryCache<MeshType>::boundary_geometry_type&
TransportGeometryCache<MeshType>::boundary_geometry() const noexcept
{
    return d_boundary_geometry;
}

template<class MeshType>
std::vector<detail::AffineLeastSquaresGradientStencil<MeshType>>
TransportGeometryCache<MeshType>::scalar_affine_stencils(
    std::function<BoundaryCondition(int, size_t)> boundary_condition,
    std::function<typename MeshType::scalar_type(int, size_t)> boundary_value) const
{
    return detail::materialize_scalar_affine_gradient_stencils<MeshType>(
        d_boundary_geometry, std::move(boundary_condition), std::move(boundary_value));
}

template<class MeshType>
std::vector<detail::VectorAffineLeastSquaresGradientStencil<MeshType>>
TransportGeometryCache<MeshType>::vector_affine_stencils(
    std::function<typename MeshType::Vec3(int, size_t)> boundary_value) const
{
    return detail::materialize_vector_affine_gradient_stencils<MeshType>(
        d_boundary_geometry, std::move(boundary_value));
}

template<TpetraTypePack Pack>
TransportSystem<Pack> transport_system(const CellField<Pack>& old_values, const FaceField<Pack>& face_fluxes,
    typename Pack::scalar_type time_step, typename Pack::scalar_type diffusivity,
    ScalarBoundaryConditionProvider<Pack> boundary_condition, ScalarBoundaryValueProvider<Pack> boundary_value,
    ScalarCellValueProvider<Pack> right_hand_source, Teuchos::RCP<typename Pack::matrix_type> cached_matrix)
{
    using scalar_type = typename Pack::scalar_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;

    const auto& mesh = old_values.mesh();
    if (&face_fluxes.mesh() != &mesh)
    {
        throw std::invalid_argument("transport_system requires face fluxes on the old-value mesh.");
    }
    if (time_step <= 0.0)
    {
        throw std::invalid_argument("transport_system requires a positive time step.");
    }
    if (diffusivity < 0.0)
    {
        throw std::invalid_argument("transport_system requires non-negative diffusivity.");
    }

    // Row map = owned cells; domain map = owned + ghost cells
    // so that neighbour columns on other ranks are valid.
    const auto prepared = detail::prepare_transport_matrix<Pack>(mesh, std::move(cached_matrix), 12);
    const auto& matrix = prepared.matrix;
    auto rhs = Teuchos::rcp(new typename Pack::vector_type(mesh.owned_cell_map(), true));
    const auto old_value_data = old_values.owned_read_view();
    const auto face_flux_data = face_fluxes.owned_read_view();

    detail::FlatMatrixRow<local_ordinal_type, scalar_type> row_values(mesh.num_local_cells(), 32);

    for (size_t owned = 0; owned < mesh.num_owned_cells(); ++owned)
    {
        const auto cell_lid = static_cast<local_ordinal_type>(owned);

        const auto volume = mesh.cell_volume(cell_lid);
        const auto transient = volume / time_step;
        const auto old_value = old_value_data(cell_lid, 0);
        scalar_type rhs_value = transient * old_value + volume * right_hand_source(cell_lid);
        row_values.clear();
        detail::add_matrix_entry(row_values, cell_lid, transient);

        for (const auto face_lid : mesh.faces(cell_lid))
        {
            const auto is_interior = mesh.is_interior_face(face_lid);
            local_ordinal_type other{};
            if (is_interior)
            {
                other = mesh.opposite_or_periodic_neighbor_cell(face_lid, cell_lid);
            }

            // === advection (upwind) ===
            const auto owner_oriented_flux = face_fluxes.is_owned_face(face_lid)
                                                 ? face_flux_data(face_fluxes.owned_row(face_lid), 0)
                                                 : scalar_type{};
            const auto out_flux = mesh.owner_cell(face_lid) == cell_lid ? owner_oriented_flux : -owner_oriented_flux;

            if (out_flux >= scalar_type{0})
            {
                detail::add_matrix_entry(row_values, cell_lid, out_flux);
            }
            else if (is_interior)
            {
                detail::add_matrix_entry(row_values, other, out_flux);
            }

            // === diffusion ===
            if (!is_interior)
            {
                continue;
            }

            row_values.ensure(other);
            if (diffusivity <= scalar_type{0})
            {
                continue;
            }

            const auto coeff = detail::interior_diffusion_coefficient(mesh, face_lid, cell_lid, other, diffusivity);
            detail::add_matrix_entry(row_values, cell_lid, coeff);
            detail::add_matrix_entry(row_values, other, -coeff);
        }

        detail::add_transport_values<Pack>(prepared, cell_lid, row_values);
        rhs->replaceLocalValue(cell_lid, rhs_value);
    }

    // Apply boundary conditions using sumIntoLocalValues (works reliably).
    for (const auto& [batch_id, boundary_batch] : mesh.boundary_batches())
    {
        for (size_t in_batch_id = 0; in_batch_id < boundary_batch.face_lids.size(); ++in_batch_id)
        {
            const auto face_lid = boundary_batch.face_lids[in_batch_id];
            if (!mesh.is_owned_face(face_lid))
            {
                continue;
            }
            if (!mesh.is_boundary_face(face_lid))
            {
                continue;
            }

            const auto owner = mesh.owner_cell(face_lid);

            // Advective boundary flux
            const auto out_flux = face_fluxes.is_owned_face(face_lid)
                                      ? face_flux_data(face_fluxes.owned_row(face_lid), 0)
                                      : scalar_type{};

            const auto boundary_face_value = boundary_value(batch_id, in_batch_id);
            const auto condition = boundary_condition(batch_id, in_batch_id);

            if (out_flux >= scalar_type{0})
            {
                local_ordinal_type col = owner;
                matrix->sumIntoLocalValues(owner, Teuchos::arrayView(&col, 1), Teuchos::arrayView(&out_flux, 1));
            }
            else
            {
                rhs->sumIntoLocalValue(owner, -out_flux * boundary_face_value);
            }

            // Diffusive boundary flux
            if (diffusivity <= scalar_type{0})
            {
                continue;
            }

            const auto coeff = detail::boundary_diffusion_coefficient(mesh, face_lid, owner, diffusivity);
            if (condition.type == BoundaryConditionType::Dirichlet)
            {
                if (coeff > scalar_type{0})
                {
                    local_ordinal_type col = owner;
                    scalar_type bval = coeff;
                    matrix->sumIntoLocalValues(owner, Teuchos::arrayView(&col, 1), Teuchos::arrayView(&bval, 1));
                    rhs->sumIntoLocalValue(owner, coeff * boundary_face_value);
                }
            }
            else if (condition.type == BoundaryConditionType::Neumann)
            {
                rhs->sumIntoLocalValue(owner, diffusivity * condition.value * mesh.face_area(face_lid));
            }
            else if (condition.type == BoundaryConditionType::Robin)
            {
                throw std::runtime_error("Robin boundary conditions are not yet implemented in transport_system.");
            }
        }
    }

    matrix->fillComplete();
    return {matrix, rhs};
}

template<TpetraTypePack Pack>
TransportSystem<Pack> transport_system(const CellField<Pack>& old_values, const FaceField<Pack>& face_fluxes,
    typename Pack::scalar_type time_step, typename Pack::scalar_type diffusivity,
    ScalarBoundaryValueProvider<Pack> boundary_value, ScalarCellValueProvider<Pack> right_hand_source,
    Teuchos::RCP<typename Pack::matrix_type> cached_matrix)
{
    auto dirichlet_condition = [](int, size_t)
    { return BoundaryCondition{BoundaryConditionType::Dirichlet, typename Pack::scalar_type{}}; };

    return transport_system<Pack>(old_values, face_fluxes, time_step, diffusivity, dirichlet_condition, boundary_value,
        right_hand_source, cached_matrix);
}

template<TpetraTypePack Pack>
TransportSystem<Pack> transport_system(const CellField<Pack>& old_values, const FaceField<Pack>& face_fluxes,
    typename Pack::scalar_type time_step, typename Pack::scalar_type diffusivity,
    ScalarBoundaryConditionProvider<Pack> boundary_condition, ScalarBoundaryValueProvider<Pack> boundary_value,
    Teuchos::RCP<typename Pack::matrix_type> cached_matrix)
{
    using scalar_type = typename Pack::scalar_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;

    auto zero_source = [](local_ordinal_type) -> scalar_type { return scalar_type{}; };

    return transport_system<Pack>(old_values, face_fluxes, time_step, diffusivity, boundary_condition, boundary_value,
        zero_source, cached_matrix);
}

template<TpetraTypePack Pack>
TransportSystem<Pack> transport_system(const CellField<Pack>& old_values, const FaceField<Pack>& face_fluxes,
    typename Pack::scalar_type time_step, typename Pack::scalar_type diffusivity,
    ScalarBoundaryValueProvider<Pack> boundary_value, Teuchos::RCP<typename Pack::matrix_type> cached_matrix)
{
    using scalar_type = typename Pack::scalar_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;

    auto zero_source = [](local_ordinal_type) -> scalar_type { return scalar_type{}; };

    return transport_system<Pack>(
        old_values, face_fluxes, time_step, diffusivity, boundary_value, zero_source, cached_matrix);
}

template<TpetraTypePack Pack>
VectorTransportSystem<Pack> transport_system(const VectorCellField<Pack>& old_values,
    const FaceField<Pack>& face_fluxes, typename Pack::scalar_type time_step, typename Pack::scalar_type diffusivity,
    VectorBoundaryConditionProvider<Pack> boundary_condition, VectorBoundaryValueProvider<Pack> boundary_value,
    VectorCellValueProvider<Pack> right_hand_source, Teuchos::RCP<typename Pack::matrix_type> cached_matrix)
{
    using scalar_type = typename Pack::scalar_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;

    constexpr size_t num_components = 3;

    const auto& mesh = old_values.mesh();
    if (&face_fluxes.mesh() != &mesh)
    {
        throw std::invalid_argument("transport_system requires face fluxes on the old-value mesh.");
    }
    if (time_step <= 0.0)
    {
        throw std::invalid_argument("transport_system requires a positive time step.");
    }
    if (diffusivity < 0.0)
    {
        throw std::invalid_argument("transport_system requires non-negative diffusivity.");
    }

    const auto prepared = detail::prepare_transport_matrix<Pack>(mesh, std::move(cached_matrix), 12);
    const auto& matrix = prepared.matrix;
    auto rhs = Teuchos::rcp(new typename Pack::multi_vector_type(mesh.owned_cell_map(), num_components, true));
    const auto old_value_data = old_values.owned_read_view();
    const auto face_flux_data = face_fluxes.owned_read_view();

    detail::FlatMatrixRow<local_ordinal_type, scalar_type> row_values(mesh.num_local_cells(), 32);

    for (size_t owned = 0; owned < mesh.num_owned_cells(); ++owned)
    {
        const auto cell_lid = static_cast<local_ordinal_type>(owned);

        const auto volume = mesh.cell_volume(cell_lid);
        const auto transient = volume / time_step;
        const auto source_value = right_hand_source(cell_lid);
        row_values.clear();
        detail::add_matrix_entry(row_values, cell_lid, transient);

        for (const auto face_lid : mesh.faces(cell_lid))
        {
            const auto is_interior = mesh.is_interior_face(face_lid);
            local_ordinal_type other{};
            if (is_interior)
            {
                other = mesh.opposite_or_periodic_neighbor_cell(face_lid, cell_lid);
            }

            const auto owner_oriented_flux = face_fluxes.is_owned_face(face_lid)
                                                 ? face_flux_data(face_fluxes.owned_row(face_lid), 0)
                                                 : scalar_type{};
            const auto out_flux = mesh.owner_cell(face_lid) == cell_lid ? owner_oriented_flux : -owner_oriented_flux;

            if (out_flux >= scalar_type{0})
            {
                detail::add_matrix_entry(row_values, cell_lid, out_flux);
            }
            else if (is_interior)
            {
                detail::add_matrix_entry(row_values, other, out_flux);
            }

            if (!is_interior)
            {
                continue;
            }

            row_values.ensure(other);
            if (diffusivity <= scalar_type{0})
            {
                continue;
            }

            const auto coeff = detail::interior_diffusion_coefficient(mesh, face_lid, cell_lid, other, diffusivity);
            detail::add_matrix_entry(row_values, cell_lid, coeff);
            detail::add_matrix_entry(row_values, other, -coeff);
        }

        detail::add_transport_values<Pack>(prepared, cell_lid, row_values);
        for (size_t comp = 0; comp < num_components; ++comp)
        {
            rhs->replaceLocalValue(
                cell_lid, comp, transient * old_value_data(cell_lid, comp) + volume * source_value.component(comp));
        }
    }

    for (const auto& [batch_id, boundary_batch] : mesh.boundary_batches())
    {
        for (size_t in_batch_id = 0; in_batch_id < boundary_batch.face_lids.size(); ++in_batch_id)
        {
            const auto face_lid = boundary_batch.face_lids[in_batch_id];
            if (!mesh.is_owned_face(face_lid))
            {
                continue;
            }
            if (!mesh.is_boundary_face(face_lid))
            {
                continue;
            }

            const auto owner = mesh.owner_cell(face_lid);
            const auto out_flux = face_fluxes.is_owned_face(face_lid)
                                      ? face_flux_data(face_fluxes.owned_row(face_lid), 0)
                                      : scalar_type{};

            const auto boundary_face_value = boundary_value(batch_id, in_batch_id);
            const auto condition = boundary_condition(batch_id, in_batch_id);

            if (out_flux >= scalar_type{0})
            {
                local_ordinal_type col = owner;
                matrix->sumIntoLocalValues(owner, Teuchos::arrayView(&col, 1), Teuchos::arrayView(&out_flux, 1));
            }
            else
            {
                for (size_t comp = 0; comp < num_components; ++comp)
                {
                    rhs->sumIntoLocalValue(owner, comp, -out_flux * boundary_face_value.component(comp));
                }
            }

            if (diffusivity <= scalar_type{0})
            {
                continue;
            }

            const auto coeff = detail::boundary_diffusion_coefficient(mesh, face_lid, owner, diffusivity);
            if (condition.type == BoundaryConditionType::Dirichlet || condition.type == BoundaryConditionType::NoSlip)
            {
                if (coeff > scalar_type{0})
                {
                    local_ordinal_type col = owner;
                    scalar_type bval = coeff;
                    matrix->sumIntoLocalValues(owner, Teuchos::arrayView(&col, 1), Teuchos::arrayView(&bval, 1));
                    for (size_t comp = 0; comp < num_components; ++comp)
                    {
                        rhs->sumIntoLocalValue(owner, comp, coeff * boundary_face_value.component(comp));
                    }
                }
            }
            else if (condition.type == BoundaryConditionType::Neumann)
            {
                for (size_t comp = 0; comp < num_components; ++comp)
                {
                    rhs->sumIntoLocalValue(
                        owner, comp, diffusivity * condition.value.component(comp) * mesh.face_area(face_lid));
                }
            }
            else if (condition.type == BoundaryConditionType::Robin)
            {
                throw std::runtime_error(
                    "Robin boundary conditions are not yet implemented in vector transport_system.");
            }
        }
    }

    matrix->fillComplete();
    return {matrix, rhs};
}

template<TpetraTypePack Pack>
VectorTransportSystem<Pack> transport_system(const VectorCellField<Pack>& old_values,
    const FaceField<Pack>& face_fluxes, typename Pack::scalar_type time_step, typename Pack::scalar_type diffusivity,
    VectorBoundaryValueProvider<Pack> boundary_value, VectorCellValueProvider<Pack> right_hand_source,
    Teuchos::RCP<typename Pack::matrix_type> cached_matrix)
{
    auto dirichlet_condition = [](int, size_t)
    { return VectorBoundaryCondition{BoundaryConditionType::Dirichlet, typename VectorCellField<Pack>::vec_type{}}; };

    return transport_system<Pack>(old_values, face_fluxes, time_step, diffusivity, dirichlet_condition, boundary_value,
        right_hand_source, cached_matrix);
}

template<TpetraTypePack Pack>
VectorTransportSystem<Pack> transport_system(const VectorCellField<Pack>& old_values,
    const FaceField<Pack>& face_fluxes, typename Pack::scalar_type time_step, typename Pack::scalar_type diffusivity,
    VectorBoundaryConditionProvider<Pack> boundary_condition, VectorBoundaryValueProvider<Pack> boundary_value,
    Teuchos::RCP<typename Pack::matrix_type> cached_matrix)
{
    using local_ordinal_type = typename Pack::local_ordinal_type;
    using vec_type = typename VectorCellField<Pack>::vec_type;

    auto zero_source = [](local_ordinal_type) -> vec_type { return vec_type{}; };

    return transport_system<Pack>(old_values, face_fluxes, time_step, diffusivity, boundary_condition, boundary_value,
        zero_source, cached_matrix);
}

template<TpetraTypePack Pack>
VectorTransportSystem<Pack> non_orthogonal_transport_system(const VectorCellField<Pack>& old_values,
    const FaceField<Pack>& face_fluxes, typename Pack::scalar_type time_step, typename Pack::scalar_type diffusivity,
    VectorBoundaryValueProvider<Pack> boundary_value, VectorCellValueProvider<Pack> right_hand_source,
    NonOrthogonalTreatment treatment, const VectorCellField<Pack>* correction_field,
    Teuchos::RCP<typename Pack::matrix_type> cached_matrix, BoundaryFaceSelector boundary_diffusion,
    const TransportGeometryCache<Mesh<Pack>>* geometry_cache)
{
    using scalar_type = typename Pack::scalar_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;

    constexpr size_t num_components = VectorCellField<Pack>::num_components;

    const auto& mesh = old_values.mesh();
    const auto non_orthogonal_weights = detail::validate_non_orthogonal_transport_selection<Pack>(
        mesh, treatment, correction_field, "non_orthogonal_transport_system");
    if (&face_fluxes.mesh() != &mesh)
    {
        throw std::invalid_argument("non_orthogonal_transport_system requires face fluxes on the old-value mesh.");
    }
    if (time_step <= scalar_type{0})
    {
        throw std::invalid_argument("non_orthogonal_transport_system requires a positive time step.");
    }
    if (diffusivity < scalar_type{0})
    {
        throw std::invalid_argument("non_orthogonal_transport_system requires non-negative diffusivity.");
    }

    const auto implicit_weight = non_orthogonal_weights.implicit;
    const auto explicit_weight = non_orthogonal_weights.explicit_;

    using geometry_cache_type = TransportGeometryCache<Mesh<Pack>>;
    typename geometry_cache_type::interior_stencils_type local_gradient_stencils;
    typename geometry_cache_type::boundary_locations_type local_boundary_locations;
    if (geometry_cache == nullptr)
    {
        local_gradient_stencils = detail::least_squares_gradient_stencils(mesh);
        local_boundary_locations = detail::boundary_face_locations(mesh);
    }
    else
    {
        geometry_cache->require_mesh(mesh);
    }
    const auto& gradient_stencils =
        geometry_cache == nullptr ? local_gradient_stencils : geometry_cache->interior_stencils();
    const auto& boundary_locations =
        geometry_cache == nullptr ? local_boundary_locations : geometry_cache->boundary_locations();

    std::unique_ptr<TensorCellField<Pack>> partition_gradients;
    if (implicit_weight > scalar_type{})
    {
        partition_gradients =
            std::make_unique<TensorCellField<Pack>>(old_values.mesh_ptr(), "partition_vector_non_orthogonal_gradient");
        const auto& lagged_field = correction_field == nullptr ? old_values : *correction_field;
        detail::evaluate_vector_interior_gradients(lagged_field, gradient_stencils, *partition_gradients);
        partition_gradients->sync_ghosts();
    }

    const auto prepared = detail::prepare_transport_matrix<Pack>(mesh, std::move(cached_matrix), 32);
    const auto& matrix = prepared.matrix;
    auto rhs = Teuchos::rcp(new typename Pack::multi_vector_type(mesh.owned_cell_map(), num_components, true));

    detail::FlatMatrixRow<local_ordinal_type, scalar_type> row_values(mesh.num_local_cells(), 64);
    const auto old_value_data = old_values.owned_read_view();
    const auto face_flux_data = face_fluxes.owned_read_view();
    using partition_gradient_view_type = decltype(partition_gradients->local_read_view());
    std::optional<partition_gradient_view_type> partition_gradient_data;
    if (partition_gradients != nullptr)
    {
        partition_gradient_data.emplace(partition_gradients->local_read_view());
    }

    for (size_t owned = 0; owned < mesh.num_owned_cells(); ++owned)
    {
        const auto cell_lid = static_cast<local_ordinal_type>(owned);
        const auto volume = mesh.cell_volume(cell_lid);
        const auto transient = volume / time_step;
        const auto source_value = right_hand_source(cell_lid);

        row_values.clear();
        detail::add_matrix_entry(row_values, cell_lid, transient);

        for (size_t comp = 0; comp < num_components; ++comp)
        {
            rhs->replaceLocalValue(
                cell_lid, comp, transient * old_value_data(cell_lid, comp) + volume * source_value.component(comp));
        }

        auto add_non_orthogonal_stencil = [&](local_ordinal_type gradient_cell_lid, scalar_type face_gradient_weight,
                                              const typename Mesh<Pack>::Vec3& tangential_area)
        {
            if (face_gradient_weight == scalar_type{0} || !mesh.is_owned_cell(gradient_cell_lid) ||
                static_cast<size_t>(gradient_cell_lid) >= gradient_stencils.size())
            {
                return;
            }

            const auto scale = -implicit_weight * diffusivity * face_gradient_weight;
            for (const auto& entry : gradient_stencils[static_cast<size_t>(gradient_cell_lid)])
            {
                detail::add_matrix_entry(row_values, entry.cell_lid, scale * entry.coefficient.dot(tangential_area));
            }
        };

        for (const auto face_lid : mesh.faces(cell_lid))
        {
            const auto is_interior = mesh.is_interior_face(face_lid);
            local_ordinal_type other{};
            if (is_interior)
            {
                other = mesh.opposite_or_periodic_neighbor_cell(face_lid, cell_lid);
                // Keep the cached graph independent of the current flux
                // direction, including the zero-diffusivity case.
                row_values.ensure(other);
            }

            const auto owner_oriented_flux = face_fluxes.is_owned_face(face_lid)
                                                 ? face_flux_data(face_fluxes.owned_row(face_lid), 0)
                                                 : scalar_type{};
            const auto out_flux = mesh.owner_cell(face_lid) == cell_lid ? owner_oriented_flux : -owner_oriented_flux;

            if (out_flux >= scalar_type{0})
            {
                detail::add_matrix_entry(row_values, cell_lid, out_flux);
            }
            else if (is_interior)
            {
                detail::add_matrix_entry(row_values, other, out_flux);
            }
            else if (mesh.is_boundary_face(face_lid) && static_cast<size_t>(face_lid) < boundary_locations.size() &&
                     boundary_locations[static_cast<size_t>(face_lid)].active)
            {
                const auto location = boundary_locations[static_cast<size_t>(face_lid)];
                const auto boundary_face_value = boundary_value(location.batch_id, location.in_batch_id);
                for (size_t comp = 0; comp < num_components; ++comp)
                {
                    rhs->sumIntoLocalValue(cell_lid, comp, -out_flux * boundary_face_value.component(comp));
                }
            }

            if (diffusivity <= scalar_type{0})
            {
                continue;
            }

            if (is_interior)
            {
                const auto coeff = detail::interior_diffusion_coefficient(mesh, face_lid, cell_lid, other, diffusivity);
                detail::add_matrix_entry(row_values, cell_lid, coeff);
                detail::add_matrix_entry(row_values, other, -coeff);

                const auto area_vector = mesh.face_area_vector_outward(face_lid, cell_lid);
                const auto d = mesh.cell_center_vector(face_lid, cell_lid);
                const auto tangential_area = detail::non_orthogonal_area_vector(area_vector, d);

                if (mesh.is_owned_cell(other) && static_cast<size_t>(other) < gradient_stencils.size())
                {
                    add_non_orthogonal_stencil(cell_lid, scalar_type{0.5}, tangential_area);
                    add_non_orthogonal_stencil(other, scalar_type{0.5}, tangential_area);
                }
                else if (implicit_weight > scalar_type{})
                {
                    add_non_orthogonal_stencil(cell_lid, scalar_type{0.5}, tangential_area);
                    if (!partition_gradient_data)
                    {
                        throw std::logic_error("Partition-face implicit vector reconstruction "
                                               "requires synchronized remote gradients.");
                    }
                    const auto remote_gradient = detail::tensor_view_value<Pack>(*partition_gradient_data, other);
                    for (size_t component = 0; component < num_components; ++component)
                    {
                        rhs->sumIntoLocalValue(cell_lid, component,
                            implicit_weight * diffusivity * scalar_type{0.5} *
                                remote_gradient[component].dot(tangential_area));
                    }
                }
                continue;
            }

            if (!mesh.is_boundary_face(face_lid) || static_cast<size_t>(face_lid) >= boundary_locations.size() ||
                !boundary_locations[static_cast<size_t>(face_lid)].active)
            {
                continue;
            }

            const auto location = boundary_locations[static_cast<size_t>(face_lid)];
            if (!boundary_diffusion(location.batch_id, location.in_batch_id))
            {
                continue;
            }
            const auto boundary_face_value = boundary_value(location.batch_id, location.in_batch_id);
            const auto coeff = detail::boundary_diffusion_coefficient(mesh, face_lid, cell_lid, diffusivity);
            if (coeff > scalar_type{0})
            {
                detail::add_matrix_entry(row_values, cell_lid, coeff);
                for (size_t comp = 0; comp < num_components; ++comp)
                {
                    rhs->sumIntoLocalValue(cell_lid, comp, coeff * boundary_face_value.component(comp));
                }
            }

            const auto area_vector = mesh.face_area_vector_outward(face_lid, cell_lid);
            const auto d = mesh.face_centroid(face_lid) - mesh.cell_centroid(cell_lid);
            const auto tangential_area = detail::non_orthogonal_area_vector(area_vector, d);
            add_non_orthogonal_stencil(cell_lid, scalar_type{1}, tangential_area);
        }

        detail::add_transport_values<Pack>(prepared, cell_lid, row_values);
    }

    if (correction_field != nullptr && explicit_weight > scalar_type{0})
    {
        add_explicit_non_orthogonal_correction<Pack>(
            *correction_field, diffusivity, *rhs, explicit_weight, boundary_diffusion, &gradient_stencils);
    }

    matrix->fillComplete();
    return {matrix, rhs};
}

template<TpetraTypePack Pack>
TransportSystem<Pack> weighted_scalar_transport_system(const CellField<Pack>& old_values,
    const FaceField<Pack>& face_fluxes, typename Pack::scalar_type time_step, const CellField<Pack>& storage_weight,
    const CellField<Pack>& advection_weight, const CellField<Pack>& diffusivity,
    ScalarBoundaryConditionProvider<Pack> boundary_condition, ScalarBoundaryValueProvider<Pack> boundary_value,
    ScalarCellValueProvider<Pack> source, NonOrthogonalTreatment treatment, const CellField<Pack>* correction_field,
    Teuchos::RCP<typename Pack::matrix_type> cached_matrix,
    std::function<typename Pack::scalar_type(typename Pack::local_ordinal_type)> implicit_sink,
    std::function<std::optional<typename Pack::scalar_type>(typename Pack::local_ordinal_type)> fixed_cell_value,
    const BoundaryCache<Pack>* boundary_diffusivity, const TransportGeometryCache<Mesh<Pack>>* geometry_cache,
    FaceCoefficientInterpolation coefficient_interpolation)
{
    using scalar_type = typename Pack::scalar_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;

    const auto& mesh = old_values.mesh();
    const auto non_orthogonal_weights = detail::validate_non_orthogonal_transport_selection<Pack>(
        mesh, treatment, correction_field, "weighted_scalar_transport_system");
    if (&face_fluxes.mesh() != &mesh || &storage_weight.mesh() != &mesh || &advection_weight.mesh() != &mesh ||
        &diffusivity.mesh() != &mesh)
    {
        throw std::invalid_argument("weighted_scalar_transport_system requires all fields on "
                                    "the transported-field mesh.");
    }
    if (time_step <= scalar_type{})
    {
        throw std::invalid_argument("weighted_scalar_transport_system requires a positive "
                                    "time step.");
    }
    validate_boundary_coefficient_cache(mesh, boundary_diffusivity, "weighted_scalar_transport_system");
    auto boundary_face_diffusivity = [&](int batch_id, size_t in_batch_id, scalar_type owner_cell_value)
    { return boundary_coefficient<Pack>(boundary_diffusivity, batch_id, in_batch_id, owner_cell_value); };

    std::vector<std::optional<scalar_type>> fixed_cell_values(mesh.num_owned_cells());
    if (fixed_cell_value)
    {
        for (size_t owned = 0; owned < mesh.num_owned_cells(); ++owned)
        {
            const auto cell_lid = static_cast<local_ordinal_type>(owned);
            auto value = fixed_cell_value(cell_lid);
            if (value.has_value() && !std::isfinite(*value))
            {
                throw std::invalid_argument("weighted scalar transport requires finite fixed-cell "
                                            "values.");
            }
            fixed_cell_values[owned] = value;
        }
    }

    const auto implicit_weight = non_orthogonal_weights.implicit;
    const auto explicit_weight = non_orthogonal_weights.explicit_;

    using geometry_cache_type = TransportGeometryCache<Mesh<Pack>>;
    typename geometry_cache_type::boundary_locations_type local_boundary_locations;
    std::vector<detail::AffineLeastSquaresGradientStencil<Mesh<Pack>>> gradient_stencils;
    if (geometry_cache == nullptr)
    {
        gradient_stencils = detail::scalar_affine_gradient_stencils(mesh, boundary_condition, boundary_value);
        local_boundary_locations = detail::boundary_face_locations(mesh);
    }
    else
    {
        geometry_cache->require_mesh(mesh);
        gradient_stencils = geometry_cache->scalar_affine_stencils(boundary_condition, boundary_value);
    }
    const auto& boundary_locations =
        geometry_cache == nullptr ? local_boundary_locations : geometry_cache->boundary_locations();

    std::unique_ptr<VectorCellField<Pack>> partition_gradients;
    if (implicit_weight > scalar_type{})
    {
        partition_gradients =
            std::make_unique<VectorCellField<Pack>>(old_values.mesh_ptr(), "partition_scalar_non_orthogonal_gradient");
        const auto& lagged_field = correction_field == nullptr ? old_values : *correction_field;
        detail::evaluate_scalar_affine_gradients(lagged_field, gradient_stencils, *partition_gradients);
        partition_gradients->sync_ghosts();
    }

    const auto prepared = detail::prepare_transport_matrix<Pack>(mesh, std::move(cached_matrix), 32);
    const auto& matrix = prepared.matrix;
    auto rhs = Teuchos::rcp(new typename Pack::vector_type(mesh.owned_cell_map(), true));

    detail::FlatMatrixRow<local_ordinal_type, scalar_type> row_values(mesh.num_local_cells(), 64);
    const auto old_value_data = old_values.owned_read_view();
    const auto face_flux_data = face_fluxes.owned_read_view();
    const auto storage_data = storage_weight.local_read_view();
    const auto advection_data = advection_weight.local_read_view();
    const auto diffusivity_data = diffusivity.local_read_view();
    using partition_gradient_view_type = decltype(partition_gradients->local_read_view());
    std::optional<partition_gradient_view_type> partition_gradient_data;
    if (partition_gradients != nullptr)
    {
        partition_gradient_data.emplace(partition_gradients->local_read_view());
    }

    for (size_t owned = 0; owned < mesh.num_owned_cells(); ++owned)
    {
        const auto cell_lid = static_cast<local_ordinal_type>(owned);
        const auto volume = mesh.cell_volume(cell_lid);
        const auto cell_storage = storage_data(cell_lid, 0);
        const auto cell_advection = advection_data(cell_lid, 0);
        const auto transient = cell_storage * volume / time_step;
        const auto sink = implicit_sink ? implicit_sink(cell_lid) : scalar_type{};
        if (!std::isfinite(cell_storage) || !std::isfinite(cell_advection) || cell_storage <= scalar_type{} ||
            cell_advection < scalar_type{})
        {
            throw std::invalid_argument("weighted scalar transport requires positive storage "
                                        "and non-negative advection weights.");
        }
        if (!std::isfinite(sink) || sink < scalar_type{})
        {
            throw std::invalid_argument("weighted scalar transport requires a finite, "
                                        "non-negative implicit sink.");
        }

        row_values.clear();
        detail::add_matrix_entry(row_values, cell_lid, transient + volume * sink);
        rhs->replaceLocalValue(cell_lid, transient * old_value_data(cell_lid, 0) + volume * source(cell_lid));

        auto add_non_orthogonal_stencil = [&](local_ordinal_type gradient_cell_lid, scalar_type gradient_weight,
                                              scalar_type face_diffusivity,
                                              const typename Mesh<Pack>::Vec3& tangential_area)
        {
            if (implicit_weight == scalar_type{} || gradient_weight == scalar_type{} ||
                !mesh.is_owned_cell(gradient_cell_lid) ||
                static_cast<size_t>(gradient_cell_lid) >= gradient_stencils.size())
            {
                return;
            }
            const auto scale = -implicit_weight * face_diffusivity * gradient_weight;
            const auto& stencil = gradient_stencils[static_cast<size_t>(gradient_cell_lid)];
            rhs->sumIntoLocalValue(cell_lid, -scale * stencil.constant.dot(tangential_area));
            for (const auto& entry : stencil.entries)
            {
                detail::add_matrix_entry(row_values, entry.cell_lid, scale * entry.coefficient.dot(tangential_area));
            }
        };

        for (const auto face_lid : mesh.faces(cell_lid))
        {
            const auto owner_oriented_flux = face_fluxes.is_owned_face(face_lid)
                                                 ? face_flux_data(face_fluxes.owned_row(face_lid), 0)
                                                 : scalar_type{};
            const auto out_flux = mesh.owner_cell(face_lid) == cell_lid ? owner_oriented_flux : -owner_oriented_flux;
            if (out_flux >= scalar_type{})
            {
                detail::add_matrix_entry(row_values, cell_lid, out_flux * cell_advection);
            }
            else if (mesh.is_interior_face(face_lid))
            {
                const auto other = mesh.opposite_or_periodic_neighbor_cell(face_lid, cell_lid);
                const auto other_advection = advection_data(other, 0);
                detail::add_matrix_entry(row_values, other, out_flux * other_advection);
            }
            else if (mesh.is_boundary_face(face_lid) && static_cast<size_t>(face_lid) < boundary_locations.size() &&
                     boundary_locations[static_cast<size_t>(face_lid)].active)
            {
                const auto location = boundary_locations[static_cast<size_t>(face_lid)];
                rhs->sumIntoLocalValue(
                    cell_lid, -out_flux * cell_advection * boundary_value(location.batch_id, location.in_batch_id));
            }

            if (mesh.is_interior_face(face_lid))
            {
                const auto other = mesh.opposite_or_periodic_neighbor_cell(face_lid, cell_lid);
                row_values.ensure(other);
                const auto face_diffusivity = detail::face_coefficient_value(mesh, face_lid, cell_lid, other,
                    diffusivity_data(cell_lid, 0), diffusivity_data(other, 0), coefficient_interpolation);
                if (face_diffusivity <= scalar_type{})
                    continue;
                const auto coefficient =
                    detail::interior_diffusion_coefficient(mesh, face_lid, cell_lid, other, face_diffusivity);
                detail::add_matrix_entry(row_values, cell_lid, coefficient);
                detail::add_matrix_entry(row_values, other, -coefficient);

                const auto tangential_area = detail::non_orthogonal_area_vector(
                    mesh.face_area_vector_outward(face_lid, cell_lid), mesh.cell_center_vector(face_lid, cell_lid));
                if (mesh.is_owned_cell(other) && static_cast<size_t>(other) < gradient_stencils.size())
                {
                    add_non_orthogonal_stencil(cell_lid, scalar_type{0.5}, face_diffusivity, tangential_area);
                    add_non_orthogonal_stencil(other, scalar_type{0.5}, face_diffusivity, tangential_area);
                }
                else if (implicit_weight > scalar_type{})
                {
                    add_non_orthogonal_stencil(cell_lid, scalar_type{0.5}, face_diffusivity, tangential_area);
                    if (!partition_gradient_data)
                    {
                        throw std::logic_error("Partition-face implicit scalar reconstruction "
                                               "requires synchronized remote gradients.");
                    }
                    rhs->sumIntoLocalValue(cell_lid,
                        implicit_weight * face_diffusivity * scalar_type{0.5} *
                            detail::vector_view_value<Pack>(*partition_gradient_data, other).dot(tangential_area));
                }
                continue;
            }

            if (!mesh.is_boundary_face(face_lid) || static_cast<size_t>(face_lid) >= boundary_locations.size() ||
                !boundary_locations[static_cast<size_t>(face_lid)].active)
            {
                continue;
            }
            const auto location = boundary_locations[static_cast<size_t>(face_lid)];
            const auto condition = boundary_condition(location.batch_id, location.in_batch_id);
            const auto face_diffusivity =
                boundary_face_diffusivity(location.batch_id, location.in_batch_id, diffusivity_data(cell_lid, 0));
            if (condition.type == BoundaryConditionType::Dirichlet)
            {
                const auto coefficient =
                    detail::boundary_diffusion_coefficient(mesh, face_lid, cell_lid, face_diffusivity);
                if (coefficient > scalar_type{})
                {
                    detail::add_matrix_entry(row_values, cell_lid, coefficient);
                    rhs->sumIntoLocalValue(
                        cell_lid, coefficient * boundary_value(location.batch_id, location.in_batch_id));
                }
                const auto tangential_area =
                    detail::non_orthogonal_area_vector(mesh.face_area_vector_outward(face_lid, cell_lid),
                        mesh.face_centroid(face_lid) - mesh.cell_centroid(cell_lid));
                add_non_orthogonal_stencil(cell_lid, scalar_type{1}, face_diffusivity, tangential_area);
            }
            else if (condition.type == BoundaryConditionType::Neumann)
            {
                rhs->sumIntoLocalValue(cell_lid, face_diffusivity * condition.value * mesh.face_area(face_lid));
            }
            else if (condition.type == BoundaryConditionType::Robin)
            {
                throw std::runtime_error(
                    "Robin boundary conditions are not yet implemented in weighted_scalar_transport_system.");
            }
        }

        if (fixed_cell_values[owned].has_value())
        {
            // Retain the assembled graph (including non-orthogonal stencil
            // entries) so a cached matrix remains reusable if constraints
            // change between calls, but make the constrained equation an
            // exact identity row.
            row_values.fill(scalar_type{});
            row_values.set(cell_lid, scalar_type{1});
            rhs->replaceLocalValue(cell_lid, *fixed_cell_values[owned]);
        }

        detail::add_transport_values<Pack>(prepared, cell_lid, row_values);
    }

    if (correction_field != nullptr && explicit_weight > scalar_type{})
    {
        add_variable_explicit_non_orthogonal_correction<Pack>(*correction_field, diffusivity, boundary_condition,
            boundary_value, *rhs, explicit_weight, boundary_face_diffusivity, &gradient_stencils,
            coefficient_interpolation);
    }
    // An explicit correction is an RHS update, so restore constrained values
    // after it to keep fixed rows exactly equal to phi = phi_fixed.
    for (size_t owned = 0; owned < fixed_cell_values.size(); ++owned)
    {
        if (fixed_cell_values[owned].has_value())
        {
            rhs->replaceLocalValue(static_cast<local_ordinal_type>(owned), *fixed_cell_values[owned]);
        }
    }

    matrix->fillComplete();
    return {matrix, rhs};
}

template<TpetraTypePack Pack>
TransportSystem<Pack> physical_temperature_transport_system(const CellField<Pack>& old_temperature,
    const FaceField<Pack>& face_fluxes, typename Pack::scalar_type time_step, const CellField<Pack>& density,
    const CellField<Pack>& specific_heat_capacity, const CellField<Pack>& thermal_conductivity,
    ScalarBoundaryConditionProvider<Pack> boundary_condition, ScalarBoundaryValueProvider<Pack> boundary_value,
    ScalarCellValueProvider<Pack> power_density, NonOrthogonalTreatment treatment,
    const CellField<Pack>* correction_field, Teuchos::RCP<typename Pack::matrix_type> cached_matrix,
    const BoundaryCache<Pack>* boundary_thermal_conductivity, const TransportGeometryCache<Mesh<Pack>>* geometry_cache,
    FaceCoefficientInterpolation coefficient_interpolation)
{
    using scalar_type = typename Pack::scalar_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;

    const auto& mesh = old_temperature.mesh();
    const auto non_orthogonal_weights = detail::validate_non_orthogonal_transport_selection<Pack>(
        mesh, treatment, correction_field, "physical_temperature_transport_system");
    if (&face_fluxes.mesh() != &mesh || &density.mesh() != &mesh || &specific_heat_capacity.mesh() != &mesh ||
        &thermal_conductivity.mesh() != &mesh)
    {
        throw std::invalid_argument("physical_temperature_transport_system requires all fields on "
                                    "the temperature mesh.");
    }
    if (time_step <= scalar_type{})
    {
        throw std::invalid_argument("physical_temperature_transport_system requires a positive "
                                    "time step.");
    }
    validate_boundary_coefficient_cache(mesh, boundary_thermal_conductivity, "physical_temperature_transport_system");

    auto boundary_conductivity = [&](int batch_id, size_t in_batch_id, scalar_type owner_cell_value)
    { return boundary_coefficient<Pack>(boundary_thermal_conductivity, batch_id, in_batch_id, owner_cell_value); };

    const auto implicit_weight = non_orthogonal_weights.implicit;
    const auto explicit_weight = non_orthogonal_weights.explicit_;

    using geometry_cache_type = TransportGeometryCache<Mesh<Pack>>;
    typename geometry_cache_type::boundary_locations_type local_boundary_locations;
    std::vector<detail::AffineLeastSquaresGradientStencil<Mesh<Pack>>> gradient_stencils;
    if (geometry_cache == nullptr)
    {
        gradient_stencils = detail::scalar_affine_gradient_stencils(mesh, boundary_condition, boundary_value);
        local_boundary_locations = detail::boundary_face_locations(mesh);
    }
    else
    {
        geometry_cache->require_mesh(mesh);
        gradient_stencils = geometry_cache->scalar_affine_stencils(boundary_condition, boundary_value);
    }
    const auto& boundary_locations =
        geometry_cache == nullptr ? local_boundary_locations : geometry_cache->boundary_locations();

    std::unique_ptr<VectorCellField<Pack>> partition_gradients;
    if (implicit_weight > scalar_type{})
    {
        partition_gradients = std::make_unique<VectorCellField<Pack>>(
            old_temperature.mesh_ptr(), "partition_temperature_non_orthogonal_gradient");
        const auto& lagged_field = correction_field == nullptr ? old_temperature : *correction_field;
        detail::evaluate_scalar_affine_gradients(lagged_field, gradient_stencils, *partition_gradients);
        partition_gradients->sync_ghosts();
    }

    const auto prepared = detail::prepare_transport_matrix<Pack>(mesh, std::move(cached_matrix), 32);
    const auto& matrix = prepared.matrix;
    auto rhs = Teuchos::rcp(new typename Pack::vector_type(mesh.owned_cell_map(), true));

    detail::FlatMatrixRow<local_ordinal_type, scalar_type> row_values(mesh.num_local_cells(), 64);
    const auto old_temperature_data = old_temperature.owned_read_view();
    const auto face_flux_data = face_fluxes.owned_read_view();
    const auto density_data = density.local_read_view();
    const auto heat_capacity_data = specific_heat_capacity.local_read_view();
    const auto conductivity_data = thermal_conductivity.local_read_view();
    using partition_gradient_view_type = decltype(partition_gradients->local_read_view());
    std::optional<partition_gradient_view_type> partition_gradient_data;
    if (partition_gradients != nullptr)
    {
        partition_gradient_data.emplace(partition_gradients->local_read_view());
    }

    for (size_t owned = 0; owned < mesh.num_owned_cells(); ++owned)
    {
        const auto cell_lid = static_cast<local_ordinal_type>(owned);
        const auto volume = mesh.cell_volume(cell_lid);
        const auto cell_capacity = density_data(cell_lid, 0) * heat_capacity_data(cell_lid, 0);
        const auto transient = cell_capacity * volume / time_step;
        row_values.clear();
        detail::add_matrix_entry(row_values, cell_lid, transient);
        rhs->replaceLocalValue(
            cell_lid, transient * old_temperature_data(cell_lid, 0) + volume * power_density(cell_lid));

        auto add_non_orthogonal_stencil = [&](local_ordinal_type gradient_cell_lid, scalar_type gradient_weight,
                                              scalar_type face_conductivity,
                                              const typename Mesh<Pack>::Vec3& tangential_area)
        {
            if (implicit_weight == scalar_type{} || gradient_weight == scalar_type{} ||
                !mesh.is_owned_cell(gradient_cell_lid) ||
                static_cast<size_t>(gradient_cell_lid) >= gradient_stencils.size())
            {
                return;
            }
            const auto scale = -implicit_weight * face_conductivity * gradient_weight;
            const auto& stencil = gradient_stencils[static_cast<size_t>(gradient_cell_lid)];
            rhs->sumIntoLocalValue(cell_lid, -scale * stencil.constant.dot(tangential_area));
            for (const auto& entry : stencil.entries)
            {
                detail::add_matrix_entry(row_values, entry.cell_lid, scale * entry.coefficient.dot(tangential_area));
            }
        };

        for (const auto face_lid : mesh.faces(cell_lid))
        {
            const auto owner_oriented_flux = face_fluxes.is_owned_face(face_lid)
                                                 ? face_flux_data(face_fluxes.owned_row(face_lid), 0)
                                                 : scalar_type{};
            const auto out_flux = mesh.owner_cell(face_lid) == cell_lid ? owner_oriented_flux : -owner_oriented_flux;

            if (out_flux >= scalar_type{})
            {
                detail::add_matrix_entry(row_values, cell_lid, out_flux * cell_capacity);
            }
            else if (mesh.is_interior_face(face_lid))
            {
                const auto other = mesh.opposite_or_periodic_neighbor_cell(face_lid, cell_lid);
                const auto other_capacity = density_data(other, 0) * heat_capacity_data(other, 0);
                detail::add_matrix_entry(row_values, other, out_flux * other_capacity);
            }
            else if (mesh.is_boundary_face(face_lid) && static_cast<size_t>(face_lid) < boundary_locations.size() &&
                     boundary_locations[static_cast<size_t>(face_lid)].active)
            {
                const auto location = boundary_locations[static_cast<size_t>(face_lid)];
                rhs->sumIntoLocalValue(
                    cell_lid, -out_flux * cell_capacity * boundary_value(location.batch_id, location.in_batch_id));
            }

            if (mesh.is_interior_face(face_lid))
            {
                const auto other = mesh.opposite_or_periodic_neighbor_cell(face_lid, cell_lid);
                row_values.ensure(other);
                const auto face_conductivity = detail::face_coefficient_value(mesh, face_lid, cell_lid, other,
                    conductivity_data(cell_lid, 0), conductivity_data(other, 0), coefficient_interpolation);
                if (face_conductivity <= scalar_type{})
                {
                    continue;
                }
                const auto coefficient =
                    detail::interior_diffusion_coefficient(mesh, face_lid, cell_lid, other, face_conductivity);
                detail::add_matrix_entry(row_values, cell_lid, coefficient);
                detail::add_matrix_entry(row_values, other, -coefficient);

                const auto tangential_area = detail::non_orthogonal_area_vector(
                    mesh.face_area_vector_outward(face_lid, cell_lid), mesh.cell_center_vector(face_lid, cell_lid));
                if (mesh.is_owned_cell(other) && static_cast<size_t>(other) < gradient_stencils.size())
                {
                    add_non_orthogonal_stencil(cell_lid, scalar_type{0.5}, face_conductivity, tangential_area);
                    add_non_orthogonal_stencil(other, scalar_type{0.5}, face_conductivity, tangential_area);
                }
                else if (implicit_weight > scalar_type{})
                {
                    add_non_orthogonal_stencil(cell_lid, scalar_type{0.5}, face_conductivity, tangential_area);
                    if (!partition_gradient_data)
                    {
                        throw std::logic_error("Partition-face implicit temperature "
                                               "reconstruction requires synchronized remote "
                                               "gradients.");
                    }
                    rhs->sumIntoLocalValue(cell_lid,
                        implicit_weight * face_conductivity * scalar_type{0.5} *
                            detail::vector_view_value<Pack>(*partition_gradient_data, other).dot(tangential_area));
                }
                continue;
            }

            if (!mesh.is_boundary_face(face_lid) || static_cast<size_t>(face_lid) >= boundary_locations.size() ||
                !boundary_locations[static_cast<size_t>(face_lid)].active)
            {
                continue;
            }
            const auto location = boundary_locations[static_cast<size_t>(face_lid)];
            const auto condition = boundary_condition(location.batch_id, location.in_batch_id);
            const auto face_conductivity =
                boundary_conductivity(location.batch_id, location.in_batch_id, conductivity_data(cell_lid, 0));
            if (condition.type == BoundaryConditionType::Dirichlet)
            {
                const auto coefficient =
                    detail::boundary_diffusion_coefficient(mesh, face_lid, cell_lid, face_conductivity);
                if (coefficient > scalar_type{})
                {
                    detail::add_matrix_entry(row_values, cell_lid, coefficient);
                    rhs->sumIntoLocalValue(
                        cell_lid, coefficient * boundary_value(location.batch_id, location.in_batch_id));
                }

                const auto tangential_area =
                    detail::non_orthogonal_area_vector(mesh.face_area_vector_outward(face_lid, cell_lid),
                        mesh.face_centroid(face_lid) - mesh.cell_centroid(cell_lid));
                add_non_orthogonal_stencil(cell_lid, scalar_type{1}, face_conductivity, tangential_area);
            }
            else if (condition.type == BoundaryConditionType::Neumann)
            {
                rhs->sumIntoLocalValue(cell_lid, face_conductivity * condition.value * mesh.face_area(face_lid));
            }
            else if (condition.type == BoundaryConditionType::Robin)
            {
                throw std::runtime_error(
                    "Robin boundary conditions are not yet implemented in physical_temperature_transport_system.");
            }
        }

        detail::add_transport_values<Pack>(prepared, cell_lid, row_values);
    }

    if (correction_field != nullptr && explicit_weight > scalar_type{})
    {
        add_variable_explicit_non_orthogonal_correction<Pack>(*correction_field, thermal_conductivity,
            boundary_condition, boundary_value, *rhs, explicit_weight, boundary_conductivity, &gradient_stencils,
            coefficient_interpolation);
    }

    matrix->fillComplete();
    return {matrix, rhs};
}

template<TpetraTypePack Pack>
VectorTransportSystem<Pack> physical_momentum_transport_system(const VectorCellField<Pack>& old_velocity,
    const FaceField<Pack>& face_fluxes, typename Pack::scalar_type time_step, const CellField<Pack>& dynamic_viscosity,
    typename Pack::scalar_type reference_density, VectorBoundaryValueProvider<Pack> boundary_value,
    VectorCellValueProvider<Pack> acceleration_source, NonOrthogonalTreatment treatment,
    const VectorCellField<Pack>* correction_field, Teuchos::RCP<typename Pack::matrix_type> cached_matrix,
    BoundaryFaceSelector boundary_diffusion, const BoundaryCache<Pack>* boundary_dynamic_viscosity,
    const TransportGeometryCache<Mesh<Pack>>* geometry_cache, FaceCoefficientInterpolation coefficient_interpolation)
{
    using scalar_type = typename Pack::scalar_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;
    constexpr size_t components = VectorCellField<Pack>::num_components;

    const auto& mesh = old_velocity.mesh();
    const auto non_orthogonal_weights = detail::validate_non_orthogonal_transport_selection<Pack>(
        mesh, treatment, correction_field, "physical_momentum_transport_system");
    if (&face_fluxes.mesh() != &mesh || &dynamic_viscosity.mesh() != &mesh)
    {
        throw std::invalid_argument("physical_momentum_transport_system requires all fields on "
                                    "the velocity mesh.");
    }
    if (time_step <= scalar_type{} || reference_density <= scalar_type{})
    {
        throw std::invalid_argument("physical_momentum_transport_system requires positive time step "
                                    "and reference density.");
    }
    validate_boundary_coefficient_cache(mesh, boundary_dynamic_viscosity, "physical_momentum_transport_system");

    auto boundary_viscosity = [&](int batch_id, size_t in_batch_id, scalar_type owner_cell_value)
    { return boundary_coefficient<Pack>(boundary_dynamic_viscosity, batch_id, in_batch_id, owner_cell_value); };

    const auto implicit_weight = non_orthogonal_weights.implicit;
    const auto explicit_weight = non_orthogonal_weights.explicit_;

    using geometry_cache_type = TransportGeometryCache<Mesh<Pack>>;
    typename geometry_cache_type::boundary_locations_type local_boundary_locations;
    std::vector<detail::VectorAffineLeastSquaresGradientStencil<Mesh<Pack>>> gradient_stencils;
    if (geometry_cache == nullptr)
    {
        gradient_stencils = detail::vector_affine_gradient_stencils(mesh, boundary_value);
        local_boundary_locations = detail::boundary_face_locations(mesh);
    }
    else
    {
        geometry_cache->require_mesh(mesh);
        gradient_stencils = geometry_cache->vector_affine_stencils(boundary_value);
    }
    const auto& boundary_locations =
        geometry_cache == nullptr ? local_boundary_locations : geometry_cache->boundary_locations();

    std::unique_ptr<TensorCellField<Pack>> partition_gradients;
    if (implicit_weight > scalar_type{})
    {
        partition_gradients = std::make_unique<TensorCellField<Pack>>(
            old_velocity.mesh_ptr(), "partition_momentum_non_orthogonal_gradient");
        const auto& lagged_field = correction_field == nullptr ? old_velocity : *correction_field;
        detail::evaluate_vector_affine_gradients(lagged_field, gradient_stencils, *partition_gradients);
        partition_gradients->sync_ghosts();
    }

    const auto prepared = detail::prepare_transport_matrix<Pack>(mesh, std::move(cached_matrix), 32);
    const auto& matrix = prepared.matrix;
    auto rhs = Teuchos::rcp(new typename Pack::multi_vector_type(mesh.owned_cell_map(), components, true));

    detail::FlatMatrixRow<local_ordinal_type, scalar_type> row_values(mesh.num_local_cells(), 64);
    const auto old_velocity_data = old_velocity.owned_read_view();
    const auto face_flux_data = face_fluxes.owned_read_view();
    const auto viscosity_data = dynamic_viscosity.local_read_view();
    using partition_gradient_view_type = decltype(partition_gradients->local_read_view());
    std::optional<partition_gradient_view_type> partition_gradient_data;
    if (partition_gradients != nullptr)
    {
        partition_gradient_data.emplace(partition_gradients->local_read_view());
    }

    for (size_t owned = 0; owned < mesh.num_owned_cells(); ++owned)
    {
        const auto cell_lid = static_cast<local_ordinal_type>(owned);
        const auto volume = mesh.cell_volume(cell_lid);
        const auto transient = volume / time_step;
        const auto source_value = acceleration_source(cell_lid);

        row_values.clear();
        detail::add_matrix_entry(row_values, cell_lid, transient);
        for (size_t component = 0; component < components; ++component)
        {
            rhs->replaceLocalValue(cell_lid, component,
                transient * old_velocity_data(cell_lid, component) + volume * source_value.component(component));
        }

        auto add_non_orthogonal_stencil = [&](local_ordinal_type gradient_cell_lid, scalar_type gradient_weight,
                                              scalar_type face_kinematic_viscosity,
                                              const typename Mesh<Pack>::Vec3& tangential_area)
        {
            if (implicit_weight == scalar_type{} || gradient_weight == scalar_type{} ||
                !mesh.is_owned_cell(gradient_cell_lid) ||
                static_cast<size_t>(gradient_cell_lid) >= gradient_stencils.size())
            {
                return;
            }
            const auto scale = -implicit_weight * face_kinematic_viscosity * gradient_weight;
            const auto& stencil = gradient_stencils[static_cast<size_t>(gradient_cell_lid)];
            for (size_t component = 0; component < components; ++component)
            {
                rhs->sumIntoLocalValue(cell_lid, component, -scale * stencil.constants[component].dot(tangential_area));
            }
            for (const auto& entry : stencil.entries)
            {
                detail::add_matrix_entry(row_values, entry.cell_lid, scale * entry.coefficient.dot(tangential_area));
            }
        };

        for (const auto face_lid : mesh.faces(cell_lid))
        {
            const auto owner_oriented_flux = face_fluxes.is_owned_face(face_lid)
                                                 ? face_flux_data(face_fluxes.owned_row(face_lid), 0)
                                                 : scalar_type{};
            const auto out_flux = mesh.owner_cell(face_lid) == cell_lid ? owner_oriented_flux : -owner_oriented_flux;
            if (out_flux >= scalar_type{})
            {
                detail::add_matrix_entry(row_values, cell_lid, out_flux);
            }
            else if (mesh.is_interior_face(face_lid))
            {
                const auto other = mesh.opposite_or_periodic_neighbor_cell(face_lid, cell_lid);
                detail::add_matrix_entry(row_values, other, out_flux);
            }
            else if (mesh.is_boundary_face(face_lid) && static_cast<size_t>(face_lid) < boundary_locations.size() &&
                     boundary_locations[static_cast<size_t>(face_lid)].active)
            {
                const auto location = boundary_locations[static_cast<size_t>(face_lid)];
                const auto face_value = boundary_value(location.batch_id, location.in_batch_id);
                for (size_t component = 0; component < components; ++component)
                {
                    rhs->sumIntoLocalValue(cell_lid, component, -out_flux * face_value.component(component));
                }
            }

            if (mesh.is_interior_face(face_lid))
            {
                const auto other = mesh.opposite_or_periodic_neighbor_cell(face_lid, cell_lid);
                row_values.ensure(other);
                const auto face_kinematic_viscosity =
                    detail::face_coefficient_value(mesh, face_lid, cell_lid, other, viscosity_data(cell_lid, 0),
                        viscosity_data(other, 0), coefficient_interpolation) /
                    reference_density;
                if (face_kinematic_viscosity <= scalar_type{})
                {
                    continue;
                }
                const auto coefficient =
                    detail::interior_diffusion_coefficient(mesh, face_lid, cell_lid, other, face_kinematic_viscosity);
                detail::add_matrix_entry(row_values, cell_lid, coefficient);
                detail::add_matrix_entry(row_values, other, -coefficient);

                const auto tangential_area = detail::non_orthogonal_area_vector(
                    mesh.face_area_vector_outward(face_lid, cell_lid), mesh.cell_center_vector(face_lid, cell_lid));
                if (mesh.is_owned_cell(other) && static_cast<size_t>(other) < gradient_stencils.size())
                {
                    add_non_orthogonal_stencil(cell_lid, scalar_type{0.5}, face_kinematic_viscosity, tangential_area);
                    add_non_orthogonal_stencil(other, scalar_type{0.5}, face_kinematic_viscosity, tangential_area);
                }
                else if (implicit_weight > scalar_type{})
                {
                    add_non_orthogonal_stencil(cell_lid, scalar_type{0.5}, face_kinematic_viscosity, tangential_area);
                    if (!partition_gradient_data)
                    {
                        throw std::logic_error("Partition-face implicit momentum reconstruction "
                                               "requires synchronized remote gradients.");
                    }
                    const auto remote_gradient = detail::tensor_view_value<Pack>(*partition_gradient_data, other);
                    for (size_t component = 0; component < components; ++component)
                    {
                        rhs->sumIntoLocalValue(cell_lid, component,
                            implicit_weight * face_kinematic_viscosity * scalar_type{0.5} *
                                remote_gradient[component].dot(tangential_area));
                    }
                }
                continue;
            }

            if (!mesh.is_boundary_face(face_lid) || static_cast<size_t>(face_lid) >= boundary_locations.size() ||
                !boundary_locations[static_cast<size_t>(face_lid)].active)
            {
                continue;
            }
            const auto location = boundary_locations[static_cast<size_t>(face_lid)];
            if (!boundary_diffusion(location.batch_id, location.in_batch_id))
            {
                continue;
            }
            const auto face_kinematic_viscosity =
                boundary_viscosity(location.batch_id, location.in_batch_id, viscosity_data(cell_lid, 0)) /
                reference_density;
            const auto coefficient =
                detail::boundary_diffusion_coefficient(mesh, face_lid, cell_lid, face_kinematic_viscosity);
            if (coefficient > scalar_type{})
            {
                detail::add_matrix_entry(row_values, cell_lid, coefficient);
                const auto face_value = boundary_value(location.batch_id, location.in_batch_id);
                for (size_t component = 0; component < components; ++component)
                {
                    rhs->sumIntoLocalValue(cell_lid, component, coefficient * face_value.component(component));
                }
            }

            const auto tangential_area =
                detail::non_orthogonal_area_vector(mesh.face_area_vector_outward(face_lid, cell_lid),
                    mesh.face_centroid(face_lid) - mesh.cell_centroid(cell_lid));
            add_non_orthogonal_stencil(cell_lid, scalar_type{1}, face_kinematic_viscosity, tangential_area);
        }

        detail::add_transport_values<Pack>(prepared, cell_lid, row_values);
    }

    if (correction_field != nullptr && explicit_weight > scalar_type{})
    {
        add_variable_explicit_non_orthogonal_correction<Pack>(*correction_field, dynamic_viscosity, boundary_value,
            *rhs, explicit_weight / reference_density, boundary_diffusion, boundary_viscosity, &gradient_stencils,
            &boundary_locations, coefficient_interpolation);
    }

    add_explicit_deviatoric_transpose_gradient_stress<Pack>(old_velocity, dynamic_viscosity, reference_density,
        boundary_value, *rhs, boundary_diffusion, boundary_viscosity, &gradient_stencils, &boundary_locations,
        coefficient_interpolation);

    matrix->fillComplete();
    return {matrix, rhs};
}

template<TpetraTypePack Pack>
VectorTransportSystem<Pack> transport_system(const VectorCellField<Pack>& old_values,
    const FaceField<Pack>& face_fluxes, typename Pack::scalar_type time_step, typename Pack::scalar_type diffusivity,
    VectorBoundaryValueProvider<Pack> boundary_value, Teuchos::RCP<typename Pack::matrix_type> cached_matrix)
{
    using local_ordinal_type = typename Pack::local_ordinal_type;
    using vec_type = typename VectorCellField<Pack>::vec_type;

    auto zero_source = [](local_ordinal_type) -> vec_type { return vec_type{}; };

    return transport_system<Pack>(
        old_values, face_fluxes, time_step, diffusivity, boundary_value, zero_source, cached_matrix);
}

} // namespace SimpleFluid::FVM
