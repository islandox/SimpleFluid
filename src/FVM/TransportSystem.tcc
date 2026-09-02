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

#include <array>
#include <cmath>
#include <exception>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace SimpleFluid::FVM
{
namespace detail
{

template<class MeshType, size_t StateCount>
std::array<int, StateCount> reduce_transport_validation_state(
    const MeshType& mesh,
    const std::array<int, StateCount>& local_state)
{
    auto global_state = local_state;
    const auto communicator = mesh.owned_cell_map()->getComm();
    if (communicator->getSize() > 1)
    {
        Teuchos::reduceAll(
            *communicator,
            Teuchos::REDUCE_MAX,
            static_cast<int>(StateCount),
            local_state.data(),
            global_state.data());
    }
    return global_state;
}

template<TpetraTypePack Pack> struct TransportMatrixRow
{
    std::vector<typename Pack::local_ordinal_type> columns;
    std::vector<typename Pack::scalar_type> values;
};

template<TpetraTypePack Pack>
TransportMatrixRow<Pack> capture_transport_row(
    const FlatMatrixRow<typename Pack::local_ordinal_type,
        typename Pack::scalar_type>& row_values)
{
    TransportMatrixRow<Pack> row;
    row.columns.assign(
        row_values.column_data(),
        row_values.column_data() + row_values.size());
    row.values.assign(
        row_values.value_data(),
        row_values.value_data() + row_values.size());
    return row;
}

template<TpetraTypePack Pack, class MeshType>
void validate_transport_matrix_graph(
    const MeshType& mesh,
    const Teuchos::RCP<typename Pack::matrix_type>& cached_matrix,
    const std::vector<TransportMatrixRow<Pack>>& rows)
{
    using local_ordinal_type = typename Pack::local_ordinal_type;
    using matrix_type = typename Pack::matrix_type;

    int incompatible_graph = 0;
    if (!cached_matrix.is_null())
    {
        try
        {
            for (size_t row = 0;
                 row < rows.size() && incompatible_graph == 0;
                 ++row)
            {
                typename matrix_type::local_inds_host_view_type
                    cached_columns;
                typename matrix_type::values_host_view_type cached_values;
                cached_matrix->getLocalRowView(
                    static_cast<local_ordinal_type>(row),
                    cached_columns,
                    cached_values);
                for (const auto required_column : rows[row].columns)
                {
                    bool found = false;
                    for (size_t entry = 0;
                         entry < cached_columns.extent(0);
                         ++entry)
                    {
                        found = found
                            || cached_columns[entry] == required_column;
                    }
                    incompatible_graph = incompatible_graph || !found;
                }
            }
        }
        catch (const std::exception&)
        {
            incompatible_graph = 1;
        }
    }

    const auto graph_state = reduce_transport_validation_state(
        mesh, std::array<int, 1>{incompatible_graph});
    if (graph_state[0] != 0)
    {
        throw std::invalid_argument(
            "transport_system cached matrix graph is incompatible with "
            "the operator.");
    }
}

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
TransportSystem<Pack> non_orthogonal_transport_system(const CellField<Pack>& old_values,
    const FaceField<Pack>& face_fluxes, typename Pack::scalar_type time_step, typename Pack::scalar_type diffusivity,
    ScalarBoundaryConditionProvider<Pack> boundary_condition, ScalarBoundaryValueProvider<Pack> boundary_value,
    ScalarCellValueProvider<Pack> right_hand_source, NonOrthogonalTreatment treatment,
    const CellField<Pack>* correction_field, Teuchos::RCP<typename Pack::matrix_type> cached_matrix,
    const TransportGeometryCache<Mesh<Pack>>* geometry_cache)
{
    using scalar_type = typename Pack::scalar_type;

    const auto& mesh = old_values.mesh();
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
    const int local_validation_state[4]{&face_fluxes.mesh() != &mesh ? 1 : 0,
        !std::isfinite(time_step) || time_step <= scalar_type{} ? 1 : 0,
        !std::isfinite(diffusivity) || diffusivity < scalar_type{} ? 1 : 0, invalid_geometry_cache};
    int validation_state[4]{};
    Teuchos::reduceAll(*mesh.owned_cell_map()->getComm(), Teuchos::REDUCE_MAX, 4, local_validation_state,
        validation_state);
    if (validation_state[0] != 0)
    {
        throw std::invalid_argument(
            "non_orthogonal_transport_system requires face fluxes on the old-value mesh.");
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

    CellField<Pack> storage_weight(old_values.mesh_ptr(), scalar_type{1}, "constant_storage_weight");
    CellField<Pack> advection_weight(old_values.mesh_ptr(), scalar_type{1}, "constant_advection_weight");
    CellField<Pack> diffusivity_field(old_values.mesh_ptr(), diffusivity, "constant_diffusivity");
    return weighted_scalar_transport_system<Pack>(old_values, face_fluxes, time_step, storage_weight,
        advection_weight, diffusivity_field, std::move(boundary_condition), std::move(boundary_value),
        std::move(right_hand_source), treatment, correction_field, std::move(cached_matrix), {}, {}, nullptr,
        geometry_cache, FaceCoefficientInterpolation::Harmonic);
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
    return weighted_scalar_transport_system<Pack>(WeightedScalarTransportRequest<Pack>{
        .old_values = old_values,
        .face_fluxes = face_fluxes,
        .time_step = time_step,
        .storage_weight = storage_weight,
        .advection_weight = advection_weight,
        .diffusivity = diffusivity,
        .boundary_condition = std::move(boundary_condition),
        .boundary_value = std::move(boundary_value),
        .source = std::move(source),
        .treatment = treatment,
        .correction_field = correction_field,
        .cached_matrix = std::move(cached_matrix),
        .implicit_sink = std::move(implicit_sink),
        .fixed_cell_value = std::move(fixed_cell_value),
        .boundary_diffusivity = boundary_diffusivity,
        .geometry_cache = geometry_cache,
        .coefficient_interpolation = coefficient_interpolation});
}

template<TpetraTypePack Pack>
TransportSystem<Pack> weighted_scalar_transport_system(const CellField<Pack>& old_values,
    const FaceField<Pack>& face_fluxes, typename Pack::scalar_type time_step, const CellField<Pack>& storage_weight,
    const CellField<Pack>& advection_weight, const CellField<Pack>& diffusivity,
    ScalarBoundaryConditionProvider<Pack> boundary_condition, ScalarBoundaryValueProvider<Pack> boundary_value,
    ScalarCellValueProvider<Pack> source, ScalarTransportDiscretization discretization,
    NonOrthogonalTreatment treatment, const CellField<Pack>* older_values, const CellField<Pack>* correction_field,
    Teuchos::RCP<typename Pack::matrix_type> cached_matrix,
    std::function<typename Pack::scalar_type(typename Pack::local_ordinal_type)> implicit_sink,
    std::function<std::optional<typename Pack::scalar_type>(typename Pack::local_ordinal_type)> fixed_cell_value,
    const BoundaryCache<Pack>* boundary_diffusivity, const TransportGeometryCache<Mesh<Pack>>* geometry_cache,
    FaceCoefficientInterpolation coefficient_interpolation)
{
    return weighted_scalar_transport_system<Pack>(WeightedScalarTransportRequest<Pack>{
        .old_values = old_values,
        .face_fluxes = face_fluxes,
        .time_step = time_step,
        .storage_weight = storage_weight,
        .advection_weight = advection_weight,
        .diffusivity = diffusivity,
        .boundary_condition = std::move(boundary_condition),
        .boundary_value = std::move(boundary_value),
        .source = std::move(source),
        .treatment = treatment,
        .discretization = discretization,
        .older_values = older_values,
        .correction_field = correction_field,
        .cached_matrix = std::move(cached_matrix),
        .implicit_sink = std::move(implicit_sink),
        .fixed_cell_value = std::move(fixed_cell_value),
        .boundary_diffusivity = boundary_diffusivity,
        .geometry_cache = geometry_cache,
        .coefficient_interpolation = coefficient_interpolation});
}

template<TpetraTypePack Pack>
TransportSystem<Pack> weighted_scalar_transport_system(WeightedScalarTransportRequest<Pack> request)
{
    using scalar_type = typename Pack::scalar_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;

    const auto& old_values = request.old_values;
    const auto& face_fluxes = request.face_fluxes;
    const auto time_step = request.time_step;
    const auto& storage_weight = request.storage_weight;
    const auto& advection_weight = request.advection_weight;
    const auto& diffusivity = request.diffusivity;
    auto boundary_condition = std::move(request.boundary_condition);
    auto boundary_value = std::move(request.boundary_value);
    auto source = std::move(request.source);
    const auto treatment = request.treatment;
    const auto discretization = request.discretization;
    const auto* older_values = request.older_values;
    const auto* correction_field = request.correction_field;
    auto cached_matrix = std::move(request.cached_matrix);
    auto implicit_sink = std::move(request.implicit_sink);
    auto fixed_cell_value = std::move(request.fixed_cell_value);
    const auto* boundary_diffusivity = request.boundary_diffusivity;
    const auto* geometry_cache = request.geometry_cache;
    const auto coefficient_interpolation = request.coefficient_interpolation;

    const auto& mesh = old_values.mesh();
    const auto older_field_state =
        older_values == nullptr
            ? 0
            : &older_values->mesh() == &mesh ? 1 : 2;
    detail::validate_scalar_transport_discretization(
        mesh,
        discretization,
        older_field_state,
        "weighted_scalar_transport_system");
    const auto transient_coefficients =
        detail::scalar_transient_coefficients<scalar_type>(
            discretization.time);
    const auto non_orthogonal_weights = detail::validate_non_orthogonal_transport_selection<Pack>(
        mesh, treatment, correction_field, "weighted_scalar_transport_system");

    const int incompatible_fields =
        &face_fluxes.mesh() != &mesh
             || &storage_weight.mesh() != &mesh
             || &advection_weight.mesh() != &mesh
             || &diffusivity.mesh() != &mesh
            ? 1
            : 0;
    int invalid_boundary_cache = 0;
    try
    {
        validate_boundary_coefficient_cache(
            mesh,
            boundary_diffusivity,
            "weighted_scalar_transport_system");
    }
    catch (...)
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
        catch (...)
        {
            invalid_geometry_cache = 1;
        }
    }
    int invalid_coefficients = 0;
    if (incompatible_fields == 0)
    {
        try
        {
            for (size_t owned = 0; owned < mesh.num_owned_cells(); ++owned)
            {
                const auto cell_lid =
                    static_cast<local_ordinal_type>(owned);
                const auto storage = storage_weight.local_value(cell_lid);
                const auto advection = advection_weight.local_value(cell_lid);
                const auto diffusion = diffusivity.local_value(cell_lid);
                invalid_coefficients = invalid_coefficients
                    || !std::isfinite(storage)
                    || !std::isfinite(advection)
                    || !std::isfinite(diffusion)
                    || storage <= scalar_type{}
                    || advection < scalar_type{}
                    || diffusion < scalar_type{};
            }
        }
        catch (...)
        {
            invalid_coefficients = 1;
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
    const int matrix_cache_state = cached_matrix.is_null() ? 0 : 1;
    const auto matrix_cache_categories =
        detail::reduce_transport_validation_state(
            mesh,
            std::array<int, 2>{
                matrix_cache_state,
                -matrix_cache_state});
    if (-matrix_cache_categories[1] != matrix_cache_categories[0])
    {
        throw std::invalid_argument(
            "transport_system requires every rank to use the same "
            "cached-matrix category.");
    }
    int invalid_matrix_cache_structure = 0;
    if (!cached_matrix.is_null())
    {
        try
        {
            const auto row_map = cached_matrix->getRowMap();
            const auto column_map = cached_matrix->getColMap();
            const auto domain_map = cached_matrix->getDomainMap();
            invalid_matrix_cache_structure =
                !cached_matrix->isFillComplete()
                || row_map.is_null()
                || column_map.is_null()
                || domain_map.is_null();
        }
        catch (...)
        {
            invalid_matrix_cache_structure = 1;
        }
    }
    const auto matrix_structure_state =
        detail::reduce_transport_validation_state(
            mesh,
            std::array<int, 1>{invalid_matrix_cache_structure});
    if (matrix_structure_state[0] != 0)
    {
        throw std::invalid_argument(
            "transport_system cached matrix is incompatible with the mesh.");
    }
    int invalid_matrix_cache_maps = 0;
    if (!cached_matrix.is_null())
    {
        const auto row_map = cached_matrix->getRowMap();
        const auto column_map = cached_matrix->getColMap();
        const auto domain_map = cached_matrix->getDomainMap();
        const int incompatible_row_map =
            !row_map->isSameAs(*mesh.owned_cell_map()) ? 1 : 0;
        const int incompatible_column_map =
            !column_map->isSameAs(*mesh.overlap_cell_map()) ? 1 : 0;
        const int incompatible_domain_map =
            !domain_map->isSameAs(*mesh.owned_cell_map()) ? 1 : 0;
        invalid_matrix_cache_maps = incompatible_row_map
            || incompatible_column_map
            || incompatible_domain_map;
    }

    const auto validation_state =
        detail::reduce_transport_validation_state(
            mesh,
            std::array<int, 9>{
                incompatible_fields,
                !std::isfinite(time_step) || time_step <= scalar_type{}
                    ? 1
                    : 0,
                invalid_boundary_cache,
                invalid_geometry_cache,
                invalid_coefficients,
                interpolation_state,
                -interpolation_state,
                source ? 0 : 1,
                invalid_matrix_cache_maps});
    if (validation_state[0] != 0)
    {
        throw std::invalid_argument(
            "weighted_scalar_transport_system requires all fields on "
            "the transported-field mesh.");
    }
    if (validation_state[1] != 0)
    {
        throw std::invalid_argument(
            "weighted_scalar_transport_system requires a positive "
            "time step.");
    }
    if (!detail::scalar_transport_value_agrees(mesh, time_step))
    {
        throw std::invalid_argument(
            "weighted_scalar_transport_system requires every rank to use "
            "the same time step.");
    }
    if (validation_state[2] != 0)
    {
        throw std::invalid_argument(
            "weighted_scalar_transport_system received an invalid "
            "boundary-coefficient cache.");
    }
    if (validation_state[3] != 0)
    {
        throw std::invalid_argument(
            "weighted_scalar_transport_system received a geometry cache "
            "on the wrong mesh.");
    }
    if (validation_state[4] != 0)
    {
        throw std::invalid_argument(
            "weighted scalar transport requires finite positive storage "
            "and finite non-negative advection and diffusion coefficients.");
    }
    if (validation_state[5] == 2
        || -validation_state[6] != validation_state[5])
    {
        throw std::invalid_argument(
            "weighted_scalar_transport_system requires every rank to use "
            "the same valid face-coefficient interpolation.");
    }
    if (validation_state[7] != 0)
    {
        throw std::invalid_argument(
            "weighted scalar transport requires a source provider.");
    }
    if (validation_state[8] != 0)
    {
        throw std::invalid_argument(
            "transport_system cached matrix is incompatible with the mesh.");
    }

    std::vector<scalar_type> sink_values(
        mesh.num_owned_cells(), scalar_type{});
    std::vector<std::optional<scalar_type>> fixed_cell_values(
        mesh.num_owned_cells());
    std::vector<scalar_type> source_values(
        mesh.num_owned_cells(), scalar_type{});
    int invalid_sink = 0;
    int invalid_fixed_value = 0;
    int invalid_source = 0;
    std::exception_ptr local_callback_error;
    try
    {
        for (size_t owned = 0; owned < mesh.num_owned_cells(); ++owned)
        {
            const auto cell_lid =
                static_cast<local_ordinal_type>(owned);
            const auto fixed = fixed_cell_value
                ? fixed_cell_value(cell_lid)
                : std::optional<scalar_type>{};
            invalid_fixed_value = invalid_fixed_value
                || (fixed.has_value() && !std::isfinite(*fixed));
            fixed_cell_values[owned] = fixed;
        }
        for (size_t owned = 0; owned < mesh.num_owned_cells(); ++owned)
        {
            const auto cell_lid =
                static_cast<local_ordinal_type>(owned);
            const auto sink = implicit_sink
                ? implicit_sink(cell_lid)
                : scalar_type{};
            const auto source_value = source(cell_lid);
            invalid_sink = invalid_sink
                || !std::isfinite(sink)
                || sink < scalar_type{};
            invalid_source = invalid_source
                || !std::isfinite(source_value);
            sink_values[owned] = sink;
            source_values[owned] = source_value;
        }
    }
    catch (...)
    {
        local_callback_error = std::current_exception();
    }
    const auto callback_validation =
        detail::reduce_transport_validation_state(
            mesh,
            std::array<int, 4>{
                local_callback_error ? 1 : 0,
                invalid_sink,
                invalid_fixed_value,
                invalid_source});
    if (callback_validation[0] != 0)
    {
        if (local_callback_error)
        {
            std::rethrow_exception(local_callback_error);
        }
        throw std::runtime_error(
            "weighted scalar transport callback failed on another rank.");
    }
    if (callback_validation[1] != 0)
    {
        throw std::invalid_argument(
            "weighted scalar transport requires a finite, non-negative "
            "implicit sink.");
    }
    if (callback_validation[2] != 0)
    {
        throw std::invalid_argument(
            "weighted scalar transport requires finite fixed-cell values.");
    }
    if (callback_validation[3] != 0)
    {
        throw std::invalid_argument(
            "weighted scalar transport requires finite source values.");
    }

    using geometry_cache_type = TransportGeometryCache<Mesh<Pack>>;
    typename geometry_cache_type::boundary_locations_type local_boundary_locations;
    if (geometry_cache == nullptr)
    {
        local_boundary_locations = detail::boundary_face_locations(mesh);
    }
    const auto& boundary_locations =
        geometry_cache == nullptr ? local_boundary_locations : geometry_cache->boundary_locations();

    std::vector<BoundaryCondition> boundary_conditions(
        boundary_locations.size());
    std::vector<scalar_type> boundary_values(
        boundary_locations.size(), scalar_type{});
    const auto physical_boundary_faces =
        detail::physical_boundary_face_mask(mesh);
    std::map<std::pair<int, size_t>, size_t> boundary_indices;
    int unsupported_boundary_condition = 0;
    int invalid_boundary_condition_value = 0;
    int invalid_boundary_value = 0;
    std::exception_ptr local_boundary_callback_error;
    try
    {
        for (size_t face = 0; face < boundary_locations.size(); ++face)
        {
            const auto& location = boundary_locations[face];
            if (!location.active || !physical_boundary_faces[face])
            {
                continue;
            }
            boundary_indices.emplace(
                std::pair{location.batch_id, location.in_batch_id}, face);
            const auto condition =
                boundary_condition(location.batch_id, location.in_batch_id);
            const auto value =
                boundary_value(location.batch_id, location.in_batch_id);
            unsupported_boundary_condition = unsupported_boundary_condition
                || (condition.type != BoundaryConditionType::Dirichlet
                    && condition.type != BoundaryConditionType::Neumann);
            invalid_boundary_condition_value =
                invalid_boundary_condition_value
                || !std::isfinite(condition.value);
            invalid_boundary_value = invalid_boundary_value
                || !std::isfinite(value);
            boundary_conditions[face] = condition;
            boundary_values[face] = value;
        }
    }
    catch (...)
    {
        local_boundary_callback_error = std::current_exception();
    }
    const auto boundary_callback_validation =
        detail::reduce_transport_validation_state(
            mesh,
            std::array<int, 4>{
                local_boundary_callback_error ? 1 : 0,
                unsupported_boundary_condition,
                invalid_boundary_condition_value,
                invalid_boundary_value});
    if (boundary_callback_validation[0] != 0)
    {
        if (local_boundary_callback_error)
        {
            std::rethrow_exception(local_boundary_callback_error);
        }
        throw std::runtime_error(
            "weighted scalar transport boundary callback failed on another rank.");
    }
    if (boundary_callback_validation[1] != 0)
    {
        throw std::invalid_argument(
            "weighted scalar transport supports only Dirichlet and Neumann boundaries.");
    }
    if (boundary_callback_validation[2] != 0)
    {
        throw std::invalid_argument(
            "weighted scalar transport requires finite boundary-condition values.");
    }
    if (boundary_callback_validation[3] != 0)
    {
        throw std::invalid_argument(
            "weighted scalar transport requires finite boundary values.");
    }

    const auto boundary_index =
        [&](int batch_id, size_t in_batch_id)
    {
        const auto iter = boundary_indices.find(
            std::pair{batch_id, in_batch_id});
        if (iter == boundary_indices.end())
        {
            throw std::logic_error(
                "weighted scalar transport requested an unknown boundary face.");
        }
        return iter->second;
    };
    const auto cached_boundary_condition =
        [&](int batch_id, size_t in_batch_id)
    {
        return boundary_conditions[boundary_index(batch_id, in_batch_id)];
    };
    const auto cached_boundary_value =
        [&](int batch_id, size_t in_batch_id)
    {
        return boundary_values[boundary_index(batch_id, in_batch_id)];
    };

    auto boundary_face_diffusivity =
        [&](int batch_id,
            size_t in_batch_id,
            scalar_type owner_cell_value)
    {
        return boundary_coefficient<Pack>(
            boundary_diffusivity,
            batch_id,
            in_batch_id,
            owner_cell_value);
    };

    const auto implicit_weight = non_orthogonal_weights.implicit;
    const auto explicit_weight = non_orthogonal_weights.explicit_;

    std::vector<detail::AffineLeastSquaresGradientStencil<Mesh<Pack>>> gradient_stencils;
    if (geometry_cache == nullptr)
    {
        gradient_stencils = detail::scalar_affine_gradient_stencils(
            mesh, cached_boundary_condition, cached_boundary_value);
    }
    else
    {
        gradient_stencils = geometry_cache->scalar_affine_stencils(
            cached_boundary_condition, cached_boundary_value);
    }

    std::unique_ptr<VectorCellField<Pack>> partition_gradients;
    if (implicit_weight > scalar_type{})
    {
        partition_gradients =
            std::make_unique<VectorCellField<Pack>>(old_values.mesh_ptr(), "partition_scalar_non_orthogonal_gradient");
        const auto& lagged_field = correction_field == nullptr ? old_values : *correction_field;
        detail::evaluate_scalar_affine_gradients(lagged_field, gradient_stencils, *partition_gradients);
        partition_gradients->sync_ghosts();
    }

    std::unique_ptr<VectorCellField<Pack>> convection_gradients;
    if (discretization.convection
        == ScalarConvectionScheme::BoundedLinearUpwind)
    {
        convection_gradients = std::make_unique<VectorCellField<Pack>>(
            old_values.mesh_ptr(),
            "bounded_linear_upwind_gradient");
        detail::evaluate_scalar_affine_gradients(
            old_values, gradient_stencils, *convection_gradients);
        convection_gradients->sync_ghosts();
    }

    auto rhs = Teuchos::rcp(new typename Pack::vector_type(mesh.owned_cell_map(), true));

    detail::FlatMatrixRow<local_ordinal_type, scalar_type> row_values(mesh.num_local_cells(), 64);
    std::vector<detail::TransportMatrixRow<Pack>> rows;
    rows.reserve(mesh.num_owned_cells());
    const auto old_value_data = old_values.owned_read_view();
    using old_value_view_type = decltype(old_value_data);
    std::optional<old_value_view_type> older_value_data;
    if (older_values != nullptr)
    {
        older_value_data.emplace(older_values->owned_read_view());
    }
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
        const auto sink = sink_values[owned];

        row_values.clear();
        detail::add_matrix_entry(
            row_values,
            cell_lid,
            transient_coefficients.diagonal * transient
                + volume * sink);
        rhs->replaceLocalValue(
            cell_lid,
            transient
                    * (transient_coefficients.previous
                           * old_value_data(cell_lid, 0)
                       + transient_coefficients.older
                           * (older_value_data
                                  ? (*older_value_data)(cell_lid, 0)
                                  : scalar_type{}))
                + volume * source_values[owned]);

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
                    cell_lid,
                    -out_flux * cell_advection
                        * cached_boundary_value(
                            location.batch_id, location.in_batch_id));
            }

            if (mesh.is_interior_face(face_lid)
                && convection_gradients != nullptr)
            {
                const auto other =
                    mesh.opposite_or_periodic_neighbor_cell(
                        face_lid, cell_lid);
                const auto upwind = out_flux >= scalar_type{}
                    ? cell_lid
                    : other;
                const auto downwind = out_flux >= scalar_type{}
                    ? other
                    : cell_lid;
                const auto upwind_value = old_values.local_value(upwind);
                const auto face_value =
                    detail::bounded_linear_upwind_face_value(
                        upwind_value,
                        old_values.local_value(downwind),
                        convection_gradients->local_value(upwind),
                        detail::cell_to_face_displacement(
                            mesh, face_lid, upwind));
                rhs->sumIntoLocalValue(
                    cell_lid,
                    detail::deferred_convection_rhs_correction(
                        out_flux,
                        advection_data(upwind, 0),
                        upwind_value,
                        face_value));
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
            const auto condition = cached_boundary_condition(
                location.batch_id, location.in_batch_id);
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
                        cell_lid,
                        coefficient
                            * cached_boundary_value(
                                location.batch_id, location.in_batch_id));
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

        rows.push_back(detail::capture_transport_row<Pack>(row_values));
    }

    if (correction_field != nullptr && explicit_weight > scalar_type{})
    {
        add_variable_explicit_non_orthogonal_correction<Pack>(*correction_field, diffusivity,
            cached_boundary_condition, cached_boundary_value, *rhs, explicit_weight, boundary_face_diffusivity,
            &gradient_stencils,
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

    detail::validate_transport_matrix_graph<Pack>(
        mesh, cached_matrix, rows);

    const auto prepared = detail::prepare_transport_matrix<Pack>(
        mesh, std::move(cached_matrix), 32);
    const auto& matrix = prepared.matrix;
    for (size_t row = 0; row < rows.size(); ++row)
    {
        const auto row_size =
            SimpleFluid::detail::checked_size_to_ordinal<Teuchos::Ordinal>(
                rows[row].columns.size(),
                "transport row entry count");
        detail::add_transport_values<Pack>(
            prepared,
            static_cast<local_ordinal_type>(row),
            Teuchos::arrayView(rows[row].columns.data(), row_size),
            Teuchos::arrayView(rows[row].values.data(), row_size));
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
