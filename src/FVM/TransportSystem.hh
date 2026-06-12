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

#include "fields/CellField.hh"
#include "fields/FaceField.hh"
#include "fields/VectorCellField.hh"
#include "FVM/NonOrthogonalCorrection.hh"
#include "FVM/OperatorDetails.hh"
#include "geometry/Mesh.hh"

#include <Teuchos_Array.hpp>
#include <Teuchos_RCP.hpp>

#include <concepts>
#include <cstddef>
#include <stdexcept>
#include <unordered_map>

namespace SimpleFluid::FVM
{
namespace detail
{

template<TpetraTypePack Pack>
struct PreparedTransportMatrix
{
    Teuchos::RCP<typename Pack::matrix_type> matrix;
    bool reused = false;
};

template<TpetraTypePack Pack, class MeshType>
PreparedTransportMatrix<Pack> prepare_transport_matrix(
    const MeshType& mesh,
    Teuchos::RCP<typename Pack::matrix_type> cached_matrix,
    size_t entries_per_row)
{
    using matrix_type = typename Pack::matrix_type;

    if (cached_matrix.is_null())
    {
        return {
            Teuchos::rcp(new matrix_type(
                mesh.owned_cell_map(),
                mesh.overlap_cell_map(),
                entries_per_row)),
            false};
    }
    if (!cached_matrix->isFillComplete()
        || !cached_matrix->getRowMap()->isSameAs(*mesh.owned_cell_map())
        || cached_matrix->getColMap().is_null()
        || !cached_matrix->getColMap()->isSameAs(*mesh.overlap_cell_map())
        || !cached_matrix->getDomainMap()->isSameAs(*mesh.owned_cell_map()))
    {
        throw std::invalid_argument(
            "transport_system cached matrix is incompatible with the mesh.");
    }

    cached_matrix->resumeFill();
    cached_matrix->setAllToScalar(typename Pack::scalar_type{});
    return {cached_matrix, true};
}

template<TpetraTypePack Pack>
void add_transport_values(
    const PreparedTransportMatrix<Pack>& prepared,
    typename Pack::local_ordinal_type row,
    const Teuchos::ArrayView<const typename Pack::local_ordinal_type>& columns,
    const Teuchos::ArrayView<const typename Pack::scalar_type>& values)
{
    if (!prepared.reused)
    {
        prepared.matrix->insertLocalValues(row, columns, values);
        return;
    }

    const auto updated =
        prepared.matrix->sumIntoLocalValues(row, columns, values);
    if (updated != static_cast<typename Pack::local_ordinal_type>(
                       columns.size()))
    {
        throw std::invalid_argument(
            "transport_system cached matrix graph is incompatible with the operator.");
    }
}

} // namespace detail

/**
 * @brief Holds the assembled left-hand-side matrix and right-hand-side
 *        vector for a semi-implicit transport step.
 *
 * @tparam Pack The Tpetra type pack.
 */
template<TpetraTypePack Pack>
struct TransportSystem
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
template<TpetraTypePack Pack>
struct VectorTransportSystem
{
    Teuchos::RCP<typename Pack::matrix_type> matrix;
    Teuchos::RCP<typename Pack::multi_vector_type> rhs;
};

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
 * @tparam BoundaryValueProvider A callable with signature
 *         scalar_type(int patch_id, local_ordinal_type boundary_face_id).
 * @tparam SourceProvider A callable with signature
 *         scalar_type(local_ordinal_type cell_lid).
 * @param old_values Previous time-step scalar field. Its overlap values
 *        must be synchronized before assembly.
 * @param face_fluxes Pre-computed face volumetric fluxes.
 * @param time_step Time-step size (must be positive).
 * @param diffusivity Constant scalar diffusivity (non-negative).
 * @param boundary_value Callable that returns the prescribed boundary
 *        value for a face.
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
template<TpetraTypePack Pack, class BoundaryValueProvider, class SourceProvider>
    requires std::invocable<SourceProvider, typename Pack::local_ordinal_type>
TransportSystem<Pack>
transport_system(const CellField<Pack>& old_values,
                 const FaceField<Pack>& face_fluxes,
                 typename Pack::scalar_type time_step,
                 typename Pack::scalar_type diffusivity,
                 BoundaryValueProvider boundary_value,
                 SourceProvider right_hand_source,
                 Teuchos::RCP<typename Pack::matrix_type> cached_matrix = Teuchos::null)
{
    using scalar_type = typename Pack::scalar_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;

    const auto& mesh = old_values.mesh();
    if (&face_fluxes.mesh() != &mesh)
    {
        throw std::invalid_argument(
            "transport_system requires face fluxes on the old-value mesh.");
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
    const auto prepared = detail::prepare_transport_matrix<Pack>(
        mesh, std::move(cached_matrix), 12);
    const auto& matrix = prepared.matrix;
    auto rhs = Teuchos::rcp(
        new typename Pack::vector_type(mesh.owned_cell_map(), true));

    Teuchos::Array<local_ordinal_type> cols;
    Teuchos::Array<scalar_type> vals;
    cols.reserve(32);
    vals.reserve(32);
    std::unordered_map<local_ordinal_type, scalar_type> row_values;
    row_values.reserve(32);

    for (size_t owned = 0; owned < mesh.num_owned_cells(); ++owned)
    {
        const auto cell_lid = static_cast<local_ordinal_type>(owned);

        const auto volume = mesh.cell_volume(cell_lid);
        const auto transient = volume / time_step;
        const auto old_value = old_values.value(cell_lid);
        scalar_type rhs_value =
            transient * old_value + volume * right_hand_source(cell_lid);
        row_values.clear();
        detail::add_matrix_entry(row_values, cell_lid, transient);

        for (const auto face_lid : mesh.faces(cell_lid))
        {
            // === advection (upwind) ===
            const auto owner_oriented_flux =
                face_fluxes.is_owned_face(face_lid)
                    ? face_fluxes.value(face_lid)
                    : scalar_type{};
            const auto out_flux = mesh.owner_cell(face_lid) == cell_lid
                                    ? owner_oriented_flux
                                    : -owner_oriented_flux;

            if (out_flux >= scalar_type{0})
            {
                detail::add_matrix_entry(
                    row_values, cell_lid, out_flux);
            }
            else if (mesh.is_interior_face(face_lid))
            {
                const auto other =
                    mesh.opposite_or_periodic_neighbor_cell(face_lid, cell_lid);
                detail::add_matrix_entry(row_values, other, out_flux);
            }

            // === diffusion ===
            if (!mesh.is_interior_face(face_lid))
            {
                continue;
            }

            const auto other =
                mesh.opposite_or_periodic_neighbor_cell(face_lid, cell_lid);
            row_values.try_emplace(other, scalar_type{});
            if (diffusivity <= scalar_type{0})
            {
                continue;
            }

            const auto coeff =
                detail::interior_diffusion_coefficient(
                    mesh, face_lid, cell_lid, other, diffusivity);
            detail::add_matrix_entry(row_values, cell_lid, coeff);
            detail::add_matrix_entry(row_values, other, -coeff);
        }

        cols.clear();
        vals.clear();
        for (const auto& [column, value] : row_values)
        {
            cols.push_back(column);
            vals.push_back(value);
        }
        detail::add_transport_values<Pack>(
            prepared, cell_lid, cols(), vals());
        rhs->replaceLocalValue(cell_lid, rhs_value);
    }

    // Apply boundary conditions using sumIntoLocalValues (works reliably).
    for (const auto& [patch_id, boundary_patch] : mesh.boundary_patches())
    {
        for (size_t in_patch_id = 0;
             in_patch_id < boundary_patch.face_lids.size(); ++in_patch_id)
        {
            const auto face_lid = boundary_patch.face_lids[in_patch_id];
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
            const auto out_flux =
                face_fluxes.is_owned_face(face_lid)
                    ? face_fluxes.value(face_lid)
                    : scalar_type{};

            const auto boundary_face_value =
                boundary_value(patch_id, in_patch_id);

            if (out_flux >= scalar_type{0})
            {
                local_ordinal_type col = owner;
                matrix->sumIntoLocalValues(
                    owner,
                    Teuchos::arrayView(&col, 1),
                    Teuchos::arrayView(&out_flux, 1));
            }
            else
            {
                rhs->sumIntoLocalValue(owner,
                                       -out_flux * boundary_face_value);
            }

            // Diffusive boundary flux
            if (diffusivity <= scalar_type{0})
            {
                continue;
            }

            const auto coeff =
                detail::boundary_diffusion_coefficient(
                    mesh, face_lid, owner, diffusivity);
            if (coeff > scalar_type{0})
            {
                local_ordinal_type col = owner;
                scalar_type bval = coeff;
                matrix->sumIntoLocalValues(
                    owner,
                    Teuchos::arrayView(&col, 1),
                    Teuchos::arrayView(&bval, 1));
                rhs->sumIntoLocalValue(owner,
                                       coeff * boundary_face_value);
            }
        }
    }

    matrix->fillComplete();
    return {matrix, rhs};
}

/**
 * @brief Assemble the scalar semi-implicit transport system with no
 *        explicit right-hand source term.
 *
 * This overload preserves the existing call pattern and delegates to the
 * source-aware transport assembly with a zero source.
 *
 * @tparam Pack Tpetra type pack.
 * @tparam BoundaryValueProvider Callable returning scalar boundary value
 *         for (patch id, boundary face id).
 * @param old_values Previous time-step scalar field.
 * @param face_fluxes Pre-computed face volumetric fluxes.
 * @param time_step Time-step size (must be positive).
 * @param diffusivity Constant scalar diffusivity (non-negative).
 * @param boundary_value Callable that returns the prescribed boundary
 *        value for a face.
 * @param cached_matrix Optional fill-complete matrix cache.
 * @return TransportSystem containing the assembled matrix and RHS vector.
 */
template<TpetraTypePack Pack, class BoundaryValueProvider>
TransportSystem<Pack>
transport_system(const CellField<Pack>& old_values,
                 const FaceField<Pack>& face_fluxes,
                 typename Pack::scalar_type time_step,
                 typename Pack::scalar_type diffusivity,
                 BoundaryValueProvider boundary_value,
                 Teuchos::RCP<typename Pack::matrix_type> cached_matrix = Teuchos::null)
{
    using scalar_type = typename Pack::scalar_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;

    auto zero_source =
        [](local_ordinal_type) -> scalar_type
    {
        return scalar_type{};
    };

    return transport_system<Pack>(
        old_values, face_fluxes, time_step, diffusivity, boundary_value,
        zero_source, cached_matrix);
}

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
 * @tparam BoundaryValueProvider A callable with signature
 *         vec_type(int patch_id, local_ordinal_type boundary_face_id).
 * @tparam SourceProvider A callable with signature
 *         vec_type(local_ordinal_type cell_lid).
 * @param old_values Previous time-step vector field. Its overlap values
 *        must be synchronized before assembly.
 * @param face_fluxes Pre-computed face volumetric fluxes.
 * @param time_step Time-step size (must be positive).
 * @param diffusivity Constant scalar diffusivity (non-negative).
 * @param boundary_value Callable that returns the prescribed boundary
 *        vector value for a face.
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
template<TpetraTypePack Pack, class BoundaryValueProvider, class SourceProvider>
    requires std::invocable<SourceProvider, typename Pack::local_ordinal_type>
VectorTransportSystem<Pack>
transport_system(const VectorCellField<Pack>& old_values,
                 const FaceField<Pack>& face_fluxes,
                 typename Pack::scalar_type time_step,
                 typename Pack::scalar_type diffusivity,
                 BoundaryValueProvider boundary_value,
                 SourceProvider right_hand_source,
                 Teuchos::RCP<typename Pack::matrix_type> cached_matrix = Teuchos::null)
{
    using scalar_type = typename Pack::scalar_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;

    constexpr size_t num_components = 3;

    const auto& mesh = old_values.mesh();
    if (&face_fluxes.mesh() != &mesh)
    {
        throw std::invalid_argument(
            "transport_system requires face fluxes on the old-value mesh.");
    }
    if (time_step <= 0.0)
    {
        throw std::invalid_argument("transport_system requires a positive time step.");
    }
    if (diffusivity < 0.0)
    {
        throw std::invalid_argument("transport_system requires non-negative diffusivity.");
    }

    const auto prepared = detail::prepare_transport_matrix<Pack>(
        mesh, std::move(cached_matrix), 12);
    const auto& matrix = prepared.matrix;
    auto rhs = Teuchos::rcp(
        new typename Pack::multi_vector_type(
            mesh.owned_cell_map(), num_components, true));

    Teuchos::Array<local_ordinal_type> cols;
    Teuchos::Array<scalar_type> vals;
    cols.reserve(32);
    vals.reserve(32);
    std::unordered_map<local_ordinal_type, scalar_type> row_values;
    row_values.reserve(32);

    for (size_t owned = 0; owned < mesh.num_owned_cells(); ++owned)
    {
        const auto cell_lid = static_cast<local_ordinal_type>(owned);

        const auto volume = mesh.cell_volume(cell_lid);
        const auto transient = volume / time_step;
        const auto old_value = old_values.value(cell_lid);
        const auto source_value = right_hand_source(cell_lid);
        row_values.clear();
        detail::add_matrix_entry(row_values, cell_lid, transient);

        for (const auto face_lid : mesh.faces(cell_lid))
        {
            const auto owner_oriented_flux =
                face_fluxes.is_owned_face(face_lid)
                    ? face_fluxes.value(face_lid)
                    : scalar_type{};
            const auto out_flux = mesh.owner_cell(face_lid) == cell_lid
                                    ? owner_oriented_flux
                                    : -owner_oriented_flux;

            if (out_flux >= scalar_type{0})
            {
                detail::add_matrix_entry(
                    row_values, cell_lid, out_flux);
            }
            else if (mesh.is_interior_face(face_lid))
            {
                const auto other =
                    mesh.opposite_or_periodic_neighbor_cell(face_lid, cell_lid);
                detail::add_matrix_entry(row_values, other, out_flux);
            }

            if (!mesh.is_interior_face(face_lid))
            {
                continue;
            }

            const auto other =
                mesh.opposite_or_periodic_neighbor_cell(face_lid, cell_lid);
            row_values.try_emplace(other, scalar_type{});
            if (diffusivity <= scalar_type{0})
            {
                continue;
            }

            const auto coeff =
                detail::interior_diffusion_coefficient(
                    mesh, face_lid, cell_lid, other, diffusivity);
            detail::add_matrix_entry(row_values, cell_lid, coeff);
            detail::add_matrix_entry(row_values, other, -coeff);
        }

        cols.clear();
        vals.clear();
        for (const auto& [column, value] : row_values)
        {
            cols.push_back(column);
            vals.push_back(value);
        }
        detail::add_transport_values<Pack>(
            prepared, cell_lid, cols(), vals());
        for (size_t comp = 0; comp < num_components; ++comp)
        {
            rhs->replaceLocalValue(cell_lid, comp,
                                   transient * old_value.component(comp)
                                 + volume * source_value.component(comp));
        }
    }

    for (const auto& [patch_id, boundary_patch] : mesh.boundary_patches())
    {
        for (size_t in_patch_id = 0;
             in_patch_id < boundary_patch.face_lids.size(); ++in_patch_id)
        {
            const auto face_lid = boundary_patch.face_lids[in_patch_id];
            if (!mesh.is_owned_face(face_lid))
            {
                continue;
            }
            if (!mesh.is_boundary_face(face_lid))
            {
                continue;
            }

            const auto owner = mesh.owner_cell(face_lid);
            const auto out_flux =
                face_fluxes.is_owned_face(face_lid)
                    ? face_fluxes.value(face_lid)
                    : scalar_type{};

            const auto boundary_face_value =
                boundary_value(patch_id, in_patch_id);

            if (out_flux >= scalar_type{0})
            {
                local_ordinal_type col = owner;
                matrix->sumIntoLocalValues(
                    owner,
                    Teuchos::arrayView(&col, 1),
                    Teuchos::arrayView(&out_flux, 1));
            }
            else
            {
                for (size_t comp = 0; comp < num_components; ++comp)
                {
                    rhs->sumIntoLocalValue(
                        owner, comp,
                        -out_flux * boundary_face_value.component(comp));
                }
            }

            if (diffusivity <= scalar_type{0})
            {
                continue;
            }

            const auto coeff =
                detail::boundary_diffusion_coefficient(
                    mesh, face_lid, owner, diffusivity);
            if (coeff > scalar_type{0})
            {
                local_ordinal_type col = owner;
                scalar_type bval = coeff;
                matrix->sumIntoLocalValues(
                    owner,
                    Teuchos::arrayView(&col, 1),
                    Teuchos::arrayView(&bval, 1));
                for (size_t comp = 0; comp < num_components; ++comp)
                {
                    rhs->sumIntoLocalValue(
                        owner, comp,
                        coeff * boundary_face_value.component(comp));
                }
            }
        }
    }

    matrix->fillComplete();
    return {matrix, rhs};
}

/**
 * @brief Assemble vector transport with selectable non-orthogonal viscous
 *        diffusion treatment.
 *
 * The transient and first-order upwind advection terms are assembled as in
 * transport_system(). Viscous diffusion uses the Phase 1 orthogonal
 * two-point stencil plus Phase 2/3 non-orthogonal treatment selected by
 * @p treatment.
 */
template<TpetraTypePack Pack, class BoundaryValueProvider, class SourceProvider>
    requires std::invocable<SourceProvider, typename Pack::local_ordinal_type>
VectorTransportSystem<Pack>
non_orthogonal_transport_system(
    const VectorCellField<Pack>& old_values,
    const FaceField<Pack>& face_fluxes,
    typename Pack::scalar_type time_step,
    typename Pack::scalar_type diffusivity,
    BoundaryValueProvider boundary_value,
    SourceProvider right_hand_source,
    NonOrthogonalTreatment treatment,
    const VectorCellField<Pack>* correction_field = nullptr,
    Teuchos::RCP<typename Pack::matrix_type> cached_matrix = Teuchos::null)
{
    using scalar_type = typename Pack::scalar_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;

    constexpr size_t num_components = VectorCellField<Pack>::num_components;

    const auto& mesh = old_values.mesh();
    if (&face_fluxes.mesh() != &mesh)
    {
        throw std::invalid_argument(
            "non_orthogonal_transport_system requires face fluxes on the old-value mesh.");
    }
    if (correction_field != nullptr
        && &correction_field->mesh() != &mesh)
    {
        throw std::invalid_argument(
            "non_orthogonal_transport_system requires correction field on the old-value mesh.");
    }
    if (time_step <= scalar_type{0})
    {
        throw std::invalid_argument(
            "non_orthogonal_transport_system requires a positive time step.");
    }
    if (diffusivity < scalar_type{0})
    {
        throw std::invalid_argument(
            "non_orthogonal_transport_system requires non-negative diffusivity.");
    }

    scalar_type implicit_weight = 0.0;
    scalar_type explicit_weight = 0.0;
    switch (treatment)
    {
        case NonOrthogonalTreatment::Explicit:
            explicit_weight = 1.0;
            break;
        case NonOrthogonalTreatment::Implicit:
            implicit_weight = 1.0;
            break;
        case NonOrthogonalTreatment::Hybrid:
            implicit_weight = 0.5;
            explicit_weight = 0.5;
            break;
    }

    const auto gradient_stencils =
        detail::least_squares_gradient_stencils(mesh);
    const auto boundary_locations = detail::boundary_face_locations(mesh);

    const auto prepared = detail::prepare_transport_matrix<Pack>(
        mesh, std::move(cached_matrix), 32);
    const auto& matrix = prepared.matrix;
    auto rhs = Teuchos::rcp(
        new typename Pack::multi_vector_type(
            mesh.owned_cell_map(), num_components, true));

    Teuchos::Array<local_ordinal_type> cols;
    Teuchos::Array<scalar_type> vals;
    cols.reserve(64);
    vals.reserve(64);

    std::unordered_map<local_ordinal_type, scalar_type> row_values;
    row_values.reserve(64);

    for (size_t owned = 0; owned < mesh.num_owned_cells(); ++owned)
    {
        const auto cell_lid = static_cast<local_ordinal_type>(owned);
        const auto volume = mesh.cell_volume(cell_lid);
        const auto transient = volume / time_step;
        const auto old_value = old_values.value(cell_lid);
        const auto source_value = right_hand_source(cell_lid);

        row_values.clear();
        detail::add_matrix_entry(row_values, cell_lid, transient);

        for (size_t comp = 0; comp < num_components; ++comp)
        {
            rhs->replaceLocalValue(cell_lid, comp,
                                   transient * old_value.component(comp)
                                 + volume * source_value.component(comp));
        }

        auto add_non_orthogonal_stencil =
            [&](local_ordinal_type gradient_cell_lid,
                scalar_type face_gradient_weight,
                const typename Mesh<Pack>::Vec3& tangential_area)
        {
            if (face_gradient_weight == scalar_type{0}
                || !mesh.is_owned_cell(gradient_cell_lid)
                || static_cast<size_t>(gradient_cell_lid)
                   >= gradient_stencils.size())
            {
                return;
            }

            const auto scale =
                -implicit_weight * diffusivity * face_gradient_weight;
            for (const auto& entry :
                 gradient_stencils[static_cast<size_t>(gradient_cell_lid)])
            {
                detail::add_matrix_entry(
                    row_values, entry.cell_lid,
                    scale * entry.coefficient.dot(tangential_area));
            }
        };

        for (const auto face_lid : mesh.faces(cell_lid))
        {
            const auto owner_oriented_flux =
                face_fluxes.is_owned_face(face_lid)
                    ? face_fluxes.value(face_lid)
                    : scalar_type{};
            const auto out_flux = mesh.owner_cell(face_lid) == cell_lid
                                    ? owner_oriented_flux
                                    : -owner_oriented_flux;

            if (out_flux >= scalar_type{0})
            {
                detail::add_matrix_entry(row_values, cell_lid, out_flux);
            }
            else if (mesh.is_interior_face(face_lid))
            {
                const auto other =
                    mesh.opposite_or_periodic_neighbor_cell(face_lid, cell_lid);
                detail::add_matrix_entry(row_values, other, out_flux);
            }
            else if (mesh.is_boundary_face(face_lid)
                     && static_cast<size_t>(face_lid)
                        < boundary_locations.size()
                     && boundary_locations[static_cast<size_t>(face_lid)].active)
            {
                const auto location =
                    boundary_locations[static_cast<size_t>(face_lid)];
                const auto boundary_face_value =
                    boundary_value(location.patch_id, location.in_patch_id);
                for (size_t comp = 0; comp < num_components; ++comp)
                {
                    rhs->sumIntoLocalValue(
                        cell_lid, comp,
                        -out_flux * boundary_face_value.component(comp));
                }
            }

            if (diffusivity <= scalar_type{0})
            {
                continue;
            }

            if (mesh.is_interior_face(face_lid))
            {
                const auto other =
                    mesh.opposite_or_periodic_neighbor_cell(face_lid, cell_lid);
                row_values.try_emplace(other, scalar_type{});
                const auto coeff =
                    detail::interior_diffusion_coefficient(
                        mesh, face_lid, cell_lid, other, diffusivity);
                detail::add_matrix_entry(row_values, cell_lid, coeff);
                detail::add_matrix_entry(row_values, other, -coeff);

                const auto area_vector =
                    mesh.face_area_vector_outward(face_lid, cell_lid);
                const auto d = mesh.cell_center_vector(face_lid, cell_lid);
                const auto tangential_area =
                    detail::non_orthogonal_area_vector(area_vector, d);

                if (mesh.is_owned_cell(other)
                    && static_cast<size_t>(other)
                       < gradient_stencils.size())
                {
                    add_non_orthogonal_stencil(
                        cell_lid, scalar_type{0.5}, tangential_area);
                    add_non_orthogonal_stencil(
                        other, scalar_type{0.5}, tangential_area);
                }
                else
                {
                    add_non_orthogonal_stencil(
                        cell_lid, scalar_type{1}, tangential_area);
                }
                continue;
            }

            if (!mesh.is_boundary_face(face_lid)
                || static_cast<size_t>(face_lid) >= boundary_locations.size()
                || !boundary_locations[static_cast<size_t>(face_lid)].active)
            {
                continue;
            }

            const auto location =
                boundary_locations[static_cast<size_t>(face_lid)];
            const auto boundary_face_value =
                boundary_value(location.patch_id, location.in_patch_id);
            const auto coeff =
                detail::boundary_diffusion_coefficient(
                    mesh, face_lid, cell_lid, diffusivity);
            if (coeff > scalar_type{0})
            {
                detail::add_matrix_entry(row_values, cell_lid, coeff);
                for (size_t comp = 0; comp < num_components; ++comp)
                {
                    rhs->sumIntoLocalValue(
                        cell_lid, comp,
                        coeff * boundary_face_value.component(comp));
                }
            }

            const auto area_vector =
                mesh.face_area_vector_outward(face_lid, cell_lid);
            const auto d =
                mesh.face_centroid(face_lid) - mesh.cell_centroid(cell_lid);
            const auto tangential_area =
                detail::non_orthogonal_area_vector(area_vector, d);
            add_non_orthogonal_stencil(cell_lid, scalar_type{1}, tangential_area);
        }

        cols.clear();
        vals.clear();
        cols.reserve(row_values.size());
        vals.reserve(row_values.size());
        for (const auto& [column, value] : row_values)
        {
            cols.push_back(column);
            vals.push_back(value);
        }

        detail::add_transport_values<Pack>(
            prepared, cell_lid, cols(), vals());
    }

    if (correction_field != nullptr && explicit_weight > scalar_type{0})
    {
        add_explicit_non_orthogonal_correction<Pack>(
            *correction_field, diffusivity, *rhs, explicit_weight);
    }

    matrix->fillComplete();
    return {matrix, rhs};
}

/**
 * @brief Assemble the vector semi-implicit transport system with no
 *        explicit right-hand source term.
 *
 * This overload preserves the existing call pattern and delegates to the
 * source-aware transport assembly with a zero vector source.
 */
template<TpetraTypePack Pack, class BoundaryValueProvider>
VectorTransportSystem<Pack>
transport_system(const VectorCellField<Pack>& old_values,
                 const FaceField<Pack>& face_fluxes,
                 typename Pack::scalar_type time_step,
                 typename Pack::scalar_type diffusivity,
                 BoundaryValueProvider boundary_value,
                 Teuchos::RCP<typename Pack::matrix_type> cached_matrix = Teuchos::null)
{
    using local_ordinal_type = typename Pack::local_ordinal_type;
    using vec_type = typename VectorCellField<Pack>::vec_type;

    auto zero_source =
        [](local_ordinal_type) -> vec_type
    {
        return vec_type{};
    };

    return transport_system<Pack>(
        old_values, face_fluxes, time_step, diffusivity, boundary_value,
        zero_source, cached_matrix);
}

} // namespace SimpleFluid::FVM
