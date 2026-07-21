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
#include "FVM/BoundaryCache.hh"
#include "FVM/NonOrthogonalCorrection.hh"
#include "FVM/OperatorDetails.hh"
#include "geometry/Mesh.hh"

#include <Teuchos_Array.hpp>
#include <Teuchos_RCP.hpp>

#include <cmath>
#include <concepts>
#include <cstddef>
#include <functional>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace SimpleFluid::FVM
{
namespace detail
{

/**
 * @brief Matrix prepared for fresh insertion or cached-graph reuse.
 * @tparam Pack Tpetra type pack providing the matrix type.
 */
template<TpetraTypePack Pack>
struct PreparedTransportMatrix
{
    Teuchos::RCP<typename Pack::matrix_type> matrix;
    bool reused = false;
};

/**
 * @brief Constrains a callable that supplies boundary-condition records.
 * @tparam Provider Callable type.
 * @tparam Condition Required result type.
 */
template<class Provider, class Condition>
concept BoundaryConditionProviderFor =
    requires(Provider provider, int batch_id, size_t in_batch_id)
{
    { provider(batch_id, in_batch_id) } -> std::convertible_to<Condition>;
};

/**
 * @brief Constrains a callable that supplies boundary-face values.
 * @tparam Provider Callable type.
 * @tparam Value Required result type.
 */
template<class Provider, class Value>
concept BoundaryValueProviderFor =
    requires(Provider provider, int batch_id, size_t in_batch_id)
{
    { provider(batch_id, in_batch_id) } -> std::convertible_to<Value>;
};

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
 * @tparam BoundaryConditionProvider A callable returning BoundaryCondition
 *         for (batch id, in-batch face id).
 * @tparam BoundaryValueProvider A callable with signature
 *         scalar_type(int batch_id, size_t boundary_face_id).
 * @tparam SourceProvider A callable with signature
 *         scalar_type(local_ordinal_type cell_lid).
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
template<TpetraTypePack Pack,
         class BoundaryConditionProvider,
         class BoundaryValueProvider,
         class SourceProvider>
    requires std::invocable<SourceProvider, typename Pack::local_ordinal_type>
          && detail::BoundaryConditionProviderFor<
                 BoundaryConditionProvider, BoundaryCondition>
          && detail::BoundaryValueProviderFor<
                 BoundaryValueProvider, typename Pack::scalar_type>
TransportSystem<Pack>
transport_system(const CellField<Pack>& old_values,
                 const FaceField<Pack>& face_fluxes,
                 typename Pack::scalar_type time_step,
                 typename Pack::scalar_type diffusivity,
                 BoundaryConditionProvider boundary_condition,
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
            const auto is_interior = mesh.is_interior_face(face_lid);
            local_ordinal_type other{};
            if (is_interior)
            {
                other =
                    mesh.opposite_or_periodic_neighbor_cell(face_lid, cell_lid);
            }

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
            else if (is_interior)
            {
                detail::add_matrix_entry(row_values, other, out_flux);
            }

            // === diffusion ===
            if (!is_interior)
            {
                continue;
            }

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
    for (const auto& [batch_id, boundary_batch] : mesh.boundary_batches())
    {
        for (size_t in_batch_id = 0;
             in_batch_id < boundary_batch.face_lids.size(); ++in_batch_id)
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
            const auto out_flux =
                face_fluxes.is_owned_face(face_lid)
                    ? face_fluxes.value(face_lid)
                    : scalar_type{};

            const auto boundary_face_value =
                boundary_value(batch_id, in_batch_id);
            const auto condition =
                boundary_condition(batch_id, in_batch_id);

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
            if (condition.type == BoundaryConditionType::Dirichlet)
            {
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
            else if (condition.type == BoundaryConditionType::Neumann)
            {
                rhs->sumIntoLocalValue(
                    owner,
                    diffusivity * condition.value
                  * mesh.face_area(face_lid));
            }
            else if (condition.type == BoundaryConditionType::Robin)
            {
                throw std::runtime_error(
                    "Robin boundary conditions are not yet implemented in transport_system.");
            }
        }
    }

    matrix->fillComplete();
    return {matrix, rhs};
}

/**
 * @brief Assemble scalar transport using value-only boundary data.
 *
 * This compatibility overload treats every boundary value as Dirichlet,
 * matching the historical transport_system() behavior.
 */
template<TpetraTypePack Pack, class BoundaryValueProvider, class SourceProvider>
    requires std::invocable<SourceProvider, typename Pack::local_ordinal_type>
          && detail::BoundaryValueProviderFor<
                 BoundaryValueProvider, typename Pack::scalar_type>
TransportSystem<Pack>
transport_system(const CellField<Pack>& old_values,
                 const FaceField<Pack>& face_fluxes,
                 typename Pack::scalar_type time_step,
                 typename Pack::scalar_type diffusivity,
                 BoundaryValueProvider boundary_value,
                 SourceProvider right_hand_source,
                 Teuchos::RCP<typename Pack::matrix_type> cached_matrix = Teuchos::null)
{
    auto dirichlet_condition =
        [](int, size_t)
    {
        return BoundaryCondition{
            BoundaryConditionType::Dirichlet,
            typename Pack::scalar_type{}};
    };

    return transport_system<Pack>(
        old_values, face_fluxes, time_step, diffusivity,
        dirichlet_condition, boundary_value, right_hand_source,
        cached_matrix);
}

/**
 * @brief Assemble scalar transport with boundary-condition-aware diffusion
 *        and zero explicit source.
 */
template<TpetraTypePack Pack,
         class BoundaryConditionProvider,
         class BoundaryValueProvider>
    requires detail::BoundaryConditionProviderFor<
                 BoundaryConditionProvider, BoundaryCondition>
          && detail::BoundaryValueProviderFor<
                 BoundaryValueProvider, typename Pack::scalar_type>
TransportSystem<Pack>
transport_system(const CellField<Pack>& old_values,
                 const FaceField<Pack>& face_fluxes,
                 typename Pack::scalar_type time_step,
                 typename Pack::scalar_type diffusivity,
                 BoundaryConditionProvider boundary_condition,
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
        old_values, face_fluxes, time_step, diffusivity,
        boundary_condition, boundary_value, zero_source, cached_matrix);
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
 *         for (batch id, boundary face id).
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
 * @tparam BoundaryConditionProvider A callable returning
 *         VectorBoundaryCondition for (batch id, in-batch face id).
 * @tparam BoundaryValueProvider A callable with signature
 *         vec_type(int batch_id, size_t boundary_face_id).
 * @tparam SourceProvider A callable with signature
 *         vec_type(local_ordinal_type cell_lid).
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
template<TpetraTypePack Pack,
         class BoundaryConditionProvider,
         class BoundaryValueProvider,
         class SourceProvider>
    requires std::invocable<SourceProvider, typename Pack::local_ordinal_type>
          && detail::BoundaryConditionProviderFor<
                 BoundaryConditionProvider, VectorBoundaryCondition>
          && detail::BoundaryValueProviderFor<
                 BoundaryValueProvider, typename VectorCellField<Pack>::vec_type>
VectorTransportSystem<Pack>
transport_system(const VectorCellField<Pack>& old_values,
                 const FaceField<Pack>& face_fluxes,
                 typename Pack::scalar_type time_step,
                 typename Pack::scalar_type diffusivity,
                 BoundaryConditionProvider boundary_condition,
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
            const auto is_interior = mesh.is_interior_face(face_lid);
            local_ordinal_type other{};
            if (is_interior)
            {
                other =
                    mesh.opposite_or_periodic_neighbor_cell(face_lid, cell_lid);
            }

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
            else if (is_interior)
            {
                detail::add_matrix_entry(row_values, other, out_flux);
            }

            if (!is_interior)
            {
                continue;
            }

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

    for (const auto& [batch_id, boundary_batch] : mesh.boundary_batches())
    {
        for (size_t in_batch_id = 0;
             in_batch_id < boundary_batch.face_lids.size(); ++in_batch_id)
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
            const auto out_flux =
                face_fluxes.is_owned_face(face_lid)
                    ? face_fluxes.value(face_lid)
                    : scalar_type{};

            const auto boundary_face_value =
                boundary_value(batch_id, in_batch_id);
            const auto condition =
                boundary_condition(batch_id, in_batch_id);

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
            if (condition.type == BoundaryConditionType::Dirichlet
                || condition.type == BoundaryConditionType::NoSlip)
            {
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
            else if (condition.type == BoundaryConditionType::Neumann)
            {
                for (size_t comp = 0; comp < num_components; ++comp)
                {
                    rhs->sumIntoLocalValue(
                        owner, comp,
                        diffusivity * condition.value.component(comp)
                      * mesh.face_area(face_lid));
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

/**
 * @brief Assemble vector transport using value-only boundary data.
 *
 * This compatibility overload treats every boundary value as Dirichlet,
 * matching the historical transport_system() behavior.
 */
template<TpetraTypePack Pack, class BoundaryValueProvider, class SourceProvider>
    requires std::invocable<SourceProvider, typename Pack::local_ordinal_type>
          && detail::BoundaryValueProviderFor<
                 BoundaryValueProvider, typename VectorCellField<Pack>::vec_type>
VectorTransportSystem<Pack>
transport_system(const VectorCellField<Pack>& old_values,
                 const FaceField<Pack>& face_fluxes,
                 typename Pack::scalar_type time_step,
                 typename Pack::scalar_type diffusivity,
                 BoundaryValueProvider boundary_value,
                 SourceProvider right_hand_source,
                 Teuchos::RCP<typename Pack::matrix_type> cached_matrix = Teuchos::null)
{
    auto dirichlet_condition =
        [](int, size_t)
    {
        return VectorBoundaryCondition{
            BoundaryConditionType::Dirichlet,
            typename VectorCellField<Pack>::vec_type{}};
    };

    return transport_system<Pack>(
        old_values, face_fluxes, time_step, diffusivity,
        dirichlet_condition, boundary_value, right_hand_source,
        cached_matrix);
}

/**
 * @brief Assemble vector transport with boundary-condition-aware diffusion
 *        and zero explicit source.
 */
template<TpetraTypePack Pack,
         class BoundaryConditionProvider,
         class BoundaryValueProvider>
    requires detail::BoundaryConditionProviderFor<
                 BoundaryConditionProvider, VectorBoundaryCondition>
          && detail::BoundaryValueProviderFor<
                 BoundaryValueProvider, typename VectorCellField<Pack>::vec_type>
VectorTransportSystem<Pack>
transport_system(const VectorCellField<Pack>& old_values,
                 const FaceField<Pack>& face_fluxes,
                 typename Pack::scalar_type time_step,
                 typename Pack::scalar_type diffusivity,
                 BoundaryConditionProvider boundary_condition,
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
        old_values, face_fluxes, time_step, diffusivity,
        boundary_condition, boundary_value, zero_source, cached_matrix);
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
template<TpetraTypePack Pack,
         class BoundaryValueProvider,
         class SourceProvider,
         class BoundaryDiffusionProvider = detail::AlwaysDiffuseBoundary>
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
    Teuchos::RCP<typename Pack::matrix_type> cached_matrix = Teuchos::null,
    BoundaryDiffusionProvider boundary_diffusion =
        BoundaryDiffusionProvider{})
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
                    boundary_value(location.batch_id, location.in_batch_id);
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
            if (!boundary_diffusion(location.batch_id,
                                    location.in_batch_id))
            {
                continue;
            }
            const auto boundary_face_value =
                boundary_value(location.batch_id, location.in_batch_id);
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
            *correction_field, diffusivity, *rhs, explicit_weight,
            boundary_diffusion);
    }

    matrix->fillComplete();
    return {matrix, rhs};
}

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
 */
template<TpetraTypePack Pack,
         class BoundaryConditionProvider,
         class BoundaryValueProvider,
         class SourceProvider>
TransportSystem<Pack>
weighted_scalar_transport_system(
    const CellField<Pack>& old_values,
    const FaceField<Pack>& face_fluxes,
    typename Pack::scalar_type time_step,
    const CellField<Pack>& storage_weight,
    const CellField<Pack>& advection_weight,
    const CellField<Pack>& diffusivity,
    BoundaryConditionProvider boundary_condition,
    BoundaryValueProvider boundary_value,
    SourceProvider source,
    NonOrthogonalTreatment treatment,
    const CellField<Pack>* correction_field = nullptr,
    Teuchos::RCP<typename Pack::matrix_type> cached_matrix = Teuchos::null,
    std::function<typename Pack::scalar_type(
        typename Pack::local_ordinal_type)> implicit_sink = {},
    std::function<std::optional<typename Pack::scalar_type>(
        typename Pack::local_ordinal_type)> fixed_cell_value = {},
    const BoundaryCache<Pack>* boundary_diffusivity = nullptr)
{
    using scalar_type = typename Pack::scalar_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;

    const auto& mesh = old_values.mesh();
    if (&face_fluxes.mesh() != &mesh
        || &storage_weight.mesh() != &mesh
        || &advection_weight.mesh() != &mesh
        || &diffusivity.mesh() != &mesh
        || (correction_field != nullptr
            && &correction_field->mesh() != &mesh))
    {
        throw std::invalid_argument(
            "weighted_scalar_transport_system requires all fields on "
            "the transported-field mesh.");
    }
    if (time_step <= scalar_type{})
    {
        throw std::invalid_argument(
            "weighted_scalar_transport_system requires a positive "
            "time step.");
    }
    validate_boundary_coefficient_cache(
        mesh, boundary_diffusivity,
        "weighted_scalar_transport_system");
    auto boundary_face_diffusivity =
        [&](int batch_id, size_t in_batch_id,
            scalar_type owner_cell_value)
    {
        return boundary_coefficient<Pack>(
            boundary_diffusivity, batch_id, in_batch_id,
            owner_cell_value);
    };

    std::vector<std::optional<scalar_type>> fixed_cell_values(
        mesh.num_owned_cells());
    if (fixed_cell_value)
    {
        for (size_t owned = 0; owned < mesh.num_owned_cells(); ++owned)
        {
            const auto cell_lid =
                static_cast<local_ordinal_type>(owned);
            auto value = fixed_cell_value(cell_lid);
            if (value.has_value() && !std::isfinite(*value))
            {
                throw std::invalid_argument(
                    "weighted scalar transport requires finite fixed-cell "
                    "values.");
            }
            fixed_cell_values[owned] = value;
        }
    }

    scalar_type implicit_weight{};
    scalar_type explicit_weight{};
    switch (treatment)
    {
        case NonOrthogonalTreatment::Explicit:
            explicit_weight = scalar_type{1};
            break;
        case NonOrthogonalTreatment::Implicit:
            implicit_weight = scalar_type{1};
            break;
        case NonOrthogonalTreatment::Hybrid:
            implicit_weight = scalar_type{0.5};
            explicit_weight = scalar_type{0.5};
            break;
    }

    const auto gradient_stencils =
        detail::scalar_affine_gradient_stencils(
            mesh, boundary_condition, boundary_value);
    const auto boundary_locations =
        detail::boundary_face_locations(mesh);
    const auto prepared = detail::prepare_transport_matrix<Pack>(
        mesh, std::move(cached_matrix), 32);
    const auto& matrix = prepared.matrix;
    auto rhs = Teuchos::rcp(
        new typename Pack::vector_type(mesh.owned_cell_map(), true));

    Teuchos::Array<local_ordinal_type> columns;
    Teuchos::Array<scalar_type> values;
    columns.reserve(64);
    values.reserve(64);
    std::unordered_map<local_ordinal_type, scalar_type> row_values;
    row_values.reserve(64);

    for (size_t owned = 0; owned < mesh.num_owned_cells(); ++owned)
    {
        const auto cell_lid =
            static_cast<local_ordinal_type>(owned);
        const auto volume = mesh.cell_volume(cell_lid);
        const auto cell_storage =
            storage_weight.local_value(cell_lid);
        const auto cell_advection =
            advection_weight.local_value(cell_lid);
        const auto transient =
            cell_storage * volume / time_step;
        const auto sink = implicit_sink
                        ? implicit_sink(cell_lid)
                        : scalar_type{};
        if (!std::isfinite(cell_storage)
            || !std::isfinite(cell_advection)
            || cell_storage <= scalar_type{}
            || cell_advection < scalar_type{})
        {
            throw std::invalid_argument(
                "weighted scalar transport requires positive storage "
                "and non-negative advection weights.");
        }
        if (!std::isfinite(sink) || sink < scalar_type{})
        {
            throw std::invalid_argument(
                "weighted scalar transport requires a finite, "
                "non-negative implicit sink.");
        }

        row_values.clear();
        detail::add_matrix_entry(
            row_values, cell_lid, transient + volume * sink);
        rhs->replaceLocalValue(
            cell_lid,
            transient * old_values.value(cell_lid)
          + volume * source(cell_lid));

        auto add_non_orthogonal_stencil =
            [&](local_ordinal_type gradient_cell_lid,
                scalar_type gradient_weight,
                scalar_type face_diffusivity,
                const typename Mesh<Pack>::Vec3& tangential_area)
        {
            if (implicit_weight == scalar_type{}
                || gradient_weight == scalar_type{}
                || !mesh.is_owned_cell(gradient_cell_lid)
                || static_cast<size_t>(gradient_cell_lid)
                    >= gradient_stencils.size())
            {
                return;
            }
            const auto scale =
                -implicit_weight
              * face_diffusivity
              * gradient_weight;
            const auto& stencil = gradient_stencils[
                static_cast<size_t>(gradient_cell_lid)];
            rhs->sumIntoLocalValue(
                cell_lid,
                -scale * stencil.constant.dot(tangential_area));
            for (const auto& entry : stencil.entries)
            {
                detail::add_matrix_entry(
                    row_values,
                    entry.cell_lid,
                    scale
                  * entry.coefficient.dot(tangential_area));
            }
        };

        for (const auto face_lid : mesh.faces(cell_lid))
        {
            const auto owner_oriented_flux =
                face_fluxes.is_owned_face(face_lid)
                    ? face_fluxes.value(face_lid)
                    : scalar_type{};
            const auto out_flux =
                mesh.owner_cell(face_lid) == cell_lid
                    ? owner_oriented_flux
                    : -owner_oriented_flux;
            if (out_flux >= scalar_type{})
            {
                detail::add_matrix_entry(
                    row_values,
                    cell_lid,
                    out_flux * cell_advection);
            }
            else if (mesh.is_interior_face(face_lid))
            {
                const auto other =
                    mesh.opposite_or_periodic_neighbor_cell(
                        face_lid, cell_lid);
                const auto other_advection =
                    advection_weight.local_value(other);
                detail::add_matrix_entry(
                    row_values,
                    other,
                    out_flux * other_advection);
            }
            else if (mesh.is_boundary_face(face_lid)
                     && static_cast<size_t>(face_lid)
                        < boundary_locations.size()
                     && boundary_locations[
                            static_cast<size_t>(face_lid)].active)
            {
                const auto location =
                    boundary_locations[
                        static_cast<size_t>(face_lid)];
                rhs->sumIntoLocalValue(
                    cell_lid,
                    -out_flux * cell_advection
                  * boundary_value(
                        location.batch_id,
                        location.in_batch_id));
            }

            if (mesh.is_interior_face(face_lid))
            {
                const auto other =
                    mesh.opposite_or_periodic_neighbor_cell(
                        face_lid, cell_lid);
                row_values.try_emplace(other, scalar_type{});
                const auto face_diffusivity =
                    detail::harmonic_face_value(
                        mesh, face_lid, cell_lid, other,
                        diffusivity);
                if (face_diffusivity <= scalar_type{})
                    continue;
                const auto coefficient =
                    detail::interior_diffusion_coefficient(
                        mesh, face_lid, cell_lid, other,
                        face_diffusivity);
                detail::add_matrix_entry(
                    row_values, cell_lid, coefficient);
                detail::add_matrix_entry(
                    row_values, other, -coefficient);

                const auto tangential_area =
                    detail::non_orthogonal_area_vector(
                        mesh.face_area_vector_outward(
                            face_lid, cell_lid),
                        mesh.cell_center_vector(
                            face_lid, cell_lid));
                if (mesh.is_owned_cell(other)
                    && static_cast<size_t>(other)
                        < gradient_stencils.size())
                {
                    add_non_orthogonal_stencil(
                        cell_lid, scalar_type{0.5},
                        face_diffusivity, tangential_area);
                    add_non_orthogonal_stencil(
                        other, scalar_type{0.5},
                        face_diffusivity, tangential_area);
                }
                else
                {
                    add_non_orthogonal_stencil(
                        cell_lid, scalar_type{1},
                        face_diffusivity, tangential_area);
                }
                continue;
            }

            if (!mesh.is_boundary_face(face_lid)
                || static_cast<size_t>(face_lid)
                    >= boundary_locations.size()
                || !boundary_locations[
                        static_cast<size_t>(face_lid)].active)
            {
                continue;
            }
            const auto location =
                boundary_locations[static_cast<size_t>(face_lid)];
            const auto condition =
                boundary_condition(
                    location.batch_id, location.in_batch_id);
            const auto face_diffusivity =
                boundary_face_diffusivity(
                    location.batch_id, location.in_batch_id,
                    diffusivity.local_value(cell_lid));
            if (condition.type == BoundaryConditionType::Dirichlet)
            {
                const auto coefficient =
                    detail::boundary_diffusion_coefficient(
                        mesh, face_lid, cell_lid, face_diffusivity);
                if (coefficient > scalar_type{})
                {
                    detail::add_matrix_entry(
                        row_values, cell_lid, coefficient);
                    rhs->sumIntoLocalValue(
                        cell_lid,
                        coefficient
                      * boundary_value(
                            location.batch_id,
                            location.in_batch_id));
                }
                const auto tangential_area =
                    detail::non_orthogonal_area_vector(
                        mesh.face_area_vector_outward(
                            face_lid, cell_lid),
                        mesh.face_centroid(face_lid)
                            - mesh.cell_centroid(cell_lid));
                add_non_orthogonal_stencil(
                    cell_lid, scalar_type{1},
                    face_diffusivity, tangential_area);
            }
            else if (condition.type == BoundaryConditionType::Neumann)
            {
                rhs->sumIntoLocalValue(
                    cell_lid,
                    face_diffusivity * condition.value
                  * mesh.face_area(face_lid));
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
            for (auto& [column, value] : row_values)
            {
                static_cast<void>(column);
                value = scalar_type{};
            }
            row_values[cell_lid] = scalar_type{1};
            rhs->replaceLocalValue(cell_lid, *fixed_cell_values[owned]);
        }

        columns.clear();
        values.clear();
        for (const auto& [column, value] : row_values)
        {
            columns.push_back(column);
            values.push_back(value);
        }
        detail::add_transport_values<Pack>(
            prepared, cell_lid, columns(), values());
    }

    if (correction_field != nullptr
        && explicit_weight > scalar_type{})
    {
        add_variable_explicit_non_orthogonal_correction<Pack>(
            *correction_field,
            diffusivity,
            boundary_condition,
            boundary_value,
            *rhs,
            explicit_weight,
            boundary_face_diffusivity);
    }
    // An explicit correction is an RHS update, so restore constrained values
    // after it to keep fixed rows exactly equal to phi = phi_fixed.
    for (size_t owned = 0; owned < fixed_cell_values.size(); ++owned)
    {
        if (fixed_cell_values[owned].has_value())
        {
            rhs->replaceLocalValue(
                static_cast<local_ordinal_type>(owned),
                *fixed_cell_values[owned]);
        }
    }

    matrix->fillComplete();
    return {matrix, rhs};
}

/**
 * @brief Assemble conservative temperature transport with physical material
 *        fields and a volumetric heat source.
 *
 * Storage and advection use volumetric heat capacity rho*cp, diffusion uses
 * distance-weighted harmonic conductivity, and the source is power density.
 */
template<TpetraTypePack Pack,
         class BoundaryConditionProvider,
         class BoundaryValueProvider,
         class SourceProvider>
TransportSystem<Pack>
physical_temperature_transport_system(
    const CellField<Pack>& old_temperature,
    const FaceField<Pack>& face_fluxes,
    typename Pack::scalar_type time_step,
    const CellField<Pack>& density,
    const CellField<Pack>& specific_heat_capacity,
    const CellField<Pack>& thermal_conductivity,
    BoundaryConditionProvider boundary_condition,
    BoundaryValueProvider boundary_value,
    SourceProvider power_density,
    NonOrthogonalTreatment treatment,
    const CellField<Pack>* correction_field = nullptr,
    Teuchos::RCP<typename Pack::matrix_type> cached_matrix = Teuchos::null,
    const BoundaryCache<Pack>* boundary_thermal_conductivity = nullptr)
{
    using scalar_type = typename Pack::scalar_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;

    const auto& mesh = old_temperature.mesh();
    if (&face_fluxes.mesh() != &mesh
        || &density.mesh() != &mesh
        || &specific_heat_capacity.mesh() != &mesh
        || &thermal_conductivity.mesh() != &mesh
        || (correction_field != nullptr
            && &correction_field->mesh() != &mesh))
    {
        throw std::invalid_argument(
            "physical_temperature_transport_system requires all fields on "
            "the temperature mesh.");
    }
    if (time_step <= scalar_type{})
    {
        throw std::invalid_argument(
            "physical_temperature_transport_system requires a positive "
            "time step.");
    }
    validate_boundary_coefficient_cache(
        mesh,
        boundary_thermal_conductivity,
        "physical_temperature_transport_system");

    auto boundary_conductivity =
        [&](int batch_id,
            size_t in_batch_id,
            scalar_type owner_cell_value)
    {
        return boundary_coefficient<Pack>(
            boundary_thermal_conductivity,
            batch_id,
            in_batch_id,
            owner_cell_value);
    };

    scalar_type implicit_weight{};
    scalar_type explicit_weight{};
    switch (treatment)
    {
        case NonOrthogonalTreatment::Explicit:
            explicit_weight = scalar_type{1};
            break;
        case NonOrthogonalTreatment::Implicit:
            implicit_weight = scalar_type{1};
            break;
        case NonOrthogonalTreatment::Hybrid:
            implicit_weight = scalar_type{0.5};
            explicit_weight = scalar_type{0.5};
            break;
    }

    const auto gradient_stencils =
        detail::scalar_affine_gradient_stencils(
            mesh, boundary_condition, boundary_value);
    const auto boundary_locations =
        detail::boundary_face_locations(mesh);
    const auto prepared = detail::prepare_transport_matrix<Pack>(
        mesh, std::move(cached_matrix), 32);
    const auto& matrix = prepared.matrix;
    auto rhs = Teuchos::rcp(
        new typename Pack::vector_type(
            mesh.owned_cell_map(), true));

    Teuchos::Array<local_ordinal_type> columns;
    Teuchos::Array<scalar_type> values;
    columns.reserve(64);
    values.reserve(64);
    std::unordered_map<local_ordinal_type, scalar_type> row_values;
    row_values.reserve(64);

    for (size_t owned = 0; owned < mesh.num_owned_cells(); ++owned)
    {
        const auto cell_lid =
            static_cast<local_ordinal_type>(owned);
        const auto volume = mesh.cell_volume(cell_lid);
        const auto cell_capacity =
            density.local_value(cell_lid)
          * specific_heat_capacity.local_value(cell_lid);
        const auto transient =
            cell_capacity * volume / time_step;
        row_values.clear();
        detail::add_matrix_entry(
            row_values, cell_lid, transient);
        rhs->replaceLocalValue(
            cell_lid,
            transient * old_temperature.value(cell_lid)
          + volume * power_density(cell_lid));

        auto add_non_orthogonal_stencil =
            [&](local_ordinal_type gradient_cell_lid,
                scalar_type gradient_weight,
                scalar_type face_conductivity,
                const typename Mesh<Pack>::Vec3& tangential_area)
        {
            if (implicit_weight == scalar_type{}
                || gradient_weight == scalar_type{}
                || !mesh.is_owned_cell(gradient_cell_lid)
                || static_cast<size_t>(gradient_cell_lid)
                    >= gradient_stencils.size())
            {
                return;
            }
            const auto scale =
                -implicit_weight
              * face_conductivity
              * gradient_weight;
            const auto& stencil = gradient_stencils[
                static_cast<size_t>(gradient_cell_lid)];
            rhs->sumIntoLocalValue(
                cell_lid,
                -scale * stencil.constant.dot(tangential_area));
            for (const auto& entry : stencil.entries)
            {
                detail::add_matrix_entry(
                    row_values,
                    entry.cell_lid,
                    scale
                  * entry.coefficient.dot(tangential_area));
            }
        };

        for (const auto face_lid : mesh.faces(cell_lid))
        {
            const auto owner_oriented_flux =
                face_fluxes.is_owned_face(face_lid)
                    ? face_fluxes.value(face_lid)
                    : scalar_type{};
            const auto out_flux =
                mesh.owner_cell(face_lid) == cell_lid
                    ? owner_oriented_flux
                    : -owner_oriented_flux;

            if (out_flux >= scalar_type{})
            {
                detail::add_matrix_entry(
                    row_values,
                    cell_lid,
                    out_flux * cell_capacity);
            }
            else if (mesh.is_interior_face(face_lid))
            {
                const auto other =
                    mesh.opposite_or_periodic_neighbor_cell(
                        face_lid, cell_lid);
                const auto other_capacity =
                    density.local_value(other)
                  * specific_heat_capacity.local_value(other);
                detail::add_matrix_entry(
                    row_values,
                    other,
                    out_flux * other_capacity);
            }
            else if (mesh.is_boundary_face(face_lid)
                     && static_cast<size_t>(face_lid)
                        < boundary_locations.size()
                     && boundary_locations[
                            static_cast<size_t>(face_lid)].active)
            {
                const auto location =
                    boundary_locations[
                        static_cast<size_t>(face_lid)];
                rhs->sumIntoLocalValue(
                    cell_lid,
                    -out_flux * cell_capacity
                  * boundary_value(
                        location.batch_id,
                        location.in_batch_id));
            }

            if (mesh.is_interior_face(face_lid))
            {
                const auto other =
                    mesh.opposite_or_periodic_neighbor_cell(
                        face_lid, cell_lid);
                row_values.try_emplace(other, scalar_type{});
                const auto face_conductivity =
                    detail::harmonic_face_value(
                        mesh, face_lid, cell_lid, other,
                        thermal_conductivity);
                if (face_conductivity <= scalar_type{})
                {
                    continue;
                }
                const auto coefficient =
                    detail::interior_diffusion_coefficient(
                        mesh, face_lid, cell_lid, other,
                        face_conductivity);
                detail::add_matrix_entry(
                    row_values, cell_lid, coefficient);
                detail::add_matrix_entry(
                    row_values, other, -coefficient);

                const auto tangential_area =
                    detail::non_orthogonal_area_vector(
                        mesh.face_area_vector_outward(
                            face_lid, cell_lid),
                        mesh.cell_center_vector(
                            face_lid, cell_lid));
                if (mesh.is_owned_cell(other)
                    && static_cast<size_t>(other)
                        < gradient_stencils.size())
                {
                    add_non_orthogonal_stencil(
                        cell_lid, scalar_type{0.5},
                        face_conductivity, tangential_area);
                    add_non_orthogonal_stencil(
                        other, scalar_type{0.5},
                        face_conductivity, tangential_area);
                }
                else
                {
                    add_non_orthogonal_stencil(
                        cell_lid, scalar_type{1},
                        face_conductivity, tangential_area);
                }
                continue;
            }

            if (!mesh.is_boundary_face(face_lid)
                || static_cast<size_t>(face_lid)
                    >= boundary_locations.size()
                || !boundary_locations[
                        static_cast<size_t>(face_lid)].active)
            {
                continue;
            }
            const auto location =
                boundary_locations[static_cast<size_t>(face_lid)];
            const auto condition =
                boundary_condition(
                    location.batch_id, location.in_batch_id);
            const auto face_conductivity =
                boundary_conductivity(
                    location.batch_id,
                    location.in_batch_id,
                    thermal_conductivity.local_value(cell_lid));
            if (condition.type == BoundaryConditionType::Dirichlet)
            {
                const auto coefficient =
                    detail::boundary_diffusion_coefficient(
                        mesh, face_lid, cell_lid,
                        face_conductivity);
                if (coefficient > scalar_type{})
                {
                    detail::add_matrix_entry(
                        row_values, cell_lid, coefficient);
                    rhs->sumIntoLocalValue(
                        cell_lid,
                        coefficient
                      * boundary_value(
                            location.batch_id,
                            location.in_batch_id));
                }

                const auto tangential_area =
                    detail::non_orthogonal_area_vector(
                        mesh.face_area_vector_outward(
                            face_lid, cell_lid),
                        mesh.face_centroid(face_lid)
                            - mesh.cell_centroid(cell_lid));
                add_non_orthogonal_stencil(
                    cell_lid, scalar_type{1},
                    face_conductivity, tangential_area);
            }
            else if (condition.type == BoundaryConditionType::Neumann)
            {
                rhs->sumIntoLocalValue(
                    cell_lid,
                    face_conductivity * condition.value
                  * mesh.face_area(face_lid));
            }
            else if (condition.type == BoundaryConditionType::Robin)
            {
                throw std::runtime_error(
                    "Robin boundary conditions are not yet implemented in physical_temperature_transport_system.");
            }
        }

        columns.clear();
        values.clear();
        for (const auto& [column, value] : row_values)
        {
            columns.push_back(column);
            values.push_back(value);
        }
        detail::add_transport_values<Pack>(
            prepared, cell_lid, columns(), values());
    }

    if (correction_field != nullptr
        && explicit_weight > scalar_type{})
    {
        add_variable_explicit_non_orthogonal_correction<Pack>(
            *correction_field,
            thermal_conductivity,
            boundary_condition,
            boundary_value,
            *rhs,
            explicit_weight,
            boundary_conductivity);
    }

    matrix->fillComplete();
    return {matrix, rhs};
}

/**
 * @brief Assemble incompressible momentum transport with a variable dynamic
 *        viscosity field and constant reference density.
 */
template<TpetraTypePack Pack,
         class BoundaryValueProvider,
         class SourceProvider,
         class BoundaryDiffusionProvider = detail::AlwaysDiffuseBoundary>
VectorTransportSystem<Pack>
physical_momentum_transport_system(
    const VectorCellField<Pack>& old_velocity,
    const FaceField<Pack>& face_fluxes,
    typename Pack::scalar_type time_step,
    const CellField<Pack>& dynamic_viscosity,
    typename Pack::scalar_type reference_density,
    BoundaryValueProvider boundary_value,
    SourceProvider acceleration_source,
    NonOrthogonalTreatment treatment,
    const VectorCellField<Pack>* correction_field = nullptr,
    Teuchos::RCP<typename Pack::matrix_type> cached_matrix = Teuchos::null,
    BoundaryDiffusionProvider boundary_diffusion =
        BoundaryDiffusionProvider{},
    const BoundaryCache<Pack>* boundary_dynamic_viscosity = nullptr)
{
    using scalar_type = typename Pack::scalar_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;
    constexpr size_t components =
        VectorCellField<Pack>::num_components;

    const auto& mesh = old_velocity.mesh();
    if (&face_fluxes.mesh() != &mesh
        || &dynamic_viscosity.mesh() != &mesh
        || (correction_field != nullptr
            && &correction_field->mesh() != &mesh))
    {
        throw std::invalid_argument(
            "physical_momentum_transport_system requires all fields on "
            "the velocity mesh.");
    }
    if (time_step <= scalar_type{}
        || reference_density <= scalar_type{})
    {
        throw std::invalid_argument(
            "physical_momentum_transport_system requires positive time step "
            "and reference density.");
    }
    validate_boundary_coefficient_cache(
        mesh,
        boundary_dynamic_viscosity,
        "physical_momentum_transport_system");

    auto boundary_viscosity =
        [&](int batch_id,
            size_t in_batch_id,
            scalar_type owner_cell_value)
    {
        return boundary_coefficient<Pack>(
            boundary_dynamic_viscosity,
            batch_id,
            in_batch_id,
            owner_cell_value);
    };

    scalar_type implicit_weight{};
    scalar_type explicit_weight{};
    switch (treatment)
    {
        case NonOrthogonalTreatment::Explicit:
            explicit_weight = scalar_type{1};
            break;
        case NonOrthogonalTreatment::Implicit:
            implicit_weight = scalar_type{1};
            break;
        case NonOrthogonalTreatment::Hybrid:
            implicit_weight = scalar_type{0.5};
            explicit_weight = scalar_type{0.5};
            break;
    }

    const auto gradient_stencils =
        detail::vector_affine_gradient_stencils(mesh, boundary_value);
    const auto boundary_locations =
        detail::boundary_face_locations(mesh);
    const auto prepared = detail::prepare_transport_matrix<Pack>(
        mesh, std::move(cached_matrix), 32);
    const auto& matrix = prepared.matrix;
    auto rhs = Teuchos::rcp(
        new typename Pack::multi_vector_type(
            mesh.owned_cell_map(), components, true));

    Teuchos::Array<local_ordinal_type> columns;
    Teuchos::Array<scalar_type> values;
    columns.reserve(64);
    values.reserve(64);
    std::unordered_map<local_ordinal_type, scalar_type> row_values;
    row_values.reserve(64);

    for (size_t owned = 0; owned < mesh.num_owned_cells(); ++owned)
    {
        const auto cell_lid =
            static_cast<local_ordinal_type>(owned);
        const auto volume = mesh.cell_volume(cell_lid);
        const auto transient = volume / time_step;
        const auto old_value = old_velocity.value(cell_lid);
        const auto source_value = acceleration_source(cell_lid);

        row_values.clear();
        detail::add_matrix_entry(
            row_values, cell_lid, transient);
        for (size_t component = 0;
             component < components;
             ++component)
        {
            rhs->replaceLocalValue(
                cell_lid,
                component,
                transient * old_value.component(component)
              + volume * source_value.component(component));
        }

        auto add_non_orthogonal_stencil =
            [&](local_ordinal_type gradient_cell_lid,
                scalar_type gradient_weight,
                scalar_type face_kinematic_viscosity,
                const typename Mesh<Pack>::Vec3& tangential_area)
        {
            if (implicit_weight == scalar_type{}
                || gradient_weight == scalar_type{}
                || !mesh.is_owned_cell(gradient_cell_lid)
                || static_cast<size_t>(gradient_cell_lid)
                    >= gradient_stencils.size())
            {
                return;
            }
            const auto scale =
                -implicit_weight
              * face_kinematic_viscosity
              * gradient_weight;
            const auto& stencil = gradient_stencils[
                static_cast<size_t>(gradient_cell_lid)];
            for (size_t component = 0; component < components; ++component)
            {
                rhs->sumIntoLocalValue(
                    cell_lid, component,
                    -scale
                  * stencil.constants[component].dot(tangential_area));
            }
            for (const auto& entry : stencil.entries)
            {
                detail::add_matrix_entry(
                    row_values,
                    entry.cell_lid,
                    scale
                  * entry.coefficient.dot(tangential_area));
            }
        };

        for (const auto face_lid : mesh.faces(cell_lid))
        {
            const auto owner_oriented_flux =
                face_fluxes.is_owned_face(face_lid)
                    ? face_fluxes.value(face_lid)
                    : scalar_type{};
            const auto out_flux =
                mesh.owner_cell(face_lid) == cell_lid
                    ? owner_oriented_flux
                    : -owner_oriented_flux;
            if (out_flux >= scalar_type{})
            {
                detail::add_matrix_entry(
                    row_values, cell_lid, out_flux);
            }
            else if (mesh.is_interior_face(face_lid))
            {
                const auto other =
                    mesh.opposite_or_periodic_neighbor_cell(
                        face_lid, cell_lid);
                detail::add_matrix_entry(
                    row_values, other, out_flux);
            }
            else if (mesh.is_boundary_face(face_lid)
                     && static_cast<size_t>(face_lid)
                        < boundary_locations.size()
                     && boundary_locations[
                            static_cast<size_t>(face_lid)].active)
            {
                const auto location =
                    boundary_locations[
                        static_cast<size_t>(face_lid)];
                const auto face_value =
                    boundary_value(
                        location.batch_id,
                        location.in_batch_id);
                for (size_t component = 0;
                     component < components;
                     ++component)
                {
                    rhs->sumIntoLocalValue(
                        cell_lid, component,
                        -out_flux
                      * face_value.component(component));
                }
            }

            if (mesh.is_interior_face(face_lid))
            {
                const auto other =
                    mesh.opposite_or_periodic_neighbor_cell(
                        face_lid, cell_lid);
                row_values.try_emplace(other, scalar_type{});
                const auto face_kinematic_viscosity =
                    detail::harmonic_face_value(
                        mesh, face_lid, cell_lid, other,
                        dynamic_viscosity)
                  / reference_density;
                if (face_kinematic_viscosity <= scalar_type{})
                {
                    continue;
                }
                const auto coefficient =
                    detail::interior_diffusion_coefficient(
                        mesh, face_lid, cell_lid, other,
                        face_kinematic_viscosity);
                detail::add_matrix_entry(
                    row_values, cell_lid, coefficient);
                detail::add_matrix_entry(
                    row_values, other, -coefficient);

                const auto tangential_area =
                    detail::non_orthogonal_area_vector(
                        mesh.face_area_vector_outward(
                            face_lid, cell_lid),
                        mesh.cell_center_vector(
                            face_lid, cell_lid));
                if (mesh.is_owned_cell(other)
                    && static_cast<size_t>(other)
                        < gradient_stencils.size())
                {
                    add_non_orthogonal_stencil(
                        cell_lid, scalar_type{0.5},
                        face_kinematic_viscosity,
                        tangential_area);
                    add_non_orthogonal_stencil(
                        other, scalar_type{0.5},
                        face_kinematic_viscosity,
                        tangential_area);
                }
                else
                {
                    add_non_orthogonal_stencil(
                        cell_lid, scalar_type{1},
                        face_kinematic_viscosity,
                        tangential_area);
                }
                continue;
            }

            if (!mesh.is_boundary_face(face_lid)
                || static_cast<size_t>(face_lid)
                    >= boundary_locations.size()
                || !boundary_locations[
                        static_cast<size_t>(face_lid)].active)
            {
                continue;
            }
            const auto location =
                boundary_locations[static_cast<size_t>(face_lid)];
            if (!boundary_diffusion(location.batch_id,
                                    location.in_batch_id))
            {
                continue;
            }
            const auto face_kinematic_viscosity =
                boundary_viscosity(
                    location.batch_id,
                    location.in_batch_id,
                    dynamic_viscosity.local_value(cell_lid))
              / reference_density;
            const auto coefficient =
                detail::boundary_diffusion_coefficient(
                    mesh, face_lid, cell_lid,
                    face_kinematic_viscosity);
            if (coefficient > scalar_type{})
            {
                detail::add_matrix_entry(
                    row_values, cell_lid, coefficient);
                const auto face_value =
                    boundary_value(
                        location.batch_id,
                        location.in_batch_id);
                for (size_t component = 0;
                     component < components;
                     ++component)
                {
                    rhs->sumIntoLocalValue(
                        cell_lid, component,
                        coefficient
                      * face_value.component(component));
                }
            }

            const auto tangential_area =
                detail::non_orthogonal_area_vector(
                    mesh.face_area_vector_outward(
                        face_lid, cell_lid),
                    mesh.face_centroid(face_lid)
                        - mesh.cell_centroid(cell_lid));
            add_non_orthogonal_stencil(
                cell_lid, scalar_type{1},
                face_kinematic_viscosity,
                tangential_area);
        }

        columns.clear();
        values.clear();
        for (const auto& [column, value] : row_values)
        {
            columns.push_back(column);
            values.push_back(value);
        }
        detail::add_transport_values<Pack>(
            prepared, cell_lid, columns(), values());
    }

    if (correction_field != nullptr
        && explicit_weight > scalar_type{})
    {
        add_variable_explicit_non_orthogonal_correction<Pack>(
            *correction_field,
            dynamic_viscosity,
            boundary_value,
            *rhs,
            explicit_weight / reference_density,
            boundary_diffusion,
            boundary_viscosity);
    }

    add_explicit_deviatoric_transpose_gradient_stress<Pack>(
        old_velocity,
        dynamic_viscosity,
        reference_density,
        boundary_value,
        *rhs,
        boundary_diffusion,
        boundary_viscosity);

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
