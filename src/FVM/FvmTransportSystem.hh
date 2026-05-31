/**
 * @file FvmTransportSystem.hh
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Semi-implicit finite-volume transport-system assembly.
 * @version 0.1
 * @date 2026-05-30
 *
 * @copyright Copyright (c) 2026
 *
 */
#pragma once

#include "fields/FaceField.hh"
#include "geometry/Mesh.hh"

#include <Teuchos_Array.hpp>
#include <Teuchos_RCP.hpp>

#include <cstddef>
#include <optional>
#include <stdexcept>
#include <vector>

namespace SimpleFluid::FvmOperators
{

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
    typename Pack::vector_type rhs;
};

/**
 * @brief Assemble the semi-implicit transport system (unsteady
 *        advection-diffusion) for a scalar field.
 *
 * The assembly uses first-order upwinding for advection and a two-point
 * flux approximation for diffusion.  Boundary values are supplied lazily
 * via the callable @p boundary_value.
 *
 * @tparam Pack The Tpetra type pack.
 * @tparam BoundaryValueProvider A callable with signature
 *         std::optional<scalar_type>(local_ordinal_type face_lid,
 *         scalar_type old_cell_value).
 * @param mesh The computational mesh.
 * @param old_values Previous time-step cell values indexed by local ID.
 * @param face_fluxes Pre-computed face volumetric fluxes.
 * @param time_step Time-step size (must be positive).
 * @param diffusivity Constant scalar diffusivity (non-negative).
 * @param boundary_value Callable that returns the prescribed boundary
 *        value for a face, or std::nullopt.
 * @param[in,out] cached_matrix Optional pre-allocated matrix to reuse.
 *        If non-null, resumeFill() is called and values are overwritten
 *        in-place, avoiding a new allocation and graph construction.
 *        The matrix must have been previously built by this function
 *        with the same mesh. On first call, pass a default-constructed
 *        (null) RCP; the returned TransportSystem will contain the new
 *        matrix which the caller should cache for subsequent calls.
 * @return TransportSystem containing the assembled matrix and RHS vector.
 * @throws std::invalid_argument if @p time_step <= 0, @p diffusivity < 0,
 *         or array sizes are inconsistent with @p mesh.
 * @throws std::runtime_error if any interior face connects coincident
 *         cell centers.
 */
template<TpetraTypePack Pack, class BoundaryValueProvider>
TransportSystem<Pack>
transport_system(const Mesh<Pack>& mesh,
                 const std::vector<typename Pack::scalar_type>& old_values,
                 const FaceField<Pack>& face_fluxes,
                 typename Pack::scalar_type time_step,
                 typename Pack::scalar_type diffusivity,
                 BoundaryValueProvider boundary_value,
                 Teuchos::RCP<typename Pack::matrix_type> cached_matrix = Teuchos::null)
{
    using matrix_type = typename Pack::matrix_type;
    using scalar_type = typename Pack::scalar_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;

    if (time_step <= 0.0)
    {
        throw std::invalid_argument("transport_system requires a positive time step.");
    }
    if (diffusivity < 0.0)
    {
        throw std::invalid_argument("transport_system requires non-negative diffusivity.");
    }
    if (old_values.size() < mesh.num_local_cells())
    {
        throw std::invalid_argument("transport_system old-value cache is too small.");
    }

    // Row map = owned cells; domain map = owned + ghost cells
    // so that neighbour columns on other ranks are valid.
    auto matrix = Teuchos::rcp(
        new matrix_type(mesh.owned_cell_map(), mesh.overlap_cell_map(), 12));
    (void)cached_matrix;
    typename Pack::vector_type rhs(mesh.owned_cell_map(), true);

    Teuchos::Array<local_ordinal_type> cols;
    Teuchos::Array<scalar_type> vals;
    cols.reserve(32);
    vals.reserve(32);

    for (std::size_t owned = 0; owned < mesh.num_owned_cells(); ++owned)
    {
        const auto cell_lid = static_cast<local_ordinal_type>(owned);

        cols.clear();
        vals.clear();
        scalar_type diagonal = mesh.cell_volume(cell_lid) / time_step;
        const auto old_value = old_values[static_cast<std::size_t>(cell_lid)];
        scalar_type rhs_value = diagonal * old_value;

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
                diagonal += out_flux;
            }
            else if (mesh.is_interior_face(face_lid))
            {
                const auto other = mesh.opposite_cell(face_lid, cell_lid);
                cols.push_back(other);
                vals.push_back(out_flux);
            }

            // === diffusion ===
            if (diffusivity <= scalar_type{0} || !mesh.is_interior_face(face_lid))
            {
                continue;
            }

            const auto distance = mesh.face_cell_center_distance(face_lid);
            if (distance <= scalar_type{0})
            {
                throw std::runtime_error(
                    "Cannot assemble diffusion across coincident cells.");
            }
            const auto coeff =
                diffusivity * mesh.face_area(face_lid) / distance;
            const auto other = mesh.opposite_cell(face_lid, cell_lid);
            diagonal += coeff;
            cols.push_back(other);
            vals.push_back(-coeff);
        }

        cols.push_back(cell_lid);
        vals.push_back(diagonal);
        matrix->insertLocalValues(cell_lid, cols(), vals());
        rhs.replaceLocalValue(cell_lid, rhs_value);
    }

    // Apply boundary conditions using sumIntoLocalValues (works reliably).
    for (const auto& [patch_id, boundary_patch] : mesh.boundary_patches())
    {
        for (std::size_t in_patch_id = 0;
             in_patch_id < boundary_patch.face_lids.size(); ++in_patch_id)
        {
            const auto face_lid = boundary_patch.face_lids[in_patch_id];
            if (!mesh.is_owned_face(face_lid))
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
                rhs.sumIntoLocalValue(owner,
                                      -out_flux * boundary_face_value);
            }

            // Diffusive boundary flux
            if (diffusivity <= scalar_type{0})
            {
                continue;
            }

            const auto distance =
                mesh.cell_to_face_distance(face_lid, owner);
            if (distance > scalar_type{0})
            {
                const auto coeff =
                    diffusivity * mesh.face_area(face_lid) / distance;

                local_ordinal_type col = owner;
                scalar_type bval = coeff;
                matrix->sumIntoLocalValues(
                    owner,
                    Teuchos::arrayView(&col, 1),
                    Teuchos::arrayView(&bval, 1));
                rhs.sumIntoLocalValue(owner,
                                      coeff * boundary_face_value);
            }
        }
    }

    matrix->fillComplete();
    return {matrix, rhs};
}

} // namespace SimpleFluid::FvmOperators
