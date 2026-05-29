/**
 * @file FvmTransportSystem.hh
 * @brief Semi-implicit finite-volume transport-system assembly.
 */
#pragma once

#include "geometry/Mesh.hh"

#include <Teuchos_Array.hpp>
#include <Teuchos_RCP.hpp>

#include <cstddef>
#include <optional>
#include <stdexcept>
#include <vector>

namespace SimpleFluid::FvmOperators
{

template<TpetraTypePack Pack>
struct TransportSystem
{
    Teuchos::RCP<typename Pack::matrix_type> matrix;
    typename Pack::vector_type rhs;
};

template<TpetraTypePack Pack, class BoundaryValueProvider>
TransportSystem<Pack>
transport_system(const Mesh<Pack>& mesh,
                 const std::vector<typename Pack::scalar_type>& old_values,
                 const std::vector<typename Pack::scalar_type>& face_fluxes,
                 typename Pack::scalar_type time_step,
                 typename Pack::scalar_type diffusivity,
                 BoundaryValueProvider boundary_value)
{
    using matrix_type = typename Pack::matrix_type;
    using global_ordinal_type = typename Pack::global_ordinal_type;
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
    if (face_fluxes.size() != mesh.num_faces())
    {
        throw std::invalid_argument("transport_system received the wrong face-flux size.");
    }

    auto matrix = Teuchos::rcp(new matrix_type(mesh.owned_cell_map(), 12));
    typename Pack::vector_type rhs(mesh.owned_cell_map(), true);
    Teuchos::Array<global_ordinal_type> cols;
    Teuchos::Array<scalar_type> vals;
    cols.reserve(32);
    vals.reserve(32);

    for (std::size_t owned = 0; owned < mesh.num_owned_cells(); ++owned)
    {
        const auto cell_lid = static_cast<local_ordinal_type>(owned);
        const auto row_gid = mesh.cell_global_id(cell_lid);

        cols.clear();
        vals.clear();
        scalar_type diagonal = mesh.cell_volume(cell_lid) / time_step;
        const auto old_value = old_values[static_cast<std::size_t>(cell_lid)];
        scalar_type rhs_value = diagonal * old_value;

        for (const auto face_lid : mesh.faces(cell_lid))
        {
            std::optional<scalar_type> cached_boundary_value;
            bool boundary_value_cached = false;
            auto face_boundary_value = [&]() -> const std::optional<scalar_type>&
            {
                if (!boundary_value_cached)
                {
                    cached_boundary_value = boundary_value(face_lid, old_value);
                    boundary_value_cached = true;
                }
                return cached_boundary_value;
            };

            const auto owner_oriented_flux =
                face_fluxes[static_cast<std::size_t>(face_lid)];
            const auto out_flux = mesh.owner_cell(face_lid) == cell_lid
                                ? owner_oriented_flux
                                : -owner_oriented_flux;

            if (out_flux >= 0.0)
            {
                diagonal += out_flux;
            }
            else if (mesh.is_interior_face(face_lid))
            {
                const auto other = mesh.opposite_cell(face_lid, cell_lid);
                cols.push_back(mesh.cell_global_id(other));
                vals.push_back(out_flux);
            }
            else
            {
                const auto& value = face_boundary_value();
                rhs_value -= out_flux * value.value_or(old_value);
            }

            if (diffusivity <= 0.0)
            {
                continue;
            }

            if (mesh.is_interior_face(face_lid))
            {
                const auto distance = mesh.face_cell_center_distance(face_lid);
                if (distance <= 0.0)
                {
                    throw std::runtime_error(
                        "Cannot assemble diffusion across coincident cells.");
                }
                const auto coeff =
                    diffusivity * mesh.face_area(face_lid) / distance;
                const auto other = mesh.opposite_cell(face_lid, cell_lid);
                diagonal += coeff;
                cols.push_back(mesh.cell_global_id(other));
                vals.push_back(-coeff);
            }
            else
            {
                const auto distance =
                    mesh.cell_to_face_distance(face_lid, cell_lid);
                const auto& value = face_boundary_value();
                if (distance > 0.0 && value.has_value())
                {
                    const auto coeff =
                        diffusivity * mesh.face_area(face_lid) / distance;
                    diagonal += coeff;
                    rhs_value += coeff * *value;
                }
            }
        }

        cols.push_back(row_gid);
        vals.push_back(diagonal);
        matrix->insertGlobalValues(row_gid, cols(), vals());
        rhs.replaceLocalValue(static_cast<local_ordinal_type>(owned), rhs_value);
    }

    matrix->fillComplete();
    return {matrix, rhs};
}

} // namespace SimpleFluid::FvmOperators
