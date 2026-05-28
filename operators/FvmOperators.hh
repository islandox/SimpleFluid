/**
 * @file FvmOperators.hh
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Finite-volume helper operators for cell and face fields.
 * @version 0.1
 * @date 2026-05-28
 *
 * @copyright Copyright (c) 2026
 *
 */
#pragma once

#include "equations/BoundaryConditions.hh"
#include "fields/CellField.hh"
#include "fields/FaceField.hh"

#include <Teuchos_Array.hpp>
#include <Teuchos_RCP.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <optional>
#include <stdexcept>
#include <vector>

namespace SimpleFluid::FvmOperators
{

namespace detail
{

/**
 * @brief Access the i-th component of a Vec3 by reference.
 *
 * @param vector 3D vector.
 * @param index Component index (0=x, 1=y, 2=z).
 * @return Reference to the selected component.
 */
inline real_t& component(MeshUtils::Vec3& vector, std::size_t index)
{
    return index == 0 ? vector.x : (index == 1 ? vector.y : vector.z);
}

/**
 * @brief Solve a 3x3 linear system using Gaussian elimination with partial pivoting.
 *
 * @param a 3x3 matrix (modified in place).
 * @param b Right-hand side vector (modified in place, overwritten with solution).
 * @return The solution vector (reference to modified b).
 */
inline MeshUtils::Vec3 solve_3x3(std::array<std::array<real_t, 3>, 3>& a,
                                 MeshUtils::Vec3& b)
{
    for (std::size_t pivot = 0; pivot < 3; ++pivot)
    {
        std::size_t best = pivot;
        for (std::size_t row = pivot + 1; row < 3; ++row)
        {
            if (std::abs(a[row][pivot]) > std::abs(a[best][pivot]))
            {
                best = row;
            }
        }

        if (std::abs(a[best][pivot]) < 1.0e-14)
        {
            b = {};
            return {};
        }

        if (best != pivot)
        {
            std::swap(a[best], a[pivot]);
            std::swap(component(b, best), component(b, pivot));
        }

        const auto inv = 1.0 / a[pivot][pivot];
        for (std::size_t col = pivot; col < 3; ++col)
        {
            a[pivot][col] *= inv;
        }
        component(b, pivot) *= inv;

        for (std::size_t row = 0; row < 3; ++row)
        {
            if (row == pivot)
            {
                continue;
            }

            const auto factor = a[row][pivot];
            for (std::size_t col = pivot; col < 3; ++col)
            {
                a[row][col] -= factor * a[pivot][col];
            }
            component(b, row) -= factor * component(b, pivot);
        }
    }

    return b;
}

inline real_t component_value(const MeshUtils::Vec3& vector, std::size_t index)
{
    return index == 0 ? vector.x : (index == 1 ? vector.y : vector.z);
}

} // namespace detail

/**
 * @brief Compute cell-centered gradients using a least-squares approach over face neighbors.
 *
 * @tparam Pack Tpetra type pack.
 * @param field Cell-centered scalar field.
 * @return Vector of gradients, one per owned cell.
 */
template<TpetraTypePack Pack>
std::vector<typename Mesh<Pack>::Vec3>
cell_gradient(const CellField<Pack>& field)
{
    using mesh_type = Mesh<Pack>;
    using local_ordinal_type = typename mesh_type::local_ordinal_type;

    const auto& mesh = field.mesh();
    std::vector<typename mesh_type::Vec3> gradients(mesh.num_owned_cells());

    for (std::size_t owned = 0; owned < mesh.num_owned_cells(); ++owned)
    {
        const auto cell_lid = static_cast<local_ordinal_type>(owned);
        const auto phi_p = field.value(cell_lid);
        const auto& center_p = mesh.cell_centroid(cell_lid);

        std::array<std::array<real_t, 3>, 3> normal{};
        typename mesh_type::Vec3 rhs{};

        for (const auto face_lid : mesh.faces(cell_lid))
        {
            if (!mesh.is_interior_face(face_lid))
            {
                continue;
            }

            const auto other = mesh.opposite_cell(face_lid, cell_lid);
            const auto d = mesh.cell_centroid(other) - center_p;
            const auto phi_delta = field.local_value(other) - phi_p;

            normal[0][0] += d.x * d.x;
            normal[0][1] += d.x * d.y;
            normal[0][2] += d.x * d.z;
            normal[1][0] += d.y * d.x;
            normal[1][1] += d.y * d.y;
            normal[1][2] += d.y * d.z;
            normal[2][0] += d.z * d.x;
            normal[2][1] += d.z * d.y;
            normal[2][2] += d.z * d.z;

            rhs.x += d.x * phi_delta;
            rhs.y += d.y * phi_delta;
            rhs.z += d.z * phi_delta;
        }

        gradients[owned] = detail::solve_3x3(normal, rhs);
    }

    return gradients;
}

/**
 * @brief Compute the cell-centered divergence of a face flux field.
 *
 * @tparam Pack Tpetra type pack.
 * @param mesh The finite-volume mesh.
 * @param flux Face-centered flux field.
 * @return Vector of divergence values, one per owned cell.
 */
template<TpetraTypePack Pack>
std::vector<typename Pack::scalar_type>
cell_divergence(const Mesh<Pack>& mesh, const FaceField<Pack>& flux)
{
    using local_ordinal_type = typename Mesh<Pack>::local_ordinal_type;

    std::vector<typename Pack::scalar_type> divergence(mesh.num_owned_cells(), 0.0);
    for (std::size_t owned = 0; owned < mesh.num_owned_cells(); ++owned)
    {
        const auto cell_lid = static_cast<local_ordinal_type>(owned);
        auto balance = typename Pack::scalar_type{};

        for (const auto face_lid : mesh.faces(cell_lid))
        {
            if (!flux.is_owned_face(face_lid))
            {
                continue;
            }

            const auto sign = mesh.owner_cell(face_lid) == cell_lid ? 1.0 : -1.0;
            balance += sign * flux.value(face_lid) * mesh.face_area(face_lid);
        }

        divergence[owned] = balance / mesh.cell_volume(cell_lid);
    }

    return divergence;
}

/**
 * @brief Build a diagonal (identity) Tpetra matrix over the given map.
 *
 * @tparam Pack Tpetra type pack.
 * @param map Tpetra map defining the matrix row distribution.
 * @param diagonal Value to place on the diagonal.
 * @return RCP to the filled identity matrix.
 */
template<TpetraTypePack Pack>
Teuchos::RCP<typename Pack::matrix_type>
identity_matrix(const Teuchos::RCP<const typename Pack::map_type>& map,
                typename Pack::scalar_type diagonal = 1.0)
{
    using matrix_type = typename Pack::matrix_type;
    using global_ordinal_type = typename Pack::global_ordinal_type;

    auto matrix = Teuchos::rcp(new matrix_type(map, 1));
    for (std::size_t row = 0; row < map->getLocalNumElements(); ++row)
    {
        const auto gid = map->getGlobalElement(
            static_cast<typename Pack::local_ordinal_type>(row));
        Teuchos::Array<global_ordinal_type> cols{gid};
        Teuchos::Array<typename Pack::scalar_type> vals{diagonal};
        matrix->insertGlobalValues(gid, cols(), vals());
    }
    matrix->fillComplete();
    return matrix;
}

/**
 * @brief Assemble a finite-volume diffusion matrix with a constant diffusivity.
 *
 * @tparam Pack Tpetra type pack.
 * @param mesh The finite-volume mesh providing connectivity and geometry.
 * @param diffusivity Constant diffusion coefficient.
 * @return RCP to the filled diffusion matrix.
 * @throws std::runtime_error if coincident cells have zero face-center distance.
 */
template<TpetraTypePack Pack>
Teuchos::RCP<typename Pack::matrix_type>
diffusion_matrix(const Mesh<Pack>& mesh, typename Pack::scalar_type diffusivity)
{
    using matrix_type = typename Pack::matrix_type;
    using global_ordinal_type = typename Pack::global_ordinal_type;
    using scalar_type = typename Pack::scalar_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;

    auto matrix = Teuchos::rcp(new matrix_type(mesh.owned_cell_map(), 8));
    for (std::size_t owned = 0; owned < mesh.num_owned_cells(); ++owned)
    {
        const auto cell_lid = static_cast<local_ordinal_type>(owned);
        const auto row_gid = mesh.cell_global_id(cell_lid);

        Teuchos::Array<global_ordinal_type> cols;
        Teuchos::Array<scalar_type> vals;
        scalar_type diagonal = 0.0;

        for (const auto face_lid : mesh.faces(cell_lid))
        {
            if (!mesh.is_interior_face(face_lid))
            {
                continue;
            }

            const auto other = mesh.opposite_cell(face_lid, cell_lid);
            const auto distance = mesh.face_cell_center_distance(face_lid);
            if (distance <= 0.0)
            {
                throw std::runtime_error("Cannot assemble diffusion across coincident cells.");
            }

            const auto coeff = diffusivity * mesh.face_area(face_lid) / distance;
            diagonal += coeff;
            cols.push_back(mesh.cell_global_id(other));
            vals.push_back(-coeff);
        }

        cols.push_back(row_gid);
        vals.push_back(diagonal);
        matrix->insertGlobalValues(row_gid, cols(), vals());
    }

    matrix->fillComplete();
    return matrix;
}

/**
 * @brief Linear system data for a semi-implicit scalar transport update.
 */
template<TpetraTypePack Pack>
struct TransportSystem
{
    Teuchos::RCP<typename Pack::matrix_type> matrix;
    typename Pack::vector_type rhs;
};

/**
 * @brief Compute owner-oriented integrated face fluxes from cell velocities.
 *
 * Interior fluxes use arithmetic face interpolation. Exterior faces default to
 * zero normal flux unless a velocity Dirichlet or no-slip condition is present.
 *
 * @tparam Pack Tpetra type pack.
 * @param mesh Finite-volume mesh.
 * @param velocity_x Cell-centered x velocity.
 * @param velocity_y Cell-centered y velocity.
 * @param velocity_z Cell-centered z velocity.
 * @param boundary_conditions Optional velocity boundary-condition set.
 * @return Integrated flux u dot n_owner times face area for each face.
 */
template<TpetraTypePack Pack>
std::vector<typename Pack::scalar_type>
face_fluxes(const Mesh<Pack>& mesh,
            const CellField<Pack>& velocity_x,
            const CellField<Pack>& velocity_y,
            const CellField<Pack>& velocity_z,
            const BoundaryConditionSet* boundary_conditions = nullptr)
{
    using scalar_type = typename Pack::scalar_type;

    if (&velocity_x.mesh() != &mesh || &velocity_y.mesh() != &mesh
        || &velocity_z.mesh() != &mesh)
    {
        throw std::invalid_argument("face_fluxes requires velocity fields on the input mesh.");
    }

    std::vector<scalar_type> fluxes(mesh.num_faces(), 0.0);
    for (std::size_t face = 0; face < mesh.num_faces(); ++face)
    {
        const auto face_lid =
            static_cast<typename Pack::local_ordinal_type>(face);
        const auto owner = mesh.owner_cell(face_lid);
        typename Mesh<Pack>::Vec3 velocity{
            velocity_x.local_value(owner),
            velocity_y.local_value(owner),
            velocity_z.local_value(owner)
        };

        if (mesh.is_interior_face(face_lid))
        {
            const auto neighbor = mesh.neighbor_cell(face_lid);
            velocity = (velocity
                      + typename Mesh<Pack>::Vec3{
                            velocity_x.local_value(neighbor),
                            velocity_y.local_value(neighbor),
                            velocity_z.local_value(neighbor)})
                     / 2.0;
        }
        else if (boundary_conditions != nullptr && mesh.is_boundary_face(face_lid))
        {
            const auto iter =
                boundary_conditions->velocity.find(mesh.boundary_name(face_lid));
            if (iter == boundary_conditions->velocity.end())
            {
                fluxes[face] = 0.0;
                continue;
            }

            if (iter->second.type == BoundaryConditionType::NoSlip)
            {
                velocity = {};
            }
            else if (iter->second.type == BoundaryConditionType::Dirichlet)
            {
                velocity = iter->second.value;
            }
            else
            {
                fluxes[face] = 0.0;
                continue;
            }
        }
        else if (!mesh.is_interior_face(face_lid))
        {
            fluxes[face] = 0.0;
            continue;
        }

        fluxes[face] = velocity.dot(mesh.face_normal(face_lid))
                     * mesh.face_area(face_lid);
    }

    return fluxes;
}

/**
 * @brief Integrated flux balance for one cell.
 *
 * Positive values indicate net outflow from the cell.
 *
 * @tparam Pack Tpetra type pack.
 * @param mesh The finite-volume mesh.
 * @param face_fluxes Owner-oriented integrated fluxes for all faces.
 * @param cell_lid Local index of the cell.
 * @return Net flux balance for the cell.
 */
template<TpetraTypePack Pack>
typename Pack::scalar_type cell_flux_balance(
    const Mesh<Pack>& mesh,
    const std::vector<typename Pack::scalar_type>& face_fluxes,
    typename Pack::local_ordinal_type cell_lid)
{
    if (face_fluxes.size() != mesh.num_faces())
    {
        throw std::invalid_argument("cell_flux_balance received the wrong face-flux size.");
    }

    typename Pack::scalar_type balance = 0.0;
    for (const auto face_lid : mesh.faces(cell_lid))
    {
        const auto sign = mesh.owner_cell(face_lid) == cell_lid ? 1.0 : -1.0;
        balance += sign * face_fluxes[static_cast<std::size_t>(face_lid)];
    }

    return balance;
}

/**
 * @brief Compute cell-centered divergence from owner-oriented integrated fluxes.
 *
 * @tparam Pack Tpetra type pack.
 * @param mesh The finite-volume mesh.
 * @param face_fluxes Owner-oriented integrated fluxes for all faces.
 * @return Vector of divergence values, one per owned cell.
 */
template<TpetraTypePack Pack>
std::vector<typename Pack::scalar_type>
cell_divergence_from_fluxes(
    const Mesh<Pack>& mesh,
    const std::vector<typename Pack::scalar_type>& face_fluxes)
{
    std::vector<typename Pack::scalar_type> divergence(mesh.num_owned_cells(), 0.0);
    for (std::size_t owned = 0; owned < mesh.num_owned_cells(); ++owned)
    {
        const auto cell_lid =
            static_cast<typename Pack::local_ordinal_type>(owned);
        divergence[owned] = cell_flux_balance(mesh, face_fluxes, cell_lid)
                          / mesh.cell_volume(cell_lid);
    }

    return divergence;
}

/**
 * @brief Assemble the integrated first-order upwind convection operator.
 * 
 * @tparam Pack Tpetra type pack.
 * @param mesh The finite-volume mesh.
 * @param face_fluxes Owner-oriented integrated fluxes for all faces.
 * @return RCP to the filled upwind convection matrix.
 *  return divergence;
 */
template<TpetraTypePack Pack>
Teuchos::RCP<typename Pack::matrix_type>
upwind_convection_matrix(
    const Mesh<Pack>& mesh,
    const std::vector<typename Pack::scalar_type>& face_fluxes)
{
    using matrix_type = typename Pack::matrix_type;
    using global_ordinal_type = typename Pack::global_ordinal_type;
    using scalar_type = typename Pack::scalar_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;

    if (face_fluxes.size() != mesh.num_faces())
    {
        throw std::invalid_argument("upwind_convection_matrix received the wrong face-flux size.");
    }

    auto matrix = Teuchos::rcp(new matrix_type(mesh.owned_cell_map(), 8));
    for (std::size_t owned = 0; owned < mesh.num_owned_cells(); ++owned)
    {
        const auto cell_lid = static_cast<local_ordinal_type>(owned);
        const auto row_gid = mesh.cell_global_id(cell_lid);

        Teuchos::Array<global_ordinal_type> cols;
        Teuchos::Array<scalar_type> vals;
        scalar_type diagonal = 0.0;

        for (const auto face_lid : mesh.faces(cell_lid))
        {
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
        }

        cols.push_back(row_gid);
        vals.push_back(diagonal);
        matrix->insertGlobalValues(row_gid, cols(), vals());
    }

    matrix->fillComplete();
    return matrix;
}

/**
 * @brief Assemble mass, upwind convection, and diffusion into one transport system.
 *
 * BoundaryValueProvider must return std::optional<scalar> for a boundary face.
 * A missing value means zero-gradient diffusion and fallback inflow value.
 * @tparam Pack Tpetra type pack.
 * @tparam BoundaryValueProvider Callable returning std::optional<scalar_type>
 *         for a boundary face local ID and fallback value.
 * @param mesh The finite-volume mesh.
 * @param old_values Previous-time solution values for all local cells.
 * @param face_fluxes Owner-oriented integrated fluxes for all faces.
 * @param time_step Time-step size.
 * @param diffusivity Constant diffusion coefficient.
 * @param boundary_value Boundary value provider callable.
 * @return TransportSystem containing the assembled matrix and RHS vector.
 *
 */
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

    for (std::size_t owned = 0; owned < mesh.num_owned_cells(); ++owned)
    {
        const auto cell_lid = static_cast<local_ordinal_type>(owned);
        const auto row_gid = mesh.cell_global_id(cell_lid);

        Teuchos::Array<global_ordinal_type> cols;
        Teuchos::Array<scalar_type> vals;
        scalar_type diagonal = mesh.cell_volume(cell_lid) / time_step;
        scalar_type rhs_value =
            diagonal * old_values[static_cast<std::size_t>(cell_lid)];

        for (const auto face_lid : mesh.faces(cell_lid))
        {
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
                const auto value = boundary_value(
                    face_lid, old_values[static_cast<std::size_t>(cell_lid)]);
                rhs_value -= out_flux * value.value_or(
                    old_values[static_cast<std::size_t>(cell_lid)]);
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
                const auto value = boundary_value(
                    face_lid, old_values[static_cast<std::size_t>(cell_lid)]);
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

/**
 * @brief Assemble a finite-volume pressure Poisson matrix with one gauge row.
 * 
 * @tparam Pack Tpetra type pack.
 * @param mesh The finite-volume mesh.
 * @param gauge_cell_gid Global ID of the cell used as the pressure gauge.
 * @return RCP to the filled pressure Poisson matrix.
 */
template<TpetraTypePack Pack>
Teuchos::RCP<typename Pack::matrix_type>
pressure_poisson_matrix(
    const Mesh<Pack>& mesh,
    typename Pack::global_ordinal_type gauge_cell_gid)
{
    using matrix_type = typename Pack::matrix_type;
    using global_ordinal_type = typename Pack::global_ordinal_type;
    using scalar_type = typename Pack::scalar_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;

    auto matrix = Teuchos::rcp(new matrix_type(mesh.owned_cell_map(), 8));
    for (std::size_t owned = 0; owned < mesh.num_owned_cells(); ++owned)
    {
        const auto cell_lid = static_cast<local_ordinal_type>(owned);
        const auto row_gid = mesh.cell_global_id(cell_lid);

        Teuchos::Array<global_ordinal_type> cols;
        Teuchos::Array<scalar_type> vals;

        if (row_gid == gauge_cell_gid)
        {
            cols.push_back(row_gid);
            vals.push_back(1.0);
            matrix->insertGlobalValues(row_gid, cols(), vals());
            continue;
        }

        scalar_type diagonal = 0.0;
        for (const auto face_lid : mesh.faces(cell_lid))
        {
            if (!mesh.is_interior_face(face_lid))
            {
                continue;
            }

            const auto distance = mesh.face_cell_center_distance(face_lid);
            if (distance <= 0.0)
            {
                throw std::runtime_error(
                    "Cannot assemble pressure Poisson matrix across coincident cells.");
            }
            const auto coeff = mesh.face_area(face_lid) / distance;
            const auto other = mesh.opposite_cell(face_lid, cell_lid);
            diagonal += coeff;
            cols.push_back(mesh.cell_global_id(other));
            vals.push_back(-coeff);
        }

        cols.push_back(row_gid);
        vals.push_back(diagonal > 0.0 ? diagonal : 1.0);
        matrix->insertGlobalValues(row_gid, cols(), vals());
    }

    matrix->fillComplete();
    return matrix;
}

} // namespace SimpleFluid::FvmOperators
