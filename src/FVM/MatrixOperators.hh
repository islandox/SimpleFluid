/**
 * @file FVM/MatrixOperators.hh
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Finite-volume matrix assembly operators (identity, diffusion, convection,
 *        pressure Poisson).
 * @version 0.1
 * @date 2026-05-30
 *
 * @copyright Copyright (c) 2026
 *
 */
#pragma once

#include "equations/BoundaryConditions.hh"
#include "FVM/details/MatrixOperatorsImpl.hh"
#include "fields/FaceField.hh"
#include "fields/FieldStored.hh"
#include "geometry/Mesh.hh"

#include <Teuchos_Array.hpp>
#include <Teuchos_RCP.hpp>

#include <concepts>
#include <cstddef>
#include <optional>
#include <type_traits>
#include <utility>

namespace SimpleFluid::FVM
{

/**
 * @brief Build an identity (or scaled-identity) square matrix over the
 *        given map.
 *
 * @tparam Pack The Tpetra type pack.
 * @param map Row/column map for the matrix.
 * @param diagonal Value to place on the diagonal (default 1.0).
 * @return Fill-complete sparse matrix with @p diagonal on the diagonal.
 */
template<TpetraTypePack Pack>
Teuchos::RCP<typename Pack::matrix_type>
identity_matrix(const Teuchos::RCP<const typename Pack::map_type>& map,
                typename Pack::scalar_type diagonal = 1.0)
{
    using matrix_type = typename Pack::matrix_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;

    auto matrix = Teuchos::rcp(new matrix_type(map, map, 1));
    for (size_t row = 0; row < map->getLocalNumElements(); ++row)
    {
        const auto lid = static_cast<local_ordinal_type>(row);
        Teuchos::Array<local_ordinal_type> cols{lid};
        Teuchos::Array<typename Pack::scalar_type> vals{diagonal};
        matrix->insertLocalValues(lid, cols(), vals());
    }
    matrix->fillComplete();
    return matrix;
}

/**
 * @brief Assemble the finite-volume diffusion matrix (negative Laplacian).
 *
 * @tparam Pack The Tpetra type pack.
 * @param mesh The computational mesh.
 * @param diffusivity Constant scalar diffusivity coefficient.
 * @return Fill-complete sparse matrix representing the diffusion operator.
 * @throws std::runtime_error if any interior face connects coincident
 *         cell centers.
 */
template<TpetraTypePack Pack>
Teuchos::RCP<typename Pack::matrix_type>
diffusion_matrix(const Mesh<Pack>& mesh, typename Pack::scalar_type diffusivity)
{
    return detail::diffusion_matrix_impl<Pack>(mesh, diffusivity);
}

/**
 * @brief Assemble the first-order upwind convection matrix from
 *        pre-computed face fluxes stored in a FaceField.
 *
 * @tparam Pack The Tpetra type pack.
 * @param mesh The computational mesh.
 * @param face_fluxes FaceField of scalar face fluxes.
 * @return Fill-complete sparse matrix representing the upwind convection
 *         operator.
 */
template<TpetraTypePack Pack>
Teuchos::RCP<typename Pack::matrix_type>
upwind_convection_matrix(
    const Mesh<Pack>& mesh,
    const FaceField<Pack>& face_fluxes)
{
    return detail::upwind_convection_matrix_impl<Pack>(
        mesh,
        [&](typename Pack::local_ordinal_type face_lid)
        {
            return face_fluxes.is_owned_face(face_lid);
        },
        [&](typename Pack::local_ordinal_type face_lid)
        {
            return face_fluxes.value(face_lid);
        });
}

/**
 * @brief Assemble the pressure Poisson matrix with pressure-correction
 *        boundary conditions and an optional gauge constraint.
 *
 * Dirichlet pressure boundaries add the homogeneous correction coefficient
 * to the owner diagonal. Neumann pressure boundaries require no matrix term.
 *
 * @tparam Pack The Tpetra type pack.
 * @tparam BoundaryConditionProvider Callable returning a pressure boundary
 *         condition for a batch ID and in-batch face index.
 * @param mesh The computational mesh.
 * @param gauge_cell_gid Optional global row-map ID used to fix the pressure
 *        level for an all-Neumann system.
 * @param boundary_condition Pressure-boundary provider.
 * @return Fill-complete sparse matrix representing the Poisson operator.
 * @throws std::invalid_argument for unsupported pressure boundary types.
 * @throws std::runtime_error if any interior face connects coincident
 *         cell centers.
 */
template<TpetraTypePack Pack, class BoundaryConditionProvider>
Teuchos::RCP<typename Pack::matrix_type>
pressure_poisson_matrix(
    const Mesh<Pack>& mesh,
    std::optional<typename Pack::global_ordinal_type> gauge_cell_gid,
    BoundaryConditionProvider boundary_condition)
{
    return detail::pressure_poisson_matrix_impl<Pack>(
        mesh, gauge_cell_gid, std::move(boundary_condition));
}

/**
 * @brief Assemble an all-Neumann pressure Poisson matrix with one gauge row.
 *
 * @tparam Pack The Tpetra type pack.
 * @param mesh The computational mesh.
 * @param gauge_cell_gid Global row-map ID of the pressure gauge cell.
 * @return Fill-complete sparse matrix representing the Poisson operator.
 */
template<TpetraTypePack Pack>
Teuchos::RCP<typename Pack::matrix_type>
pressure_poisson_matrix(
    const Mesh<Pack>& mesh,
    typename Pack::global_ordinal_type gauge_cell_gid)
{
    auto homogeneous_neumann =
        [](int, size_t)
    {
        return BoundaryCondition{};
    };
    return pressure_poisson_matrix<Pack>(
        mesh, gauge_cell_gid, homogeneous_neumann);
}

/**
 * @brief Assemble diffusion on a mapped orthogonal/semi-structured mesh
 *        interface.
 *
 * The explicit @p Pack parameter preserves the established call surface:
 * `diffusion_matrix<Pack>(mesh, diffusivity)`.
 */
template<TpetraTypePack Pack, class MeshType>
    requires (!std::same_as<
        std::remove_cv_t<MeshType>, Mesh<Pack>>)
Teuchos::RCP<typename Pack::matrix_type>
diffusion_matrix(
    const MeshType& mesh,
    typename Pack::scalar_type diffusivity)
{
    return detail::diffusion_matrix_impl<Pack>(mesh, diffusivity);
}

/**
 * @brief Assemble upwind convection from a mesh-aware stored face field.
 * @note Synchronize @p face_fluxes before distributed assembly so overlap
 *       partition faces contain their owning rank's accepted flux.
 */
template<TpetraTypePack Pack, class MeshType>
Teuchos::RCP<typename Pack::matrix_type>
upwind_convection_matrix(
    const MeshType& mesh,
    const ScalarFaceFieldStored<Pack, MeshType>& face_fluxes)
{
    if (&mesh != &face_fluxes.mesh())
    {
        throw std::invalid_argument(
            "upwind_convection_matrix requires the face field mesh.");
    }
    return detail::upwind_convection_matrix_impl<Pack>(
        mesh,
        [](typename Pack::local_ordinal_type)
        {
            return true;
        },
        [&](typename Pack::local_ordinal_type face_lid)
        {
            return face_fluxes.is_owned(face_lid)
                 ? face_fluxes.value(face_lid)
                 : face_fluxes.local_value(face_lid);
        });
}

/** @brief Assemble pressure Poisson on a mapped static/runtime mesh. */
template<TpetraTypePack Pack,
         class MeshType,
         class BoundaryConditionProvider>
    requires (!std::same_as<
        std::remove_cv_t<MeshType>, Mesh<Pack>>)
Teuchos::RCP<typename Pack::matrix_type>
pressure_poisson_matrix(
    const MeshType& mesh,
    std::optional<typename Pack::global_ordinal_type> gauge_cell_gid,
    BoundaryConditionProvider boundary_condition)
{
    return detail::pressure_poisson_matrix_impl<Pack>(
        mesh, gauge_cell_gid, std::move(boundary_condition));
}

/** @brief Assemble all-Neumann pressure Poisson on a mapped mesh. */
template<TpetraTypePack Pack, class MeshType>
    requires (!std::same_as<
        std::remove_cv_t<MeshType>, Mesh<Pack>>)
Teuchos::RCP<typename Pack::matrix_type>
pressure_poisson_matrix(
    const MeshType& mesh,
    typename Pack::global_ordinal_type gauge_cell_gid)
{
    auto homogeneous_neumann = [](int, size_t)
    {
        return BoundaryCondition{};
    };
    return detail::pressure_poisson_matrix_impl<Pack>(
        mesh,
        std::optional<typename Pack::global_ordinal_type>{gauge_cell_gid},
        homogeneous_neumann);
}

} // namespace SimpleFluid::FVM
