/** @file VoFProjectionWorkspace.cc
 * @brief Reusable canonical-subface pressure matrix with explicit numeric invalidation.
 */
#include "solvers/details/VoFProjectionWorkspace.hh"
#include <Teuchos_CommHelpers.hpp>
#include <algorithm>
#include <cmath>
#include <map>
#include <stdexcept>

namespace SimpleFluid::detail
{
using LO = DefaultTpetraTypes::local_ordinal_type;
using GO = DefaultTpetraTypes::global_ordinal_type;
VoFProjectionWorkspace::VoFProjectionWorkspace(SP<const MeshHandle<>> mesh)
    : rhs(mesh->owned_cell_map(), false), solution(mesh->owned_cell_map(), false), mesh_(std::move(mesh))
{
    const auto execution = mesh_->acquire_execution_view();
    for (std::size_t c = 0; c < mesh_->num_owned_cells(); ++c)
        row_capacity_ = std::max(row_capacity_, mesh_->faces(static_cast<LO>(c)).size() + 1);
}
void VoFProjectionWorkspace::refresh(const std::vector<double>& mobility,
    const std::vector<double>* susceptibility, double dt, GO gauge)
{
    const bool first = matrix.is_null();
    if (first)
        matrix = Teuchos::rcp(new Pack::matrix_type(mesh_->owned_cell_map(), row_capacity_));
    else
    {
        if (!matrix->isFillActive())
            matrix->resumeFill();
        matrix->setAllToScalar(0);
    }
    bool valid = true;
    {
        const auto execution = mesh_->acquire_execution_view();
        for (std::size_t c = 0; c < mesh_->num_owned_cells(); ++c)
        {
            const auto lid = static_cast<LO>(c);
            const auto gid = mesh_->cell_global_id(lid);
            const bool fixed = !susceptibility && gid == gauge;
            double diagonal = fixed ? 1 : (susceptibility ? (*susceptibility)[c] / (dt * dt) : 0);
            std::map<GO, double> row;
            for (const auto face : mesh_->faces(lid))
            {
                if (mesh_->is_geometry_exterior_face(face))
                    continue;
                const auto neighbor = mesh_->opposite_cell(face, lid);
                const auto ngid = mesh_->cell_global_id(neighbor);
                auto& value = row[ngid]; // Include zero gauge entries in the reusable graph.
                if (fixed)
                    continue;
                const auto normal = mesh_->face_normal(face);
                const double distance = std::abs(mesh_->face_center_vector(face, lid).dot(normal)) +
                                        std::abs(mesh_->face_center_vector(face, neighbor).dot(normal));
                const double coefficient = mobility[face] * mesh_->face_area(face) / distance;
                valid = valid && std::isfinite(coefficient) && coefficient > 0;
                diagonal += coefficient;
                if (susceptibility || ngid != gauge)
                    value -= coefficient;
            }
            valid = valid && std::isfinite(diagonal) && diagonal > 0;
            row[gid] += diagonal;
            Teuchos::Array<GO> columns;
            Teuchos::Array<double> values;
            for (const auto& [column, value] : row)
            {
                columns.push_back(column);
                values.push_back(value);
            }
            if (first)
                matrix->insertGlobalValues(gid, columns(), values());
            else
                matrix->replaceGlobalValues(gid, columns(), values());
        }
    }
    int local = valid, global = 0;
    Teuchos::reduceAll(*mesh_->owned_cell_map()->getComm(), Teuchos::REDUCE_MIN, 1, &local, &global);
    if (!global)
        throw std::runtime_error("VoF pressure preconditioner has invalid mobility or susceptibility.");
    matrix->fillComplete();
    solver.notify_operator_values_changed();
}
} // namespace SimpleFluid::detail
