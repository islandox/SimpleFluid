#pragma once

#include "AssembledPressureFaces.hh"
#include "FVM/details/CachedGradientRows.hh"
#include "equations/CollectiveValidation.hh"

#include <TpetraExt_MatrixMatrix.hpp>

#include <array>
#include <limits>

namespace SimpleFluid::detail
{
template<TpetraTypePack Pack, class Mesh, class Topology, class Metrics, class Cache>
void assemble_full_pressure_operator(const Mesh& mesh, const Topology& topology, const Metrics& metrics,
    const BoundaryConditionMap& flux_conditions, const BoundaryConditionMap& gradient_conditions,
    const Cache& cache, const std::optional<typename Pack::global_ordinal_type>& gauge,
    const typename Pack::matrix_type& compact, Teuchos::RCP<const typename Pack::map_type>& channel_map,
    Teuchos::RCP<const typename Pack::import_type>& face_import, Teuchos::RCP<typename Pack::matrix_type>& result)
{
    using global_type = typename Pack::global_ordinal_type;
    using local_type = typename Pack::local_ordinal_type;
    using scalar_type = typename Pack::scalar_type;
    using matrix_type = typename Pack::matrix_type;
    using map_type = typename Pack::map_type;
    const auto cells = mesh.owned_cell_map();
    const auto overlap = mesh.overlap_cell_map();
    Teuchos::Array<global_type> channels;
    Teuchos::Array<std::size_t> gradient_widths;
    collective_detail::collective_local_validation(mesh, "Pressure reconstruction graph", [&] {
        cache.require_mesh(mesh);
        if (cells->getLocalNumElements() > static_cast<size_t>(std::numeric_limits<local_type>::max()) / 4)
            throw std::overflow_error("Pressure reconstruction map exceeds local ordinal range.");
        if (channel_map.is_null())
        {
            channels.reserve(4 * cells->getLocalNumElements());
            for (const auto gid : cells->getLocalElementList())
            {
                if (gid < 0 || gid > (std::numeric_limits<global_type>::max() - 3) / 4)
                    throw std::overflow_error("Pressure reconstruction map exceeds global ordinal range.");
                for (global_type component = 0; component != 4; ++component)
                    channels.push_back(4 * gid + component);
            }
        }
        gradient_widths.resize(4 * cells->getLocalNumElements());
        FVM::detail::visit_cached_gradient_rows<true>(cache, [&](size_t row, const auto& geometry) {
            gradient_widths[4 * row] = 1;
            for (size_t component = 1; component != 4; ++component)
                gradient_widths[4 * row + component] = 1 + geometry.interior_samples.size();
        });
    });
    if (channel_map.is_null())
        channel_map = Teuchos::rcp(new map_type(Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid(),
            channels(), global_type{}, cells->getComm()));
    if (face_import.is_null())
        face_import = Teuchos::rcp(new typename Pack::import_type(mesh.owned_face_map(), mesh.overlap_face_map()));
    Teuchos::RCP<matrix_type> reconstruction;
    collective_detail::collective_local_validation(mesh, "Pressure gradient coefficients", [&] {
        reconstruction = Teuchos::rcp(new matrix_type(channel_map, gradient_widths()));
        Teuchos::Array<global_type> columns;
        std::array<Teuchos::Array<scalar_type>, 3> values;
        FVM::detail::visit_cached_gradient_rows<true>(cache, [&](size_t row, const auto& geometry) {
            const auto gid = cells->getGlobalElement(static_cast<local_type>(row));
            const scalar_type one{1};
            reconstruction->insertGlobalValues(4 * gid, Teuchos::arrayView(&gid, 1), Teuchos::arrayView(&one, 1));
            columns.clear();
            for (auto& component : values) component.clear();
            vec3<scalar_type> diagonal{};
            for (const auto& sample : geometry.interior_samples)
            {
                columns.push_back(overlap->getGlobalElement(sample.other_lid));
                for (size_t component = 0; component != 3; ++component)
                    values[component].push_back(sample.weight.component(component));
                diagonal = diagonal - sample.weight;
            }
            for (const auto& sample : geometry.boundary_samples)
            {
                const auto condition = gradient_conditions.find(mesh.boundary_batch_name(sample.location.batch_id));
                if (condition == gradient_conditions.end()) continue;
                if (condition->second.type == BoundaryConditionType::Dirichlet)
                    diagonal = diagonal - sample.weight;
                else if (condition->second.type != BoundaryConditionType::Neumann)
                    throw std::invalid_argument("Unsupported assembled pressure-gradient boundary condition.");
            }
            columns.push_back(gid);
            for (size_t component = 0; component != 3; ++component)
            {
                values[component].push_back(diagonal.component(component));
                reconstruction->insertGlobalValues(4 * gid + 1 + static_cast<global_type>(component),
                    columns(), values[component]());
            }
        });
    });
    reconstruction->fillComplete(cells, channel_map);

    const auto balance = assembled_pressure_face_matrix<Pack>(mesh, topology, metrics, flux_conditions,
        gauge, compact, channel_map, *face_import);
    Teuchos::RCP<Teuchos::ParameterList> parameters;
    collective_detail::collective_local_validation(mesh, "Pressure matrix product preparation", [&] {
        if (result.is_null()) result = Teuchos::rcp(new matrix_type(cells, 0));
        parameters = Teuchos::parameterList();
        parameters->set("MM Skip Explicit Zeros", false);
    });
    Tpetra::MatrixMatrix::Multiply(*balance, false, *reconstruction, false, *result, true,
        "assembled corrected pressure", parameters);
}
}
