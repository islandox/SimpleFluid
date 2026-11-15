/** @file FVM/details/BasicVectorTransportSystem.tcc
 * @brief Native single-matrix vector transport with condition/value boundary callbacks.
 */
#pragma once

#include "FVM/details/FieldStoredTransportSystem.hh"
#include "FVM/TransportSystem.hh"
#include "FVM/details/TransportAssemblyGeometry.hh"
#include "FVM/details/TransportRhs.tcc"
#include "FVM/details/TransportValidation.hh"

#include <Teuchos_CommHelpers.hpp>
#include <array>
#include <exception>
#include <cmath>
#include <utility>
#include <vector>

namespace SimpleFluid::FVM::detail
{
template<TpetraTypePack Pack, SimpleFluid::MeshType MeshType>
SIMPLEFLUID_LOCAL void validate_basic_vector_transport_matrix_graph(
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

template<TpetraTypePack Pack, SimpleFluid::MeshType MeshType>
SIMPLEFLUID_LOCAL StoredPreparedTransportMatrix<Pack> prepare_basic_vector_transport_matrix(
    const MeshType& mesh, Teuchos::RCP<typename Pack::matrix_type> cached_matrix, size_t entries_per_row,
    const std::vector<TransportMatrixRow<Pack>>* required_rows)
{
    using matrix_type = typename Pack::matrix_type;

    validate_transport_matrix_maps<Pack>(mesh, cached_matrix, mesh.num_owned_cells());
    if (required_rows != nullptr)
        validate_basic_vector_transport_matrix_graph<Pack, MeshType>(mesh, cached_matrix, *required_rows);

    Teuchos::RCP<matrix_type> matrix = cached_matrix;
    std::exception_ptr allocation_error;
    if (matrix.is_null())
    {
        try
        {
            matrix = Teuchos::rcp(new matrix_type(mesh.owned_cell_map(), mesh.overlap_cell_map(), entries_per_row));
        }
        catch (...)
        {
            allocation_error = std::current_exception();
        }
    }
    const auto allocation_state = reduce_transport_validation_state(mesh, std::array<int, 1>{allocation_error ? 1 : 0});
    if (allocation_state[0])
    {
        if (allocation_error) std::rethrow_exception(allocation_error);
        throw std::runtime_error("Native basic-vector matrix allocation failed on another rank.");
    }
    if (cached_matrix.is_null())
        return {matrix, false};

    cached_matrix->resumeFill();
    cached_matrix->setAllToScalar(typename Pack::scalar_type{});
    return {cached_matrix, true};
}

template<TpetraTypePack Pack, SimpleFluid::MeshType MeshType>
SIMPLEFLUID_LOCAL void add_basic_vector_transport_values(const StoredPreparedTransportMatrix<Pack>& prepared,
    typename Pack::local_ordinal_type row,
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

template<TpetraTypePack Pack, SimpleFluid::MeshType MeshType>
SIMPLEFLUID_LOCAL void finish_basic_vector_transport_matrix(const StoredPreparedTransportMatrix<Pack>& prepared)
{
    if (prepared.reused)
    {
        // Keep the existing domain/import and make the range identical to
        // this graph's row map on every rank. Tpetra otherwise branches on
        // local map pointer identity while rebuilding its exporter.
        prepared.matrix->fillComplete(
            prepared.matrix->getDomainMap(), prepared.matrix->getRowMap());
    }
    else
    {
        prepared.matrix->fillComplete();
    }
}

template<TpetraTypePack Pack, SimpleFluid::MeshType MeshType>
SIMPLEFLUID_LOCAL void add_basic_vector_transport_values(const StoredPreparedTransportMatrix<Pack>& prepared,
    typename Pack::local_ordinal_type row,
    const FlatMatrixRow<typename Pack::local_ordinal_type, typename Pack::scalar_type>& row_values)
{
    const auto row_size =
        SimpleFluid::detail::checked_size_to_ordinal<Teuchos::Ordinal>(row_values.size(), "transport row entry count");
    add_basic_vector_transport_values<Pack, MeshType>(
        prepared, row, Teuchos::arrayView(row_values.column_data(), row_size),
        Teuchos::arrayView(row_values.value_data(), row_size));
}

template<TpetraTypePack Pack, SimpleFluid::MeshType MeshType>
using BasicTransportRowFaces = TransportAssemblyRowFaces<TransportAssemblyGeometry<MeshType>,
    AssemblyFaceRow<typename Pack::local_ordinal_type, typename Pack::scalar_type>>;

// Concrete native face/flux views avoid function-pointer dispatch inside face loops.
template<TpetraTypePack Pack, SimpleFluid::MeshType MeshType>
SIMPLEFLUID_LOCAL void accumulate_basic_vector_transport_face_row(
    const decltype(std::declval<const ScalarFaceFieldStored<Pack, MeshType>&>().local_read_view())& face_flux_data,
    typename Pack::scalar_type diffusivity, typename Pack::local_ordinal_type cell_lid,
    const BasicTransportRowFaces<Pack, MeshType>& visit_faces,
    FlatMatrixRow<typename Pack::local_ordinal_type, typename Pack::scalar_type>& row_values)
{
    using scalar_type = typename Pack::scalar_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;

    visit_faces(
        [&](const auto& face, const auto& metrics)
        {
            using face_type = std::remove_cvref_t<decltype(face)>;
            const auto face_lid = face.face_lid;
            constexpr bool is_interior = face_type::interior;
            local_ordinal_type other{};
            if constexpr (is_interior)
            {
                other = face.other;
            }

            // Physical boundaries are handled once by the separate ordered sweep.
            if constexpr (face_type::boundary)
                return;

            // === advection (upwind) ===
            const auto owner_oriented_flux = face_flux_data(face_lid, 0);
            const auto out_flux = face_type::owned_orientation ? owner_oriented_flux : -owner_oriented_flux;

            if (out_flux >= scalar_type{0})
            {
                add_matrix_entry(row_values, cell_lid, out_flux);
            }
            else if constexpr (is_interior)
            {
                add_matrix_entry(row_values, other, out_flux);
            }

            // === diffusion ===
            if constexpr (!is_interior)
            {
                return;
            }

            row_values.ensure(other);
            if (diffusivity <= scalar_type{0})
            {
                return;
            }

            const auto coeff = interior_diffusion_coefficient(metrics, face_lid, cell_lid, other, diffusivity);
            add_matrix_entry(row_values, cell_lid, coeff);
            add_matrix_entry(row_values, other, -coeff);
        });
}

template<TpetraTypePack Pack, SimpleFluid::MeshType MeshType, class Terms, class Rhs, class Context>
SIMPLEFLUID_LOCAL Teuchos::RCP<typename Pack::matrix_type> assemble_basic_vector_transport_system(
    Terms terms, Rhs& rhs, Context context)
{
    const auto& mesh = terms.history.old_values.mesh();
    const auto& face_fluxes = terms.face_fluxes;
    const auto time_step = terms.history.time_step;
    const auto diffusivity = terms.coefficients.diffusivity;
    auto cached_matrix = std::move(context.cached_matrix);
    using scalar_type = typename Pack::scalar_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;

    validate_transport_scalar_inputs(mesh, &face_fluxes.mesh() == &mesh, time_step, diffusivity);

    // Row map = owned cells; domain map = owned + ghost cells
    // so that neighbour columns on other ranks are valid.
    TransportAssemblyGeometry<MeshType> local_assembly_geometry;
    const auto& assembly_geometry = select_transport_assembly_geometry(mesh, local_assembly_geometry);
    // Callbacks, local geometry, arithmetic and allocation must succeed on every
    // rank before a cached matrix is resumed or cleared. These rows and ordered
    // diagonal increments preserve the original matrix summation sequence.
    validate_transport_matrix_maps<Pack>(mesh, cached_matrix, mesh.num_owned_cells());
    std::vector<TransportMatrixRow<Pack>> rows;
    std::vector<scalar_type> diagonal_values;
    std::vector<std::pair<local_ordinal_type, scalar_type>> boundary_matrix_updates;
    std::exception_ptr staging_error;
    try
    {
        rows.resize(mesh.num_owned_cells());
        diagonal_values.resize(mesh.num_owned_cells());
        rhs.initialize(mesh);
        const auto face_flux_data = face_fluxes.local_read_view();
        FlatMatrixRow<local_ordinal_type, scalar_type> row_values(mesh.num_local_cells(), 32);

        visit_transport_assembly_rows(mesh, assembly_geometry,
            [&](local_ordinal_type cell_lid, scalar_type volume, auto&& visit_faces)
        {
            const auto transient = volume / time_step;
            rhs.begin_cell(cell_lid, transient, volume);
            row_values.clear();
            add_matrix_entry(row_values, cell_lid, transient);

            accumulate_basic_vector_transport_face_row<Pack, MeshType>(
                face_flux_data, diffusivity, cell_lid, visit_faces, row_values);

            auto& row = rows[static_cast<size_t>(cell_lid)];
            row = capture_transport_row<Pack>(row_values);
            (void) SimpleFluid::detail::checked_size_to_ordinal<Teuchos::Ordinal>(
                row.values.size(), "basic-vector transport row entry count");
            for (size_t entry = 0; entry < row.values.size(); ++entry)
            {
                if (!std::isfinite(row.values[entry]))
                    throw std::invalid_argument("Native basic-vector matrix contributions must be finite.");
                if (row.columns[entry] == cell_lid)
                    diagonal_values[static_cast<size_t>(cell_lid)] = row.values[entry];
            }
            rhs.finish_cell(cell_lid, transient, volume);
        });

        const auto stage_diagonal = [&](local_ordinal_type owner, scalar_type value)
        {
            auto& total = diagonal_values.at(static_cast<size_t>(owner));
            total += value;
            if (!std::isfinite(value) || !std::isfinite(total))
                throw std::invalid_argument("Native basic-vector boundary matrix contributions must be finite.");
            boundary_matrix_updates.emplace_back(owner, value);
        };

        // Preserve the separate boundary sweep and VALUE-before-CONDITION order.
        visit_transport_boundary_batches(mesh, assembly_geometry,
            [&](int batch_id, const auto boundary_faces, const auto& metrics)
        {
            for (const auto& entry : boundary_faces)
            {
                const auto face_lid = entry.face_lid;
                const auto owner = entry.owner;
                const auto in_batch_id = entry.in_batch_id;
                const auto out_flux = face_flux_data(face_lid, 0);
                const auto condition_type = rhs.begin_boundary(batch_id, in_batch_id);
                if (!std::isfinite(out_flux))
                    throw std::invalid_argument("Native basic-vector boundary fluxes must be finite.");

                if (out_flux >= scalar_type{0})
                    stage_diagonal(owner, out_flux);
                else
                    rhs.add_boundary_value(owner, -out_flux);

                if (diffusivity <= scalar_type{0})
                    continue;

                const auto coeff = boundary_diffusion_coefficient(metrics, face_lid, owner, diffusivity);
                if (!std::isfinite(coeff))
                    throw std::invalid_argument("Native basic-vector boundary diffusion must be finite.");
                if (rhs.is_dirichlet(condition_type))
                {
                    if (coeff > scalar_type{0})
                    {
                        stage_diagonal(owner, coeff);
                        rhs.add_boundary_value(owner, coeff);
                    }
                }
                else if (condition_type == BoundaryConditionType::Neumann)
                    rhs.add_neumann(owner, diffusivity, metrics, face_lid);
                else if (condition_type == BoundaryConditionType::Robin)
                    rhs.throw_robin();
            }
        });

        for (size_t component = 0; component < Rhs::num_components; ++component)
        {
            const auto values = rhs.values->getData(component);
            for (size_t row = 0; row < rows.size(); ++row)
                if (!std::isfinite(values[row]))
                    throw std::invalid_argument("Native basic-vector RHS contributions must be finite.");
        }
    }
    catch (...)
    {
        staging_error = std::current_exception();
    }
    const auto staging_state = reduce_transport_validation_state(mesh, std::array<int, 1>{staging_error ? 1 : 0});
    if (staging_state[0])
    {
        if (staging_error) std::rethrow_exception(staging_error);
        throw std::runtime_error("Native basic-vector assembly staging failed on another rank.");
    }

    const auto prepared = prepare_basic_vector_transport_matrix<Pack, MeshType>(
        mesh, std::move(cached_matrix), 12, &rows);
    const auto& matrix = prepared.matrix;
    for (size_t row = 0; row < rows.size(); ++row)
    {
        const auto count = SimpleFluid::detail::checked_size_to_ordinal<Teuchos::Ordinal>(
            rows[row].values.size(), "basic-vector transport row entry count");
        add_basic_vector_transport_values<Pack, MeshType>(prepared, static_cast<local_ordinal_type>(row),
            Teuchos::arrayView(rows[row].columns.data(), count), Teuchos::arrayView(rows[row].values.data(), count));
    }
    for (const auto& [owner, value] : boundary_matrix_updates)
    {
        local_ordinal_type column = owner;
        matrix->sumIntoLocalValues(owner, Teuchos::arrayView(&column, 1), Teuchos::arrayView(&value, 1));
    }

    finish_basic_vector_transport_matrix<Pack, MeshType>(prepared);
    return matrix;
}

template<TpetraTypePack Pack, SimpleFluid::MeshType MeshType>
VectorTransportSystem<Pack> assemble_stored_basic_vector(VectorTerms<Pack, MeshType> terms,
    BasicVectorTransportBoundary<Pack> boundary, BasicVectorAssemblyOptions,
    BasicVectorAssemblyContext<Pack> context)
{
    const auto& old_values = terms.history.old_values;
    auto boundary_condition = std::move(boundary.condition);
    auto boundary_value = std::move(boundary.value);
    auto right_hand_source = std::move(terms.source);

    BasicVectorTransportRhs<Pack, MeshType> rhs(old_values, boundary_condition, boundary_value, right_hand_source);
    const auto matrix = assemble_basic_vector_transport_system<Pack, MeshType>(terms, rhs, std::move(context));
    return {matrix, rhs.values};
}
} // namespace SimpleFluid::FVM::detail
