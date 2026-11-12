/** @file testTemperaturePreparedResponse.cc
 * @brief Native temperature response parity and direct coefficient-view contracts.
 */
#include "equations/TemperatureDiffusionEquation.hh"
#include "geometry/mesh/PartitionedMeshBase.hh"
#include "geometry/unitTests/test_mesh_helpers.hh"
#include "parallel/MeshPartitioner.hh"
#include "utils/testing_environment.hh"

#include <Teuchos_CommHelpers.hpp>
#include <gtest/gtest.h>

#include <algorithm>
#include <vector>

namespace
{
using namespace SimpleFluid;
using namespace SimpleFluid::FVM;
using Pack = DefaultTpetraTypes;
using Handle = MeshHandle<Pack>;
using Cells = ScalarCellFieldStored<Pack>;
using Faces = ScalarFaceFieldStored<Pack>;
using Cache = MeshBoundaryCache<Pack, Handle>;
using Equation = TemperatureDiffusionEquation<Pack, Handle>;
using LO = Pack::local_ordinal_type;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);

SP<const Handle> make_mesh(bool skewed)
{
    using Native = Meshes::UnstructuredMesh;
    const auto reference = test::make_unstructured_hex_line(8, 0.125);
    auto nodes = reference->nodes();
    if (skewed)
        for (auto& node : nodes)
            node.x += 0.3 * node.y;
    Arr<Native::CellDefinition> cells;
    for (size_t i = 0; i < reference->num_cells(); ++i)
    {
        const auto cell = reference->cell_id(i);
        cells.push_back({reference->cell_type(cell), reference->cell_nodes(cell)});
    }
    Arr<Native::BoundaryFaceDefinition> boundaries;
    for (const auto& [batch, faces] : reference->boundary_batches())
        for (const auto face : faces.face_lids)
            boundaries.push_back({reference->face_nodes(face), batch, reference->boundary_names().at(batch)});
    auto native = std::make_shared<Native>(nodes, cells, boundaries);
    const auto comm = Tpetra::getDefaultComm();
    auto partition = MeshPartitioner<Pack>::partition(*native, comm);
    using Partitioned = Meshes::PartitionedMesh<Native, Pack>;
    return std::make_shared<Handle>(std::make_shared<Partitioned>(native, std::move(partition.indexer), comm));
}

TEST(TemperaturePreparedResponseTest, NativeResponseMatchesPowerAndHomogeneousSkewAssembly)
{
    for (const bool skewed : {false, true})
    {
        SCOPED_TRACE(skewed);
        const auto mesh = make_mesh(skewed);
        BoussinesqModelOptions options;
        options.density = 2.0;
        options.specific_heat_capacity = 3.0;
        options.thermal_conductivity = 4.0;
        MaterialPropertyFields<Pack, Handle> material(mesh, options, {});
        for (size_t row = 0; row < mesh->num_owned_cells(); ++row)
        {
            const auto cell = static_cast<LO>(row);
            material.thermal_conductivity.set_owned_value(cell, 2.0 + mesh->cell_centroid(cell).x);
        }
        material.thermal_conductivity.sync_ghosts();
        Cells baseline(mesh, 1e12, "large_baseline"), zero(mesh, 0.0, "zero");
        Faces flux(mesh, 0.0, "carrier_flux");
        Cache support{{}, mesh}, boundary_k{{}, mesh};
        for (const auto& [batch, faces] : mesh->boundary_batches())
        {
            if (mesh->boundary_batch_name(batch) == "ymax")
                support.value[batch] = ArrReal(faces.face_lids.size(), 0.0);
            for (const auto face : faces.face_lids)
                boundary_k.value[batch].push_back(3.0 + 0.2 * mesh->face_centroid(face).x);
        }
        const auto frozen_boundary_k = boundary_k;
        BoundaryConditionSet boundaries;
        if (skewed)
        {
            boundaries.temperature["xmin"] = {BoundaryConditionType::Dirichlet, 310.0};
            boundaries.temperature["xmax"] = {BoundaryConditionType::Robin, 300.0, 0.4};
        }
        TransportGeometryCache<Handle> geometry(*mesh);
        const int local_non_orthogonal = geometry.has_non_orthogonal_faces();
        int global_non_orthogonal = 0;
        Teuchos::reduceAll(
            *mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_MAX, 1, &local_non_orthogonal, &global_non_orthogonal);
        if (skewed)
            EXPECT_NE(global_non_orthogonal, 0);
        const auto source = [](LO) { return 0.0; };
        for (auto treatment :
            {NonOrthogonalTreatment::Implicit, NonOrthogonalTreatment::Explicit, NonOrthogonalTreatment::Hybrid})
            for (auto interpolation : {FaceCoefficientInterpolation::Harmonic, FaceCoefficientInterpolation::Linear})
            {
                Equation equation(mesh, boundaries);
                equation.set_boundary_heat_flux(support);
                equation.prepare_physical_advance(
                    baseline, flux, 0.1, material, treatment, nullptr, &boundary_k, interpolation);
                const auto baseline_rhs = equation.prepared_rhs();
                const auto baseline_norm = baseline_rhs->norm2();
                const auto matrix = equation.prepared_operator();
                const auto matrix_norm = matrix->getFrobeniusNorm();
                Teuchos::RCP<const Pack::vector_type> retained;
                double retained_norm = 0.0;
                for (const double sign : {1.0, -2.0, 0.25})
                {
                    auto increment = support;
                    std::vector<double> expected(mesh->num_owned_cells(), 0.0);
                    for (auto& [batch, values] : increment.value)
                    {
                        const auto& faces = mesh->boundary_face_batch(batch).face_lids;
                        for (size_t slot = 0; slot < values.size(); ++slot)
                        {
                            values[slot] = sign * (0.3 + mesh->face_centroid(faces[slot]).x);
                            const auto owner = mesh->owner_cell(faces[slot]);
                            if (mesh->is_owned_cell(owner))
                                expected[owner] -= values[slot] * mesh->face_area(faces[slot]);
                        }
                    }
                    const auto response = equation.prepared_boundary_response(increment);
                    EXPECT_GT(response->normInf(), 1e-6);
                    if (!skewed)
                        for (size_t row = 0; row < expected.size(); ++row)
                            EXPECT_NEAR(response->getData()[row], expected[row], 1e-12);
                    const auto condition = [&](int batch, size_t slot)
                    {
                        const auto prescribed = increment.value.find(batch);
                        if (prescribed != increment.value.end())
                            return BoundaryCondition{BoundaryConditionType::Neumann,
                                -prescribed->second[slot] / frozen_boundary_k.value.at(batch)[slot]};
                        const auto found = boundaries.temperature.find(mesh->boundary_batch_name(batch));
                        auto result = found == boundaries.temperature.end() ? BoundaryCondition{} : found->second;
                        result.value = 0.0;
                        return result;
                    };
                    const auto reference = physical_temperature_transport_system<Pack>(
                        zero, flux, 0.1, material.density, material.specific_heat_capacity,
                        material.thermal_conductivity, condition, [](int, size_t) { return 0.0; }, source, treatment,
                        treatment == NonOrthogonalTreatment::Implicit ? nullptr : &zero, Teuchos::null,
                        &frozen_boundary_k, &geometry, interpolation);
                    Pack::vector_type error(*response, Teuchos::Copy);
                    error.update(-1.0, *reference.rhs, 1.0);
                    EXPECT_LT(error.normInf(), 1e-11);
                    EXPECT_EQ(matrix.get(), equation.prepared_operator().get());
                    EXPECT_EQ(matrix_norm, matrix->getFrobeniusNorm());
                    EXPECT_EQ(baseline_norm, baseline_rhs->norm2());
                    if (!retained.is_null())
                        EXPECT_EQ(retained_norm, retained->norm2());
                    retained = response;
                    retained_norm = response->norm2();
                }
            }
    }
}
} // namespace
