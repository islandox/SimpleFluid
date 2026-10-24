/** Orthogonal MPI field imports and independent conservative transport-row oracles. */
#include "FVM/Operators.hh"
#include "FVM/details/VectorLaplacian.hh"
#include "geometry/MeshHandle.hh"
#include "solvers/BelosLinearSolver.hh"
#include "utils/testing_environment.hh"
#include <Teuchos_CommHelpers.hpp>
#include <cmath>
#include <gtest/gtest.h>
#include <map>
#include <memory>
#include <numbers>
#include <type_traits>
#include <vector>

namespace
{
using namespace SimpleFluid;
using Pack = DefaultTpetraTypes;
using Handle = MeshHandle<Pack>;
using LO = Pack::local_ordinal_type;
using GO = Pack::global_ordinal_type;
using Vec = Handle::Vec3;
using Scalar = ScalarCellFieldStored<Pack>;
using Vector = VectorCellFieldStored<Pack>;
using Flux = ScalarFaceFieldStored<Pack>;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);
double global_sum(double value)
{
    double result = 0;
    Teuchos::reduceAll(*Tpetra::getDefaultComm(), Teuchos::REDUCE_SUM, 1, &value, &result);
    return result;
}
void expect_vector(const Vec& a, const Vec& b, double tolerance = 2e-12)
{
    EXPECT_NEAR(a.x, b.x, tolerance);
    EXPECT_NEAR(a.y, b.y, tolerance);
    EXPECT_NEAR(a.z, b.z, tolerance);
}

/** Raw edge arithmetic supplies the extensive metric independently of handle caches. */
template<class Native> double cross_section(const Native& native, const auto& id)
{
    const auto& edges = native.cell_edges();
    const double first = [&]
    {
        if constexpr (std::is_same_v<Native, Handle::Cylindrical>)
            return .5 * (edges[0][id.i + 1] * edges[0][id.i + 1] - edges[0][id.i] * edges[0][id.i]);
        else
            return edges[0][id.i + 1] - edges[0][id.i];
    }();
    return first * (edges[1][id.j + 1] - edges[1][id.j]);
}
template<class Native> double volume(const Native& native, const auto& cell)
{
    return cross_section(native, cell) * (native.cell_edges()[2][cell.k + 1] - native.cell_edges()[2][cell.k]);
}
template<class Native> double axial_flux(const Native& native, const auto& face)
{
    if (face.orientation != Meshes::OrthogonalIndexer::K_FACE)
        return 0.;
    return .3 * cross_section(native, face) * (face.k == 0 ? -1 : 1);
}
template<class Native> void verify(const std::shared_ptr<Native>& native)
{
    Handle::DistributionOptions options;
    options.allow_empty_partitions = true;
    auto mesh = std::make_shared<Handle>(native, options);
    EXPECT_FALSE(mesh->has_materialized_indexer());
    EXPECT_FALSE(mesh->has_materialized_connectivity());
    const auto compact = mesh->decomposed_orthogonal_topology();
    if (Tpetra::getDefaultComm()->getSize() > 1)
    {
        ASSERT_TRUE(compact);
        EXPECT_EQ(&compact->source_topology(), &native->topology());
        EXPECT_EQ(compact->lookup_storage_estimate(), 0U);
    }
    Scalar scalar(mesh, -91., "source_cell_ids");
    Vector vector(mesh, Vec{-91., -91., -91.}, "source_vectors");
    Flux flux(mesh, -91., "source_face_ids");
    std::vector<GO> local(native->num_cells(), -1), algebraic_to_source(native->num_cells());
    for (std::size_t c = 0; c < mesh->num_owned_cells(); ++c)
    {
        const auto lid = LO(c);
        const auto source = mesh->cell_geometry_global_id(lid);
        const double value = double(source) + .5;
        scalar.set_owned_value(lid, value);
        vector.set_owned_value(lid, Vec{value, 2 * value, -value});
        local.at(std::size_t(mesh->cell_global_id(lid))) = source;
    }
    Teuchos::reduceAll(
        *Tpetra::getDefaultComm(), Teuchos::REDUCE_MAX, int(local.size()), local.data(), algebraic_to_source.data());
    for (std::size_t f = 0; f < mesh->num_owned_faces(); ++f)
        flux.set_owned_value(LO(f), double(mesh->face_geometry_global_id(LO(f))) + .25);
    scalar.sync_ghosts();
    vector.sync_ghosts();
    flux.sync_ghosts();
    for (std::size_t c = 0; c < mesh->num_local_cells(); ++c)
    {
        const auto lid = LO(c);
        const double value = double(mesh->cell_geometry_global_id(lid)) + .5;
        EXPECT_DOUBLE_EQ(scalar.local_value(lid), value);
        expect_vector(vector.local_value(lid), Vec{value, 2 * value, -value});
    }
    for (std::size_t f = 0; f < mesh->num_faces(); ++f)
        EXPECT_DOUBLE_EQ(flux.local_value(LO(f)), double(mesh->face_geometry_global_id(LO(f))) + .25);

    if constexpr (std::is_same_v<Native, Handle::Cylindrical>)
    {
        double renumbered_incidences = 0.;
        for (std::size_t c = 0; c < mesh->num_owned_cells(); ++c)
        {
            const auto cell_lid = LO(c);
            const auto source_cell = native->indexer().cell_id(std::size_t(mesh->cell_geometry_global_id(cell_lid)));
            Vec integrated_sum{};
            for (const auto face_lid : mesh->faces(cell_lid))
            {
                const auto source_face =
                    native->indexer().face_id(std::size_t(mesh->face_geometry_global_id(face_lid)));
                auto expected = native->face_area_vector_outward(source_face, source_cell);
                if (source_face.orientation == Native::R_FACE)
                {
                    const auto& theta = native->cell_edges()[Native::THETA];
                    const double half_angle = .5 * (theta[source_face.j + 1] - theta[source_face.j]);
                    expected = expected * (std::sin(half_angle) / half_angle);
                }
                const auto actual = FVM::detail::laplacian_face_area(*mesh, face_lid, cell_lid);
                expect_vector(actual, expected);
                integrated_sum = integrated_sum + actual;
                renumbered_incidences += mesh->face_global_id(face_lid) != mesh->face_geometry_global_id(face_lid);
            }
            expect_vector(integrated_sum, Vec{});
        }
        if (native->num_cells() > 1 && Tpetra::getDefaultComm()->getSize() > 1)
            EXPECT_GT(global_sum(renumbered_incidences), 0.);
    }

    // Source IDs intentionally vary across I/J/K cuts. A wrong algebraic-to-source
    // translation cannot pass this full-row upwind oracle merely by preserving constants.
    for (std::size_t f = 0; f < mesh->num_owned_faces(); ++f)
    {
        const auto lid = LO(f);
        const auto face = native->indexer().face_id(std::size_t(mesh->face_geometry_global_id(lid)));
        const bool exterior = native->neighbor_cell(face) == Native::invalid_cell_id();
        flux.set_owned_value(lid, exterior ? 0. : axial_flux(*native, face));
    }
    flux.sync_ghosts();
    constexpr double dt = .03;
    const auto system = FVM::transport_system<Pack>(
        scalar, flux, dt, 0., [](int, std::size_t) { return BoundaryCondition{BoundaryConditionType::Neumann, 0.}; },
        [](int, std::size_t) { return 0.; }, [](LO) { return 0.; });
    for (std::size_t c = 0; c < mesh->num_owned_cells(); ++c)
    {
        const auto lid = LO(c);
        const auto source = mesh->cell_geometry_global_id(lid);
        const auto cell = native->indexer().cell_id(std::size_t(source));
        const double cell_volume = volume(*native, cell);
        std::map<GO, double> expected{{source, cell_volume / dt}}, actual;
        for (const auto face : native->faces(cell))
        {
            const auto neighbor = native->neighbor_cell(face);
            if (neighbor == Native::invalid_cell_id())
                continue;
            const auto owner = native->owner_cell(face);
            const double outward = (owner == cell ? 1 : -1) * axial_flux(*native, face);
            const auto upwind = outward >= 0 ? cell : native->opposite_cell(face, cell);
            expected[GO(native->indexer().cell_ordinal(upwind))] += outward;
        }
        Pack::matrix_type::local_inds_host_view_type columns;
        Pack::matrix_type::values_host_view_type values;
        system.matrix->getLocalRowView(lid, columns, values);
        for (std::size_t i = 0; i < columns.size(); ++i)
        {
            const auto algebraic = system.matrix->getColMap()->getGlobalElement(columns[i]);
            actual[algebraic_to_source.at(std::size_t(algebraic))] += values[i];
        }
        for (const auto [column, value] : actual)
            EXPECT_NEAR(value, expected.contains(column) ? expected.at(column) : 0., 2e-11);
        for (const auto [column, value] : expected)
        {
            EXPECT_TRUE(actual.contains(column));
            if (actual.contains(column))
                EXPECT_NEAR(actual.at(column), value, 2e-11);
        }
        EXPECT_NEAR(system.rhs->getData()[c], cell_volume / dt * (double(source) + .5), 2e-10);
    }
    Pack::vector_type solution(mesh->owned_cell_map(), true);
    BelosLinearSolver<Pack> solver;
    LinearSolverOptions solver_options;
    solver_options.tolerance = 1e-12;
    solver_options.max_iterations = 300;
    ASSERT_TRUE(solver.solve(system.matrix, *system.rhs, solution, solver_options));
    const auto values = solution.getData();
    double before = 0., after = 0.;
    for (std::size_t c = 0; c < mesh->num_owned_cells(); ++c)
    {
        const auto lid = LO(c);
        const double cell_volume =
            volume(*native, native->indexer().cell_id(std::size_t(mesh->cell_geometry_global_id(lid))));
        before += cell_volume * scalar.value(lid);
        after += cell_volume * values[c];
    }
    EXPECT_NEAR(global_sum(after), global_sum(before), 2e-9);

    // Axial affine reconstruction is exact in Cartesian and cylindrical geometry;
    // it does not assume planar Cartesian affine exactness on curved radial faces.
    for (std::size_t c = 0; c < mesh->num_owned_cells(); ++c)
        scalar.set_owned_value(LO(c), 1.2 + .4 * mesh->cell_centroid(LO(c)).z);
    scalar.sync_ghosts();
    auto boundary_value = [&](int batch, std::size_t index)
    { return 1.2 + .4 * mesh->face_centroid(mesh->boundary_face_batch(batch).face_lids.at(index)).z; };
    auto boundary = [&](int batch, std::size_t index)
    { return BoundaryCondition{BoundaryConditionType::Dirichlet, boundary_value(batch, index)}; };
    Vector gradient(mesh, "axial_gradient");
    FVM::CellGradientCache<Pack, Handle> cache(mesh);
    FVM::cell_gradient(scalar, boundary, boundary_value, gradient, cache);
    for (std::size_t c = 0; c < mesh->num_owned_cells(); ++c)
        expect_vector(gradient.value(LO(c)), Vec{0., 0., .4}, 2e-10);
    EXPECT_EQ(mesh->decomposed_orthogonal_topology().get(), compact.get());
    EXPECT_FALSE(mesh->has_materialized_indexer());
    EXPECT_FALSE(mesh->legacy_mesh());
    EXPECT_FALSE(mesh->has_materialized_connectivity());
}
} // namespace
TEST(OrthogonalDistributedOperatorsTest, CartesianImportsAxialGradientAndConservativeUpwindRows)
{
    const Vec3D<ArrReal> edges{{{0., .1, .8, 1.4, 2.}, {0., .3, .9, 1.6, 2.}, {0., .2, .7, 1.5, 2.}}};
    for (const bool periodic : {false, true})
        verify(std::make_shared<Handle::Cartesian>(edges, Vec3D<bool>{periodic, periodic, false}));
}
TEST(OrthogonalDistributedOperatorsTest, CylindricalImportsAxialGradientAndConservativeUpwindRows)
{
    for (const double angle : {1.3, 2 * std::numbers::pi})
    {
        const Vec3D<ArrReal> edges{
            {{1., 1.1, 1.6, 2.1, 3.}, {0., .1 * angle, .4 * angle, .7 * angle, angle}, {0., .2, .7, 1.5, 2.}}};
        verify(std::make_shared<Handle::Cylindrical>(edges));
    }
}
TEST(OrthogonalDistributedOperatorsTest, EmptyOwnersParticipateInImportsAndTransport)
{
    const Vec3D<ArrReal> edges{{{1., 2.}, {0., .5}, {0., 1.}}};
    verify(std::make_shared<Handle::Cartesian>(edges));
    verify(std::make_shared<Handle::Cylindrical>(edges));
}