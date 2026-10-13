/** @file testTransportTraverse.cc
 * @brief Ordered transport traversal, live inputs, and retained geometry contracts.
 */
#include "FVM/Operators.hh"
#include "geometry/MeshReorderingFactory.hh"
#include "geometry/PlanarALEMeshMotion.hh"
#include "geometry/unitTests/region_mesh_helpers.hh"
#include "geometry/unitTests/test_mesh_helpers.hh"
#include "geometry/unitTests/test_skewed_prism_mesh_helpers.hh"
#include "utils/testing_environment.hh"
#include <gtest/gtest.h>
#include <Teuchos_CommHelpers.hpp>

#include <array>
#include <limits>
#include <optional>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace
{
using namespace SimpleFluid;
using namespace SimpleFluid::FVM;
using Pack = DefaultTpetraTypes;
using Handle = MeshHandle<Pack>;
using Local = Pack::local_ordinal_type;
using Vec = MeshUtils::Vec3;
using Matrix = Teuchos::RCP<Pack::matrix_type>;
testing::Environment* const environment =
    testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);

void expect_exact_vector(const char* metric, const Vec& actual, const Vec& expected)
{
    SCOPED_TRACE(metric);
    EXPECT_EQ(actual.x, expected.x);
    EXPECT_EQ(actual.y, expected.y);
    EXPECT_EQ(actual.z, expected.z);
}

// Query canonical point geometry independently of the batch descriptors. The
// incidence order, multiplicity, orientation, and classification are all part
// of the assembly contract, including repeated periodic neighbours.
template<class MeshType>
void expect_canonical_geometry(const MeshType& mesh, const TransportGeometryCache<MeshType>& cache)
{
    const auto& geometry = cache.assembly_geometry();
    const auto locations = FVM::detail::boundary_face_locations(mesh);
    ASSERT_EQ(geometry.volumes.size(), mesh.num_owned_cells());
    ASSERT_EQ(geometry.face_offsets.size(), mesh.num_owned_cells() + 1);
    size_t incidence = 0;
    for (size_t owned = 0; owned < mesh.num_owned_cells(); ++owned)
    {
        const auto local = static_cast<Local>(owned);
        const auto cell = FVM::detail::query_cell_id(mesh, local);
        EXPECT_EQ(geometry.volumes[owned], mesh.cell_volume(cell));
        EXPECT_EQ(geometry.face_offsets[owned], incidence);
        for (const auto face : mesh.faces(cell))
        {
            SCOPED_TRACE(::testing::Message() << "cell=" << owned << " incidence=" << incidence);
            ASSERT_LT(incidence, geometry.faces.size());
            const auto& actual = geometry.faces[incidence++];
            EXPECT_EQ(actual.face_lid, FVM::detail::packed_face_local_id(mesh, face));
            EXPECT_EQ(actual.interior, mesh.is_interior_face(face));
            EXPECT_EQ(actual.boundary, mesh.is_boundary_face(face));
            EXPECT_EQ(actual.owned_orientation, mesh.owner_cell(face) == cell);
            if (actual.interior)
                EXPECT_EQ(actual.other, FVM::detail::packed_cell_local_id(mesh,
                    mesh.opposite_or_periodic_neighbor_cell(face, cell)));
            if (actual.boundary)
            {
                const auto& location = locations.at(actual.face_lid);
                ASSERT_TRUE(location.active);
                EXPECT_EQ(geometry.boundary_indices.at({location.batch_id, location.in_batch_id}),
                    static_cast<size_t>(actual.face_lid));
            }
        }
    }
    EXPECT_EQ(geometry.face_offsets.back(), incidence);
    EXPECT_EQ(geometry.faces.size(), incidence);

    size_t row = 0, face_index = 0;
    FVM::detail::visit_transport_assembly_rows(mesh, geometry,
        [&](Local cell, auto volume, const auto& visit_faces)
    {
        EXPECT_EQ(cell, static_cast<Local>(row++));
        EXPECT_EQ(volume, geometry.volumes[cell]);
        visit_faces([&](const auto& face, const auto& metrics)
        {
            using Face = std::remove_cvref_t<decltype(face)>;
            // These must be type properties, usable without a branch in the
            // numerical loop, while callbacks still follow canonical order.
            constexpr bool interior = Face::interior;
            constexpr bool boundary = Face::boundary;
            constexpr bool owner = Face::owned_orientation;
            ASSERT_LT(face_index, geometry.faces.size());
            const auto& expected = geometry.faces[face_index++];
            EXPECT_EQ(face.face_lid, expected.face_lid);
            EXPECT_EQ(interior, expected.interior);
            EXPECT_EQ(boundary, expected.boundary);
            EXPECT_EQ(owner, expected.owned_orientation);
            if constexpr (interior) EXPECT_EQ(face.other, expected.other);

            SCOPED_TRACE(::testing::Message() << "metric cell=" << cell << " face=" << face.face_lid);
            const auto face_id = FVM::detail::query_face_id(mesh, face.face_lid);
            const auto cell_id = FVM::detail::query_cell_id(mesh, cell);
            const auto point_face_center_vector = [&](const auto selected_cell)
            {
                if constexpr (requires { mesh.face_center_vector(face_id, selected_cell); })
                    return mesh.face_center_vector(face_id, selected_cell);
                else return mesh.face_centroid(face_id) - mesh.cell_centroid(selected_cell);
            };
            EXPECT_EQ(metrics.face_area(face.face_lid), mesh.face_area(face_id));
            expect_exact_vector("face_centroid", metrics.face_centroid(face.face_lid), mesh.face_centroid(face_id));
            expect_exact_vector("face_normal", metrics.face_normal(face.face_lid), mesh.face_normal(face_id));
            expect_exact_vector("face_normal_outward", metrics.face_normal_outward(face.face_lid, cell),
                mesh.face_normal_outward(face_id, cell_id));
            expect_exact_vector("face_area_vector_outward", metrics.face_area_vector_outward(face.face_lid, cell),
                mesh.face_area_vector_outward(face_id, cell_id));
            expect_exact_vector("cell_centroid", metrics.cell_centroid(cell), mesh.cell_centroid(cell_id));
            expect_exact_vector("face_center_vector", metrics.face_center_vector(face.face_lid, cell),
                point_face_center_vector(cell_id));
            EXPECT_EQ(metrics.cell_to_face_distance(face.face_lid, cell), mesh.cell_to_face_distance(face_id, cell_id));
            if constexpr (interior)
            {
                const auto other_id = FVM::detail::query_cell_id(mesh, face.other);
                expect_exact_vector("neighbor_centroid", metrics.cell_centroid(face.other), mesh.cell_centroid(other_id));
                expect_exact_vector("neighbor_face_center_vector", metrics.face_center_vector(face.face_lid, face.other),
                    point_face_center_vector(other_id));
                expect_exact_vector("cell_center_vector", metrics.cell_center_vector(face.face_lid, cell),
                    mesh.cell_center_vector(face_id, cell_id));
                expect_exact_vector("neighbor_cell_center_vector", metrics.cell_center_vector(face.face_lid, face.other),
                    mesh.cell_center_vector(face_id, other_id));
                EXPECT_EQ(metrics.cell_to_face_distance(face.face_lid, face.other),
                    mesh.cell_to_face_distance(face_id, other_id));
                EXPECT_EQ(metrics.face_cell_center_distance(face.face_lid), mesh.face_cell_center_distance(face_id));
                EXPECT_EQ(FVM::detail::interior_diffusion_coefficient(metrics, face.face_lid, cell, face.other, 0.7),
                    FVM::detail::interior_diffusion_coefficient(mesh, face.face_lid, cell, face.other, 0.7));
                for (auto interpolation : {FaceCoefficientInterpolation::Harmonic, FaceCoefficientInterpolation::Linear})
                    EXPECT_EQ(FVM::detail::face_coefficient_value(metrics, face.face_lid, cell, face.other, 0.7, 1.3,
                                  interpolation),
                        FVM::detail::face_coefficient_value(mesh, face.face_lid, cell, face.other, 0.7, 1.3,
                            interpolation));
            }
            else if constexpr (boundary)
            {
                EXPECT_EQ(FVM::detail::boundary_diffusion_coefficient(metrics, face.face_lid, cell, 0.7),
                    FVM::detail::boundary_diffusion_coefficient(mesh, face.face_lid, cell, 0.7));
                EXPECT_EQ(FVM::detail::robin_boundary_weight(metrics, face.face_lid, cell, 0.4),
                    FVM::detail::robin_boundary_weight(mesh, face.face_lid, cell, 0.4));
            }
        });
    });
    EXPECT_EQ(row, mesh.num_owned_cells());
    EXPECT_EQ(face_index, geometry.faces.size());
}

template<class System> auto snapshot(const System& system)
{
    std::vector<std::vector<std::pair<Local, double>>> rows;
    std::vector<std::vector<double>> rhs;
    for (size_t row = 0; row < system.matrix->getLocalNumRows(); ++row)
    {
        Pack::matrix_type::local_inds_host_view_type columns;
        Pack::matrix_type::values_host_view_type values;
        system.matrix->getLocalRowView(static_cast<Local>(row), columns, values);
        auto& entries = rows.emplace_back();
        for (size_t entry = 0; entry < columns.extent(0); ++entry)
            entries.emplace_back(columns(entry), values(entry));
    }
    for (size_t component = 0; component < system.rhs->getNumVectors(); ++component)
    {
        const auto& multivector = static_cast<const Pack::multi_vector_type&>(*system.rhs);
        const auto data = multivector.getData(component);
        rhs.emplace_back(data.begin(), data.end());
    }
    return std::pair{rows, rhs};
}

template<class System>
void expect_same_system(const System& actual, const System& expected, bool reused_graph = false)
{
    ASSERT_TRUE(actual.matrix->getRowMap()->isSameAs(*expected.matrix->getRowMap()));
    ASSERT_TRUE(actual.matrix->getColMap()->isSameAs(*expected.matrix->getColMap()));
    const auto a = snapshot(actual), e = snapshot(expected);
    EXPECT_EQ(a.second, e.second);
    ASSERT_EQ(a.first.size(), e.first.size());
    if (!reused_graph)
    {
        // Compare ordered columns and exact values, rather than only a norm.
        EXPECT_EQ(a.first, e.first);
        return;
    }
    // A frozen graph may retain now-unused stencil columns after a BC change.
    // Their values must be exactly zero, and every other coefficient exact.
    const auto columns = actual.matrix->getColMap()->getLocalNumElements();
    for (size_t row = 0; row < a.first.size(); ++row)
    {
        std::vector<double> av(columns), ev(columns);
        for (const auto& [column, value] : a.first[row]) av.at(column) = value;
        for (const auto& [column, value] : e.first[row]) ev.at(column) = value;
        EXPECT_EQ(av, ev);
    }
}

template<class MeshType> struct Assembly
{
    using Fields = MeshFieldTraits<Pack, MeshType>;
    using Scalar = typename Fields::scalar_cell_type;
    using Vector = typename Fields::vector_cell_type;
    using Flux = typename Fields::scalar_face_type;
    SP<const MeshType> mesh;
    Scalar scalar, weight, diffusion;
    Vector velocity;
    Flux flux;
    int generation = 0;
    bool invalid_source = false;
    std::vector<std::tuple<char, int, size_t>> callbacks;

    explicit Assembly(SP<const MeshType> selected)
        : mesh(std::move(selected)), scalar(mesh, "traverse_scalar"), weight(mesh, "traverse_weight"),
          diffusion(mesh, "traverse_diffusion"), velocity(mesh, "traverse_velocity"), flux(mesh, "traverse_flux") {}

    void update(int next_generation)
    {
        generation = next_generation;
        for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
        {
            const auto cell = static_cast<Local>(owned);
            const auto p = mesh->cell_centroid(cell);
            scalar.set_owned_value(cell, 2 + p.x * p.y + 0.3 * p.z + generation);
            weight.set_owned_value(cell, 1.5 + 0.2 * p.x + 0.1 * generation);
            diffusion.set_owned_value(cell, 0.4 + 0.1 * p.z + 0.05 * generation);
            velocity.set_owned_value(cell, Vec{1 + p.x * p.y, 0.2 * p.x - p.z, p.y * p.z + generation});
        }
        for (const auto face : flux.owned_face_ids())
        {
            const auto value = mesh->face_area_vector(face).dot(Vec{generation % 2 ? -0.25 : 0.25, 0.05, -0.1});
            if constexpr (requires { flux.set_owned_value(face, value); }) flux.set_owned_value(face, value);
            else flux.set_value(face, value);
        }
        scalar.sync_ghosts(); weight.sync_ghosts(); diffusion.sync_ghosts(); velocity.sync_ghosts();
        if constexpr (requires { flux.sync_ghosts(); }) flux.sync_ghosts();
    }

    auto weighted(const TransportGeometryCache<MeshType>& geometry, NonOrthogonalTreatment treatment,
        Matrix matrix = Teuchos::null)
    {
        callbacks.clear();
        return weighted_scalar_transport_system<Pack>(MeshWeightedScalarTransportRequest<Pack, MeshType>{
            .old_values = scalar,
            .face_fluxes = flux,
            .time_step = 0.3,
            .storage_weight = weight,
            .advection_weight = weight,
            .diffusivity = diffusion,
            .boundary_condition = [&](int batch, size_t index)
            {
                callbacks.emplace_back('c', batch, index);
                return BoundaryCondition{(batch + generation) % 2 ? BoundaryConditionType::Dirichlet
                                                                 : BoundaryConditionType::Neumann, 0.125};
            },
            .boundary_value = [&](int batch, size_t index)
            {
                callbacks.emplace_back('v', batch, index);
                return 0.75 + generation;
            },
            .source = [&](Local cell)
            {
                callbacks.emplace_back('s', cell, 0);
                return invalid_source && Tpetra::getDefaultComm()->getRank() == 0
                    ? std::numeric_limits<double>::quiet_NaN() : 0.2;
            },
            .treatment = treatment,
            .discretization = {ScalarTimeScheme::BackwardEuler, ScalarConvectionScheme::BoundedLinearUpwind},
            .correction_field = treatment == NonOrthogonalTreatment::Implicit ? nullptr : &scalar,
            .cached_matrix = std::move(matrix),
            .implicit_sink = [](Local) { return 0.1; },
            .fixed_cell_value = [&](Local cell) -> std::optional<double>
            {
                return mesh->owned_cell_map()->getGlobalElement(cell) % 3 == 0
                    ? std::optional<double>{1.25 + generation} : std::nullopt;
            },
            .geometry_cache = &geometry,
            .coefficient_interpolation = generation % 2 ? FaceCoefficientInterpolation::Linear
                                                       : FaceCoefficientInterpolation::Harmonic});
    }

    auto temperature(const TransportGeometryCache<MeshType>& geometry, NonOrthogonalTreatment treatment,
        Matrix matrix = Teuchos::null)
    {
        return physical_temperature_transport_system<Pack>(scalar, flux, 0.3, weight, weight, diffusion,
            [&](int batch, size_t)
            {
                return BoundaryCondition{(batch + generation) % 2 ? BoundaryConditionType::Robin
                                                                 : BoundaryConditionType::Dirichlet,
                    2.0 + generation, 0.4 + 0.1 * generation};
            },
            [&](int, size_t) { return 2.0 + generation; }, [](Local) { return 0.2; }, treatment,
            treatment == NonOrthogonalTreatment::Implicit ? nullptr : &scalar, std::move(matrix), nullptr, &geometry);
    }

    auto momentum(const TransportGeometryCache<MeshType>& geometry, NonOrthogonalTreatment treatment,
        Matrix matrix = Teuchos::null)
    {
        return physical_momentum_transport_system<Pack>(velocity, flux, 0.3, diffusion, 1.75,
            [&](int, size_t) { return Vec{0.2 + generation, -0.3, 0.4}; },
            [](Local) { return Vec{0.01, -0.02, 0.03}; }, treatment,
            treatment == NonOrthogonalTreatment::Implicit ? nullptr : &velocity, std::move(matrix),
            FVM::detail::AlwaysDiffuseBoundary{}, nullptr, &geometry);
    }
};

template<class MeshType> void compare_repeated_assemblies(SP<const MeshType> mesh)
{
    Assembly<MeshType> state(mesh);
    TransportGeometryCache<MeshType> retained(*mesh);
    const auto* traverse = &retained.traverse_plan();
    for (auto treatment : {NonOrthogonalTreatment::Explicit, NonOrthogonalTreatment::Implicit,
             NonOrthogonalTreatment::Hybrid})
    {
        SCOPED_TRACE(static_cast<int>(treatment));
        Matrix weighted_matrix, temperature_matrix, momentum_matrix;
        for (int generation = 0; generation < 2; ++generation)
        {
            SCOPED_TRACE(generation);
            state.update(generation);
            TransportGeometryCache<MeshType> fresh(*mesh);
            expect_canonical_geometry(*mesh, retained);
            const auto expected = state.weighted(fresh, treatment);
            const auto callbacks = state.callbacks;
            const auto actual = state.weighted(retained, treatment);
            expect_same_system(actual, expected);
            EXPECT_EQ(state.callbacks, callbacks);
            const auto reused = state.weighted(retained, treatment, weighted_matrix);
            expect_same_system(reused, expected, true);
            EXPECT_EQ(state.callbacks, callbacks);
            if (!weighted_matrix.is_null()) EXPECT_EQ(reused.matrix.get(), weighted_matrix.get());
            weighted_matrix = reused.matrix;

            const auto expected_temperature = state.temperature(fresh, treatment);
            expect_same_system(state.temperature(retained, treatment), expected_temperature);
            const auto reused_temperature = state.temperature(retained, treatment, temperature_matrix);
            expect_same_system(reused_temperature, expected_temperature, true);
            if (!temperature_matrix.is_null()) EXPECT_EQ(reused_temperature.matrix.get(), temperature_matrix.get());
            temperature_matrix = reused_temperature.matrix;

            const auto expected_momentum = state.momentum(fresh, treatment);
            expect_same_system(state.momentum(retained, treatment), expected_momentum);
            const auto reused_momentum = state.momentum(retained, treatment, momentum_matrix);
            expect_same_system(reused_momentum, expected_momentum, true);
            if (!momentum_matrix.is_null()) EXPECT_EQ(reused_momentum.matrix.get(), momentum_matrix.get());
            momentum_matrix = reused_momentum.matrix;
            EXPECT_EQ(&retained.traverse_plan(), traverse);

            // A rank-local invalid callback result must fail collectively before
            // writing even one row of the retained matrix.
            const auto before = snapshot(reused);
            state.invalid_source = true;
            EXPECT_THROW(state.weighted(retained, treatment, weighted_matrix), std::invalid_argument);
            EXPECT_EQ(snapshot(reused), before);
            EXPECT_TRUE(weighted_matrix->isFillComplete());
            state.invalid_source = false;
        }
    }
}
// Canonical serial XY/Z topology with mixed cell degrees and repeated side
// patch IDs. The middle quad supplies interior cells once there are three layers.
auto layered_transport_mesh(size_t layers, bool triangles)
{
    using Native = Meshes::SemiStructuredXY_Z;
    Arr<Vec> nodes;
    const std::array<real_t, 4> xy{0, 0.4, 1.3, 2.1};
    for (unsigned y = 0; y < 4; ++y)
        for (unsigned x = 0; x < 4; ++x) nodes.push_back({xy[x], xy[y], 0});
    Arr<Arr<unsigned>> cells;
    for (unsigned y = 0; y < 3; ++y)
        for (unsigned x = 0; x < 3; ++x)
        {
            const unsigned a = x + 4 * y, b = a + 1, c = b + 4, d = a + 4;
            if (triangles || (x == 0 && y == 0))
            {
                cells.push_back({a, b, d});
                cells.push_back({b, c, d});
            }
            else cells.push_back({a, b, c, d});
        }
    Arr<Native::BoundaryEdge> edges;
    for (unsigned i = 0; i < 3; ++i)
    {
        edges.push_back({i, i + 1, "south"});
        edges.push_back({12 + i, 13 + i, "north"});
        edges.push_back({4 * i, 4 * (i + 1), i % 2 ? "odd" : "even"});
        edges.push_back({3 + 4 * i, 3 + 4 * (i + 1), i % 2 ? "odd" : "even"});
    }
    Arr<real_t> z{0};
    for (size_t layer = 0; layer < layers; ++layer) z.push_back(z.back() + 0.3 + 0.17 * layer);
    return std::make_shared<Handle>(std::make_shared<Native>(nodes, cells, z, edges));
}
} // namespace

TEST(TransportTraverseTest, CompositeIncidenceMatchesCanonicalOrderAndClassification)
{
    for (const auto& geometry : {test::two_regions(3), test::mixed_regions(), test::coarse_fine_regions(),
             test::periodic_regions(), test::cylindrical_regions(), test::independent_extruded_regions()})
    {
        auto mesh = std::make_shared<Handle>(geometry);
        const auto materialized = mesh->has_materialized_indexer();
        TransportGeometryCache<Handle> cache(*mesh);
        expect_canonical_geometry(*mesh, cache);
        EXPECT_EQ(mesh->connectivity_storage_bytes(), 0U);
        EXPECT_EQ(mesh->has_materialized_indexer(), materialized);
    }
}

TEST(TransportTraverseTest, NonconformingAndPeriodicRowsReuseWithLivePhysicalInputs)
{
    for (const auto& geometry : {test::coarse_fine_regions(), test::periodic_regions(), test::mixed_regions()})
        compare_repeated_assemblies<Handle>(std::make_shared<Handle>(geometry));
}

TEST(TransportTraverseTest, NativeReorderedAndLegacyRowsReuseWithLivePhysicalInputs)
{
    auto native = std::make_shared<Meshes::OrthogonalCartesian3D>(
        Vec3D<ArrReal>{{{0, 0.5, 1, 1.5, 2}, {0, 0.5, 1}, {0, 0.5, 1}}});
    compare_repeated_assemblies<Handle>(std::make_shared<Handle>(native));
    auto reordered = MeshReorderingFactory<>::selected_cells_first(
        std::make_shared<Handle>(native), [](auto, const Vec& point) { return point.y > 0.5; });
    compare_repeated_assemblies(reordered.mesh);
    compare_repeated_assemblies<Mesh<Pack>>(test::make_skewed_prism_mesh<Pack>());
}

TEST(TransportTraverseTest, MotionAndRollbackRefreshMetricsAndRetainTraversePlan)
{
    auto mesh = std::make_shared<Handle>(test::coarse_fine_regions());
    Assembly<Handle> state(mesh);
    state.update(0);
    TransportGeometryCache<Handle> cache(*mesh);
    const auto* traverse = &cache.traverse_plan();
    const auto old_geometry = cache.shared_geometry();
    const auto batches = cache.assembly_geometry().batches;
    EXPECT_EQ(cache.assembly_geometry().traverse_plan.get(), traverse);
    const auto original = state.weighted(cache, NonOrthogonalTreatment::Hybrid);
    const auto original_values = snapshot(original);
    PlanarALEMeshMotion<> motion(mesh);
    EXPECT_THROW(FVM::detail::visit_transport_assembly_rows(*mesh, cache.assembly_geometry(),
        [](Local, auto, const auto& visit_faces)
        { visit_faces([](const auto&, const auto&) { throw std::runtime_error("injected assembly failure"); }); }),
        std::runtime_error);
    // The failed callback must release its geometry lease before mutation.
    motion.begin_trial(1.35, 0.2);
    EXPECT_THROW(state.weighted(cache, NonOrthogonalTreatment::Hybrid, original.matrix), std::invalid_argument);
    EXPECT_EQ(snapshot(original), original_values);
    cache.refresh();
    EXPECT_EQ(&cache.traverse_plan(), traverse);
    EXPECT_EQ(cache.assembly_geometry().batches, batches);
    EXPECT_NE(cache.shared_geometry(), old_geometry);
    expect_canonical_geometry(*mesh, cache);
    TransportGeometryCache<Handle> fresh(*mesh);
    expect_same_system(state.weighted(cache, NonOrthogonalTreatment::Hybrid),
        state.weighted(fresh, NonOrthogonalTreatment::Hybrid));
    motion.rollback_trial();
    EXPECT_THROW(cache.assembly_geometry(), std::invalid_argument);
    cache.refresh();
    EXPECT_EQ(&cache.traverse_plan(), traverse);
    EXPECT_EQ(cache.assembly_geometry().batches, batches);
    expect_canonical_geometry(*mesh, cache);
    EXPECT_EQ(snapshot(state.weighted(cache, NonOrthogonalTreatment::Hybrid)), original_values);
}

TEST(TransportTraverseTest, RepeatedPeriodicNeighbourAccumulatesBothUniformDiffusionFaces)
{
    using namespace Meshes;
    StructuredPatchInterface closure{{0, 0}, {1, 1}};
    closure.periodic_translation = Vec{2, 0, 0};
    auto geometry = std::make_shared<MultiRegionMesh>(std::vector<MultiRegionMesh::Region>{
        cartesian_region("left", {{{0, 1}, {0, 1}, {0, 1}}}),
        cartesian_region("right", {{{1, 2}, {0, 1}, {0, 1}}})},
        std::vector<MultiRegionMesh::Interface>{StructuredPatchInterface{{0, 1}, {1, 0}}, closure});
    SP<const Handle> mesh = std::make_shared<Handle>(geometry);
    ScalarCellFieldStored<Pack> scalar(mesh, "scalar");
    VectorCellFieldStored<Pack> velocity(mesh, "velocity");
    ScalarFaceFieldStored<Pack> flux(mesh, 0.0, "zero_flux");
    for (size_t cell = 0; cell < mesh->num_owned_cells(); ++cell)
    {
        const auto gid = mesh->owned_cell_map()->getGlobalElement(cell);
        scalar.set_owned_value(cell, 2.0 + 2.0 * gid);
        velocity.set_owned_value(cell, Vec{2.0 + 2.0 * gid, 3.0 + 2.0 * gid, 4.0 + 2.0 * gid});
    }
    scalar.sync_ghosts(); velocity.sync_ghosts(); flux.sync_ghosts();
    TransportGeometryCache<Handle> cache(*mesh);
    expect_canonical_geometry(*mesh, cache);
    const auto scalar_system = transport_system<Pack>(scalar, flux, 0.25, 0.5,
        [](int, size_t) { return BoundaryCondition{BoundaryConditionType::Neumann, 0.0}; },
        [](int, size_t) { return 0.0; }, [](Local) { return 0.0; });
    const auto vector_system = non_orthogonal_transport_system<Pack>(velocity, flux, 0.25, 0.5,
        [](int, size_t) { return Vec{}; }, [](Local) { return Vec{}; }, NonOrthogonalTreatment::Explicit,
        nullptr, Teuchos::null, [](int, size_t) { return false; }, &cache);
    const auto check = [&](const auto& system)
    {
        const auto values = snapshot(system);
        for (size_t cell = 0; cell < mesh->num_owned_cells(); ++cell)
        {
            const auto gid = mesh->owned_cell_map()->getGlobalElement(cell);
            ASSERT_EQ(values.first[cell].size(), 2U);
            for (const auto& [column, coefficient] : values.first[cell])
                EXPECT_EQ(coefficient, system.matrix->getColMap()->getGlobalElement(column) == gid ? 5.0 : -1.0);
            for (size_t component = 0; component < values.second.size(); ++component)
                EXPECT_EQ(values.second[component][cell], 4.0 * (2.0 + component + 2.0 * gid));
        }
    };
    check(scalar_system);
    check(vector_system);
}

TEST(TransportTraverseTest, RankLocalInvalidatedPlanRejectsBeforeCachedMatrixMutation)
{
    auto mesh = std::make_shared<Handle>(test::two_regions(2));
    Assembly<Handle> state(mesh);
    state.update(0);
    using Cache = TransportGeometryCache<Handle>;
    auto plan = std::make_shared<MeshTraversePlan<Handle>>(*mesh, MeshTraverseEntities::AllWithCellFaces);
    Cache cache(*mesh);
    cache.refresh(std::make_shared<Cache::SharedGeometry>(*mesh, plan));
    const auto system = state.weighted(cache, NonOrthogonalTreatment::Hybrid);
    const auto before = snapshot(system);
    if (Tpetra::getDefaultComm()->getRank() == 0) plan->invalidate();
    EXPECT_THROW(state.weighted(cache, NonOrthogonalTreatment::Hybrid, system.matrix), std::invalid_argument);
    EXPECT_EQ(snapshot(system), before);
    EXPECT_TRUE(system.matrix->isFillComplete());
}

TEST(TransportTraverseTest, RankAsymmetricCacheSelectionCoordinatesLocalPlanFailure)
{
    auto mesh = std::make_shared<Handle>(test::two_regions(2));
    const auto comm = mesh->owned_cell_map()->getComm();
    using Cache = TransportGeometryCache<Handle>;
    using Geometry = FVM::detail::TransportAssemblyGeometry<Handle>;
    auto plan = std::make_shared<MeshTraversePlan<Handle>>(*mesh, MeshTraverseEntities::AllWithCellFaces);
    Cache cache(*mesh);
    cache.refresh(std::make_shared<Cache::SharedGeometry>(*mesh, plan));
    const Cache* selected_cache = comm->getRank() == 0 ? &cache : nullptr;
    Geometry local;
    const Geometry* selected = nullptr;
    EXPECT_NO_THROW(selected = &FVM::detail::select_transport_assembly_geometry(*mesh, local, selected_cache));
    EXPECT_EQ(selected, comm->getRank() == 0 ? &cache.assembly_geometry() : &local);
    if (selected) EXPECT_EQ(selected->volumes.size(), mesh->num_owned_cells());

    if (comm->getRank() == 0) plan->invalidate();
    int rejected = 0;
    try
    {
        // Other ranks still construct valid local geometry; they must observe
        // rank zero's failed cache validation before proceeding to assembly.
        FVM::detail::select_transport_assembly_geometry(*mesh, local, selected_cache);
    }
    catch (const std::exception&) { rejected = 1; }
    int rejected_ranks = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &rejected, &rejected_ranks);
    EXPECT_EQ(rejected_ranks, comm->getSize());
}

TEST(TransportTraverseTest, EmptyRanksPreserveRowsAndCollectiveRejection)
{
    auto geometry = std::make_shared<Meshes::MultiRegionMesh>(std::vector<Meshes::MultiRegionMesh::Region>{
        Meshes::cartesian_region("single", {{{0, 1}, {0, 1}, {0, 1}}})},
        std::vector<Meshes::MultiRegionMesh::Interface>{});
    SP<const Handle> mesh = std::make_shared<Handle>(geometry);
    compare_repeated_assemblies(mesh);
    if (Tpetra::getDefaultComm()->getRank() > 0) EXPECT_EQ(mesh->num_owned_cells(), 0U);
}

TEST(TransportTraverseTest, LayeredRowsPreserveCanonicalIncidenceAndLiveAssemblyInputs)
{
    if (Tpetra::getDefaultComm()->getSize() != 1) GTEST_SKIP() << "XY/Z remains serial only.";
    for (size_t layers : {1U, 2U, 5U})
        for (bool triangles : {false, true})
        {
            SCOPED_TRACE(::testing::Message() << "layers=" << layers << " triangles=" << triangles);
            auto mesh = layered_transport_mesh(layers, triangles);
            const auto materialized = mesh->has_materialized_indexer();
            TransportGeometryCache<Handle> cache(*mesh);
            EXPECT_TRUE(cache.traverse_plan().uses_layered_ranges());
            EXPECT_FALSE(cache.traverse_plan().uses_structured_ranges());
            expect_canonical_geometry(*mesh, cache);
            EXPECT_EQ(mesh->connectivity_storage_bytes(), 0U);
            EXPECT_EQ(mesh->has_materialized_indexer(), materialized);
            compare_repeated_assemblies<Handle>(mesh);
        }
}

TEST(TransportTraverseTest, LayeredMotionAndRollbackRefreshMetricsAndRetainTraversePlan)
{
    if (Tpetra::getDefaultComm()->getSize() != 1) GTEST_SKIP() << "XY/Z remains serial only.";
    for (size_t layers : {1U, 2U, 5U})
        for (bool triangles : {false, true})
        {
            SCOPED_TRACE(::testing::Message() << "layers=" << layers << " triangles=" << triangles);
            auto mesh = layered_transport_mesh(layers, triangles);
            Assembly<Handle> state(mesh);
            state.update(0);
            TransportGeometryCache<Handle> cache(*mesh);
            const auto* traverse = &cache.traverse_plan();
            const auto topology = traverse->layered_layout();
            const auto plan_bytes = traverse->storage_bytes();
            const auto old_geometry = cache.shared_geometry();
            const auto batches = cache.assembly_geometry().batches;
            const auto original = state.weighted(cache, NonOrthogonalTreatment::Hybrid);
            const auto original_values = snapshot(original);
            PlanarALEMeshMotion<> motion(mesh);
            EXPECT_THROW(FVM::detail::visit_transport_assembly_rows(*mesh, cache.assembly_geometry(),
                [](Local, auto, const auto& visit_faces)
                { visit_faces([](const auto&, const auto&) { throw std::runtime_error("injected layered assembly failure"); }); }),
                std::runtime_error);
            // Callback failure releases the execution lease before geometry mutation.
            motion.begin_trial(1.35, 0.2);
            EXPECT_THROW(state.weighted(cache, NonOrthogonalTreatment::Hybrid, original.matrix), std::invalid_argument);
            EXPECT_EQ(snapshot(original), original_values);
            cache.refresh();
            EXPECT_EQ(&cache.traverse_plan(), traverse);
            EXPECT_EQ(traverse->layered_layout(), topology);
            EXPECT_EQ(traverse->storage_bytes(), plan_bytes);
            EXPECT_EQ(cache.assembly_geometry().batches, batches);
            EXPECT_NE(cache.shared_geometry(), old_geometry);
            expect_canonical_geometry(*mesh, cache);
            TransportGeometryCache<Handle> fresh(*mesh);
            expect_same_system(state.weighted(cache, NonOrthogonalTreatment::Hybrid),
                state.weighted(fresh, NonOrthogonalTreatment::Hybrid));
            motion.rollback_trial();
            EXPECT_THROW(cache.assembly_geometry(), std::invalid_argument);
            cache.refresh();
            EXPECT_EQ(&cache.traverse_plan(), traverse);
            EXPECT_EQ(traverse->layered_layout(), topology);
            EXPECT_EQ(traverse->storage_bytes(), plan_bytes);
            EXPECT_EQ(cache.assembly_geometry().batches, batches);
            expect_canonical_geometry(*mesh, cache);
            EXPECT_EQ(snapshot(state.weighted(cache, NonOrthogonalTreatment::Hybrid)), original_values);
        }
}
