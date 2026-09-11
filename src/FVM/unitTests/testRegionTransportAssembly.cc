/** @file testRegionTransportAssembly.cc @brief Region CFD transport/reference parity. */
#include <gtest/gtest.h>

#include "FVM/Operators.hh"
#include "geometry/MeshHandle.hh"
#include "geometry/PlanarALEMeshMotion.hh"
#include "geometry/unitTests/region_mesh_helpers.hh"
#include "utils/testing_environment.hh"

#include <Teuchos_Array.hpp>

#include <cmath>
#include <vector>

namespace
{
using namespace SimpleFluid;
using namespace SimpleFluid::FVM;
using Pack = DefaultTpetraTypes;
using Handle = MeshHandle<Pack>;
using LO = Pack::local_ordinal_type;
using Vec = MeshUtils::Vec3;

// Retain precisely the same providers, local IDs and maps while selecting the
// generic cached-row/metric path. No production execution switch is needed.
struct ReferenceHandle : Handle
{
    using Handle::Handle;
    bool supports_region_execution() const noexcept { return false; }
};

testing::Environment* const environment =
    testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);

struct Systems
{
    TransportSystem<Pack> scalar;
    VectorTransportSystem<Pack> momentum;
};

template<class MeshType> struct AssemblyState
{
    SP<const MeshType> mesh;
    ScalarCellFieldStored<Pack, MeshType> old, storage, diffusion;
    VectorCellFieldStored<Pack, MeshType> velocity;
    ScalarFaceFieldStored<Pack, MeshType> flux;
    TransportGeometryCache<MeshType> geometry;
    FVM::detail::StoredTransportSymbolicPlan<Pack> plan;
    Teuchos::RCP<Pack::matrix_type> scalar_matrix, momentum_matrix;
    std::vector<std::pair<int, size_t>> boundary_calls;
    std::vector<LO> source_calls;

    explicit AssemblyState(SP<const MeshType> selected)
        : mesh(std::move(selected)), old(mesh, "old"), storage(mesh, "storage"),
          diffusion(mesh, "diffusion"), velocity(mesh, "velocity"), flux(mesh, "flux"), geometry(*mesh) {}

    Systems assemble(int generation, FaceCoefficientInterpolation interpolation)
    {
        for (size_t c = 0; c < mesh->num_owned_cells(); ++c)
        {
            const auto p = mesh->cell_centroid(c);
            old.set_owned_value(c, 1 + 0.2 * p.x + 0.1 * p.x * p.y + 0.05 * std::sin(p.z) + generation);
            storage.set_owned_value(c, 2 + 0.05 * p.z + 0.1 * generation);
            diffusion.set_owned_value(c, 1.1 + 0.1 * p.x + 0.01 * p.y + 0.2 * generation);
            velocity.set_owned_value(c, Vec{1 + 0.1 * p.x * p.y, 0.2 * std::sin(p.x), 0.3 * p.z});
        }
        for (size_t f = 0; f < mesh->num_owned_faces(); ++f)
            flux.set_owned_value(f, mesh->face_area_vector(f).dot(Vec{generation == 1 ? -0.2 : 0.2, -0.05, 0.1}));
        old.sync_ghosts();
        storage.sync_ghosts();
        diffusion.sync_ghosts();
        velocity.sync_ghosts();
        flux.sync_ghosts();
        boundary_calls.clear();
        source_calls.clear();
        const auto scalar = weighted_scalar_transport_system<Pack>(
            FieldStoredWeightedScalarTransportRequest<Pack, MeshType>{
                .old_values = old,
                .face_fluxes = flux,
                .time_step = 0.25,
                .storage_weight = storage,
                .advection_weight = storage,
                .diffusivity = diffusion,
                .boundary_condition = [&](int batch, size_t in_batch)
                {
                    boundary_calls.emplace_back(batch, in_batch);
                    return BoundaryCondition{batch % 2 ? BoundaryConditionType::Dirichlet
                                                      : BoundaryConditionType::Neumann, 0.13};
                },
                .boundary_value = [generation](int, size_t) { return 0.75 + generation; },
                .source = [&](LO cell) { source_calls.push_back(cell); return 0.17; },
                .treatment = NonOrthogonalTreatment::Hybrid,
                .discretization = {ScalarTimeScheme::BackwardEuler, ScalarConvectionScheme::BoundedLinearUpwind},
                .correction_field = &old,
                .cached_matrix = scalar_matrix,
                .implicit_sink = [](LO) { return 0.025; },
                .geometry_cache = &geometry,
                .coefficient_interpolation = interpolation,
                .symbolic_plan = &plan});
        const auto momentum = physical_momentum_transport_system<Pack>(velocity, flux, 0.25, diffusion, 2.3,
            [](int, size_t) { return Vec{0.3, -0.2, 0.1}; },
            [](LO) { return Vec{0.01, 0.02, -0.03}; }, NonOrthogonalTreatment::Hybrid,
            &velocity, momentum_matrix, FVM::detail::AlwaysDiffuseBoundary{}, nullptr, &geometry, interpolation);
        if (!scalar_matrix.is_null()) EXPECT_EQ(scalar.matrix.get(), scalar_matrix.get());
        if (!momentum_matrix.is_null()) EXPECT_EQ(momentum.matrix.get(), momentum_matrix.get());
        scalar_matrix = scalar.matrix;
        momentum_matrix = momentum.matrix;
        return {scalar, momentum};
    }
};

template<class System> void expect_same_system(const System& actual, const System& expected)
{
    ASSERT_EQ(actual.matrix->getGlobalNumEntries(), expected.matrix->getGlobalNumEntries());
    ASSERT_TRUE(actual.matrix->getRowMap()->isSameAs(*expected.matrix->getRowMap()));
    ASSERT_TRUE(actual.matrix->getColMap()->isSameAs(*expected.matrix->getColMap()));
    for (size_t row = 0; row < actual.matrix->getLocalNumRows(); ++row)
    {
        Pack::matrix_type::local_inds_host_view_type ac, ec;
        Pack::matrix_type::values_host_view_type av, ev;
        actual.matrix->getLocalRowView(row, ac, av);
        expected.matrix->getLocalRowView(row, ec, ev);
        ASSERT_EQ(ac.extent(0), ec.extent(0));
        for (size_t i = 0; i < ac.extent(0); ++i)
        {
            EXPECT_EQ(ac(i), ec(i));
            EXPECT_NEAR(av(i), ev(i), 3e-13 * (1 + std::abs(ev(i))));
        }
    }
    const auto columns = actual.rhs->getNumVectors();
    Pack::multi_vector_type input(actual.matrix->getDomainMap(), columns), action(actual.rhs->getMap(), columns),
        reference_action(expected.rhs->getMap(), columns), difference(actual.rhs->getMap(), columns);
    {
        auto values = input.getLocalViewHost(Tpetra::Access::OverwriteAll);
        for (size_t row = 0; row < input.getLocalLength(); ++row)
            for (size_t c = 0; c < columns; ++c)
                values(row, c) = std::sin(0.3 * (1 + input.getMap()->getGlobalElement(row)) + c);
    }
    actual.matrix->apply(input, action);
    expected.matrix->apply(input, reference_action);
    Teuchos::Array<double> errors(columns), reference_norms(columns);
    difference.update(1.0, action, -1.0, reference_action, 0.0);
    difference.norm2(errors()); reference_action.norm2(reference_norms());
    for (size_t c = 0; c < columns; ++c) EXPECT_LE(errors[c], 3e-13 * (1 + reference_norms[c]));
    difference.update(1.0, *actual.rhs, -1.0, *expected.rhs, 0.0);
    difference.norm2(errors()); expected.rhs->norm2(reference_norms());
    for (size_t c = 0; c < columns; ++c) EXPECT_LE(errors[c], 3e-13 * (1 + reference_norms[c]));
}

void expect_pressure_geometry(const Handle& actual, const ReferenceHandle& expected)
{
    const auto a = FVM::detail::stored_pressure_face_geometry<Pack>(actual);
    const auto e = FVM::detail::stored_pressure_face_geometry<Pack>(expected);
    ASSERT_EQ(a.size(), e.size());
    for (size_t f = 0; f < a.size(); ++f)
    {
        EXPECT_EQ(a[f].face, e[f].face); EXPECT_EQ(a[f].owner, e[f].owner); EXPECT_EQ(a[f].neighbor, e[f].neighbor);
        EXPECT_EQ(a[f].interior, e[f].interior); EXPECT_EQ(a[f].boundary, e[f].boundary);
        EXPECT_EQ(a[f].owned_owner, e[f].owned_owner);
        EXPECT_NEAR((a[f].normal - e[f].normal).norm(), 0, 2e-14);
        EXPECT_NEAR((a[f].owner_normal - e[f].owner_normal).norm(), 0, 2e-14);
        EXPECT_NEAR((a[f].area_vector - e[f].area_vector).norm(), 0, 2e-14);
        EXPECT_DOUBLE_EQ(a[f].area, e[f].area);
        EXPECT_DOUBLE_EQ(a[f].owner_weight, e[f].owner_weight);
        EXPECT_DOUBLE_EQ(a[f].neighbor_weight, e[f].neighbor_weight);
        EXPECT_DOUBLE_EQ(a[f].center_projection, e[f].center_projection);
        EXPECT_DOUBLE_EQ(a[f].distance_squared, e[f].distance_squared);
        EXPECT_DOUBLE_EQ(a[f].boundary_diffusion, e[f].boundary_diffusion);
    }
}

void compare(AssemblyState<Handle>& actual, AssemblyState<ReferenceHandle>& expected, int generation)
{
    const auto interpolation = generation % 2 ? FaceCoefficientInterpolation::Linear : FaceCoefficientInterpolation::Harmonic;
    const auto a = actual.assemble(generation, interpolation);
    const auto e = expected.assemble(generation, interpolation);
    expect_same_system(a.scalar, e.scalar);
    expect_same_system(a.momentum, e.momentum);
    EXPECT_EQ(actual.boundary_calls, expected.boundary_calls);
    EXPECT_EQ(actual.source_calls, expected.source_calls);
    expect_pressure_geometry(*actual.mesh, *expected.mesh);
    EXPECT_EQ(actual.mesh->connectivity_storage_bytes(), 0U);
    EXPECT_FALSE(actual.mesh->has_materialized_indexer());
}
} // namespace

TEST(RegionTransportAssemblyTest, ScalarTurbulenceAndMomentumPreserveGenericActionsAndReuse)
{
    for (const auto& geometry : {test::two_regions(3), test::mixed_regions(), test::coarse_fine_regions(),
             test::periodic_regions(), test::cylindrical_regions(), test::independent_extruded_regions()})
    {
        AssemblyState<Handle> actual(std::make_shared<Handle>(geometry));
        AssemblyState<ReferenceHandle> expected(std::make_shared<ReferenceHandle>(geometry));
        for (int generation = 0; generation < 2; ++generation) compare(actual, expected, generation);
    }
}

TEST(RegionTransportAssemblyTest, MotionAndRollbackRejectStaleCachesAndRefreshResolvedMetrics)
{
    for (const auto& geometry : {test::two_regions(), test::coarse_fine_regions(), test::periodic_regions()})
    {
        auto mesh = std::make_shared<Handle>(geometry);
        AssemblyState<Handle> actual(mesh);
        AssemblyState<ReferenceHandle> expected(std::make_shared<ReferenceHandle>(geometry));
        compare(actual, expected, 0);
        PlanarALEMeshMotion<> motion(mesh);
        motion.begin_trial(1.35, 0.2);
        EXPECT_THROW(actual.geometry.assembly_geometry(), std::invalid_argument);
        EXPECT_THROW(expected.geometry.assembly_geometry(), std::invalid_argument);
        actual.geometry.refresh(); expected.geometry.refresh();
        compare(actual, expected, 1);
        motion.rollback_trial();
        EXPECT_THROW(actual.geometry.assembly_geometry(), std::invalid_argument);
        actual.geometry.refresh(); expected.geometry.refresh();
        compare(actual, expected, 2);
    }
}

TEST(RegionTransportAssemblyTest, EmptyRanksPreserveFieldMapsAndCollectiveAssembly)
{
    const auto geometry = test::two_regions(1);
    AssemblyState<Handle> actual(std::make_shared<Handle>(geometry));
    AssemblyState<ReferenceHandle> expected(std::make_shared<ReferenceHandle>(geometry));
    compare(actual, expected, 0);
}
