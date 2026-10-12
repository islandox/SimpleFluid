/**
 * @file testStoredPressureFaceFluxCache.cc
 * @brief Analytic pressure-flux checks across workspace reuse and ALE epochs.
 */

#include <gtest/gtest.h>

#include "FVM/FaceFlux.hh"
#include "geometry/DirectBatchRange.hh"
#include "geometry/PlanarALEMeshMotion.hh"
#include "geometry/mesh/UnstructuredMesh.hh"
#include "geometry/unitTests/region_mesh_helpers.hh"
#include "utils/testing_environment.hh"

#include <Teuchos_CommHelpers.hpp>

#include <algorithm>
#include <limits>
#include <span>
#include <type_traits>
#include <memory>
#include <unordered_map>
#include <vector>

namespace
{

using Pack = SimpleFluid::DefaultTpetraTypes;
using Mesh = SimpleFluid::MeshHandle<Pack>;
using ScalarField = SimpleFluid::ScalarCellFieldStored<Pack>;
using VectorField = SimpleFluid::VectorCellFieldStored<Pack>;
using FluxField = SimpleFluid::ScalarFaceFieldStored<Pack>;
using Workspace = SimpleFluid::FVM::FieldStoredPressureWeightedFaceFluxWorkspace<Pack>;
using Vec = SimpleFluid::vec3<double>;
using BoundaryType = SimpleFluid::BoundaryConditionType;

template<class Map>
struct CountingBoundaryMap : Map
{
    explicit CountingBoundaryMap(const Map& entries) : Map(entries) {}

    auto find(const typename Map::key_type& key) const
    {
        ++lookups;
        return Map::find(key);
    }

    mutable size_t lookups = 0;
};

struct CountingVelocityBoundaryCache
{
    using Cache = SimpleFluid::FVM::FieldStoredVelocityBoundaryCache<Pack, Mesh>;

    explicit CountingVelocityBoundaryCache(const Cache& cache)
        : value(cache.value), type(cache.type), type_by_name(cache.type_by_name), mesh(cache.mesh)
    {
    }

    CountingBoundaryMap<decltype(Cache::value)> value;
    CountingBoundaryMap<decltype(Cache::type)> type;
    decltype(Cache::type_by_name) type_by_name;
    SimpleFluid::SP<const Mesh> mesh;
};

using utils_test::KokkosEnvironment;
testing::Environment* const kokkos_environment = testing::AddGlobalTestEnvironment(new KokkosEnvironment);

auto make_mesh()
{
    return std::make_shared<Mesh>(std::make_shared<Mesh::Cartesian>(
        SimpleFluid::Vec3D<SimpleFluid::ArrReal>{{
            {0.0, 0.25, 0.75, 1.5, 3.0}, {0.0, 0.5, 2.0}, {0.0, 0.25, 1.0, 2.0, 4.0}}}));
}

Vec affine_velocity(double z, double shift)
{
    return {0.25 + shift + 0.1 * z, -0.3 + 0.2 * z, 0.4 - shift - 0.125 * z};
}

/**
 * @brief Quadratic axial pressure has an analytic two-point face derivative.
 *
 * A graded mesh makes that derivative differ from the linearly interpolated
 * exact cell gradient, exercising an actual Rhie--Chow correction. Physical
 * boundary values change independently of the reusable geometry workspace.
 */
std::vector<double> check_quadratic_flux(const std::shared_ptr<Mesh>& mesh, Workspace& workspace,
    double height, double shift)
{
    VectorField velocity(mesh, "velocity");
    ScalarField pressure(mesh, "pressure");
    VectorField gradient(mesh, "pressure_gradient");
    FluxField normal(mesh, "normal_flux");
    FluxField flux(mesh, "pressure_flux");
    FluxField expected_flux(mesh, "expected_pressure_flux");
    for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
    {
        const auto cell = static_cast<Pack::local_ordinal_type>(owned);
        const auto z = mesh->cell_centroid(cell).z;
        velocity.set_owned_value(cell, affine_velocity(z, shift));
        pressure.set_owned_value(cell, 3.0 * z * z);
        gradient.set_owned_value(cell, {0.0, 0.0, 6.0 * z});
    }
    velocity.sync_ghosts();
    pressure.sync_ghosts();
    gradient.sync_ghosts();

    SimpleFluid::BoundaryConditionSet conditions;
    conditions.velocity["xmin"] = {BoundaryType::Slip, {}};
    conditions.velocity["xmax"] = {BoundaryType::Dirichlet, {1.0 + shift, -2.0, 0.5}};
    conditions.velocity["ymin"] = {BoundaryType::NoSlip, {}};
    conditions.velocity["ymax"] = {BoundaryType::Neumann, {}};
    conditions.velocity["zmin"] = {BoundaryType::Dirichlet, {-1.0, 2.0, 0.25 - shift}};
    conditions.velocity["zmax"] = {BoundaryType::Neumann, {}};
    const SimpleFluid::SP<const Mesh> const_mesh = mesh;
    auto boundaries = SimpleFluid::FVM::cache_velocity_boundary_conditions<Pack>(const_mesh, conditions);
    for (const auto& [batch_id, batch] : mesh->boundary_batches())
    {
        if (mesh->boundary_batch_name(batch_id) == "ymax")
        {
            for (auto& value : boundaries.value.at(batch_id))
            {
                value = {0.5, 0.75 + shift, -0.5};
            }
        }
    }
    SimpleFluid::BoundaryConditionMap pressure_boundaries;
    pressure_boundaries["zmax"] = {BoundaryType::Dirichlet, 3.0 * height * height};
    SimpleFluid::FVM::face_fluxes(velocity, boundaries, normal);
    normal.sync_ghosts();
    constexpr double coefficient = 0.13;
    flux.put_value(173.0);
    SimpleFluid::FVM::pressure_weighted_face_fluxes(
        velocity, pressure, gradient, coefficient, boundaries, pressure_boundaries, workspace, flux);

    // Fringe overlap faces can have an adjacent cell outside the local cell
    // map. Evaluate their analytical value on the owning rank, then import it.
    for (const auto face : flux.owned_face_ids())
    {
        const auto owner = mesh->owner_cell(face);
        const auto zo = mesh->cell_centroid(owner).z;
        const auto zf = mesh->face_centroid(face).z;
        const auto area = mesh->face_area(face);
        const auto normal_vector = mesh->face_normal(face);
        double expected = normal.local_value(face);
        if (mesh->is_interior_face(face))
        {
            const auto neighbor = mesh->opposite_or_periodic_neighbor_cell(face, owner);
            const auto zn = mesh->cell_centroid(neighbor).z;
            const auto direct = 3.0 * (zo + zn) * normal_vector.z * area;
            const auto interpolated = 6.0 * zf * normal_vector.z * area;
            expected = affine_velocity(zf, shift).dot(normal_vector) * area
                     - coefficient * (direct - interpolated);
        }
        else if (normal_vector.z > 0.5)
        {
            const auto direct = 3.0 * (height + zo) * area;
            const auto interpolated = 6.0 * zo * area;
            expected = affine_velocity(zo, shift).z * area - coefficient * (direct - interpolated);
        }
        expected_flux.set_owned_value(face, expected);
    }
    expected_flux.sync_ghosts();
    std::vector<double> result;
    for (size_t local = 0; local < mesh->num_faces(); ++local)
    {
        const auto face = static_cast<Pack::local_ordinal_type>(local);
        EXPECT_NEAR(flux.local_value(face), expected_flux.local_value(face), 3.0e-13) << "face " << face;
        result.push_back(flux.local_value(face));
    }

    // The disabled correction must not consume the supplied gradient at all.
    gradient.put_value({std::numeric_limits<double>::quiet_NaN(), 0.0, 0.0});
    flux.put_value(-173.0);
    SimpleFluid::FVM::pressure_weighted_face_fluxes(
        velocity, pressure, gradient, 0.0, boundaries, pressure_boundaries, workspace, flux);
    for (size_t local = 0; local < mesh->num_faces(); ++local)
    {
        const auto face = static_cast<Pack::local_ordinal_type>(local);
        EXPECT_DOUBLE_EQ(flux.local_value(face), normal.local_value(face));
    }
    return result;
}

std::vector<double> reconstruct_flux(const std::shared_ptr<Mesh>& mesh, Workspace& workspace,
    double scale, double boundary_shift)
{
    VectorField velocity(mesh, "shared_geometry_velocity");
    ScalarField pressure(mesh, "shared_geometry_pressure");
    FluxField flux(mesh, "shared_geometry_flux");
    velocity.put_value({0.2, 0.1, -0.3});
    for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
    {
        const auto cell = static_cast<Pack::local_ordinal_type>(owned);
        const auto center = mesh->cell_centroid(cell);
        pressure.set_owned_value(cell, scale * (0.2 * center.x + center.z * center.z));
    }
    pressure.sync_ghosts();
    SimpleFluid::BoundaryConditionSet conditions;
    for (const auto& [batch_id, batch] : mesh->boundary_batches())
        conditions.velocity[mesh->boundary_batch_name(batch_id)] = {BoundaryType::Neumann, {}};
    const SimpleFluid::SP<const Mesh> const_mesh = mesh;
    const auto boundaries = SimpleFluid::FVM::cache_velocity_boundary_conditions<Pack>(const_mesh, conditions);
    SimpleFluid::BoundaryConditionMap pressure_boundaries;
    pressure_boundaries["zmax"] = {BoundaryType::Dirichlet, 16.0 * scale + boundary_shift};
    SimpleFluid::FVM::pressure_weighted_face_fluxes(
        velocity, pressure, 0.13, boundaries, pressure_boundaries, workspace, flux);
    std::vector<double> result;
    for (size_t local = 0; local < mesh->num_faces(); ++local)
        result.push_back(flux.local_value(static_cast<Pack::local_ordinal_type>(local)));
    return result;
}

std::vector<double> gradient_values(const Workspace& workspace)
{
    const auto& gradient = workspace.pressure_gradient();
    const auto values = gradient.local_read_view();
    std::vector<double> result;
    for (size_t owned = 0; owned < gradient.num_owned_cells(); ++owned)
        for (size_t component = 0; component < 3; ++component)
            result.push_back(values(owned, component));
    return result;
}

} // namespace

TEST(StoredPressureFaceFluxCacheTest, DirectRangeValidatesPairedLengthsAndExposesImmutableRecords)
{
    struct Topology { int face; };
    struct Metrics { double area; };
    using Range = SimpleFluid::DirectBatchRange<Topology, Metrics>;
    const std::vector<Topology> topology{{7}, {2}, {9}};
    const std::vector<Metrics> metrics{{0.5}, {1.5}, {2.5}};
    const auto range = Range::checked(topology, metrics);
    EXPECT_EQ(range.size(), 3U);
    EXPECT_FALSE(range.empty());
    EXPECT_EQ(range.topology().data(), topology.data());
    EXPECT_EQ(range.metrics().data(), metrics.data());
    std::vector<int> face_ids;
    double weighted_area = 0;
    range.apply([&](auto& face, auto& metric)
    {
        static_assert(std::is_const_v<std::remove_reference_t<decltype(face)>>);
        static_assert(std::is_const_v<std::remove_reference_t<decltype(metric)>>);
        face_ids.push_back(face.face);
        weighted_area += face.face * metric.area;
    });
    EXPECT_EQ(face_ids, (std::vector<int>{7, 2, 9}));
    EXPECT_DOUBLE_EQ(weighted_area, 29.0);
    EXPECT_THROW(static_cast<void>(Range::checked(
        std::span<const Topology>(topology), std::span<const Metrics>(metrics).first(2))),
        std::invalid_argument);
    const auto empty = Range::checked({}, {});
    EXPECT_TRUE(empty.empty());
    empty.apply([](const auto&, const auto&) { ADD_FAILURE() << "An empty range invoked its operation."; });
}

TEST(StoredPressureFaceFluxCacheTest, DirectBatchesCoverOwnedFacesAndPreserveBoundarySlots)
{
    auto mesh = make_mesh();
    Workspace workspace(mesh);
    EXPECT_FALSE(workspace.has_materialized_face_geometry());
    const auto execution = workspace.acquire_execution();
    static_assert(!std::is_move_constructible_v<decltype(execution)>);
    std::vector<unsigned> visits(mesh->num_faces(), 0);
    size_t visited = 0;
    const auto check_common = [&](const auto& topology, const auto& metrics)
    {
        static_assert(std::is_const_v<std::remove_reference_t<decltype(topology)>>);
        static_assert(std::is_const_v<std::remove_reference_t<decltype(metrics)>>);
        ASSERT_TRUE(mesh->is_owned_face(topology.face));
        EXPECT_EQ(topology.owner, mesh->owner_cell(topology.face));
        EXPECT_DOUBLE_EQ(metrics.area, mesh->face_area(topology.face));
        const auto normal = mesh->face_normal(topology.face);
        EXPECT_DOUBLE_EQ(metrics.normal.x, normal.x);
        EXPECT_DOUBLE_EQ(metrics.normal.y, normal.y);
        EXPECT_DOUBLE_EQ(metrics.normal.z, normal.z);
        ++visits.at(static_cast<size_t>(topology.face));
        ++visited;
    };

    // Both streams must advance directly through contiguous records; field
    // IDs still refer to the mesh and need not equal the packed position.
    const void* previous_topology = nullptr;
    const void* previous_metrics = nullptr;
    execution.interior().apply([&](const auto& topology, const auto& metrics)
    {
        check_common(topology, metrics);
        EXPECT_FALSE(mesh->is_geometry_exterior_face(topology.face));
        EXPECT_EQ(topology.neighbor,
            mesh->opposite_or_periodic_neighbor_cell(topology.face, topology.owner));
        if (previous_topology != nullptr)
            EXPECT_EQ(static_cast<const char*>(previous_topology) + sizeof(topology),
                reinterpret_cast<const char*>(&topology));
        if (previous_metrics != nullptr)
            EXPECT_EQ(static_cast<const char*>(previous_metrics) + sizeof(metrics),
                reinterpret_cast<const char*>(&metrics));
        previous_topology = &topology;
        previous_metrics = &metrics;
    });

    size_t next_begin = 0;
    for (const auto& group : execution.boundary_groups())
    {
        EXPECT_EQ(group.begin, next_begin);
        EXPECT_EQ(group.name, mesh->boundary_batch_name(group.id));
        const auto& original = mesh->boundary_batches().at(group.id).face_lids;
        std::vector<Pack::local_ordinal_type> expected_faces;
        for (const auto face : original)
            if (mesh->is_owned_face(face)) expected_faces.push_back(face);
        std::ranges::sort(expected_faces);
        std::vector<Pack::local_ordinal_type> actual_faces;
        size_t required_values = 0;
        previous_topology = nullptr;
        previous_metrics = nullptr;
        execution.boundary(group).apply([&](const auto& topology, const auto& metrics)
        {
            check_common(topology, metrics);
            EXPECT_TRUE(mesh->is_boundary_face(topology.face));
            EXPECT_EQ(mesh->boundary_id(topology.face), group.id);
            ASSERT_LT(topology.in_batch_id, original.size());
            EXPECT_EQ(original[topology.in_batch_id], topology.face);
            required_values = std::max(required_values, topology.in_batch_id + 1);
            actual_faces.push_back(topology.face);
            if (previous_topology != nullptr)
                EXPECT_EQ(static_cast<const char*>(previous_topology) + sizeof(topology),
                    reinterpret_cast<const char*>(&topology));
            if (previous_metrics != nullptr)
                EXPECT_EQ(static_cast<const char*>(previous_metrics) + sizeof(metrics),
                    reinterpret_cast<const char*>(&metrics));
            previous_topology = &topology;
            previous_metrics = &metrics;
        });
        EXPECT_EQ(actual_faces, expected_faces);
        EXPECT_EQ(actual_faces.size(), group.count);
        EXPECT_EQ(required_values, group.required_values);
        next_begin += group.count;
    }
    execution.zero_flux_faces().apply([&](const auto&, const auto&)
    {
        ADD_FAILURE() << "Every Cartesian exterior face belongs to a boundary batch.";
    });
    EXPECT_EQ(visited, mesh->num_owned_faces());
    for (size_t face = 0; face < mesh->num_faces(); ++face)
        EXPECT_EQ(visits[face], mesh->is_owned_face(static_cast<Pack::local_ordinal_type>(face)) ? 1U : 0U);
    EXPECT_FALSE(workspace.has_materialized_face_geometry());
}

TEST(StoredPressureFaceFluxCacheTest, ReuseReadsUpdatedFieldsAndPhysicalBoundaryValues)
{
    auto mesh = make_mesh();
    Workspace workspace(mesh);
    const auto geometry = workspace.shared_geometry();
    const auto* topology = &workspace.topology_groups();
    EXPECT_FALSE(workspace.has_materialized_face_geometry());
    const auto first = check_quadratic_flux(mesh, workspace, 4.0, 0.0);
    const auto second = check_quadratic_flux(mesh, workspace, 4.0, 0.375);
    EXPECT_EQ(workspace.shared_geometry(), geometry);
    EXPECT_EQ(&workspace.topology_groups(), topology);
    EXPECT_FALSE(workspace.has_materialized_face_geometry());
    EXPECT_NE(first, second);
    EXPECT_EQ(check_quadratic_flux(mesh, workspace, 4.0, 0.0), first);
}

TEST(StoredPressureFaceFluxCacheTest, ReuseReadsUpdatedBoundaryTypes)
{
    auto mesh = make_mesh();
    Workspace workspace(mesh);
    const auto* topology = &workspace.topology_groups();
    VectorField velocity(mesh, "changing_boundary_velocity");
    ScalarField pressure(mesh, "changing_boundary_pressure");
    VectorField gradient(mesh, "changing_boundary_gradient");
    FluxField flux(mesh, "changing_boundary_flux");
    const Vec owner_velocity{0.25, -0.5, 1.25};
    const Vec prescribed_velocity{0.75, -0.25, -0.5};
    velocity.put_value(owner_velocity);
    pressure.put_value(2.0);
    gradient.put_value({});
    SimpleFluid::BoundaryConditionSet conditions;
    for (const auto& [batch_id, batch] : mesh->boundary_batches())
        conditions.velocity[mesh->boundary_batch_name(batch_id)] = {BoundaryType::Dirichlet, prescribed_velocity};
    const SimpleFluid::SP<const Mesh> const_mesh = mesh;
    auto boundaries = SimpleFluid::FVM::cache_velocity_boundary_conditions<Pack>(const_mesh, conditions);
    const SimpleFluid::BoundaryConditionMap pressure_boundaries;

    for (const auto type : {BoundaryType::Slip, BoundaryType::Dirichlet,
                           BoundaryType::NoSlip, BoundaryType::Neumann, BoundaryType::Slip})
    {
        for (auto& [batch_id, values] : boundaries.value)
        {
            boundaries.type[batch_id] = type;
            boundaries.type_by_name[mesh->boundary_batch_name(batch_id)] = type;
            for (auto& value : values)
                value = type == BoundaryType::NoSlip ? Vec{} : prescribed_velocity;
        }
        SimpleFluid::FVM::pressure_weighted_face_fluxes(
            velocity, pressure, gradient, 0.13, boundaries, pressure_boundaries, workspace, flux);
        EXPECT_EQ(&workspace.topology_groups(), topology);
        for (const auto face : flux.owned_face_ids())
        {
            if (!mesh->is_boundary_face(face)) continue;
            const auto normal = mesh->face_normal(face);
            const auto expected = type == BoundaryType::Slip || type == BoundaryType::NoSlip
                ? 0.0 : prescribed_velocity.dot(normal) * mesh->face_area(face);
            EXPECT_DOUBLE_EQ(flux.local_value(face), expected) << "face " << face;
        }
    }
}

TEST(StoredPressureFaceFluxCacheTest, BoundaryBatchesResolveCacheOnceAndPreserveLiveValueSlots)
{
    auto mesh = make_mesh();
    Workspace workspace(mesh);
    VectorField velocity(mesh, "batch_velocity");
    ScalarField pressure(mesh, "batch_pressure");
    VectorField gradient(mesh, "batch_gradient");
    FluxField normal(mesh, "batch_normal_flux");
    FluxField flux(mesh, "batch_pressure_flux");
    FluxField expected(mesh, "batch_expected_flux");
    const Vec owner_velocity{0.25, -0.5, 1.25};
    constexpr double owner_pressure = 2.0;
    constexpr double coefficient = 0.13;
    velocity.put_value(owner_velocity);
    pressure.put_value(owner_pressure);
    gradient.put_value({});
    const SimpleFluid::SP<const Mesh> const_mesh = mesh;

    size_t owned_boundary_batches = 0;
    for (const auto& [batch_id, batch] : mesh->boundary_batches())
    {
        bool has_owned_face = false;
        for (const auto face : batch.face_lids)
            has_owned_face = has_owned_face || mesh->is_owned_face(face);
        owned_boundary_batches += has_owned_face;
    }

    // Reuse the topology while removing and restoring cache entries. Distinct
    // values in every original local batch slot detect accidental compaction
    // of owned faces away from the cache's owned/overlap slot numbering.
    for (const int round : {0, 1, 2})
    {
        SimpleFluid::BoundaryConditionSet conditions;
        conditions.velocity["xmin"] = {BoundaryType::Slip, {}};
        conditions.velocity["xmax"] = {BoundaryType::Dirichlet, {}};
        conditions.velocity["ymin"] = {BoundaryType::Dirichlet, {}};
        conditions.velocity["ymax"] = {BoundaryType::Neumann, {}};
        conditions.velocity["zmin"] = {BoundaryType::Neumann, {}};
        conditions.velocity["zmax"] = {BoundaryType::NoSlip, {}};
        auto boundaries = SimpleFluid::FVM::cache_velocity_boundary_conditions<Pack>(const_mesh, conditions);
        for (const auto& [batch_id, batch] : mesh->boundary_batches())
        {
            const auto& name = mesh->boundary_batch_name(batch_id);
            auto& values = boundaries.value.at(batch_id);
            for (size_t slot = 0; slot < values.size(); ++slot)
            {
                const auto distinct = 0.125 * static_cast<double>(slot + 1) + 0.25 * round;
                values[slot] = name == "zmax" ? Vec{} : Vec{distinct, -2.0 * distinct, 3.0 * distinct};
            }
            if (round == 1)
            {
                if (name == "ymin") boundaries.type.erase(batch_id);
                if (name == "xmax" || name == "ymax") boundaries.value.erase(batch_id);
                if (name == "ymin" || name == "zmin") values.resize(values.size() / 2);
                // Slip still requires an available slot, even though its
                // numerical value comes from the owner-cell projection.
                if (name == "xmin") values.resize(values.size() / 2);
            }
        }

        SimpleFluid::BoundaryConditionMap pressure_boundaries;
        if (round == 1)
        {
            // Dirichlet pressure overrides missing/short velocity values.
            pressure_boundaries["ymax"] = {BoundaryType::Dirichlet, owner_pressure + 0.75};
            pressure_boundaries["zmin"] = {BoundaryType::Dirichlet, owner_pressure - 0.5};
        }
        SimpleFluid::FVM::face_fluxes(velocity, boundaries, normal);
        normal.sync_ghosts();
        for (const auto face : expected.owned_face_ids())
            expected.set_owned_value(face, normal.local_value(face));
        for (const auto& [batch_id, batch] : mesh->boundary_batches())
        {
            const auto condition = pressure_boundaries.find(mesh->boundary_batch_name(batch_id));
            if (condition == pressure_boundaries.end()) continue;
            for (const auto face : batch.face_lids)
            {
                if (!mesh->is_owned_face(face)) continue;
                const auto owner = mesh->owner_cell(face);
                if (static_cast<size_t>(owner) >= mesh->num_owned_cells()) continue;
                const auto distance = (mesh->face_centroid(face) - mesh->cell_centroid(owner)).norm();
                const auto pressure_drop = static_cast<double>(condition->second.value) - owner_pressure;
                expected.set_owned_value(face,
                    (owner_velocity.dot(mesh->face_normal(face)) - coefficient * pressure_drop / distance) *
                    mesh->face_area(face));
            }
        }
        expected.sync_ghosts();

        CountingVelocityBoundaryCache counted(boundaries);
        flux.put_value(173.0);
        SimpleFluid::FVM::detail::pressure_weighted_stored_face_fluxes_impl<Pack>(
            velocity, pressure, coefficient, counted, &pressure_boundaries, workspace, &gradient, flux);
        EXPECT_EQ(counted.value.lookups, owned_boundary_batches) << "round " << round;
        EXPECT_EQ(counted.type.lookups, owned_boundary_batches) << "round " << round;
        for (size_t local = 0; local < mesh->num_faces(); ++local)
        {
            const auto face = static_cast<Pack::local_ordinal_type>(local);
            EXPECT_NEAR(flux.local_value(face), expected.local_value(face), 3.0e-13)
                << "round " << round << ", face " << face;
        }
    }
}

TEST(StoredPressureFaceFluxCacheTest, FullyPeriodicFacesUseOnlyTheInteriorKernel)
{
    auto mesh = std::make_shared<Mesh>(std::make_shared<Mesh::Cartesian>(
        SimpleFluid::Vec3D<SimpleFluid::ArrReal>{{{0, 1, 2, 3}, {0, 1, 2, 3}, {0, 1, 2, 3}}},
        SimpleFluid::Vec3D<bool>{{true, true, true}}));
    Workspace workspace(mesh);
    VectorField velocity(mesh, "periodic_velocity");
    ScalarField pressure(mesh, "periodic_pressure");
    VectorField gradient(mesh, "periodic_gradient");
    FluxField flux(mesh, "periodic_flux");
    const Vec uniform_velocity{0.25, -0.5, 1.25};
    velocity.put_value(uniform_velocity);
    pressure.put_value(2.0);
    gradient.put_value({});
    const SimpleFluid::SP<const Mesh> const_mesh = mesh;
    CountingVelocityBoundaryCache boundaries(SimpleFluid::FVM::cache_velocity_boundary_conditions<Pack>(
        const_mesh, SimpleFluid::BoundaryConditionSet{}));
    const SimpleFluid::BoundaryConditionMap pressure_boundaries;
    SimpleFluid::FVM::detail::pressure_weighted_stored_face_fluxes_impl<Pack>(
        velocity, pressure, 0.13, boundaries, &pressure_boundaries, workspace, &gradient, flux);
    EXPECT_EQ(boundaries.value.lookups, 0U);
    EXPECT_EQ(boundaries.type.lookups, 0U);
    for (size_t local = 0; local < mesh->num_faces(); ++local)
    {
        const auto face = static_cast<Pack::local_ordinal_type>(local);
        EXPECT_FALSE(mesh->is_geometry_exterior_face(face));
        EXPECT_DOUBLE_EQ(flux.local_value(face),
            uniform_velocity.dot(mesh->face_normal(face)) * mesh->face_area(face));
    }
}

TEST(StoredPressureFaceFluxCacheTest, UntaggedExteriorOverwritesOutputWithoutBoundaryLookups)
{
    if (Tpetra::getDefaultComm()->getSize() != 1)
        GTEST_SKIP() << "Serial unstructured fixture with untagged exterior faces.";

    using Geometry = SimpleFluid::Meshes::UnstructuredMesh;
    auto geometry = std::make_shared<Geometry>(
        SimpleFluid::Arr<Geometry::Vec3>{{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0},
                                       {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1},
                                       {2, 0, 0}, {2, 1, 0}, {2, 0, 1}, {2, 1, 1}},
        SimpleFluid::Arr<Geometry::CellDefinition>{
            {Geometry::CellType::HEXAHEDRON, {0, 1, 2, 3, 4, 5, 6, 7}},
            {Geometry::CellType::HEXAHEDRON, {1, 8, 9, 2, 5, 10, 11, 6}}});
    auto mesh = std::make_shared<Mesh>(geometry);
    Workspace workspace(mesh);
    VectorField velocity(mesh, "untagged_velocity");
    ScalarField pressure(mesh, "untagged_pressure");
    VectorField gradient(mesh, "untagged_gradient");
    FluxField flux(mesh, "untagged_flux");
    const Vec uniform_velocity{0.25, -0.5, 1.25};
    velocity.put_value(uniform_velocity);
    pressure.put_value(2.0);
    gradient.put_value({});
    flux.put_value(173.0);
    const SimpleFluid::SP<const Mesh> const_mesh = mesh;
    CountingVelocityBoundaryCache boundaries(SimpleFluid::FVM::cache_velocity_boundary_conditions<Pack>(
        const_mesh, SimpleFluid::BoundaryConditionSet{}));
    const SimpleFluid::BoundaryConditionMap pressure_boundaries;
    SimpleFluid::FVM::detail::pressure_weighted_stored_face_fluxes_impl<Pack>(
        velocity, pressure, 0.13, boundaries, &pressure_boundaries, workspace, &gradient, flux);
    EXPECT_EQ(boundaries.value.lookups, 0U);
    EXPECT_EQ(boundaries.type.lookups, 0U);
    size_t exterior_faces = 0;
    size_t interior_faces = 0;
    for (const auto face : flux.owned_face_ids())
    {
        if (mesh->is_geometry_exterior_face(face))
        {
            ++exterior_faces;
            EXPECT_FALSE(mesh->is_boundary_face(face));
            EXPECT_DOUBLE_EQ(flux.local_value(face), 0.0) << "face " << face;
        }
        else
        {
            ++interior_faces;
            EXPECT_DOUBLE_EQ(flux.local_value(face),
                uniform_velocity.dot(mesh->face_normal(face)) * mesh->face_area(face));
        }
    }
    EXPECT_EQ(exterior_faces, 10U);
    EXPECT_EQ(interior_faces, 1U);
}

TEST(StoredPressureFaceFluxCacheTest, ExecutionPreflightFailureIsCollectiveAndPreservesOutput)
{
    struct FailingWorkspace : Workspace
    {
        using Workspace::Workspace;
        bool reject_acquisition = false;

        auto acquire_execution() const
        {
            if (reject_acquisition)
                throw std::logic_error("Injected pressure execution acquisition failure.");
            return Workspace::acquire_execution();
        }
    };

    auto mesh = make_mesh();
    const auto comm = mesh->owned_cell_map()->getComm();
    FailingWorkspace workspace(mesh);
    VectorField velocity(mesh, "preflight_velocity");
    ScalarField pressure(mesh, "preflight_pressure");
    VectorField gradient(mesh, "preflight_gradient");
    FluxField flux(mesh, "preflight_flux");
    velocity.put_value({});
    pressure.put_value(2.0);
    gradient.put_value({});
    const SimpleFluid::SP<const Mesh> const_mesh = mesh;
    const auto boundaries = SimpleFluid::FVM::cache_velocity_boundary_conditions<Pack>(
        const_mesh, SimpleFluid::BoundaryConditionSet{});
    const SimpleFluid::BoundaryConditionMap pressure_boundaries;
    const auto evaluate = [&]
    {
        SimpleFluid::FVM::detail::pressure_weighted_stored_face_fluxes_impl<Pack>(
            velocity, pressure, 0.13, boundaries, &pressure_boundaries, workspace, &gradient, flux);
    };
    const auto expect_collective_failure = [&]
    {
        flux.put_value(173.0);
        int caught = 0;
        try
        {
            evaluate();
        }
        catch (const std::invalid_argument&)
        {
            caught = 1;
        }
        int total_caught = 0;
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &caught, &total_caught);
        EXPECT_EQ(caught, 1);
        EXPECT_EQ(total_caught, comm->getSize());
        for (size_t local = 0; local < mesh->num_faces(); ++local)
            EXPECT_DOUBLE_EQ(flux.local_value(static_cast<Pack::local_ordinal_type>(local)), 173.0);
    };

    // Rank zero rejects acquisition while peers retain valid executions.
    workspace.reject_acquisition = comm->getRank() == 0;
    expect_collective_failure();
    workspace.reject_acquisition = false;
    EXPECT_NO_THROW(evaluate());
    for (size_t local = 0; local < mesh->num_faces(); ++local)
        EXPECT_DOUBLE_EQ(flux.local_value(static_cast<Pack::local_ordinal_type>(local)), 0.0);

    // A genuinely stale snapshot on one rank must take the same collective
    // failure path, before peers synchronize fields or write any output.
    SimpleFluid::PlanarALEMeshMotion<Pack> motion(mesh);
    motion.begin_trial(6.0, 1.0);
    if (comm->getRank() != 0)
        workspace.refresh_geometry();
    expect_collective_failure();
    workspace.refresh_geometry();
    EXPECT_NO_THROW(evaluate());
    for (size_t local = 0; local < mesh->num_faces(); ++local)
        EXPECT_DOUBLE_EQ(flux.local_value(static_cast<Pack::local_ordinal_type>(local)), 0.0);
    EXPECT_FALSE(workspace.has_materialized_face_geometry());
}

TEST(StoredPressureFaceFluxCacheTest, MotionAndRollbackRequireRefreshWithPrecomputedGradient)
{
    auto mesh = make_mesh();
    Workspace workspace(mesh);
    const auto* topology = &workspace.topology_groups();
    const auto original = check_quadratic_flux(mesh, workspace, 4.0, 0.0);
    SimpleFluid::PlanarALEMeshMotion<Pack> motion(mesh);
    motion.begin_trial(6.0, 1.0);
    EXPECT_THROW(static_cast<void>(workspace.face_geometry()), std::invalid_argument);
    EXPECT_THROW(check_quadratic_flux(mesh, workspace, 6.0, 0.0), std::invalid_argument);
    workspace.refresh_geometry();
    EXPECT_EQ(&workspace.topology_groups(), topology);
    const auto moved = check_quadratic_flux(mesh, workspace, 6.0, 0.0);
    EXPECT_NE(moved, original);
    Workspace fresh(mesh);
    EXPECT_EQ(moved, check_quadratic_flux(mesh, fresh, 6.0, 0.0));

    motion.rollback_trial();
    EXPECT_THROW(static_cast<void>(workspace.face_geometry()), std::invalid_argument);
    workspace.refresh_geometry();
    EXPECT_EQ(&workspace.topology_groups(), topology);
    EXPECT_EQ(check_quadratic_flux(mesh, workspace, 4.0, 0.0), original);
}

TEST(StoredPressureFaceFluxCacheTest, SharedGeometryKeepsScratchAndBoundaryReconstructionIndependent)
{
    auto mesh = make_mesh();
    auto first = std::make_unique<Workspace>(mesh);
    Workspace second(mesh, first->shared_geometry());
    Workspace fresh(mesh);
    EXPECT_EQ(second.shared_geometry(), first->shared_geometry());
    EXPECT_EQ(&second.gradient_cache(), &first->gradient_cache());
    EXPECT_FALSE(second.has_materialized_face_geometry());
    EXPECT_FALSE(first->has_materialized_face_geometry());
    EXPECT_EQ(&second.topology_groups(), &first->topology_groups());
    EXPECT_NE(&second.pressure_gradient(), &first->pressure_gradient());

    const auto first_flux = reconstruct_flux(mesh, *first, 1.0, 0.0);
    const auto first_gradient = gradient_values(*first);
    const auto second_flux = reconstruct_flux(mesh, second, 2.0, 0.75);
    EXPECT_EQ(second_flux, reconstruct_flux(mesh, fresh, 2.0, 0.75));
    EXPECT_NE(second_flux, first_flux);
    EXPECT_NE(gradient_values(second), first_gradient);
    EXPECT_EQ(gradient_values(*first), first_gradient);
    EXPECT_EQ(reconstruct_flux(mesh, *first, 1.0, 0.0), first_flux);

    EXPECT_FALSE(first->has_materialized_face_geometry());
    EXPECT_FALSE(second.has_materialized_face_geometry());
    EXPECT_FALSE(fresh.has_materialized_face_geometry());
    const auto* ordered_geometry = first->face_geometry().data();
    EXPECT_TRUE(first->has_materialized_face_geometry());
    EXPECT_TRUE(second.has_materialized_face_geometry());
    EXPECT_EQ(second.face_geometry().data(), ordered_geometry);

    // The shared snapshot owns its mesh and survives the creating workspace.
    first.reset();
    EXPECT_EQ(reconstruct_flux(mesh, second, 2.0, 0.75), second_flux);
}

TEST(StoredPressureFaceFluxCacheTest, RefreshReplacesSharedSnapshotAndRejectsForeignOrStaleGeometry)
{
    auto mesh = make_mesh();
    Workspace workspace(mesh);
    const auto original_geometry = workspace.shared_geometry();
    const auto* original_topology = &workspace.topology_groups();
    workspace.refresh_geometry();
    EXPECT_EQ(workspace.shared_geometry(), original_geometry);
    Workspace retained(mesh, original_geometry);
    const auto original_flux = check_quadratic_flux(mesh, workspace, 4.0, 0.0);
    const auto foreign = make_mesh();
    EXPECT_THROW(static_cast<void>(Workspace(foreign, original_geometry)), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(Workspace(mesh, {})), std::invalid_argument);

    SimpleFluid::PlanarALEMeshMotion<Pack> motion(mesh);
    motion.begin_trial(6.0, 1.0);
    EXPECT_THROW(static_cast<void>(Workspace(mesh, original_geometry)), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(retained.face_geometry()), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(retained.boundary_locations()), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(retained.gradient_cache().interior_geometry()), std::invalid_argument);
    workspace.refresh_geometry();
    EXPECT_NE(workspace.shared_geometry(), original_geometry);
    EXPECT_EQ(&workspace.topology_groups(), original_topology);
    EXPECT_FALSE(workspace.has_materialized_face_geometry());
    const auto moved_geometry = workspace.shared_geometry();
    workspace.refresh_geometry();
    EXPECT_EQ(workspace.shared_geometry(), moved_geometry);
    EXPECT_EQ(retained.shared_geometry(), original_geometry);
    EXPECT_THROW(static_cast<void>(retained.face_geometry()), std::invalid_argument);
    Workspace current(mesh, workspace.shared_geometry());
    EXPECT_THROW(retained.refresh_geometry(original_geometry), std::invalid_argument);
    EXPECT_THROW(retained.refresh_geometry(Workspace(foreign).shared_geometry()), std::invalid_argument);
    retained.refresh_geometry(moved_geometry);
    EXPECT_EQ(retained.shared_geometry(), moved_geometry);
    EXPECT_EQ(&retained.topology_groups(), original_topology);
    EXPECT_NE(&retained.pressure_gradient(), &workspace.pressure_gradient());
    EXPECT_EQ(check_quadratic_flux(mesh, workspace, 6.0, 0.0), check_quadratic_flux(mesh, current, 6.0, 0.0));

    motion.rollback_trial();
    EXPECT_THROW(static_cast<void>(current.face_geometry()), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(Workspace(mesh, original_geometry)), std::invalid_argument);
    retained.refresh_geometry();
    EXPECT_NE(retained.shared_geometry(), original_geometry);
    EXPECT_EQ(&retained.topology_groups(), original_topology);
    EXPECT_FALSE(retained.has_materialized_face_geometry());
    EXPECT_EQ(check_quadratic_flux(mesh, retained, 4.0, 0.0), original_flux);
    EXPECT_THROW(static_cast<void>(workspace.face_geometry()), std::invalid_argument);
}

TEST(StoredPressureFaceFluxCacheTest, OrderedCompatibilityGeometryIsLazyAndOldSnapshotRemainsImmutable)
{
    auto mesh = make_mesh();
    Workspace workspace(mesh);
    Workspace shared(mesh, workspace.shared_geometry());
    const auto original_snapshot = workspace.shared_geometry();
    const auto* original_topology = &workspace.topology_groups();
    EXPECT_FALSE(workspace.has_materialized_face_geometry());
    EXPECT_FALSE(shared.has_materialized_face_geometry());
    const auto& original = workspace.face_geometry();
    const auto original_copy = original;
    EXPECT_TRUE(workspace.has_materialized_face_geometry());
    EXPECT_TRUE(shared.has_materialized_face_geometry());
    EXPECT_EQ(original.data(), shared.face_geometry().data());
    std::vector<Pack::local_ordinal_type> expected_ids;
    for (size_t face = 0; face < mesh->num_faces(); ++face)
        if (mesh->is_owned_face(static_cast<Pack::local_ordinal_type>(face)))
            expected_ids.push_back(static_cast<Pack::local_ordinal_type>(face));
    std::vector<Pack::local_ordinal_type> actual_ids;
    for (const auto& face : original) actual_ids.push_back(face.face);
    EXPECT_EQ(actual_ids, expected_ids);

    SimpleFluid::PlanarALEMeshMotion<Pack> motion(mesh);
    motion.begin_trial(6.0, 1.0);
    workspace.refresh_geometry();
    EXPECT_NE(workspace.shared_geometry(), original_snapshot);
    EXPECT_EQ(&workspace.topology_groups(), original_topology);
    EXPECT_FALSE(workspace.has_materialized_face_geometry());
    EXPECT_TRUE(shared.has_materialized_face_geometry());
    EXPECT_THROW(static_cast<void>(shared.acquire_execution()), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(shared.face_geometry()), std::invalid_argument);
    const auto& moved = workspace.face_geometry();
    EXPECT_NE(moved.data(), original.data());
    ASSERT_EQ(moved.size(), original.size());
    bool changed_area = false;
    for (size_t face = 0; face < original.size(); ++face)
    {
        EXPECT_EQ(original[face].face, original_copy[face].face);
        EXPECT_DOUBLE_EQ(original[face].area, original_copy[face].area);
        EXPECT_DOUBLE_EQ(original[face].center_projection, original_copy[face].center_projection);
        EXPECT_DOUBLE_EQ(original[face].distance_squared, original_copy[face].distance_squared);
        EXPECT_DOUBLE_EQ(original[face].boundary_diffusion, original_copy[face].boundary_diffusion);
        EXPECT_EQ(moved[face].face, original[face].face);
        changed_area = changed_area || moved[face].area != original[face].area;
    }
    EXPECT_TRUE(changed_area);

    motion.rollback_trial();
    workspace.refresh_geometry();
    EXPECT_EQ(&workspace.topology_groups(), original_topology);
    EXPECT_FALSE(workspace.has_materialized_face_geometry());
    const auto& rolled_back = workspace.face_geometry();
    ASSERT_EQ(rolled_back.size(), original_copy.size());
    for (size_t face = 0; face < rolled_back.size(); ++face)
    {
        EXPECT_EQ(rolled_back[face].face, original_copy[face].face);
        EXPECT_DOUBLE_EQ(rolled_back[face].area, original_copy[face].area);
        EXPECT_DOUBLE_EQ(rolled_back[face].center_projection, original_copy[face].center_projection);
        EXPECT_DOUBLE_EQ(rolled_back[face].distance_squared, original_copy[face].distance_squared);
        EXPECT_DOUBLE_EQ(rolled_back[face].boundary_diffusion, original_copy[face].boundary_diffusion);
    }
}

TEST(StoredPressureFaceFluxCacheTest, ExecutionRetainsSnapshotAndCompositeGeometryLease)
{
    auto mesh = std::make_shared<Mesh>(SimpleFluid::test::two_regions());
    auto workspace = std::make_unique<Workspace>(mesh);
    std::weak_ptr<const Workspace::SharedGeometry> retained_snapshot = workspace->shared_geometry();
    SimpleFluid::PlanarALEMeshMotion<Pack> motion(mesh);
    Workspace::shared_geometry_type snapshot;
    {
        const auto execution = workspace->acquire_execution();
        EXPECT_THROW(motion.begin_trial(2.0, 0.1), std::logic_error);
        EXPECT_FALSE(motion.has_active_trial());
        workspace.reset();
        EXPECT_FALSE(retained_snapshot.expired());
        size_t interior = 0;
        execution.interior().apply([&](const auto& topology, const auto& metrics)
        {
            ++interior;
            EXPECT_FALSE(mesh->is_geometry_exterior_face(topology.face));
            EXPECT_DOUBLE_EQ(metrics.area, mesh->face_area(topology.face));
        });
        EXPECT_GT(interior, 0U);
        snapshot = retained_snapshot.lock();
    }
    EXPECT_NO_THROW(motion.begin_trial(2.0, 0.1));
    motion.rollback_trial();
    // Sharing a previous snapshot still rejects the advanced geometry epoch,
    // including a rollback to numerically identical coordinates.
    EXPECT_THROW(static_cast<void>(Workspace(mesh, snapshot)), std::invalid_argument);
}

TEST(StoredPressureFaceFluxCacheTest, ExplicitTopologyInvalidationRejectsSharedSnapshots)
{
    auto mesh = make_mesh();
    Workspace workspace(mesh);
    Workspace shared(mesh, workspace.shared_geometry());
    const auto snapshot = workspace.shared_geometry();
    const auto& legacy_plan = workspace.traverse_plan();
    workspace.invalidate_topology();
    EXPECT_THROW(static_cast<void>(legacy_plan.acquire()), std::logic_error);
    EXPECT_THROW(static_cast<void>(workspace.acquire_execution()), std::logic_error);
    EXPECT_THROW(static_cast<void>(shared.acquire_execution()), std::logic_error);
    EXPECT_THROW(workspace.refresh_geometry(), std::logic_error);
    EXPECT_THROW(shared.refresh_geometry(), std::logic_error);
    EXPECT_THROW(static_cast<void>(Workspace(mesh, snapshot)), std::logic_error);
    Workspace rebuilt(mesh);
    EXPECT_NO_THROW(static_cast<void>(rebuilt.acquire_execution()));
    EXPECT_FALSE(rebuilt.has_materialized_face_geometry());
}
