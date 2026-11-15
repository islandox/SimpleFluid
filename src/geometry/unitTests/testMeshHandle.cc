/**
 * @file testMeshHandle.cc
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Tests for the variant-backed mesh runtime.
 * @version 0.1
 * @date 2026-06-21
 *
 * @copyright Copyright (c) 2026
 *
 */

#include <gtest/gtest.h>

#include "geometry/MeshHandle.hh"
#include "geometry/unitTests/fixtures/LegacySTKFixture.hh"
#include "geometry/mesh/PartitionedMeshBase.hh"
#include "geometry/unitTests/test_mesh_helpers.hh"
#include "utils/testing_environment.hh"

#include <Teuchos_CommHelpers.hpp>
#include <Teuchos_DefaultMpiComm.hpp>
#include <stk_mesh/base/FEMHelpers.hpp>
#include <stk_mesh/base/Field.hpp>

#include <algorithm>
#include <array>
#include <exception>
#include <filesystem>
#include <memory>
#include <map>
#include <numbers>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace
{

using Pack = SimpleFluid::DefaultTpetraTypes;
using Handle = SimpleFluid::MeshHandle<Pack>;
using Cartesian = SimpleFluid::Meshes::OrthogonalCartesian3D;
using Cylindrical = SimpleFluid::Meshes::OrthogonalCylindrial3D;
using SemiStructured = SimpleFluid::Meshes::SemiStructuredXY_Z;
using Unstructured = SimpleFluid::Meshes::UnstructuredMesh;
using ConcreteSTK = SimpleFluid::test::LegacySTKFixture<Pack>;
using PartitionedUnstructured =
    SimpleFluid::Meshes::PartitionedMesh<Unstructured, Pack>;

using utils_test::KokkosEnvironment;
testing::Environment* const kokkos_environment =
    testing::AddGlobalTestEnvironment(new KokkosEnvironment);

/** Build the frozen derived-class oracle independently of the native factory. */
SimpleFluid::SP<ConcreteSTK> make_frozen_two_hex_fixture()
{
    auto mesh = std::make_shared<ConcreteSTK>();
    const auto comm = Tpetra::getDefaultComm();
    auto meta = mesh->meta();
    auto bulk = mesh->bulk();
    auto& coordinates = meta->declare_field<double>(stk::topology::NODE_RANK, "coordinates");
    stk::mesh::put_field_on_mesh(coordinates, meta->universal_part(), 3, nullptr);
    meta->set_coordinate_field(&coordinates);
    auto& hexes = meta->declare_part_with_topology("hexes", stk::topology::HEX_8);
    const std::array<const char*, 6> names{"xmin", "xmax", "ymin", "ymax", "zmin", "zmax"};
    std::array<stk::mesh::Part*, 6> boundaries{};
    for (size_t b = 0; b < boundaries.size(); ++b)
        boundaries[b] = &meta->declare_part(names[b], meta->side_rank());
    meta->commit();
    std::exception_ptr error;
    bulk->modification_begin();
    if (comm->getRank() == 0)
    {
        try
        {
            const auto first = stk::mesh::declare_element(*bulk, hexes, 1, {1, 2, 3, 4, 5, 6, 7, 8});
            const auto second = stk::mesh::declare_element(*bulk, hexes, 2, {2, 9, 10, 3, 6, 11, 12, 7});
            const auto side = [&](stk::mesh::Entity cell, unsigned ordinal, size_t boundary)
            { bulk->declare_element_side(cell, ordinal, stk::mesh::PartVector{boundaries[boundary]}); };
            side(first, 3, 0);
            side(second, 1, 1);
            for (const auto cell : {first, second})
            {
                side(cell, 0, 2);
                side(cell, 2, 3);
                side(cell, 4, 4);
                side(cell, 5, 5);
            }
            const std::array<std::array<double, 3>, 12> points{{{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}, {0, 0, 1},
                {1, 0, 1}, {1, 1, 1}, {0, 1, 1}, {2, 0, 0}, {2, 1, 0}, {2, 0, 1}, {2, 1, 1}}};
            for (size_t n = 0; n < points.size(); ++n)
            {
                const auto node = bulk->get_entity(stk::topology::NODE_RANK, n + 1);
                auto* values = stk::mesh::field_data(coordinates, node);
                if (!values)
                    throw std::runtime_error("Frozen two-HEX fixture has no coordinate storage.");
                std::copy(points[n].begin(), points[n].end(), values);
            }
        }
        catch (...)
        {
            error = std::current_exception();
        }
    }
    bulk->modification_end();
    const int failed = error ? 1 : 0;
    int any_failed = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &failed, &any_failed);
    if (any_failed)
    {
        if (error)
            std::rethrow_exception(error);
        throw std::runtime_error("Frozen two-HEX fixture population failed on rank zero.");
    }
    mesh->assemble();
    return mesh;
}

} // namespace

/**
 * @brief Verifies variant visitation dispatches correctly for every concrete
 * mesh alternative supported by MeshHandle.
 */
TEST(MeshHandleTest, VisitsEveryConcreteMeshAlternative)
{
    const auto cartesian = std::make_shared<Cartesian>(
        SimpleFluid::Vec3D<SimpleFluid::ArrReal>{{
            {0.0, 1.0, 2.0},
            {0.0, 1.0},
            {0.0, 1.0}}});
    const auto cylindrical = std::make_shared<Cylindrical>(
        SimpleFluid::Vec3D<SimpleFluid::ArrReal>{{
            {1.0, 2.0},
            {0.0, std::numbers::pi},
            {0.0, 1.0}}});
    const auto semi_structured = std::make_shared<SemiStructured>(
        SimpleFluid::Arr<SemiStructured::Vec3>{
            {0.0, 0.0, 0.0},
            {1.0, 0.0, 0.0},
            {0.0, 1.0, 0.0}},
        SimpleFluid::Arr<SimpleFluid::Arr<unsigned>>{{0, 1, 2}},
        SimpleFluid::ArrReal{0.0, 1.0});
    const auto unstructured = std::make_shared<Unstructured>(
        SimpleFluid::Arr<Unstructured::Vec3>{
            {0.0, 0.0, 0.0},
            {1.0, 0.0, 0.0},
            {1.0, 1.0, 0.0},
            {0.0, 1.0, 0.0},
            {0.0, 0.0, 1.0},
            {1.0, 0.0, 1.0},
            {1.0, 1.0, 1.0},
            {0.0, 1.0, 1.0}},
        SimpleFluid::Arr<Unstructured::CellDefinition>{
            {Unstructured::CellType::HEXAHEDRON,
             {0, 1, 2, 3, 4, 5, 6, 7}}},
        SimpleFluid::Arr<Unstructured::BoundaryFaceDefinition>{
            {{3, 7, 4, 0}, 6, "xmin"}});

    const Handle cartesian_handle(cartesian);
    const Handle cylindrical_handle(cylindrical);
    const Handle semi_structured_handle(semi_structured);
    const Handle unstructured_handle(unstructured);

    EXPECT_EQ(
        cartesian_handle.visit(
            [](const auto& mesh) { return mesh.num_cells(); }),
        cartesian->num_cells());
    EXPECT_EQ(cylindrical_handle.num_owned_cells(), 1U);
    EXPECT_EQ(semi_structured_handle.num_owned_cells(), 1U);
    EXPECT_EQ(unstructured_handle.num_owned_cells(), 1U);

    for (const auto* handle :
         {&cartesian_handle, &cylindrical_handle,
          &semi_structured_handle, &unstructured_handle})
    {
        EXPECT_TRUE(handle->has_mutable_geometry());
        EXPECT_EQ(
            handle->indexer().num_owned_cells(),
            handle->num_owned_cells());
        EXPECT_EQ(
            handle->indexer().num_local_faces(),
            handle->num_faces());
        EXPECT_EQ(
            handle->owned_cell_map()->getLocalNumElements(),
            handle->num_owned_cells());
        EXPECT_EQ(
            handle->overlap_face_map()->getLocalNumElements(),
            handle->num_faces());
        EXPECT_FALSE(handle->boundary_batches().empty());
        const auto topology = handle->vtu_topology();
        ASSERT_TRUE(topology);
        EXPECT_EQ(
            topology->cell_offsets.size(),
            handle->num_owned_cells());
        EXPECT_FALSE(topology->points.empty());
    }

    EXPECT_EQ(
        unstructured_handle.indexer().num_local_nodes(),
        unstructured->num_nodes());
    for (size_t local = 0;
         local < unstructured_handle.num_local_cells();
         ++local)
    {
        const auto local_id = static_cast<Pack::local_ordinal_type>(local);
        const auto global_id =
            unstructured_handle.indexer().cell_global_id(local_id);
        EXPECT_EQ(
            unstructured_handle.indexer().cell_local_id(global_id),
            local_id);
    }
}

/**
 * @brief Mutable-backed handles expose the original geometry while const
 *        visitation remains deeply read-only.
 */
TEST(MeshHandleTest, RetainsMutableGeometryWithConstObserverVisitation)
{
    static_assert(std::is_same_v<
                  Handle::MutableCartesianPtr,
                  SimpleFluid::SP<Cartesian>>);
    static_assert(std::is_same_v<
                  Handle::MutableCylindricalPtr,
                  SimpleFluid::SP<Cylindrical>>);
    static_assert(std::is_same_v<
                  Handle::MutableSemiStructuredPtr,
                  SimpleFluid::SP<SemiStructured>>);
    static_assert(std::is_same_v<
                  Handle::MutableUnstructuredPtr,
                  SimpleFluid::SP<Unstructured>>);
    static_assert(!std::is_constructible_v<Handle, SimpleFluid::SP<ConcreteSTK>>);
    static_assert(!std::is_constructible_v<Handle, SimpleFluid::SP<const ConcreteSTK>>);
    static_assert(std::is_constructible_v<
                  PartitionedUnstructured,
                  std::nullptr_t,
                  Handle::unstructured_indexer_type>);

    auto cartesian = std::make_shared<Cartesian>(
        SimpleFluid::Vec3D<SimpleFluid::ArrReal>{{
            {0.0, 1.0},
            {0.0, 1.0},
            {0.0, 1.0}}});
    Handle handle(cartesian);

    ASSERT_TRUE(handle.has_mutable_geometry());
    auto* mutable_address = handle.visit_mutable(
        []<class Handle>(Handle& mesh) -> void*
        {
            static_assert(!std::is_const_v<Handle>);
            return std::addressof(mesh);
        });
    EXPECT_EQ(mutable_address, cartesian.get());

    const auto* const_address = std::as_const(handle).visit(
        []<class Handle>(Handle& mesh) -> const void*
        {
            static_assert(std::is_const_v<Handle>);
            return std::addressof(mesh);
        });
    EXPECT_EQ(const_address, cartesian.get());

    Handle::CartesianPtr read_only_geometry = cartesian;
    Handle read_only_handle(std::move(read_only_geometry));
    EXPECT_FALSE(read_only_handle.has_mutable_geometry());
    EXPECT_THROW(
        read_only_handle.visit_mutable([](auto&) {}),
        std::logic_error);
    EXPECT_THROW(
        static_cast<void>(Handle(Handle::UnstructuredPtr{}, Handle::unstructured_indexer_type{})),
        std::invalid_argument);
    EXPECT_THROW(
        static_cast<void>(PartitionedUnstructured(
            nullptr, Handle::unstructured_indexer_type{})),
        std::invalid_argument);
}

/** @brief Partition adaptation preserves mutable ownership and const opt-out. */
TEST(MeshHandleTest, RetainsMutablePartitionGeometry)
{
    const auto mesh = SimpleFluid::test::make_unstructured_hex_line(1);
    using Indexer = Handle::unstructured_indexer_type;
    std::vector<Indexer::cell_id_t> owned_cells;
    std::vector<Indexer::face_id_t> owned_faces;
    std::vector<Indexer::node_id_t> nodes;
    for (size_t cell = 0; cell < mesh->num_cells(); ++cell)
    {
        owned_cells.push_back(mesh->cell_id(cell));
    }
    for (size_t face = 0; face < mesh->num_faces(); ++face)
    {
        owned_faces.push_back(mesh->face_id(face));
    }
    for (size_t node = 0; node < mesh->num_nodes(); ++node)
    {
        nodes.push_back(mesh->node_id(node));
    }

    const auto partitioned = std::make_shared<PartitionedUnstructured>(
        mesh,
        Indexer(std::move(owned_cells), {}, std::move(owned_faces), {},
            std::move(nodes)));
    Handle mutable_handle(partitioned);
    EXPECT_TRUE(mutable_handle.has_mutable_geometry());
    EXPECT_EQ(
        mutable_handle.visit_mutable(
            [](auto& geometry) -> void*
            {
                return std::addressof(geometry);
            }),
        mesh.get());

    SimpleFluid::SP<const PartitionedUnstructured> read_only_partition =
        partitioned;
    EXPECT_FALSE(Handle(read_only_partition).has_mutable_geometry());
}

/** @brief The preserved one-argument constructor builds default native MPI ownership. */
TEST(MeshHandleTest, SemiStructuredConstructionPreservesDefaultDistributedOwnership)
{
    const auto communicator = Tpetra::getDefaultComm();
    SimpleFluid::ArrReal z_edges;
    for (int layer = 0; layer <= communicator->getSize(); ++layer)
        z_edges.push_back(0.5 * layer);
    const auto mesh = std::make_shared<SemiStructured>(
        SimpleFluid::Arr<SemiStructured::Vec3>{
            {0.0, 0.0, 0.0},
            {1.0, 0.0, 0.0},
            {0.0, 1.0, 0.0}},
        SimpleFluid::Arr<SimpleFluid::Arr<unsigned>>{{0, 1, 2}},
        z_edges);
    const Handle handle(mesh);
    EXPECT_EQ(handle.num_owned_cells(), 1U);
    EXPECT_EQ(handle.cell_global_id(0), communicator->getRank());
    EXPECT_EQ(handle.cell_geometry_global_id(0), communicator->getRank());
    EXPECT_TRUE(handle.owned_cell_map()->isContiguous());
    EXPECT_TRUE(handle.owned_face_map()->isContiguous());
    EXPECT_TRUE(handle.owned_cell_map()->isOneToOne());
    EXPECT_TRUE(handle.owned_face_map()->isOneToOne());
    EXPECT_EQ(handle.owned_cell_map()->getGlobalNumElements(), static_cast<std::size_t>(communicator->getSize()));
    EXPECT_EQ(handle.visit([](const auto& geometry) -> const void* { return &geometry; }), mesh.get());
}

/** @brief Existing partitions retain their non-default communicator. */
TEST(MeshHandleTest, PreservesPartitionedMeshCommunicator)
{
    const auto world = Tpetra::getDefaultComm();
    if (world->getSize() < 2)
    {
        GTEST_SKIP() << "Requires at least two MPI ranks.";
    }

    const auto self = Teuchos::rcp(
        new Teuchos::MpiComm<int>(MPI_COMM_SELF));
    const auto mesh = SimpleFluid::test::make_unstructured_hex_line(1);
    using Indexer = Handle::unstructured_indexer_type;
    std::vector<Indexer::cell_id_t> owned_cells;
    std::vector<Indexer::face_id_t> owned_faces;
    std::vector<Indexer::node_id_t> nodes;
    const auto original_cell_base = static_cast<Pack::global_ordinal_type>(1000 * (world->getRank() + 1));
    const auto original_face_base = static_cast<Pack::global_ordinal_type>(2000 * (world->getRank() + 1));
    const auto original_node_base = static_cast<Pack::global_ordinal_type>(3000 * (world->getRank() + 1));
    for (size_t cell = 0; cell < mesh->num_cells(); ++cell)
    {
        owned_cells.push_back(static_cast<Indexer::cell_id_t>(original_cell_base + cell));
    }
    for (size_t face = 0; face < mesh->num_faces(); ++face)
    {
        owned_faces.push_back(static_cast<Indexer::face_id_t>(original_face_base + face));
    }
    for (size_t node = 0; node < mesh->num_nodes(); ++node)
    {
        nodes.push_back(static_cast<Indexer::node_id_t>(original_node_base + node));
    }

    const auto partitioned = std::make_shared<PartitionedUnstructured>(
        mesh,
        Indexer(std::move(owned_cells), {}, std::move(owned_faces), {},
            std::move(nodes)),
        self);
    Handle handle(partitioned);

    EXPECT_EQ(partitioned->mutable_mesh_ptr().get(), mesh.get());
    EXPECT_TRUE(handle.has_mutable_geometry());
    SimpleFluid::SP<const PartitionedUnstructured> read_only_partition =
        partitioned;
    EXPECT_FALSE(Handle(read_only_partition).has_mutable_geometry());
    EXPECT_EQ(handle.owned_cell_map()->getComm()->getSize(), 1);
    EXPECT_EQ(handle.overlap_cell_map()->getComm()->getSize(), 1);
    EXPECT_EQ(handle.owned_face_map()->getComm()->getSize(), 1);
    EXPECT_EQ(handle.overlap_face_map()->getComm()->getSize(), 1);
    const auto& original = partitioned->indexer();
    const auto& exposed = handle.indexer();
    for (std::size_t row = 0; row < handle.num_local_cells(); ++row)
    {
        const auto lid = static_cast<Pack::local_ordinal_type>(row);
        const auto global = static_cast<Pack::global_ordinal_type>(original.cell_global_id(lid));
        EXPECT_EQ(global, original_cell_base + row);
        EXPECT_EQ(handle.cell_global_id(lid), global);
        EXPECT_EQ(handle.cell_geometry_global_id(lid), global);
        EXPECT_EQ(exposed.cell_global_id(lid), global);
        EXPECT_NE(global, static_cast<Pack::global_ordinal_type>(row));
    }
    for (std::size_t row = 0; row < handle.num_faces(); ++row)
    {
        const auto lid = static_cast<Pack::local_ordinal_type>(row);
        const auto global = static_cast<Pack::global_ordinal_type>(original.face_global_id(lid));
        EXPECT_EQ(global, original_face_base + row);
        EXPECT_EQ(handle.face_global_id(lid), global);
        EXPECT_EQ(handle.face_geometry_global_id(lid), global);
        EXPECT_EQ(exposed.face_global_id(lid), global);
    }
}

/** Native factories retain geometry, IDs and maps independently of the frozen STK oracle. */
TEST(MeshHandleTest, NativeFactoryMatchesIndependentSTKGeometryOracle)
{
    const auto oracle = make_frozen_two_hex_fixture();
    const auto handle = SimpleFluid::test::build_mesh<Pack>(SimpleFluid::test::make_two_hex_database());
    ASSERT_TRUE(oracle);
    ASSERT_TRUE(handle);
    ASSERT_TRUE(std::holds_alternative<Handle::UnstructuredPtr>(handle->variant()));
    const auto geometry = std::get<Handle::UnstructuredPtr>(handle->variant());
    EXPECT_TRUE(handle->has_mutable_geometry());
    EXPECT_EQ(handle->visit_mutable([](auto& mesh) -> const void* { return &mesh; }), geometry.get());
    EXPECT_EQ(handle->num_owned_cells(), oracle->num_owned_cells());
    EXPECT_EQ(handle->num_faces(), oracle->num_faces());
    for (size_t cell = 0; cell < handle->num_local_cells(); ++cell)
    {
        const auto lid = static_cast<Pack::local_ordinal_type>(cell);
        const auto source = handle->cell_geometry_global_id(lid);
        const auto reference = oracle->global_to_local_cell(source);
        ASSERT_NE(reference, SimpleFluid::invalid_id<Pack::local_ordinal_type>());
        EXPECT_EQ(handle->cell_centroid(lid), oracle->cell_centroid(reference));
        EXPECT_EQ(handle->cell_volume(lid), oracle->cell_volume(reference));
        EXPECT_EQ(handle->faces(lid).size(), oracle->faces(reference).size());
        EXPECT_EQ(handle->cell_global_id(lid), handle->overlap_cell_map()->getGlobalElement(lid));
        EXPECT_EQ(handle->indexer().cell_local_id(source), lid);
    }
    std::map<Pack::global_ordinal_type, Pack::local_ordinal_type> oracle_faces;
    for (size_t face = 0; face < oracle->num_faces(); ++face)
    {
        const auto lid = static_cast<Pack::local_ordinal_type>(face);
        oracle_faces.emplace(oracle->face_global_id(lid), lid);
    }
    for (size_t face = 0; face < handle->num_faces(); ++face)
    {
        const auto lid = static_cast<Pack::local_ordinal_type>(face);
        const auto found = oracle_faces.find(handle->face_geometry_global_id(lid));
        ASSERT_NE(found, oracle_faces.end());
        const auto reference = found->second;
        EXPECT_EQ(handle->face_centroid(lid), oracle->face_centroid(reference));
        EXPECT_EQ(handle->face_area(lid), oracle->face_area(reference));
        EXPECT_EQ(handle->is_boundary_face(lid), oracle->is_boundary_face(reference));
        if (handle->is_boundary_face(lid))
            EXPECT_EQ(handle->boundary_batch_name(handle->boundary_id(lid)), oracle->boundary_name(reference));
    }
    const auto topology = handle->vtu_topology();
    ASSERT_TRUE(topology);
    EXPECT_EQ(topology->cell_offsets.size(), handle->num_owned_cells());
    EXPECT_FALSE(topology->points.empty());
    EXPECT_EQ(handle->connectivity_storage_bytes(), 0U);
}

/**
 * @brief Verifies repeated connectivity requests reuse the materialized
 * cell-to-face cache when available.
 */
TEST(MeshHandleTest, ComputesCellFaceRangesWithoutMaterialization)
{
    const auto mesh = std::make_shared<Cartesian>(
        SimpleFluid::Vec3D<SimpleFluid::ArrReal>{{
            {0.0, 1.0, 2.0},
            {0.0, 1.0},
            {0.0, 1.0}}});
    const Handle handle(mesh);

    static_assert(std::ranges::random_access_range<decltype(handle.faces(0))>);

    const auto first = handle.faces(0);
    const auto second = handle.faces(0);
    EXPECT_EQ(handle.connectivity_storage_bytes(), 0U);
    EXPECT_FALSE(handle.has_materialized_indexer());
    EXPECT_EQ(first.size(), 6U);
    EXPECT_TRUE(std::ranges::equal(first, second));
#ifdef CHECK_BOUNDS_ENABLED
    EXPECT_THROW(handle.faces(-1), std::out_of_range);
#endif
}

/**
 * @brief Verifies orthogonal slab partitioning constructs consistent owned
 * and ghost entity maps through MeshHandle.
 */
TEST(MeshHandleTest, BuildsOrthogonalSlabOwnershipAndGhostMaps)
{
    const auto mesh = std::make_shared<Cartesian>(
        SimpleFluid::Vec3D<SimpleFluid::ArrReal>{{
            {0.0, 1.0, 2.0, 3.0, 4.0},
            {0.0, 1.0},
            {0.0, 1.0}}});
    const Handle handle(
        mesh,
        Handle::DistributionOptions{
            .ghost_layers = 1,
            .partition = 0,
            .partitions = 2});

    EXPECT_EQ(handle.num_owned_cells(), 2U);
    EXPECT_EQ(handle.num_local_cells(), 3U);
    EXPECT_EQ(handle.owned_cell_map()->getLocalNumElements(), 2U);
    EXPECT_EQ(handle.overlap_cell_map()->getLocalNumElements(), 3U);
    EXPECT_TRUE(handle.is_owned_cell(0));
    EXPECT_FALSE(handle.is_owned_cell(2));
    EXPECT_EQ(handle.indexer().num_local_nodes(), 16U);
    for (size_t local = 0;
         local < handle.indexer().num_local_nodes();
         ++local)
    {
        const auto local_id = static_cast<Pack::local_ordinal_type>(local);
        const auto global_id = handle.indexer().node_global_id(local_id);
        EXPECT_EQ(handle.indexer().node_local_id(global_id), local_id);
    }
}

/**
 * @brief Verifies every CRTP mesh alternative can be exported through the
 * type-erased MeshHandle interface.
 */
TEST(MeshHandleTest, ExportsEveryCRTPMeshAlternative)
{
    const auto cartesian_file = "mesh_handle_cartesian.vtu";
    const auto cylindrical_file = "mesh_handle_cylindrical.vtu";
    const auto semi_structured_file =
        "mesh_handle_semi_structured.vtu";
    const auto unstructured_file =
        "mesh_handle_unstructured.vtu";
    const auto cartesian = std::make_shared<Cartesian>(
        SimpleFluid::Vec3D<SimpleFluid::ArrReal>{{
            {0.0, 1.0},
            {0.0, 1.0},
            {0.0, 1.0}}});
    const auto cylindrical = std::make_shared<Cylindrical>(
        SimpleFluid::Vec3D<SimpleFluid::ArrReal>{{
            {1.0, 2.0},
            {0.0,
             0.5 * std::numbers::pi,
             std::numbers::pi,
             1.5 * std::numbers::pi,
             2.0 * std::numbers::pi},
            {0.0, 1.0}}});
    const auto semi_structured = std::make_shared<SemiStructured>(
        SimpleFluid::Arr<SemiStructured::Vec3>{
            {0.0, 0.0, 0.0},
            {1.0, 0.0, 0.0},
            {0.0, 1.0, 0.0}},
        SimpleFluid::Arr<SimpleFluid::Arr<unsigned>>{{0, 1, 2}},
        SimpleFluid::ArrReal{0.0, 1.0});
    const auto unstructured = std::make_shared<Unstructured>(
        SimpleFluid::Arr<Unstructured::Vec3>{
            {0.0, 0.0, 0.0},
            {1.0, 0.0, 0.0},
            {1.0, 1.0, 0.0},
            {0.0, 1.0, 0.0},
            {0.0, 0.0, 1.0},
            {1.0, 0.0, 1.0},
            {1.0, 1.0, 1.0},
            {0.0, 1.0, 1.0}},
        SimpleFluid::Arr<Unstructured::CellDefinition>{
            {Unstructured::CellType::HEXAHEDRON,
             {0, 1, 2, 3, 4, 5, 6, 7}}});

    Handle(cartesian).export_vtu(cartesian_file);
    Handle(cylindrical).export_vtu(cylindrical_file);
    Handle(semi_structured).export_vtu(semi_structured_file);
    Handle(unstructured).export_vtu(unstructured_file);

    EXPECT_TRUE(std::filesystem::exists(cartesian_file));
    EXPECT_TRUE(std::filesystem::exists(cylindrical_file));
    EXPECT_TRUE(std::filesystem::exists(semi_structured_file));
    EXPECT_TRUE(std::filesystem::exists(unstructured_file));

    std::filesystem::remove(cartesian_file);
    std::filesystem::remove(cylindrical_file);
    std::filesystem::remove(semi_structured_file);
    std::filesystem::remove(unstructured_file);
}

TEST(MeshHandleTest, PolygonalXY_ZExportUsesOrientedPolyhedra)
{
    if (Tpetra::getDefaultComm()->getSize() != 1) GTEST_SKIP() << "Standalone XY-Z geometry is serial.";
    const auto mesh = std::make_shared<SemiStructured>(
        SimpleFluid::Arr<SemiStructured::Vec3>{{0,0,0},{2,0,0},{3,1,0},{1.5,2,0},{0,1,0}},
        SimpleFluid::Arr<SimpleFluid::Arr<unsigned>>{{0,1,2,3,4}}, SimpleFluid::ArrReal{0,1,3});
    const Handle handle(mesh);
    const auto output = handle.vtu_topology();
    EXPECT_EQ(output->cell_types, (SimpleFluid::VTUWriter::UInt8Data{42,42}));
    EXPECT_EQ(output->face_offsets, (SimpleFluid::VTUWriter::Int64Data{38,76}));
    size_t cursor = 0;
    for (size_t c = 0; c < 2; ++c)
    {
        ASSERT_EQ(output->faces.at(cursor++), 7);
        double volume = 0;
        for (size_t f = 0; f < 7; ++f)
        {
            const auto count = static_cast<size_t>(output->faces.at(cursor++));
            EXPECT_EQ(count, f < 2 ? 5U : 4U);
            SimpleFluid::Arr<SemiStructured::Vec3> loop;
            for (size_t i = 0; i < count; ++i)
                loop.push_back(output->points.at(output->faces.at(cursor++)));
            if (f == 0) EXPECT_LT(SimpleFluid::MeshUtils::face_area_vector(loop).z, 0);
            if (f == 1) EXPECT_GT(SimpleFluid::MeshUtils::face_area_vector(loop).z, 0);
            for (size_t i = 1; i + 1 < count; ++i)
                volume += loop[0].dot(loop[i].cross(loop[i + 1])) / 6;
        }
        EXPECT_EQ(output->face_offsets[c], cursor);
        EXPECT_NEAR(volume, mesh->cell_volume(mesh->cell_id(c)), 1e-13);
    }
    const auto path = std::filesystem::temp_directory_path() / "simplefluid_polygonal_xy_z.vtu";
    handle.export_vtu(path.string());
    std::filesystem::remove(path);
}

TEST(MeshHandleTest, ImplicitRangesPreservePeriodicAndGhostTraversal)
{
    const auto visible_native_faces = [](const Handle& mesh, const auto& native_mesh, size_t cell)
    {
        std::vector<Pack::global_ordinal_type> faces;
        const auto native_cell = native_mesh.cell_id(mesh.cell_geometry_global_id(cell));
        for (const auto face : native_mesh.cell_faces(native_cell))
        {
            const auto global = static_cast<Pack::global_ordinal_type>(native_mesh.face_local_id(face));
            if (mesh.overlap_face_map()->getLocalElement(global) != Handle::invalid_local_id())
                faces.push_back(global);
        }
        return faces;
    };
    const auto check_visitation = [](const Handle& mesh)
    {
        for (size_t index = 0; index < mesh.num_local_cells(); ++index)
        {
            const auto cell = static_cast<Pack::local_ordinal_type>(index);
            const auto expected = mesh.faces(cell);
            std::vector<Pack::local_ordinal_type> visited;
            mesh.visit_cell_faces(cell, [&](auto face)
            {
                visited.push_back(face);
                // Nested visitors must not overwrite the outer traversal.
                size_t nested_count = 0;
                mesh.visit_cell_faces(cell, [&](auto) { ++nested_count; });
                EXPECT_EQ(nested_count, expected.size());
            });
            EXPECT_TRUE(std::ranges::equal(visited, expected));
            visited.clear();
            mesh.visit_cell_faces(cell, [&](auto face, auto native_face, auto native_cell, const auto& native,
                                           const auto& to_local_cell)
            {
                visited.push_back(face);
                EXPECT_EQ(to_local_cell(native.owner_cell(native_face)), mesh.owner_cell(face));
                EXPECT_EQ(to_local_cell(native.neighbor_cell(native_face)), mesh.neighbor_cell(face));
                EXPECT_DOUBLE_EQ(native.face_area(native_face), mesh.face_area(face));
                EXPECT_DOUBLE_EQ(native.cell_volume(native_cell), mesh.cell_volume(cell));
                const auto native_center = native.cell_centroid(native_cell);
                const auto handle_center = mesh.cell_centroid(cell);
                // Apply ULP tolerance at each coordinate's magnitude, including
                // near-zero cylindrical components affected by roundoff.
                EXPECT_DOUBLE_EQ(native_center.x, handle_center.x);
                EXPECT_DOUBLE_EQ(native_center.y, handle_center.y);
                EXPECT_DOUBLE_EQ(native_center.z, handle_center.z);
            });
            EXPECT_TRUE(std::ranges::equal(visited, expected));
        }
        EXPECT_EQ(mesh.connectivity_storage_bytes(), 0U);
        EXPECT_FALSE(mesh.has_materialized_indexer());
    };
    auto cylinder=std::make_shared<Cylindrical>(SimpleFluid::Vec3D<SimpleFluid::ArrReal>{{
        {1,2,3},{0,std::numbers::pi,2*std::numbers::pi},{0,1,2}}});
    Handle periodic(cylinder);
    check_visitation(periodic);
    for(size_t c=0;c<periodic.num_cells();++c)
    {
        const auto native=visible_native_faces(periodic,*cylinder,c);
        const auto range=periodic.faces(c);
        const auto other_cell=(c+1)%periodic.num_cells();
        const auto other=periodic.faces(other_cell);
        ASSERT_EQ(range.size(),native.size());
        for(size_t i=0;i<range.size();++i) EXPECT_EQ(periodic.face_global_id(range[i]),native[i]);
        EXPECT_EQ(other.size(),visible_native_faces(periodic,*cylinder,other_cell).size());
        if(periodic.overlap_face_map()->getComm()->getSize()==1) EXPECT_EQ(other.size(),6U);
    }
    auto cart=std::make_shared<Cartesian>(SimpleFluid::Vec3D<SimpleFluid::ArrReal>{{{0,1,2,3,4,5,6},{0,1,2},{0,1,2}}});
    Handle partition(cart,{.ghost_layers=1,.partition=0,.partitions=2});
    check_visitation(partition);
    bool absent_remote_neighbor=false;
    for(size_t c=0;c<partition.num_cells();++c)
    {
        const auto native=visible_native_faces(partition,*cart,c);
        const auto range=partition.faces(c);
        ASSERT_EQ(range.size(),native.size());
        for(size_t i=0;i<range.size();++i)
        {
            EXPECT_EQ(partition.face_global_id(range[i]),native[i]);
            if(partition.neighbor_cell(range[i])==Handle::invalid_local_id() && !partition.is_geometry_exterior_face(range[i]))
            { absent_remote_neighbor=true; EXPECT_FALSE(partition.is_boundary_face(range[i])); }
        }
    }
    EXPECT_TRUE(absent_remote_neighbor);
    EXPECT_EQ(partition.connectivity_storage_bytes(),0U);
    EXPECT_FALSE(partition.has_materialized_indexer());
}
