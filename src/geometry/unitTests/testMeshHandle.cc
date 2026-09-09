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
#include "geometry/STKMesh.hh"
#include "geometry/mesh/PartitionedMeshBase.hh"
#include "geometry/unitTests/test_mesh_helpers.hh"
#include "utils/testing_environment.hh"

#include <Teuchos_DefaultMpiComm.hpp>

#include <algorithm>
#include <filesystem>
#include <memory>
#include <numbers>
#include <span>
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
using ConcreteSTK = SimpleFluid::STKMesh<Pack>;
using PartitionedUnstructured =
    SimpleFluid::Meshes::PartitionedMesh<Unstructured, Pack>;

using utils_test::KokkosEnvironment;
testing::Environment* const kokkos_environment =
    testing::AddGlobalTestEnvironment(new KokkosEnvironment);

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
    static_assert(std::is_same_v<
                  Handle::MutableSTKAdapterPtr,
                  SimpleFluid::SP<Handle::STKAdapter>>);
    static_assert(std::is_constructible_v<
                  Handle,
                  SimpleFluid::SP<ConcreteSTK>>);
    static_assert(std::is_constructible_v<
                  Handle,
                  SimpleFluid::SP<const ConcreteSTK>>);
    static_assert(std::is_constructible_v<
                  Handle::STKAdapter,
                  SimpleFluid::SP<ConcreteSTK>>);
    static_assert(std::is_constructible_v<
                  Handle::STKAdapter,
                  SimpleFluid::SP<const ConcreteSTK>>);
    static_assert(std::is_constructible_v<
                  Handle::STKAdapter,
                  std::nullptr_t>);
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
        []<class Mesh>(Mesh& mesh) -> void*
        {
            static_assert(!std::is_const_v<Mesh>);
            return std::addressof(mesh);
        });
    EXPECT_EQ(mutable_address, cartesian.get());

    const auto* const_address = std::as_const(handle).visit(
        []<class Mesh>(Mesh& mesh) -> const void*
        {
            static_assert(std::is_const_v<Mesh>);
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
        static_cast<void>(Handle::STKAdapter(nullptr)),
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

/** @brief Semi-structured handles enforce their documented serial contract. */
TEST(MeshHandleTest, SemiStructuredConstructionIsSerialOnly)
{
    const auto mesh = std::make_shared<SemiStructured>(
        SimpleFluid::Arr<SemiStructured::Vec3>{
            {0.0, 0.0, 0.0},
            {1.0, 0.0, 0.0},
            {0.0, 1.0, 0.0}},
        SimpleFluid::Arr<SimpleFluid::Arr<unsigned>>{{0, 1, 2}},
        SimpleFluid::ArrReal{0.0, 1.0});
    const auto communicator = Tpetra::getDefaultComm();

    if (communicator->getSize() == 1)
    {
        EXPECT_NO_THROW(static_cast<void>(Handle(mesh)));
    }
    else
    {
        EXPECT_THROW(
            static_cast<void>(Handle(mesh)),
            std::runtime_error);
    }
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
}

/**
 * @brief Verifies wrapping a legacy STK mesh preserves maps, connectivity,
 * and geometric queries.
 */
TEST(MeshHandleTest, PreservesLegacySTKMapsAndGeometry)
{
    auto legacy = SimpleFluid::test::build_mesh<Pack>(
        SimpleFluid::test::make_two_hex_database());
    auto concrete_legacy =
        std::dynamic_pointer_cast<ConcreteSTK>(legacy);
    ASSERT_TRUE(concrete_legacy);
    Handle handle(legacy);
    Handle derived_handle(concrete_legacy);

    EXPECT_TRUE(handle.is_stk());
    EXPECT_TRUE(handle.has_mutable_geometry());
    EXPECT_EQ(handle.legacy_mesh(), legacy);
    EXPECT_EQ(
        handle.visit_mutable(
            [](auto& mesh) -> SimpleFluid::Mesh<Pack>*
            {
                using Mesh = std::remove_cvref_t<decltype(mesh)>;
                if constexpr (std::is_same_v<Mesh, Handle::STKAdapter>)
                {
                    return mesh.mutable_mesh_ptr().get();
                }
                return nullptr;
            }),
        legacy.get());
    SimpleFluid::SP<const SimpleFluid::Mesh<Pack>> read_only_legacy = legacy;
    EXPECT_FALSE(Handle(read_only_legacy).has_mutable_geometry());
    SimpleFluid::SP<const ConcreteSTK> read_only_concrete = concrete_legacy;
    Handle read_only_derived_handle(read_only_concrete);
    EXPECT_FALSE(read_only_derived_handle.has_mutable_geometry());
    EXPECT_EQ(read_only_derived_handle.legacy_mesh().get(), legacy.get());
    EXPECT_TRUE(derived_handle.has_mutable_geometry());
    EXPECT_EQ(derived_handle.legacy_mesh().get(), legacy.get());
    EXPECT_EQ(handle.owned_cell_map(), legacy->owned_cell_map());
    EXPECT_EQ(handle.overlap_cell_map(), legacy->overlap_cell_map());
    EXPECT_EQ(handle.owned_face_map(), legacy->owned_face_map());
    EXPECT_EQ(handle.cell_global_id(0),
              legacy->owned_cell_map()->getGlobalElement(0));
    EXPECT_EQ(
        handle.cell_geometry_global_id(0),
        legacy->cell_global_id(0));
    EXPECT_EQ(handle.face_global_id(0),
              legacy->owned_face_map()->getGlobalElement(0));
    EXPECT_EQ(handle.num_owned_cells(), legacy->num_owned_cells());
    EXPECT_EQ(handle.cell_centroid(0), legacy->cell_centroid(0));
    const auto topology = handle.vtu_topology();
    ASSERT_TRUE(topology);
    EXPECT_EQ(topology->cell_offsets.size(), handle.num_owned_cells());
    EXPECT_FALSE(topology->points.empty());
    const auto handle_faces = handle.faces(0);
    const auto& legacy_faces = legacy->faces(0);
    EXPECT_EQ(handle_faces.size(), legacy_faces.size());
    ASSERT_FALSE(handle_faces.empty());
    EXPECT_EQ(handle_faces.data(), &legacy_faces[0]);

    // The historical indexer API remains available, but is no longer needed
    // by normal legacy geometry and field access.
    EXPECT_EQ(
        handle.indexer().num_local_cells(),
        legacy->num_local_cells());
    EXPECT_EQ(handle.indexer().num_local_faces(), legacy->num_faces());
}

/**
 * @brief Verifies repeated connectivity requests reuse the materialized
 * cell-to-face cache when available.
 */
TEST(MeshHandleTest, ReusesMaterializedCellFaceConnectivity)
{
    const auto mesh = std::make_shared<Cartesian>(
        SimpleFluid::Vec3D<SimpleFluid::ArrReal>{{
            {0.0, 1.0, 2.0},
            {0.0, 1.0},
            {0.0, 1.0}}});
    const Handle handle(mesh);

    static_assert(std::is_same_v<
        decltype(handle.faces(0)),
        std::span<const Handle::local_ordinal_type>>);

    const auto first = handle.faces(0);
    const auto second = handle.faces(0);
    EXPECT_EQ(first.data(), second.data());
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
