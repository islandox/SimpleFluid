/** @file testSTKMeshInput.cc
 * @brief Actual Exodus input returns native topology, metrics, and field maps.
 */
#include "geometry/MeshFactory.hh"
#include "io/details/STKBackendBridge.hh"
#include "utils/testing_environment.hh"

#include <Ioss_Property.h>
#include <Ioss_Utils.h>
#include <Teuchos_CommHelpers.hpp>
#include <Teuchos_DefaultMpiComm.hpp>
#include <gtest/gtest.h>
#include <stk_io/IossBridge.hpp>
#include <stk_io/StkMeshIoBroker.hpp>
#include <stk_mesh/base/FEMHelpers.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/MeshBuilder.hpp>
#include <stk_mesh/base/SideSetEntry.hpp>

#include <array>
#include <chrono>
#include <cmath>
#include <exception>
#include <filesystem>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace
{
using namespace SimpleFluid;
using Pack = DefaultTpetraTypes;
using Handle = MeshHandle<Pack>;
using Point = vec3<>;
using Comm = Teuchos::Comm<int>;
using ID = stk::mesh::EntityId;
using Geometry = Meshes::UnstructuredMesh;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);

enum class Shape
{
    SingleHex,
    WarpedHexPair,
    MixedHexWedge,
    Tetrahedron
};

STKMeshInputOptions boundary_options()
{
    STKMeshInputOptions options;
    options.boundary_name_to_id = {{"wall", 7}, {"empty", 19}};
    options.auto_assign_boundary_ids = false;
    return options;
}

/** Raw STK is scoped to this test fixture writer and never supplies runtime geometry. */
void write_exodus(const std::string& filename, Shape shape, MPI_Comm raw_comm = MPI_COMM_SELF,
    bool populate_cells = true, bool decomposed_output = false)
{
    stk::mesh::MeshBuilder builder(raw_comm);
    builder.set_spatial_dimension(3);
    auto meta = builder.create_meta_data();
    auto bulk = builder.create(meta);
    auto& coordinates = meta->declare_field<double>(stk::topology::NODE_RANK, "coordinates");
    stk::mesh::put_field_on_mesh(coordinates, meta->universal_part(), 3, nullptr);
    meta->set_coordinate_field(&coordinates);
    const stk::topology first_topology = shape == Shape::Tetrahedron ? stk::topology::TET_4 : stk::topology::HEX_8;
    auto& first_block = meta->declare_part_with_topology("block_1", first_topology);
    stk::io::put_io_part_attribute(first_block);
    stk::mesh::Part* second_block = nullptr;
    if (shape == Shape::MixedHexWedge)
    {
        second_block = &meta->declare_part_with_topology("block_2", stk::topology::WEDGE_6);
        stk::io::put_io_part_attribute(*second_block);
    }
    auto& wall = meta->declare_part("wall", meta->side_rank());
    stk::io::put_io_part_attribute(wall);
    std::vector<const stk::mesh::Part*> blocks{&first_block};
    if (second_block)
        blocks.push_back(second_block);
    meta->set_surface_to_block_mapping(&wall, blocks);
    meta->commit();
    auto& sideset = bulk->create_sideset(wall);
    std::map<ID, Point> points;
    if (shape == Shape::SingleHex)
        points = {{11, {0, 0, 0}}, {12, {2, 0, 0}}, {13, {2, 3, 0}}, {14, {0, 3, 0}}, {15, {0, 0, 4}}, {16, {2, 0, 4}},
            {17, {2, 3, 4}}, {18, {0, 3, 4}}};
    else if (shape == Shape::Tetrahedron)
        points = {{11, {0, 0, 0}}, {12, {1, 0, 0}}, {13, {0, 1, 0}}, {14, {0, 0, 1}}};
    else
        points = {{11, {0, 0, 0}}, {12, {1, 0, 0}}, {13, {1, 1, 0}}, {14, {0, 1, 0}}, {15, {0, 0, 1}}, {16, {1, 0, 1}},
            {17, {shape == Shape::WarpedHexPair ? 1.1 : 1.0, 1, 1}}, {18, {0, 1, 1}}, {19, {2, 0, 0}}, {20, {2, 1, 0}},
            {21, {2, 0, 1}}, {22, {2, 1, 1}}};
    bulk->modification_begin();
    if (populate_cells)
    {
        const stk::mesh::EntityIdVector first_nodes = shape == Shape::Tetrahedron
                                                          ? stk::mesh::EntityIdVector{11, 12, 13, 14}
                                                          : stk::mesh::EntityIdVector{11, 12, 13, 14, 15, 16, 17, 18};
        const auto first = stk::mesh::declare_element(*bulk, first_block, 101, first_nodes);
        auto add_side = [&](stk::mesh::Entity element, unsigned side)
        {
            bulk->declare_element_side(element, side, stk::mesh::PartVector{&wall});
            sideset.add({element, static_cast<stk::mesh::ConnectivityOrdinal>(side)});
        };
        const bool has_second = shape == Shape::WarpedHexPair || shape == Shape::MixedHexWedge;
        for (unsigned s = 0; s < first_topology.num_sides(); ++s)
            if (!has_second || s != 1)
                add_side(first, s);
        if (shape == Shape::WarpedHexPair)
        {
            const auto second = stk::mesh::declare_element(*bulk, first_block, 205, {12, 19, 20, 13, 16, 21, 22, 17});
            for (unsigned s = 0; s < 6; ++s)
                if (s != 3)
                    add_side(second, s);
        }
        if (shape == Shape::MixedHexWedge)
        {
            const auto second = stk::mesh::declare_element(*bulk, *second_block, 205, {12, 19, 13, 16, 21, 17});
            for (unsigned s = 0; s < 5; ++s)
                if (s != 2)
                    add_side(second, s);
        }
        for (const auto& [id, p] : points)
        {
            const auto node = bulk->get_entity(stk::topology::NODE_RANK, id);
            if (!bulk->is_valid(node))
                continue;
            auto* x = stk::mesh::field_data(coordinates, node);
            if (!x)
                throw std::runtime_error("Exodus test fixture has no node coordinate storage.");
            x[0] = p.x;
            x[1] = p.y;
            x[2] = p.z;
        }
    }
    bulk->modification_end();
    stk::io::StkMeshIoBroker writer(raw_comm);
    writer.set_bulk_data(*bulk);
    if (decomposed_output)
        simplefluid_trilinos_stk_io_set_string_property(&writer, "COMPOSE_RESULTS", "NO");
    const auto output = writer.create_output_mesh(filename, stk::io::WRITE_RESULTS);
    writer.write_output_mesh(output);
}

/** Agree writer failures before entering reader collectives on the supplied communicator. */
class MeshFile
{
public:
    MeshFile(Shape shape, Teuchos::RCP<const Comm> communicator, bool decomposed = false)
        : comm(std::move(communicator)), pieces(decomposed)
    {
        std::exception_ptr error;
        if (comm->getRank() == 0)
        {
            try
            {
                const auto token = std::chrono::steady_clock::now().time_since_epoch().count();
                path = (std::filesystem::temp_directory_path() /
                        ("simplefluid_stk_input_" + std::to_string(Tpetra::getDefaultComm()->getRank()) + "_" +
                            std::to_string(token) + ".e"))
                           .string();
                if (!pieces)
                    write_exodus(path, shape);
            }
            catch (...)
            {
                error = std::current_exception();
            }
        }
        const int failed = error ? 1 : 0;
        int any_failed = 0;
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &failed, &any_failed);
        if (any_failed)
        {
            if (error)
                std::rethrow_exception(error);
            throw std::runtime_error("Exodus test fixture writer failed on communicator rank zero.");
        }
        int size = static_cast<int>(path.size());
        Teuchos::broadcast(*comm, 0, 1, &size);
        path.resize(static_cast<size_t>(size));
        Teuchos::broadcast(*comm, 0, size, path.data());
        if (pieces)
        {
            error = {};
            try
            {
                const auto* mpi = dynamic_cast<const Teuchos::MpiComm<int>*>(comm.getRawPtr());
                if (!mpi)
                    throw std::invalid_argument("Parallel Exodus fixture requires an MPI communicator.");
                // Every rank creates identical metadata and participates in all
                // STK/Ioss collectives, including ranks without any elements.
                write_exodus(path, shape, *mpi->getRawMpiComm(), comm->getRank() == 0, true);
            }
            catch (...)
            {
                error = std::current_exception();
            }
            const int writer_failed = error ? 1 : 0;
            Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &writer_failed, &any_failed);
            if (any_failed)
            {
                if (error)
                    std::rethrow_exception(error);
                throw std::runtime_error("Parallel Exodus fixture writer failed on another rank.");
            }
        }
    }
    ~MeshFile()
    {
        comm->barrier();
        std::error_code ignored;
        if (pieces)
            std::filesystem::remove(
                SimpleFluid::detail::stk_decomposed_filename(path, comm->getRank(), comm->getSize()), ignored);
        if (comm->getRank() == 0)
            std::filesystem::remove(path, ignored);
    }
    const std::string& filename() const { return path; }

private:
    Teuchos::RCP<const Comm> comm;
    std::string path;
    bool pieces = false;
};

void expect_native(const Handle& mesh, const Teuchos::RCP<const Comm>& comm, size_t total)
{
    EXPECT_TRUE(mesh.visit(
        [](const auto& geometry) { return std::is_same_v<std::remove_cvref_t<decltype(geometry)>, Geometry>; }));
    EXPECT_EQ(mesh.owned_cell_map()->getComm()->getSize(), comm->getSize());
    EXPECT_EQ(mesh.owned_cell_map()->getGlobalNumElements(), total);
    EXPECT_EQ(mesh.boundary_batch_name(7), "wall");
    EXPECT_EQ(mesh.boundary_batch_name(19), "empty");
    EXPECT_TRUE(mesh.boundary_face_batch(19).face_lids.empty());
    unsigned long long local = mesh.num_owned_cells(), prefix = 0;
    Teuchos::scan(*comm, Teuchos::REDUCE_SUM, 1, &local, &prefix);
    const auto first = prefix - local;
    for (size_t c = 0; c < mesh.num_owned_cells(); ++c)
        EXPECT_EQ(mesh.cell_global_id(static_cast<int>(c)), static_cast<Pack::global_ordinal_type>(first + c));
}

void expect_single_hex(const Handle& mesh)
{
    for (size_t c = 0; c < mesh.num_local_cells(); ++c)
    {
        const auto local = static_cast<int>(c);
        EXPECT_EQ(mesh.cell_geometry_global_id(local), 101);
        EXPECT_DOUBLE_EQ(mesh.cell_volume(local), 24);
        const auto center = mesh.cell_centroid(local);
        EXPECT_DOUBLE_EQ(center.x, 1);
        EXPECT_DOUBLE_EQ(center.y, 1.5);
        EXPECT_DOUBLE_EQ(center.z, 2);
    }
    const auto& nodes = mesh.indexer().node_global_ids();
    for (const auto id : nodes)
        EXPECT_TRUE(id >= 11 && id <= 18);
    if (mesh.num_local_cells())
        EXPECT_EQ(std::set<Pack::global_ordinal_type>(nodes.begin(), nodes.end()).size(), 8);
    mesh.visit(
        [&](const auto& geometry)
        {
            if constexpr (std::is_same_v<std::remove_cvref_t<decltype(geometry)>, Geometry>)
                for (size_t c = 0; c < geometry.num_cells(); ++c)
                    EXPECT_EQ(geometry.cell_type(c), MeshUtils::CellType::HEXAHEDRON);
        });
}
/** Agree pure local file operations before the next collective input call. */
template<class Operation>
void agree_fixture_file_operation(const Comm& comm, const char* context, Operation&& operation)
{
    std::exception_ptr error;
    try
    {
        std::forward<Operation>(operation)();
    }
    catch (...)
    {
        error = std::current_exception();
    }
    const int failed = error ? 1 : 0;
    int any_failed = 0;
    Teuchos::reduceAll(comm, Teuchos::REDUCE_MAX, 1, &failed, &any_failed);
    if (any_failed)
    {
        if (error)
            std::rethrow_exception(error);
        throw std::runtime_error(context);
    }
}

} // namespace

TEST(STKMeshInputTest, SparseHexFileReturnsNativeMetricsSourceIdsAndEmptyBoundaryCatalog)
{
    const auto comm = Tpetra::getDefaultComm();
    MeshFile file(Shape::SingleHex, comm);
    const auto mesh = MeshFactory::from_stk_file<Pack>(file.filename(), boundary_options(), comm);
    expect_native(*mesh, comm, 1);
    expect_single_hex(*mesh);
    int local_empty = mesh->num_owned_cells() == 0 ? 1 : 0, empty = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local_empty, &empty);
    EXPECT_EQ(empty, comm->getSize() - 1);
}

TEST(STKMeshInputTest, WarpedSharedQuadKeepsElectedSourceTriangleMoment)
{
    const auto comm = Tpetra::getDefaultComm();
    MeshFile file(Shape::WarpedHexPair, comm);
    const auto mesh = MeshFactory::from_stk_file<Pack>(file.filename(), boundary_options(), comm);
    expect_native(*mesh, comm, 2);
    const std::array<Point, 4> p{{{1, 0, 0}, {1, 1, 0}, {1.1, 1, 1}, {1, 0, 1}}};
    const auto a = (p[1] - p[0]).cross(p[2] - p[0]);
    const auto b = (p[2] - p[0]).cross(p[3] - p[0]);
    const auto expected =
        ((p[0] + p[1] + p[2]) * (a.norm() / 3) + (p[0] + p[2] + p[3]) * (b.norm() / 3)) / (a.norm() + b.norm());
    const auto area = (a + b).norm() / 2;
    int local_owned_shared = 0, owned_shared = 0;
    int local_adjacent_owner_rank = mesh->num_owned_cells() > 0 ? 1 : 0, adjacent_owner_ranks = 0;
    std::array<double, 3> local_signed_area{}, global_signed_area{};
    const auto normal = (a + b) / (a + b).norm();
    for (size_t f = 0; f < mesh->num_faces(); ++f)
    {
        const auto face = static_cast<int>(f);
        if (mesh->face_geometry_global_id(face) != 1012)
            continue;
        EXPECT_FALSE(mesh->is_boundary_face(face));
        EXPECT_EQ(mesh->cell_geometry_global_id(mesh->owner_cell(face)), 101);
        const auto actual = mesh->face_centroid(face);
        EXPECT_NEAR(actual.x, expected.x, 1e-14);
        EXPECT_NEAR(actual.y, expected.y, 1e-14);
        EXPECT_NEAR(actual.z, expected.z, 1e-14);
        EXPECT_NEAR(mesh->face_area(face), area, 1e-14);
        const auto actual_normal = mesh->face_normal(face);
        EXPECT_NEAR(actual_normal.x, normal.x, 1e-14);
        EXPECT_NEAR(actual_normal.y, normal.y, 1e-14);
        EXPECT_NEAR(actual_normal.z, normal.z, 1e-14);
        local_owned_shared += mesh->is_owned_face(face) ? 1 : 0;
        for (const auto cell : {mesh->owner_cell(face), mesh->neighbor_cell(face)})
            if (mesh->is_owned_cell(cell))
            {
                const auto outward = mesh->face_area_vector_outward(face, cell);
                local_signed_area[0] += outward.x;
                local_signed_area[1] += outward.y;
                local_signed_area[2] += outward.z;
            }
    }
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local_owned_shared, &owned_shared);
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local_adjacent_owner_rank, &adjacent_owner_ranks);
    // The preserved overlap policy owns this face on each rank owning an
    // incident cell. Owner-oriented outward contributions still cancel once
    // per owned cell across that overlap (one rank serial, normally two in MPI).
    EXPECT_EQ(owned_shared, adjacent_owner_ranks);
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 3, local_signed_area.data(), global_signed_area.data());
    for (const auto component : global_signed_area)
        EXPECT_NEAR(component, 0, 1e-14);
}

TEST(STKMeshInputTest, MixedHexWedgeFileKeepsCellTypesAndKnownVolumes)
{
    const auto comm = Tpetra::getDefaultComm();
    MeshFile file(Shape::MixedHexWedge, comm);
    const auto mesh = MeshFactory::from_stk_file<Pack>(file.filename(), boundary_options(), comm);
    expect_native(*mesh, comm, 2);
    double local_volume = 0, total_volume = 0;
    for (size_t c = 0; c < mesh->num_owned_cells(); ++c)
    {
        const auto id = mesh->cell_geometry_global_id(static_cast<int>(c));
        EXPECT_TRUE(id == 101 || id == 205);
        EXPECT_DOUBLE_EQ(mesh->cell_volume(static_cast<int>(c)), id == 101 ? 1.0 : .5);
        local_volume += mesh->cell_volume(static_cast<int>(c));
    }
    mesh->visit(
        [&](const auto& geometry)
        {
            if constexpr (std::is_same_v<std::remove_cvref_t<decltype(geometry)>, Geometry>)
                for (size_t c = 0; c < geometry.num_cells(); ++c)
                {
                    const auto id = mesh->cell_geometry_global_id(static_cast<int>(c));
                    EXPECT_EQ(geometry.cell_type(c),
                        id == 101 ? MeshUtils::CellType::HEXAHEDRON : MeshUtils::CellType::TRIPRISM);
                }
        });
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local_volume, &total_volume);
    EXPECT_DOUBLE_EQ(total_volume, 1.5);
}

TEST(STKMeshInputTest, UnsupportedTetrahedronFileFailsOnEveryRank)
{
    const auto comm = Tpetra::getDefaultComm();
    MeshFile file(Shape::Tetrahedron, comm);
    EXPECT_ANY_THROW(MeshFactory::from_stk_file<Pack>(file.filename(), boundary_options(), comm));
}

TEST(STKMeshInputTest, RankDivergentFilenameAndOptionsFailBeforeOpeningInput)
{
    const auto comm = Tpetra::getDefaultComm();
    if (comm->getSize() < 2)
        GTEST_SKIP();
    MeshFile file(Shape::SingleHex, comm);
    const auto filename = file.filename() + (comm->getRank() == 1 ? ".different" : "");
    EXPECT_ANY_THROW(MeshFactory::from_stk_file<Pack>(filename, boundary_options(), comm));
    auto options = boundary_options();
    if (comm->getRank() == 1)
        options.boundary_name_to_id["wall"] = 8;
    EXPECT_ANY_THROW(MeshFactory::from_stk_file<Pack>(file.filename(), options, comm));
}

TEST(STKMeshInputTest, FileReaderAndNativeMapsUseSuppliedSplitCommunicator)
{
    const auto world = Tpetra::getDefaultComm();
    if (world->getSize() < 2)
        GTEST_SKIP();
    const auto comm = world->split(world->getRank() % 2, world->getRank());
    MeshFile file(Shape::SingleHex, comm);
    const auto mesh = MeshFactory::from_stk_file<Pack>(file.filename(), boundary_options(), comm);
    expect_native(*mesh, comm, 1);
    expect_single_hex(*mesh);
    EXPECT_EQ(mesh->owned_cell_map()->getComm()->getRank(), comm->getRank());
}

TEST(STKMeshInputTest, ExplicitAndAutomaticModesReadActualDecomposedFilesWithoutBaseFile)
{
    const auto comm = Tpetra::getDefaultComm();
    if (comm->getSize() < 2)
        GTEST_SKIP();
    MeshFile file(Shape::SingleHex, comm, true);
    // COMPOSE_RESULTS=NO must produce the complete rank-piece set; the base
    // file is deliberately absent, so neither reader path can fall back to it.
    EXPECT_FALSE(std::filesystem::exists(file.filename()));
    const auto piece = SimpleFluid::detail::stk_decomposed_filename(file.filename(), comm->getRank(), comm->getSize());
    const int local_piece_exists = std::filesystem::exists(piece) ? 1 : 0;
    int pieces_exist = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local_piece_exists, &pieces_exist);
    EXPECT_EQ(pieces_exist, comm->getSize());
    auto options = boundary_options();
    options.decomposition_method = "";
    const auto explicit_mesh = MeshFactory::from_stk_file<Pack>(file.filename(), options, comm);
    expect_native(*explicit_mesh, comm, 1);
    expect_single_hex(*explicit_mesh);
    options.decomposition_method = "auto";
    const auto automatic_mesh = MeshFactory::from_stk_file<Pack>(file.filename(), options, comm);
    expect_native(*automatic_mesh, comm, 1);
    expect_single_hex(*automatic_mesh);
}

TEST(STKMeshInputTest, CompletePiecesWinOverExistingBaseFile)
{
    const auto comm = Tpetra::getDefaultComm();
    if (comm->getSize() < 2)
        GTEST_SKIP();
    MeshFile file(Shape::SingleHex, comm, true);
    agree_fixture_file_operation(*comm, "Competing Exodus base-file writer failed on rank zero.",
        [&]
        {
            if (comm->getRank() == 0)
                write_exodus(file.filename(), Shape::MixedHexWedge);
        });
    // The valid competing base contains two cells and V=1.5. Only the full
    // decomposed set contains the single sparse HEX with V=24 and source101.
    EXPECT_TRUE(std::filesystem::exists(file.filename()));
    auto options = boundary_options();
    options.decomposition_method = "auto";
    const auto mesh = MeshFactory::from_stk_file<Pack>(file.filename(), options, comm);
    expect_native(*mesh, comm, 1);
    expect_single_hex(*mesh);
}

TEST(STKMeshInputTest, PartialPiecesRejectEvenWithUsableBase)
{
    const auto comm = Tpetra::getDefaultComm();
    if (comm->getSize() < 2)
        GTEST_SKIP();
    MeshFile file(Shape::SingleHex, comm, true);
    agree_fixture_file_operation(*comm, "Competing Exodus base-file writer failed on rank zero.",
        [&]
        {
            if (comm->getRank() == 0)
                write_exodus(file.filename(), Shape::MixedHexWedge);
        });
    EXPECT_TRUE(std::filesystem::exists(file.filename()));
    agree_fixture_file_operation(*comm, "Removing the selected Exodus rank piece failed on another rank.",
        [&]
        {
            if (comm->getRank() == comm->getSize() - 1)
            {
                const auto piece =
                    SimpleFluid::detail::stk_decomposed_filename(file.filename(), comm->getRank(), comm->getSize());
                if (!std::filesystem::remove(piece))
                    throw std::runtime_error("Expected rank piece was absent before partial-set regression.");
            }
        });
    auto options = boundary_options();
    options.decomposition_method = "auto";
    try
    {
        (void) MeshFactory::from_stk_file<Pack>(file.filename(), options, comm);
        ADD_FAILURE() << "A partial decomposed set must reject the usable base-file fallback.";
    }
    catch (const std::invalid_argument& error)
    {
        EXPECT_NE(std::string(error.what()).find("incomplete"), std::string::npos);
    }
    catch (const std::exception& error)
    {
        ADD_FAILURE() << "Expected incomplete-set invalid_argument, got: " << error.what();
    }
}
