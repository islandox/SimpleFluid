/** @file testCellFluxBalanceCache.cc @brief Ordered flux balances and mapped view ownership. */
#include <gtest/gtest.h>

#include "FVM/CellOperators.hh"
#include "fields/MeshFieldTraits.hh"
#include "geometry/PlanarALEMeshMotion.hh"
#include "geometry/unitTests/test_mesh_helpers.hh"
#include "utils/testing_environment.hh"
#include <Teuchos_CommHelpers.hpp>

namespace
{
using Pack = SimpleFluid::DefaultTpetraTypes;
using Mesh = SimpleFluid::MeshHandle<Pack>;
using Flux = SimpleFluid::ScalarFaceFieldStored<Pack>;
using utils_test::KokkosEnvironment;
testing::Environment* const kokkos_environment = testing::AddGlobalTestEnvironment(new KokkosEnvironment);

auto make_mesh()
{
    return std::make_shared<Mesh>(std::make_shared<Mesh::Cartesian>(SimpleFluid::Vec3D<SimpleFluid::ArrReal>{{
        {0.0, 0.25, 0.75, 1.5, 3.0}, {0.0, 0.5, 2.0}, {0.0, 0.25, 1.0, 2.0, 4.0}}}));
}

void fill_flux(Flux& flux)
{
    for (const auto face : flux.owned_face_ids())
        flux.set_owned_value(face, 0.25 + 0.03125 * flux.owned_map()->getGlobalElement(face));
    flux.sync_ghosts();
}

template<class Value>
double expected_balance(const Mesh& mesh, Pack::local_ordinal_type cell, Value value)
{
    double result = 0.0;
    for (const auto face : mesh.faces(cell))
        result += (mesh.owner_cell(face) == cell ? 1.0 : -1.0) * value(face);
    return result;
}
}

TEST(CellFluxBalanceCacheTest, UsesSuppliedOverlapAndOwnedViewsIncludingRemoteFaces)
{
    auto mesh = make_mesh();
    Flux flux(mesh, 0.0, "flux");
    fill_flux(flux);
    SimpleFluid::FVM::CellFluxBalanceCache<Pack, Mesh> cache(flux);
    const auto view = SimpleFluid::FVM::face_flux_balance_read_view(flux);
    Pack::vector_type supplied(flux.overlap_map(), true);
    supplied.update(1.0, flux.overlap_data(), 0.0);
    supplied.scale(2.0);
    const auto supplied_view = supplied.getLocalViewHost(Tpetra::Access::ReadOnly);
    Pack::vector_type supplied_owned(flux.owned_map(), true);
    supplied_owned.update(1.0, flux.owned_data(), 0.0);
    supplied_owned.scale(3.0);
    const auto owned_view = supplied_owned.getLocalViewHost(Tpetra::Access::ReadOnly);
    int local_remote_faces = 0;
    for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
    {
        const auto cell = static_cast<Pack::local_ordinal_type>(owned);
        const auto original = expected_balance(*mesh, cell, [&](auto face) { return flux.local_value(face); });
        EXPECT_DOUBLE_EQ(cache.balance(view, cell), original);
        EXPECT_DOUBLE_EQ(SimpleFluid::FVM::cell_flux_balance<Pack>(*mesh, flux, view, cell), original);
        const auto doubled = expected_balance(*mesh, cell, [&](auto face) { return 2.0 * flux.local_value(face); });
        EXPECT_DOUBLE_EQ(SimpleFluid::FVM::cell_flux_balance<Pack>(*mesh, flux, supplied_view, cell), doubled);
        const auto mixed = expected_balance(*mesh, cell, [&](auto face)
        {
            if (!flux.is_owned_face(face))
                ++local_remote_faces;
            return (flux.is_owned_face(face) ? 3.0 : 1.0) * flux.local_value(face);
        });
        EXPECT_DOUBLE_EQ(SimpleFluid::FVM::cell_flux_balance<Pack>(*mesh, flux, owned_view, cell), mixed);
    }
    int remote_faces = 0;
    const auto comm = mesh->owned_cell_map()->getComm();
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local_remote_faces, &remote_faces);
    if (comm->getSize() > 1)
        EXPECT_GT(remote_faces, 0);
    EXPECT_THROW(cache.balance(view, -1), std::out_of_range);
    Pack::multi_vector_type invalid(flux.overlap_map(), 2, true);
    EXPECT_THROW(SimpleFluid::FVM::cell_flux_balance<Pack>(*mesh, flux,
        invalid.getLocalViewHost(Tpetra::Access::ReadOnly), 0), std::invalid_argument);
}

TEST(CellFluxBalanceCacheTest, TopologySlotsSurviveMotionAndRollback)
{
    auto mesh = make_mesh();
    Flux flux(mesh, 0.0, "flux");
    fill_flux(flux);
    SimpleFluid::FVM::CellFluxBalanceCache<Pack, Mesh> cache(flux);
    SimpleFluid::PlanarALEMeshMotion<Pack> motion(mesh);
    const auto check = [&]
    {
        const auto view = flux.local_read_view();
        for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
        {
            const auto cell = static_cast<Pack::local_ordinal_type>(owned);
            EXPECT_DOUBLE_EQ(cache.balance(view, cell),
                expected_balance(*mesh, cell, [&](auto face) { return flux.local_value(face); }));
        }
    };
    motion.begin_trial(5.0, 1.0);
    check();
    motion.rollback_trial();
    check();
}

TEST(CellFluxBalanceCacheTest, PreservesLegacyOwnedFaceConvention)
{
    using LegacyMesh = SimpleFluid::Mesh<Pack>;
    auto mesh = SimpleFluid::test::build_mesh<Pack>(SimpleFluid::test::make_box_database(4, 2, 4, 0.25));
    SimpleFluid::FaceField<Pack> flux(mesh, 0.0, "flux");
    for (const auto face : flux.owned_face_ids())
        flux.set_value(face, 0.125 + 0.03125 * face);
    SimpleFluid::FVM::CellFluxBalanceCache<Pack, LegacyMesh> cache(flux);
    const auto view = SimpleFluid::FVM::face_flux_balance_read_view(flux);
    for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
    {
        const auto cell = static_cast<Pack::local_ordinal_type>(owned);
        EXPECT_DOUBLE_EQ(cache.balance(view, cell), SimpleFluid::FVM::cell_flux_balance<Pack>(*mesh, flux, cell));
    }
}
