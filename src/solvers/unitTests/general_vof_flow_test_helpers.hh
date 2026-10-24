/** Shared test-local native fixtures; no test cases or environment registration. */
#pragma once
#include "geometry/mesh/AMR/CartesianAMR.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveHierarchy.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveNative.hh"
#include "geometry/mesh/AMR/ExtrudedAMR.hh"
#include "geometry/unitTests/polyhedral_amr_helpers.hh"
#include "geometry/unitTests/region_mesh_helpers.hh"
#include "solvers/details/GeneralVoFFiniteProjectionArithmetic.hh"
#include "solvers/details/GeneralVoFFlowWorkspace.hh"
#include "utils/testing_environment.hh"
#include <Teuchos_CommHelpers.hpp>
#include <bit>
#include <cmath>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
namespace
{
using namespace SimpleFluid;
namespace AMR = SimpleFluid::Meshes::AMR;
using LO = DefaultTpetraTypes::local_ordinal_type;
SP<const MeshHandle<>> skew_native(unsigned layers = 3)
{
    AMR::ExtrudedAMRGrid grid;
    for (unsigned j = 0; j < 4; ++j)
        for (unsigned i = 0; i < 4; ++i)
            grid.xy_nodes.push_back({double(i) + .3 * j, double(j), 0});
    for (unsigned j = 0; j < 3; ++j)
        for (unsigned i = 0; i < 3; ++i)
        {
            const unsigned a = j * 4 + i;
            grid.base_cells.push_back({a, a + 1, a + 5, a + 4});
        }
    for (unsigned i = 0; i < 3; ++i)
    {
        grid.boundaries.push_back({i, i + 1, "wall"});
        grid.boundaries.push_back({12 + i, 13 + i, "wall"});
        grid.boundaries.push_back({4 * i, 4 * (i + 1), "wall"});
        grid.boundaries.push_back({4 * i + 3, 4 * (i + 1) + 3, "wall"});
    }
    grid.z_edges = {0, 1, 2, 3};
    grid.z_edges.resize(layers + 1);
    auto source = std::make_shared<AMR::ExtrudedAMR>(grid);
    std::vector<std::uint64_t> owned;
    if (Tpetra::getDefaultComm()->getRank() == 0)
        for (std::uint64_t c = 0; c < 9 * layers; ++c)
            owned.push_back(c);
    auto bootstrap = source->detach_owned(owned, {'f', 'l', 'o', 'w'});
    source.reset();
    auto schema = bootstrap.kernel->descriptor();
    auto hierarchy = std::make_shared<const AMR::DistributedAdaptiveHierarchy>(
        std::move(bootstrap.owned), std::move(bootstrap.families), bootstrap.kernel, Tpetra::getDefaultComm());
    return AMR::assemble_adaptive_native(*hierarchy, schema).mesh;
}
SP<const MeshHandle<>> boundary_polyhedron(bool pair)
{
    const auto native = pair ? test::polyhedral_pair() : test::polyhedral_tetrahedron();
    AMR::PolyhedralAMR source(test::polyhedral_grid(native));
    std::vector<std::uint64_t> owned;
    if (Tpetra::getDefaultComm()->getRank() == 0)
        for (std::uint64_t c = 0; c < native->num_cells(); ++c)
            owned.push_back(c);
    auto bootstrap = source.detach_owned(owned, {'b', 'o', 'u', 'n', 'd', 'a', 'r', 'y'});
    const auto schema = bootstrap.kernel->descriptor();
    const AMR::DistributedAdaptiveHierarchy hierarchy(
        std::move(bootstrap.owned), std::move(bootstrap.families), bootstrap.kernel, Tpetra::getDefaultComm());
    return AMR::assemble_adaptive_native(hierarchy, schema).mesh;
}
} // namespace
namespace
{
// Independent uncached least-squares assembly for the fully sampled fixture.
std::vector<VoF::Vector> uncached_gradient(const MeshHandle<>& mesh, const std::vector<double>& scalar,
    const std::vector<double>& halo, const std::vector<VoF::Vector>& spans)
{
    std::vector<VoF::Vector> result(scalar.size());
    const auto lease = mesh.acquire_execution_view();
    for (std::size_t c = 0; c < scalar.size(); ++c)
    {
        std::array<std::array<double, 4>, 3> m{};
        for (const auto fid : mesh.faces(LO(c)))
        {
            if (mesh.is_geometry_exterior_face(fid))
                continue;
            const auto displacement = mesh.cell_center_vector(fid, LO(c));
            VoF::Vector r{displacement.x, displacement.y, displacement.z};
            for (unsigned a = 0; a < 3; ++a)
                r[a] /= spans[c][a];
            const double w = 1 / (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]);
            const double delta = halo[mesh.opposite_cell(fid, LO(c))] - scalar[c];
            for (unsigned a = 0; a < 3; ++a)
            {
                m[a][3] += w * r[a] * delta;
                for (unsigned b = 0; b < 3; ++b)
                    m[a][b] += w * r[a] * r[b];
            }
        }
        for (unsigned a = 0; a < 3; ++a)
        {
            unsigned pivot = a;
            for (unsigned b = a + 1; b < 3; ++b)
                if (std::abs(m[b][a]) > std::abs(m[pivot][a]))
                    pivot = b;
            std::swap(m[a], m[pivot]);
            const double d = m[a][a];
            for (unsigned b = a; b < 4; ++b)
                m[a][b] /= d;
            for (unsigned row = 0; row < 3; ++row)
                if (row != a)
                {
                    const double factor = m[row][a];
                    for (unsigned b = a; b < 4; ++b)
                        m[row][b] -= factor * m[a][b];
                }
        }
        for (unsigned a = 0; a < 3; ++a)
            result[c][a] = m[a][3] / spans[c][a];
    }
    return result;
}
} // namespace
