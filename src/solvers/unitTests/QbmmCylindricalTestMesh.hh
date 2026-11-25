#pragma once

#include "geometry/mesh/OrthogonalCylindrial3D.hh"
#include "geometry/mesh/SemiStructuredXY_Z.hh"
#include "solvers/LowMachMixtureSolver.hh"

#include <cmath>
#include <cstdlib>
#include <gtest/gtest.h>
#include <numbers>

namespace SimpleFluid::Testing::QbmmCylinder
{
using Flow = Reacting::LowMachMixtureSolver;
enum class Geometry
{
    AnnularSector,
    NativeAnnulus,
    PolygonalDisk,
    PolygonalAnnulus
};
constexpr std::array geometries{
    Geometry::AnnularSector, Geometry::NativeAnnulus, Geometry::PolygonalDisk, Geometry::PolygonalAnnulus};
constexpr std::array names{"native_sector", "native_annulus", "polygon_disk", "polygon_annulus"};

inline bool native(Geometry kind)
{
    return kind == Geometry::AnnularSector || kind == Geometry::NativeAnnulus;
}
inline double inner_radius(Geometry kind)
{
    return kind == Geometry::PolygonalDisk ? 0. : 1.;
}
inline unsigned wedges(Geometry kind)
{
    return kind == Geometry::AnnularSector ? 4u : 8u;
}
inline double span(Geometry kind)
{
    return kind == Geometry::AnnularSector ? 1. : 2. * std::numbers::pi;
}

inline ArrReal edges(unsigned count, double low, double high)
{
    ArrReal out;
    for (unsigned i = 0; i <= count; ++i)
        out.push_back(std::lerp(low, high, double(i) / count));
    return out;
}

inline SP<Flow::Mesh> mesh(Geometry kind, unsigned rings, const ArrReal& z)
{
    const auto radial = edges(rings, inner_radius(kind), 2.);
    const unsigned sectors = wedges(kind);
    const auto angles = edges(sectors, kind == Geometry::AnnularSector ? -.5 : 0.,
        kind == Geometry::AnnularSector ? .5 : 2. * std::numbers::pi);
    const size_t ranks = Tpetra::getDefaultComm()->getSize();
    if (const char* expected = std::getenv("SIMPLEFLUID_Q8D_EXPECTED_RANKS"))
        EXPECT_EQ(std::to_string(ranks), expected);
    ::testing::Test::RecordProperty("mpi_ranks", static_cast<int>(ranks));
    Flow::Mesh::DistributionOptions distribution{.allow_empty_partitions = true};
    if (native(kind))
    {
        // Split in theta so a periodic seam is also an MPI interface.
        distribution.orthogonal_partitions = std::array<size_t, 3>{1, ranks, 1};
        return std::make_shared<Flow::Mesh>(
            std::make_shared<Flow::Mesh::Cylindrical>(Vec3D<ArrReal>{radial, angles, z}), distribution);
    }

    using Semi = Meshes::SemiStructuredXY_Z;
    const bool disk = kind == Geometry::PolygonalDisk;
    Arr<Semi::Vec3> nodes;
    Arr<Arr<unsigned>> cells;
    if (disk)
        nodes.push_back({0., 0., 0.});
    for (unsigned i = disk ? 1 : 0; i <= rings; ++i)
        for (unsigned j = 0; j < sectors; ++j)
            nodes.push_back({radial[i] * std::cos(angles[j]), radial[i] * std::sin(angles[j]), 0.});
    const auto vertex = [&](unsigned i, unsigned j)
    { return disk ? 1 + (i - 1) * sectors + j % sectors : i * sectors + j % sectors; };
    for (unsigned i = 0; i < rings; ++i)
        for (unsigned j = 0; j < sectors; ++j)
            if (disk && i == 0)
                cells.push_back({0, vertex(1, j), vertex(1, j + 1)});
            else
                cells.push_back({vertex(i, j), vertex(i + 1, j), vertex(i + 1, j + 1), vertex(i, j + 1)});
    distribution.xy_partitions = ranks;
    distribution.z_partitions = 1;
    return std::make_shared<Flow::Mesh>(std::make_shared<Semi>(nodes, cells, z), distribution);
}
} // namespace SimpleFluid::Testing::QbmmCylinder
