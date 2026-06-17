/**
 * @file testOrthogonalDiffusionProfile.cc
 * @brief Performance profiling: 100³ orthogonal mesh diffusion solve.
 *
 * Exercises the template-heavy MeshBase / OrthoMeshTopo code paths by
 * assembling and solving a scalar diffusion problem on a 1M-cell mesh.
 * Designed for sampling profilers (macOS sample / Instruments).
 */

#include <gtest/gtest.h>

#include "geometry/mesh/OrthogonalCartesian3D.hh"
#include "solvers/BelosLinearSolver.hh"
#include "FVM/OperatorDetails.hh"
#include "utils/testing_environment.hh"

#include <Tpetra_CrsMatrix.hpp>
#include <Tpetra_Map.hpp>
#include <Tpetra_MultiVector.hpp>
#include <Tpetra_Vector.hpp>
#include <Teuchos_Array.hpp>
#include <Teuchos_Comm.hpp>
#include <Teuchos_RCP.hpp>

#include <cmath>
#include <chrono>
#include <iostream>

namespace
{

using Pack = SimpleFluid::TpetraTypes<>;
using GO = typename Pack::global_ordinal_type;
using LO = typename Pack::local_ordinal_type;
using Scalar = typename Pack::scalar_type;
using Map = typename Pack::map_type;
using Matrix = typename Pack::matrix_type;
using Vector = typename Pack::vector_type;
using MultiVector = typename Pack::multi_vector_type;

using CartMesh = SimpleFluid::Meshes::OrthogonalCartesian3D;
using CellID = CartMesh::CellID;

using utils_test::KokkosEnvironment;
testing::Environment* const kokkos_environment =
    testing::AddGlobalTestEnvironment(new KokkosEnvironment);

// ---------------------------------------------------------------------------
// Helper: linear spacing
// ---------------------------------------------------------------------------
SimpleFluid::ArrReal linspace(double start, double end, size_t n)
{
    SimpleFluid::ArrReal v(n + 1);
    for (size_t i = 0; i <= n; ++i) {
        v[i] = start + (end - start) * static_cast<double>(i) / static_cast<double>(n);
    }
    return v;
}

// ---------------------------------------------------------------------------
// Manufactured solution: u = sin(πx)sin(πy)sin(πz)  with  u=0 on boundary
// -∇²u = 3π² u  →  f = 3π² sin(πx)sin(πy)sin(πz)
// ---------------------------------------------------------------------------
constexpr double pi = 3.14159265358979323846;

Scalar manufactured_u(const CellID& cell, const CartMesh& mesh)
{
    const auto c = mesh.cell_centroid(cell);
    return std::sin(pi * c.x) * std::sin(pi * c.y) * std::sin(pi * c.z);
}

Scalar manufactured_source(const CellID& cell, const CartMesh& mesh)
{
    return 3.0 * pi * pi * manufactured_u(cell, mesh);
}

// ---------------------------------------------------------------------------
// Build a Tpetra contiguous uniform map for the mesh's owned cells
// ---------------------------------------------------------------------------
Teuchos::RCP<const Map> build_owned_map(const CartMesh& mesh,
                                        const Teuchos::RCP<const Teuchos::Comm<int>>& comm)
{
    return Teuchos::rcp(new Map(
        Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid(),
        static_cast<size_t>(mesh.num_owned_cells()),
        static_cast<GO>(0),
        comm));
}

// ---------------------------------------------------------------------------
// Assemble the diffusion matrix and RHS
// ---------------------------------------------------------------------------
std::pair<Teuchos::RCP<Matrix>, Teuchos::RCP<MultiVector>>
assemble_diffusion(const CartMesh& mesh,
                   const Teuchos::RCP<const Map>& owned_map,
                   Scalar diffusivity = 1.0)
{
    auto matrix = Teuchos::rcp(new Matrix(owned_map, owned_map, 7));
    auto rhs = Teuchos::rcp(new MultiVector(owned_map, 1, true));
    auto rhs_2d = rhs->getLocalViewHost(Tpetra::Access::ReadWrite);

    Teuchos::Array<LO> cols(7);
    Teuchos::Array<Scalar> vals(7);

    // --- Interior contributions ---
    for (size_t local_id = 0; local_id < mesh.num_owned_cells(); ++local_id)
    {
        const auto cell = mesh.cell_id(local_id);
        const auto cell_lid = static_cast<LO>(local_id);

        Scalar diagonal = 0.0;
        LO num_cols = 0;

        for (const auto face_id : mesh.faces(cell))
        {
            if (!mesh.is_interior_face(face_id)) continue;

            const auto neighbor_id =
                mesh.opposite_or_periodic_neighbor_cell(face_id, cell);
            const auto neighbor_lid =
                static_cast<LO>(mesh.cell_local_id(neighbor_id));

            const auto d =
                mesh.cell_centroid(neighbor_id) - mesh.cell_centroid(cell);
            const auto d2 = d.x * d.x + d.y * d.y + d.z * d.z;
            const auto area = mesh.face_area(face_id);
            const auto& normal = mesh.face_normal_outward(face_id, cell);
            const auto proj = area * (normal.x * d.x + normal.y * d.y + normal.z * d.z) / d2;
            const auto coeff = diffusivity * proj;

            diagonal += coeff;
            cols[num_cols] = neighbor_lid;
            vals[num_cols] = -coeff;
            ++num_cols;
        }

        cols[num_cols] = cell_lid;
        vals[num_cols] = diagonal;
        ++num_cols;

        matrix->insertLocalValues(
            cell_lid,
            cols.view(0, num_cols),
            vals.view(0, num_cols));

        rhs_2d(cell_lid, 0) =
            mesh.cell_volume(cell) * manufactured_source(cell, mesh);
    }

    // --- Boundary contributions (homogeneous Dirichlet) ---
    for (int batch_id : mesh.boundary_batch_ids())
    {
        for (auto face_id : mesh.boundary_face_batch(batch_id))
        {
            const auto face_lid = mesh.face_local_id(face_id);
            if (!mesh.is_boundary_face(face_lid)) continue;

            const auto owner_id = mesh.owner_cell(face_id);
            const auto owner_lid = static_cast<LO>(mesh.cell_local_id(owner_id));

            const auto d =
                mesh.face_centroid(face_id) - mesh.cell_centroid(owner_id);
            const auto d2 = d.x * d.x + d.y * d.y + d.z * d.z;
            if (d2 <= Scalar{0}) continue;

            const auto& normal = mesh.face_normal_outward(face_id, owner_id);
            const auto area = mesh.face_area(face_id);
            const auto proj = area * (normal.x * d.x + normal.y * d.y + normal.z * d.z) / d2;
            const auto coeff = diffusivity * proj;

            if (coeff > Scalar{0})
            {
                matrix->sumIntoLocalValues(
                    owner_lid,
                    Teuchos::arrayView(&owner_lid, 1),
                    Teuchos::arrayView(&coeff, 1));
                rhs_2d(owner_lid, 0) += coeff * Scalar{0};
            }
        }
    }

    matrix->fillComplete();
    return {matrix, rhs};
}

} // namespace

// ---------------------------------------------------------------------------
// Test: 100³ diffusion solve with manufactured solution
// ---------------------------------------------------------------------------
TEST(OrthogonalDiffusionProfile, DISABLED_HundredCubedManufacturedSolution)
{
    constexpr size_t n = 100;
    const auto edges_x = linspace(0.0, 1.0, n);
    const auto edges_y = linspace(0.0, 1.0, n);
    const auto edges_z = linspace(0.0, 1.0, n);

    std::cout << "Building " << n << "×" << n << "×" << n
              << " mesh (" << n*n*n << " cells)..." << std::flush;
    auto t0 = std::chrono::steady_clock::now();

    CartMesh mesh({{edges_x, edges_y, edges_z}});

    auto t1 = std::chrono::steady_clock::now();
    std::cout << " " << std::chrono::duration<double>(t1 - t0).count()
              << " s\n";

    std::cout << "  cells: " << mesh.num_cells()
              << "  faces: " << mesh.num_faces()
              << "  nodes: " << mesh.num_nodes() << "\n";

    // Build Tpetra map
    auto comm = Tpetra::getDefaultComm();
    auto owned_map = build_owned_map(mesh, comm);

    SimpleFluid::LinearSolverOptions opts;
    opts.max_iterations = 500;
    opts.tolerance = 1e-8;
    opts.verbosity = Belos::Errors;

    // --- Assemble and solve multiple times for profiling ---
    std::cout << "Profiling: assembling & solving 5 iterations..." << std::flush;
    auto t_prof0 = std::chrono::steady_clock::now();
    for (int iter = 0; iter < 5; ++iter)
    {
        auto [m, r] = assemble_diffusion(mesh, owned_map);
        Teuchos::RCP<const Matrix> const_m = m;
        MultiVector xi(owned_map, 1, true);
        SimpleFluid::solve_linear_system<Pack>(const_m, *r, xi, opts);
        if (iter == 0)
        {
            // Verify only the first iteration
            Scalar max_err = 0.0;
            Scalar l2_err = 0.0;
            auto xi_2d = xi.getLocalViewHost(Tpetra::Access::ReadOnly);
            for (size_t local_id = 0; local_id < mesh.num_owned_cells(); ++local_id)
            {
                const auto cell = mesh.cell_id(local_id);
                const auto exact = manufactured_u(cell, mesh);
                const auto err = std::abs(xi_2d(local_id, 0) - exact);
                if (err > max_err) max_err = err;
                l2_err += err * err * mesh.cell_volume(cell);
            }
            l2_err = std::sqrt(l2_err);
            auto t1 = std::chrono::steady_clock::now();
            std::cout << "\n  Verify: L2=" << l2_err
                      << "  max=" << max_err << "\n";
            EXPECT_LT(max_err, 1e-4);
            EXPECT_LT(l2_err, 1e-4);
        }
    }
    auto t_prof1 = std::chrono::steady_clock::now();
    std::cout << "  Total 5 iters: "
              << std::chrono::duration<double>(t_prof1 - t_prof0).count() << " s\n";
}
