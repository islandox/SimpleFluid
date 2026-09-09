/**
 * @file region_diffusion.cc
 * @author islandox
 * @brief Diffusion through two compact conforming Cartesian regions.
 * @version 0.1
 * @date 2026-09-09
 * @copyright Copyright (c) 2026
 */
#include "geometry/MeshHandle.hh"
#include "FVM/Operators.hh"
#include "solvers/BelosLinearSolver.hh"
#include <Tpetra_Core.hpp>
#include <Teuchos_CommHelpers.hpp>
#include <iostream>
#include <iomanip>

int main(int argc, char** argv)
{
    Tpetra::ScopeGuard guard(&argc, &argv);
    using namespace SimpleFluid;
    using namespace Meshes;
    const auto comm=Tpetra::getDefaultComm();
    try
    {
        auto composite = std::make_shared<MultiRegionMesh>(std::vector<MultiRegionMesh::Region>{
            cartesian_region("left", {{{0,0.25,0.5,0.75,1},{0,0.5,1},{0,0.5,1}}}),
            cartesian_region("right", {{{1,1.25,1.5,1.75,2},{0,0.5,1},{0,0.5,1}}})},
            std::vector<MultiRegionMesh::Interface>{StructuredPatchInterface{{0,1},{1,0}}});
        SP<const MeshHandle<>> mesh = std::make_shared<MeshHandle<>>(composite);
        auto condition = [&](int b, size_t i)
        {
            const auto f = mesh->boundary_face_batch(b).face_lids[i];
            return BoundaryCondition{BoundaryConditionType::Dirichlet, mesh->face_centroid(f).x};
        };
        const auto system = FVM::diffusion_system<DefaultTpetraTypes>(*mesh, 1.0, condition, [](int){return 0.0;});
        DefaultTpetraTypes::vector_type solution(mesh->owned_cell_map(), true);
        BelosLinearSolver<> solver;
        LinearSolverOptions options; options.tolerance = 1e-12;
        if (!solver.solve(system.matrix, *system.rhs, solution, options)) return 2;
        const auto values = solution.getData();
        real_t error = 0, exterior_flux = 0;
        for (size_t c=0;c<mesh->num_owned_cells();++c) error=std::max(error,std::abs(values[c]-mesh->cell_centroid(c).x));
        for (const auto& [b,batch]:mesh->boundary_batches())
            for (auto f:batch.face_lids)
            {
                if(!mesh->is_owned_face(f)) continue;
                const auto c=mesh->owner_cell(f);
                exterior_flux += mesh->face_area(f)*(values[c]-mesh->face_centroid(f).x)/mesh->cell_to_face_distance(f,c);
            }
        real_t global_error=0,global_flux=0;
        Teuchos::reduceAll(*comm,Teuchos::REDUCE_MAX,1,&error,&global_error);
        Teuchos::reduceAll(*comm,Teuchos::REDUCE_SUM,1,&exterior_flux,&global_flux);
        error=global_error; exterior_flux=global_flux;
        const auto report=mesh->storage_report();
        if(comm->getRank()==0)
        std::cout << std::setprecision(12) << "cells=" << composite->num_cells() << " faces=" << composite->num_faces()
            << " mpi_ranks=" << comm->getSize()
            << " max_error=" << error << " net_outward_diffusive_flux=" << exterior_flux
            << "\nrank0_topology_bytes=" << report.topology << " geometry_bytes=" << report.geometry
            << " interface_bytes=" << report.interfaces << " indexing_bytes=" << report.indexing
            << " boundary_bytes=" << report.boundaries << " compatibility_bytes=" << report.compatibility
            << " measured_mesh_bytes=" << report.measured_bytes() << " map_id_payload_estimate=" << report.required_maps_estimate << '\n';
        const auto output = mesh->vtu_topology();
        const size_t output_bytes=output->points.capacity()*sizeof(MeshUtils::Vec3)
            +(output->connectivity.capacity()+output->cell_offsets.capacity())*sizeof(global_index_t)
            +output->cell_types.capacity()*sizeof(uint8_t);
        VTUWriter writer(output);
        writer.add_scalar_cell_data("phi", {values.begin(),values.end()});
        VTUWriter::Int64Data region_ids,cell_ids;
        for(size_t c=0;c<mesh->num_owned_cells();++c)
        {
            const auto id=mesh->cell_geometry_global_id(c);
            region_ids.push_back(composite->native_cell(id).first); cell_ids.push_back(id);
        }
        writer.add_int64_cell_data("topology_region",std::move(region_ids));
        writer.add_int64_cell_data("cell_gid",std::move(cell_ids));
        const std::string filename=argc>1?argv[1]:"region_diffusion.vtu";
        const auto piece=VTUWriter::rank_piece_filename(filename,comm->getRank(),comm->getSize());
        int failed=0,any_failed=0;
        std::exception_ptr output_error;
        try { writer.write(piece); } catch(...) { failed=1; output_error=std::current_exception(); }
        Teuchos::reduceAll(*comm,Teuchos::REDUCE_MAX,1,&failed,&any_failed);
        if(any_failed)
        {
            if(output_error) std::rethrow_exception(output_error);
            throw std::runtime_error("A rank could not write its region VTU piece.");
        }
        if(comm->getSize()>1 && comm->getRank()==0)
        {
            std::vector<std::string> pieces;
            for(int rank=0;rank<comm->getSize();++rank) pieces.push_back(VTUWriter::rank_piece_filename(filename,rank,comm->getSize()));
            try { writer.write_parallel_index(VTUWriter::parallel_index_filename(filename),pieces); }
            catch(...) { failed=1; output_error=std::current_exception(); }
        }
        Teuchos::reduceAll(*comm,Teuchos::REDUCE_MAX,1,&failed,&any_failed);
        if(any_failed)
        {
            if(output_error) std::rethrow_exception(output_error);
            throw std::runtime_error("Could not publish the region PVTU index.");
        }
        if(comm->getRank()==0)
            std::cout << "rank0_output_topology_bytes=" << output_bytes << " output="
                << (comm->getSize()>1?VTUWriter::parallel_index_filename(filename):filename) << '\n';
        int materialized=mesh->connectivity_storage_bytes()!=0;
        Teuchos::reduceAll(*comm,Teuchos::REDUCE_MAX,1,&materialized,&any_failed);
        return error < 1e-10 && std::abs(exterior_flux)<1e-10 && any_failed==0 ? 0 : 3;
    }
    catch(const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
