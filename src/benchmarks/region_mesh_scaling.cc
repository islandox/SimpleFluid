/**
 * @file region_mesh_scaling.cc
 * @author islandox
 * @brief Opt-in compact mesh construction, traversal and assembly scaling.
 * @version 0.1
 * @date 2026-09-09
 * @copyright Copyright (c) 2026
 */
#include "geometry/MeshHandle.hh"
#include "FVM/Operators.hh"
#include <Tpetra_Core.hpp>
#include <chrono>
#include <iostream>
#include <iomanip>

int main(int argc,char** argv)
{
    Tpetra::ScopeGuard guard(&argc,&argv);
    using namespace SimpleFluid;
    using namespace Meshes;
    using Clock=std::chrono::steady_clock;
    if(Tpetra::getDefaultComm()->getSize()!=1) { std::cerr<<"Serial benchmark only.\n"; return 1; }
    const size_t limit=argc>1?std::stoul(argv[1]):8;
    if(limit<2 || limit>256) { std::cerr<<"Maximum axis size must be in [2,256].\n"; return 1; }
    std::cout<<"mode,n,cells,construct_seconds,query_seconds,assembly_seconds,measured_mesh_bytes,topology_bytes,geometry_bytes,interfaces_bytes,indexing_bytes,compatibility_bytes,boundary_bytes,index_tree_estimate,map_id_payload_estimate,checksum\n";
    for(size_t n=2;n<=limit;n*=2)
    {
        ArrReal axis,left,right,whole;
        for(size_t i=0;i<=n;++i) { axis.push_back(real_t(i)/n); left.push_back(real_t(i)/n); right.push_back(1+real_t(i)/n); }
        whole=left; whole.insert(whole.end(),right.begin()+1,right.end());
        for(const std::string mode:{"native","multi_region","materialized"})
        {
            const auto start=Clock::now();
            SP<MeshHandle<>> mesh;
            if(mode=="multi_region")
                mesh=std::make_shared<MeshHandle<>>(std::make_shared<MultiRegionMesh>(std::vector<MultiRegionMesh::Region>{
                    cartesian_region("left",{{left,axis,axis}}),cartesian_region("right",{{right,axis,axis}})},
                    std::vector<MultiRegionMesh::Interface>{StructuredPatchInterface{{0,1},{1,0}}}));
            else
            {
                mesh=std::make_shared<MeshHandle<>>(std::make_shared<OrthogonalCartesian3D>(Vec3D<ArrReal>{{whole,axis,axis}}));
                if(mode=="materialized") { mesh->materialize_cell_faces(); static_cast<void>(mesh->indexer()); }
            }
            const auto built=Clock::now();
            real_t checksum=0;
            for(size_t c=0;c<mesh->num_cells();++c)
            {
                checksum+=mesh->cell_volume(c);
                auto query=[&](const auto& faces) { for(auto f:faces) checksum+=mesh->face_area(f); };
                if(mode=="materialized") query(mesh->materialized_faces(c)); else query(mesh->faces(c));
            }
            const auto queried=Clock::now();
            const auto matrix=FVM::diffusion_matrix<DefaultTpetraTypes>(*mesh,1.0);
            const auto assembled=Clock::now();
            const auto r=mesh->storage_report();
            std::cout<<std::setprecision(10)<<mode<<','<<n<<','<<mesh->num_cells()<<','
                <<std::chrono::duration<double>(built-start).count()<<','<<std::chrono::duration<double>(queried-built).count()<<','
                <<std::chrono::duration<double>(assembled-queried).count()<<','<<r.measured_bytes()<<','<<r.topology<<','<<r.geometry<<','
                <<r.interfaces<<','<<r.indexing<<','<<r.compatibility<<','<<r.boundaries<<','<<r.indexing_estimate<<','<<r.required_maps_estimate<<','<<checksum<<'\n';
        }
    }
}
