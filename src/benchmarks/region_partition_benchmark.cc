/**
 * @file region_partition_benchmark.cc
 * @author islandox
 * @brief Per-rank region ownership, communication and diffusion diagnostics.
 * @version 0.1
 * @date 2026-09-18
 * @copyright Copyright (c) 2026
 */
#include "FVM/Operators.hh"
#include "geometry/MeshHandle.hh"
#include <Teuchos_CommHelpers.hpp>
#include <Tpetra_Core.hpp>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <set>

namespace
{
using namespace SimpleFluid;
using namespace SimpleFluid::Meshes;
using Pack = DefaultTpetraTypes;
using Handle = MeshHandle<Pack>;
using Clock = std::chrono::steady_clock;

/** @brief Uniform coordinates on a closed interval. */
ArrReal axis(size_t n, real_t lower=0, real_t upper=1)
{
    ArrReal values;
    for(size_t i=0;i<=n;++i) values.push_back(lower+(upper-lower)*i/n);
    return values;
}
/** @brief Connected slabs in spatial or even/odd descriptor order. */
SP<MultiRegionMesh> slabs(size_t n,size_t count,bool shuffled)
{
    std::vector<size_t> order(count),inverse(count);
    std::iota(order.begin(),order.end(),0);
    if(shuffled) std::stable_partition(order.begin(),order.end(),[](size_t r){return r%2==0;});
    std::vector<MultiRegionMesh::Region> regions;
    std::vector<MultiRegionMesh::Interface> interfaces;
    for(size_t r=0;r<count;++r)
    {
        const auto physical=order[r];
        inverse[physical]=r;
        regions.push_back(cartesian_region("r"+std::to_string(physical),
            {{axis(n/count,real_t(physical)/count,real_t(physical+1)/count),axis(n),axis(n)}}));
    }
    for(size_t r=1;r<count;++r)
        interfaces.emplace_back(StructuredPatchInterface{{inverse[r-1],1},{inverse[r],0}});
    return std::make_shared<MultiRegionMesh>(std::move(regions),std::move(interfaces));
}
/** @brief Two half-domain blocks with nested 1:ratio planar subdivision. */
SP<MultiRegionMesh> refined(size_t n,size_t ratio)
{
    auto coarse=cartesian_region("coarse",{{axis(n/2,0,.5),axis(n),axis(n)}});
    auto fine=cartesian_region("fine",{{axis(n/2,.5,1),axis(ratio*n),axis(n)}});
    NonconformingInterface interface{0,1,1,0,{}};
    std::vector<std::vector<uint64_t>> tiles(n*n);
    const auto bucket=[n](const auto& region,size_t face)
    {
        const auto p=region.geometry().face_centroid(face);
        return std::min(n-1,size_t(p.z*n))*n+std::min(n-1,size_t(p.y*n));
    };
    for(size_t f=0;f<fine.layout().faces;++f)
        if(fine.topology().boundary_id(f)==0) tiles[bucket(fine,f)].push_back(f);
    for(size_t f=0;f<coarse.layout().faces;++f)
        if(coarse.topology().boundary_id(f)==1) interface.faces.push_back({f,tiles[bucket(coarse,f)]});
    return std::make_shared<MultiRegionMesh>(std::vector<MultiRegionMesh::Region>{coarse,fine},
        std::vector<MultiRegionMesh::Interface>{interface});
}
/** @brief Parse a bounded positive integer before allocation. */
size_t integer(const char* text,size_t maximum)
{
    const std::string value(text);
    if(value.empty() || value.find_first_not_of("0123456789")!=std::string::npos)
        throw std::invalid_argument("Arguments must be unsigned integers.");
    const auto result=std::stoull(value);
    if(!result || result>maximum) throw std::invalid_argument("Argument outside diagnostic bounds.");
    return result;
}
}

int main(int argc,char** argv)
{
    Tpetra::ScopeGuard guard(&argc,&argv);
    const auto comm=Tpetra::getDefaultComm();
    try
    {
        if(argc!=5) throw std::invalid_argument(
            "usage: region_partition_benchmark ordered|shuffled|refined axis_cells regions_or_ratio repeats");
        const std::string mode=argv[1];
        const size_t n=integer(argv[2],128),parameter=integer(argv[3],128),repeats=integer(argv[4],100);
        if(n<2 || n%2 || (mode!="ordered" && mode!="shuffled" && mode!="refined")
            || (mode=="refined" ? (parameter<2 || parameter>4) : (parameter>n || n%parameter)))
            throw std::invalid_argument("Even axis size required; slab count must divide it; refinement ratio must be 2..4.");
        const size_t ranks=comm->getSize(),rank=comm->getRank();
        const size_t cells=mode=="refined" ? n*n*n/2*(1+parameter) : n*n*n;
        if(cells/ranks<10000)
            throw std::invalid_argument("Diagnostic requires at least 10000 owned cells on every rank.");
        auto geometry=mode=="refined" ? refined(n,parameter) : slabs(n,parameter,mode=="shuffled");
        auto mesh=std::make_shared<Handle>(geometry);
        // This diagnostic measures initialize_composite's contiguous policy.
        const size_t quotient=cells/ranks,remainder=cells%ranks;
        const auto rank_of=[&](size_t gid)
        {
            const size_t split=(quotient+1)*remainder;
            return gid<split ? gid/(quotient+1) : remainder+(gid-split)/quotient;
        };
        const size_t expected=quotient+(rank<remainder),begin=quotient*rank+std::min(rank,remainder);
        if(mesh->num_owned_cells()!=expected || size_t(mesh->cell_global_id(0))!=begin)
            throw std::logic_error("Ownership policy changed; update this diagnostic.");
        const bool initial_indexer=mesh->has_materialized_indexer();
        const auto report=[&](const char* type,const std::vector<std::pair<std::string,double>>& values)
        {
            std::vector<double> local,all(values.size()*ranks);
            for(const auto& [name,value]:values) local.push_back(value);
            Teuchos::gatherAll(*comm,int(local.size()),local.data(),int(all.size()),all.data());
            if(!rank) for(size_t p=0;p<ranks;++p)
            {
                std::cout<<std::setprecision(17)<<"{\"type\":\""<<type<<"\",\"rank\":"<<p;
                for(size_t i=0;i<values.size();++i)
                    std::cout<<",\""<<values[i].first<<"\":"<<all[p*values.size()+i];
                std::cout<<"}\n";
            }
        };
        if(!rank) std::cout<<"{\"type\":\"metadata\",\"commit\":\""<<SIMPLEFLUID_GIT_COMMIT
            <<"\",\"dirty\":"<<SIMPLEFLUID_GIT_DIRTY<<",\"build\":\""<<SIMPLEFLUID_BUILD_TYPE
            <<"\",\"compiler\":\""<<SIMPLEFLUID_COMPILER<<"\",\"mode\":\""<<mode
            <<"\",\"axis_cells\":"<<n<<",\"parameter\":"<<parameter<<",\"ranks\":"<<ranks
            <<",\"global_cells\":"<<cells<<",\"repeats\":"<<repeats
            <<",\"warmups\":1,\"operator\":\"constant_scalar_diffusion\",\"solver\":\"none\"}\n";
        std::set<size_t> touched_regions,neighbor_ranks;
        size_t incidences=0,maximum_incidence=0,cut_faces=0,cut_region_faces=0,owned_region_faces=0;
        double cut_area=0,checksum=0;
        comm->barrier();
        const auto traversal_start=Clock::now();
        {
            const auto execution=mesh->acquire_execution_view();
            for(size_t c=0;c<mesh->num_owned_cells();++c)
            {
                const auto region=geometry->native_cell(mesh->cell_global_id(c)).first;
                touched_regions.insert(region);
                const auto faces=mesh->faces(c);
                incidences+=faces.size();
                maximum_incidence=std::max(maximum_incidence,faces.size());
                for(const auto f:faces)
                {
                    checksum+=mesh->face_area(f);
                    if(!mesh->is_interior_face(f)) continue;
                    const auto other=mesh->opposite_cell(f,c);
                    if(other==Handle::invalid_local_id()) throw std::logic_error("Missing first-layer ghost.");
                    if(mesh->is_owned_cell(other)) continue;
                    ++cut_faces;
                    cut_area+=mesh->face_area(f);
                    const auto gid=mesh->cell_global_id(other);
                    neighbor_ranks.insert(rank_of(gid));
                    cut_region_faces+=geometry->native_cell(gid).first!=region;
                }
            }
            for(size_t f=0;f<mesh->num_owned_faces();++f)
                if(mesh->is_interior_face(f))
                    owned_region_faces+=geometry->native_cell(mesh->cell_global_id(mesh->owner_cell(f))).first
                        !=geometry->native_cell(mesh->cell_global_id(mesh->neighbor_cell(f))).first;
        }
        const auto traversal_seconds=std::chrono::duration<double>(Clock::now()-traversal_start).count();
        auto condition=[](int,size_t){return BoundaryCondition{BoundaryConditionType::Dirichlet,1.};};
        auto source=[](int){return 0.;};
        auto system=FVM::diffusion_system<Pack>(*mesh,1.,condition,source);
        Pack::vector_type input(mesh->owned_cell_map(),true),action(mesh->owned_cell_map(),true);
        for(size_t c=0;c<mesh->num_owned_cells();++c)
        {
            const auto p=mesh->cell_centroid(c);
            input.replaceLocalValue(c,1+p.x+2*p.y-.5*p.z);
        }
        system.matrix->apply(input,action);
        for(size_t repeat=0;repeat<repeats;++repeat)
        {
            comm->barrier();
            const auto start=Clock::now();
            auto next=FVM::diffusion_system<Pack>(*mesh,1.,condition,source);
            const double assembly=std::chrono::duration<double>(Clock::now()-start).count();
            system=std::move(next);
            system.matrix->apply(input,action);
            comm->barrier();
            const auto apply_start=Clock::now();
            constexpr size_t applications=20;
            for(size_t i=0;i<applications;++i) system.matrix->apply(input,action);
            const double apply=std::chrono::duration<double>(Clock::now()-apply_start).count()/applications;
            report("sample",{{"repeat",double(repeat)},{"assembly_seconds",assembly},{"apply_seconds",apply}});
        }
        size_t maximum_row=0;
        for(size_t c=0;c<mesh->num_owned_cells();++c)
            maximum_row=std::max(maximum_row,system.matrix->getNumEntriesInLocalRow(c));
        report("partition",{{"owned_cells",double(mesh->num_owned_cells())},
            {"ghost_cells",double(mesh->num_local_cells()-mesh->num_owned_cells())},
            {"owned_faces",double(mesh->num_owned_faces())},{"owned_regions",double(touched_regions.size())},
            {"neighbor_ranks",double(neighbor_ranks.size())},{"face_incidences",double(incidences)},
            {"max_cell_faces",double(maximum_incidence)},{"cut_faces",double(cut_faces)},
            {"cut_region_faces",double(cut_region_faces)},{"cut_area",cut_area},
            {"owned_region_faces",double(owned_region_faces)},
            {"matrix_nnz",double(system.matrix->getLocalNumEntries())},{"max_row_nnz",double(maximum_row)},
            {"diagnostic_traversal_seconds",traversal_seconds},{"area_checksum",checksum}});
        const auto reference=FVM::detail::diffusion_system_reference_impl<Pack>(*mesh,1.,condition,source);
        Pack::vector_type expected_action(mesh->owned_cell_map(),true),difference(mesh->owned_cell_map(),true);
        reference.matrix->apply(input,expected_action);
        difference.update(1.,action,-1.,expected_action,0.);
        const auto action_error=difference.norm2()/std::max(1.,expected_action.norm2());
        difference.update(1.,*system.rhs,-1.,*reference.rhs,0.);
        const auto rhs_error=difference.norm2()/std::max(1.,reference.rhs->norm2());
        const auto action_norm=action.norm2(),rhs_norm=system.rhs->norm2();
        input.putScalar(1.);
        system.matrix->apply(input,action);
        action.update(-1.,*system.rhs,1.);
        const auto constant_error=action.normInf();
        const int expanded=mesh->connectivity_storage_bytes()!=0 || mesh->has_materialized_indexer()!=initial_indexer;
        int any_expanded=0;
        Teuchos::reduceAll(*comm,Teuchos::REDUCE_MAX,1,&expanded,&any_expanded);
        if(!std::isfinite(action_error) || !std::isfinite(rhs_error) || !std::isfinite(constant_error)
            || action_error>1e-12 || rhs_error>1e-12 || constant_error>1e-11 || any_expanded)
            throw std::runtime_error("Assembly equivalence, constant-field or compact-storage check failed.");
        if(!rank) std::cout<<std::setprecision(17)<<"{\"type\":\"validation\",\"passed\":true,\"action_relative_error\":"
            <<action_error<<",\"rhs_relative_error\":"<<rhs_error<<",\"constant_residual_max\":"<<constant_error
            <<",\"action_norm2\":"<<action_norm<<",\"rhs_norm2\":"<<rhs_norm
            <<",\"global_nnz\":"<<system.matrix->getGlobalNumEntries()<<"}\n";
    }
    catch(const std::exception& error)
    {
        std::cerr<<"rank "<<comm->getRank()<<": "<<error.what()<<'\n';
        // A rank-local failure must not strand peers in a collective.
        MPI_Abort(MPI_COMM_WORLD,1);
        return 1;
    }
}
