/**
 * @file ExtrudedRegionProviders.cc
 * @author islandox
 * @brief Factored incidence and base-plane metric implementations for independent extrusions.
 * @version 0.1
 * @date 2026-09-10
 * @copyright Copyright (c) 2026
 */
#include "geometry/mesh/ExtrudedRegionProviders.hh"
#include <cmath>
#include <numeric>
namespace SimpleFluid::Meshes
{
ExtrudedTopology::ExtrudedTopology(SemiStructMeshTopo topology):d_topology(std::move(topology))
{
    if(indexer().axial_periodic || !indexer().num_layers || !indexer().num_cells_per_layer)
        throw std::invalid_argument("Extrusion templates require nonempty nonperiodic axial layers.");
}
ExtrudedTopology::ExtrudedTopology(unsigned nodes,const Arr<Arr<unsigned>>& cells,unsigned layers,
    const Arr<SemiStructMeshTopo::BoundaryEdge>& boundaries):ExtrudedTopology(SemiStructMeshTopo(nodes,cells,layers,boundaries)) {}
RegionLayout ExtrudedTopology::layout() const
{
    const auto& i=indexer();
    return {RegionLayout::Family::Extruded,i.total_cells(),i.total_faces(),i.total_nodes(),
        {i.num_cells_per_layer,i.num_side_faces_per_layer,i.num_layers},this};
}
EntityRange<uint64_t> ExtrudedTopology::base_cell_nodes(size_t c) const
{
    return {this,c,d_topology.cell_side_faces(c).size(),[](const void* source,size_t c,size_t i)->ID
    {
        const auto& t=*static_cast<const ExtrudedTopology*>(source);
        const auto& edge=t.d_topology.side_face(t.d_topology.cell_side_faces(c)[i]);
        return edge.nodes[edge.owner==c?0:1];
    }};
}
EntityRange<uint64_t> ExtrudedTopology::cell_faces(size_t c) const
{
    if(c>=layout().cells) throw std::out_of_range("Extruded cell ordinal out of bounds.");
    const auto id=indexer().cell_id(c);
    return {this,c,d_topology.cell_side_faces(id.ij).size()+2,[](const void* source,size_t c,size_t i)->ID
    {
        const auto& t=*static_cast<const ExtrudedTopology*>(source);
        return t.indexer().face_ordinal(t.d_topology.cell_faces(t.indexer().cell_id(c))[i]);
    }};
}
EntityRange<uint64_t> ExtrudedTopology::cell_nodes(size_t c) const
{
    if(c>=layout().cells) throw std::out_of_range("Extruded cell ordinal out of bounds.");
    const auto id=indexer().cell_id(c);
    return {this,c,2*base_cell_nodes(id.ij).size(),[](const void* source,size_t c,size_t i)->ID
    {
        const auto& t=*static_cast<const ExtrudedTopology*>(source); const auto id=t.indexer().cell_id(c);
        const auto nodes=t.base_cell_nodes(id.ij);
        return t.indexer().node_ordinal({static_cast<unsigned>(nodes[i%nodes.size()]),static_cast<unsigned>(id.k+i/nodes.size())});
    }};
}
EntityRange<uint64_t> ExtrudedTopology::face_nodes(size_t f) const
{
    if(f>=layout().faces) throw std::out_of_range("Extruded face ordinal out of bounds.");
    const auto id=indexer().face_id(f);
    const auto count=id.orientation==SemiStructuredIndexer::AXIAL?base_cell_nodes(id.ij).size():4;
    return {this,f,count,[](const void* source,size_t f,size_t i)->ID
    {
        const auto& t=*static_cast<const ExtrudedTopology*>(source); const auto id=t.indexer().face_id(f);
        if(id.orientation==SemiStructuredIndexer::AXIAL)
            return t.indexer().node_ordinal({static_cast<unsigned>(t.base_cell_nodes(id.ij)[i]),id.k});
        const auto& edge=t.d_topology.side_face(id.ij);
        return t.indexer().node_ordinal({edge.nodes[(i==1 || i==2)?1:0],id.k+(i>=2?1U:0U)});
    }};
}
uint64_t ExtrudedTopology::owner_cell(size_t f) const { return indexer().cell_ordinal(d_topology.owner_cell(indexer().face_id(f))); }
uint64_t ExtrudedTopology::neighbor_cell(size_t f) const
{
    const auto c=d_topology.neighbor_cell(indexer().face_id(f));
    return c==SemiStructuredIndexer::CellID{}?UnstructuredMesh::invalid_ordinal:indexer().cell_ordinal(c);
}
MeshUtils::CellType ExtrudedTopology::cell_type(size_t c) const
{
    const auto n=base_cell_nodes(indexer().cell_id(c).ij).size();
    return n==3?MeshUtils::CellType::TRIPRISM:n==4?MeshUtils::CellType::HEXAHEDRON:MeshUtils::CellType::INVALID;
}
ExtrudedGeometry::ExtrudedGeometry(std::shared_ptr<const ExtrudedTopology> topology,Arr<Vec3> nodes,ArrReal z)
    :d_topology(std::move(topology)),d_nodes(std::move(nodes)),d_z(std::move(z))
{
    if(!d_topology || d_nodes.size()!=d_topology->indexer().num_nodes_per_layer
        || d_z.size()!=static_cast<size_t>(d_topology->indexer().num_layers)+1)
        throw std::invalid_argument("Extruded geometry and template extents mismatch.");
    for(const auto p:d_nodes)
        if(!std::isfinite(p.x) || !std::isfinite(p.y) || p.z!=0) throw std::invalid_argument("Extruded XY coordinates must be finite and lie at z=0.");
    for(size_t k=0;k<d_z.size();++k)
        if(!std::isfinite(d_z[k]) || (k && d_z[k]<=d_z[k-1])) throw std::invalid_argument("Extruded Z edges must be finite and increasing.");
    for(size_t c=0;c<d_topology->indexer().num_cells_per_layer;++c)
    {
        const auto loop=d_topology->base_cell_nodes(c); const auto origin=d_nodes[loop[0]];
        real_t area2=0,cx=0,cy=0;
        for(size_t i=0;i<loop.size();++i)
        {
            const auto a=d_nodes[loop[i]]-origin,b=d_nodes[loop[(i+1)%loop.size()]]-origin;
            const auto cross=a.x*b.y-b.x*a.y;
            area2+=cross; cx+=(a.x+b.x)*cross; cy+=(a.y+b.y)*cross;
        }
        if(!std::isfinite(area2) || !(area2>0) || !std::isfinite(cx) || !std::isfinite(cy)) throw std::invalid_argument("Extruded cells require positive counter-clockwise geometry.");
        for(size_t i=0;i<loop.size();++i)
        {
            const auto a=d_nodes[loop[i]],b=d_nodes[loop[(i+1)%loop.size()]],c=d_nodes[loop[(i+2)%loop.size()]];
            if((b.x-a.x)*(c.y-b.y)-(b.y-a.y)*(c.x-b.x)<=0)
                throw std::invalid_argument("Independent extruded base cells must be strictly convex and counter-clockwise.");
        }
        d_areas.push_back(area2/2); d_cell_centers.push_back(origin+Vec3{cx/(3*area2),cy/(3*area2),0});
    }
    for(const auto& edge:d_topology->native_topology().side_faces())
    {
        const auto a=d_nodes[edge.nodes[0]], b=d_nodes[edge.nodes[1]], d=b-a;
        const auto length=std::hypot(d.x,d.y);
        if(!std::isfinite(length) || !(length>0)) throw std::invalid_argument("Extruded base edges require positive finite length.");
        d_lengths.push_back(length); d_edge_centers.push_back({std::midpoint(a.x,b.x),std::midpoint(a.y,b.y),0}); d_edge_normals.push_back({d.y/length,-d.x/length,0});
    }
}
real_t ExtrudedGeometry::cell_volume(size_t c) const
{
    const auto id=d_topology->indexer().cell_id(c); return d_areas.at(id.ij)*(d_z.at(id.k+1)-d_z.at(id.k));
}
ExtrudedGeometry::Vec3 ExtrudedGeometry::cell_centroid(size_t c) const
{
    const auto id=d_topology->indexer().cell_id(c); auto center=d_cell_centers.at(id.ij);
    center.z=std::midpoint(d_z.at(id.k),d_z.at(id.k+1)); return center;
}
real_t ExtrudedGeometry::face_area(size_t f) const
{
    const auto id=d_topology->indexer().face_id(f);
    return id.orientation==SemiStructuredIndexer::AXIAL?d_areas.at(id.ij):d_lengths.at(id.ij)*(d_z.at(id.k+1)-d_z.at(id.k));
}
ExtrudedGeometry::Vec3 ExtrudedGeometry::face_centroid(size_t f) const
{
    const auto id=d_topology->indexer().face_id(f);
    auto center=id.orientation==SemiStructuredIndexer::AXIAL?d_cell_centers.at(id.ij):d_edge_centers.at(id.ij);
    center.z=id.orientation==SemiStructuredIndexer::AXIAL?d_z.at(id.k):std::midpoint(d_z.at(id.k),d_z.at(id.k+1));
    return center;
}
ExtrudedGeometry::Vec3 ExtrudedGeometry::face_area_vector(size_t f) const
{
    const auto id=d_topology->indexer().face_id(f);
    if(id.orientation==SemiStructuredIndexer::AXIAL) return {0,0,(id.k==0?-1.0:1.0)*face_area(f)};
    return d_edge_normals.at(id.ij)*face_area(f);
}
ExtrudedGeometry::Vec3 ExtrudedGeometry::node_coordinates(size_t n) const
{
    const auto id=d_topology->indexer().node_id(n); auto p=d_nodes.at(id.ij); p.z=d_z.at(id.k); return p;
}
MeshStorageReport ExtrudedGeometry::storage_report() const
{
    return {.geometry=(d_nodes.capacity()+d_cell_centers.capacity()+d_edge_centers.capacity()+d_edge_normals.capacity())*sizeof(Vec3)
        +(d_z.capacity()+d_areas.capacity()+d_lengths.capacity())*sizeof(real_t)};
}
EntityRange<EntityRange<uint64_t>> ExtrudedGeometry::xy_cell_nodes() const
{
    return {d_topology.get(),0,d_topology->indexer().num_cells_per_layer,[](const void* t,size_t,size_t c)
        { return static_cast<const ExtrudedTopology*>(t)->base_cell_nodes(c); }};
}
ExtrudedRegion extruded_region(std::string name,std::shared_ptr<const ExtrudedTopology> topology,Arr<MeshUtils::Vec3> nodes,ArrReal z)
{
    auto geometry=std::make_shared<const ExtrudedGeometry>(topology,std::move(nodes),std::move(z));
    return {std::move(name),std::move(topology),std::move(geometry)};
}
} // namespace SimpleFluid::Meshes
