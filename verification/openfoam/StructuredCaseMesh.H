// Coordinate-based IDs and bounds for shared, nonuniform verification meshes.
#ifndef StructuredCaseMesh_H
#define StructuredCaseMesh_H
#include "fvCFD.H"
#include <algorithm>
struct StructuredCaseMesh
{
    Foam::scalarList x,y,z;
    explicit StructuredCaseMesh(const Foam::fvMesh& mesh)
    {
        Foam::IOdictionary dictionary(Foam::IOobject("referenceMesh",mesh.time().constant(),mesh,Foam::IOobject::MUST_READ,Foam::IOobject::NO_WRITE));
        x=Foam::scalarList(dictionary.lookup("xEdges"));y=Foam::scalarList(dictionary.lookup("yEdges"));z=Foam::scalarList(dictionary.lookup("zEdges"));
        if(mesh.nCells()!=(x.size()-1)*(y.size()-1)*(z.size()-1)||y.size()!=2)
            FatalErrorInFunction<<"Mesh counts differ from shared edges"<<Foam::exit(Foam::FatalError);
    }
    static Foam::label interval(const Foam::scalarList& edges,Foam::scalar center)
    {
        auto it=std::upper_bound(edges.begin(),edges.end(),center);
        if(it==edges.begin()||it==edges.end())FatalErrorInFunction<<"Cell outside shared mesh"<<Foam::exit(Foam::FatalError);
        return Foam::label(it-edges.begin()-1);
    }
    Foam::label nx()const{return x.size()-1;}
    Foam::label nz()const{return z.size()-1;}
};
#endif
