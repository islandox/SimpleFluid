module;

#if !defined(SIMPLEFLUID_USE_STD_MODULE)
#include "cmake/StandardHeaders.hh"
#endif

#include "geometry/MeshUtils.hh"
#include "geometry/mesh/MeshIndexTypes.hh"
#include "geometry/mesh/BoundaryFaceBatch.hh"
#include "geometry/mesh/StructuredBatchView.hh"
#include "geometry/mesh/LocalGlobalIndexer.hh"
#include "geometry/mesh/OrthogonalIndexer.hh"
#include "geometry/mesh/OrthogonalLocalGlobalIndexer.hh"
#include "geometry/mesh/SemiStructuredIndexer.hh"
#include "geometry/mesh/MeshBase.hh"
#include "geometry/mesh/OrthoMeshTopo.hh"
#include "geometry/mesh/SemiStructMeshTopo.hh"
#include "geometry/mesh/OrthoMeshPartitioner.hh"
#include "geometry/mesh/OrthogonalCartesian3D.hh"
#include "geometry/mesh/OrthogonalCylindrial3D.hh"
#include "geometry/mesh/SemiStructuredXY_Z.hh"
#include "geometry/mesh/UnstructuredMesh.hh"
#include "geometry/mesh/PartitionedMeshBase.hh"
#include "geometry/mesh/STKMeshAdapter.hh"
#include "geometry/Mesh.hh"
#include "geometry/STKMesh.hh"
#include "geometry/MeshHandle.hh"
#include "geometry/MeshFactory.hh"
#include "geometry/BoundaryLayerMeshFactory.hh"
#include "geometry/MeshQuality.hh"
#include "geometry/WallYPlusStatistics.hh"
#include "geometry/YPlusBoundaryLayerController.hh"
#include "geometry/mesh/FrontalDelaunay2D.hh"
#include "io/VTUWriter.hh"
#include "parallel/MPI_interface.hh"
#include "parallel/MeshPartitioner.hh"

export module SimpleFluid.Mesh;

#if defined(SIMPLEFLUID_USE_STD_MODULE)
import std;
#endif

export import SimpleFluid.Core;

export namespace SimpleFluid
{
using ::SimpleFluid::invalid_id;
using ::SimpleFluid::MeshIndexTypePack;
using ::SimpleFluid::MeshIndexer;
using ::SimpleFluid::MeshIndexTypes;
using ::SimpleFluid::MeshBase;
using ::SimpleFluid::MeshClass;
using ::SimpleFluid::Mesh;
using ::SimpleFluid::STKMeshContainer;
using ::SimpleFluid::STKMesh;
using ::SimpleFluid::MeshHandle;
using ::SimpleFluid::MeshFactory;
using ::SimpleFluid::BoundaryLayerMeshFactory;
using ::SimpleFluid::MeshQualityMetrics;
using ::SimpleFluid::MeshQualityLimits;
using ::SimpleFluid::MeshQualityAssessment;
using ::SimpleFluid::MeshQualityGate;
using ::SimpleFluid::evaluate_mesh_quality;
using ::SimpleFluid::WallYPlusControlStatistic;
using ::SimpleFluid::WallYPlusSample;
using ::SimpleFluid::WallYPlusStatistics;
using ::SimpleFluid::WallYPlusSamplesByPatch;
using ::SimpleFluid::reduce_wall_y_plus_statistics;
using ::SimpleFluid::YPlusBoundaryLayerControllerOptions;
using ::SimpleFluid::YPlusBoundaryLayerHeightUpdate;
using ::SimpleFluid::YPlusBoundaryLayerUpdate;
using ::SimpleFluid::YPlusBoundaryLayerController;
using ::SimpleFluid::YPlusBoundaryLayerAdaptationStatus;
using ::SimpleFluid::YPlusBoundaryLayerAdaptationOptions;
using ::SimpleFluid::YPlusBoundaryLayerCycleReport;
using ::SimpleFluid::YPlusBoundaryLayerAdaptationReport;
using ::SimpleFluid::YPlusBoundaryLayerAdaptationDriver;
using ::SimpleFluid::VTUWriter;
using ::SimpleFluid::MeshPartitioner;
}

export namespace SimpleFluid::MeshUtils
{
using ::SimpleFluid::MeshUtils::CellType;
using ::SimpleFluid::MeshUtils::FaceType;
using ::SimpleFluid::MeshUtils::Vec3;
using ::SimpleFluid::MeshUtils::vtu_cell_type_code;
using ::SimpleFluid::MeshUtils::average;
using ::SimpleFluid::MeshUtils::tetra_volume;
using ::SimpleFluid::MeshUtils::hex_volume;
using ::SimpleFluid::MeshUtils::wedge_volume;
using ::SimpleFluid::MeshUtils::face_area_vector;
using ::SimpleFluid::MeshUtils::consec_diff;
using ::SimpleFluid::MeshUtils::consec_mid;
}

export namespace SimpleFluid::Meshes
{
using ::SimpleFluid::Meshes::BoundaryFaceBatch;
using ::SimpleFluid::Meshes::FrontalDelaunay2D;
using ::SimpleFluid::Meshes::LocalGlobalIndexer;
using ::SimpleFluid::Meshes::OrthogonalIndexer;
using ::SimpleFluid::Meshes::OrthogonalMeshIndexTypePack;
using ::SimpleFluid::Meshes::OrthogonalMeshIndexTypes;
using ::SimpleFluid::Meshes::SemiStructuredIndexer;
using ::SimpleFluid::Meshes::SemiStructuredMeshIndexTypes;
using ::SimpleFluid::Meshes::OrthoMeshPartitioner;
using ::SimpleFluid::Meshes::OrthoMeshTopo;
using ::SimpleFluid::Meshes::SemiStructMeshTopo;
using ::SimpleFluid::Meshes::OrthogonalCartesian3D;
using ::SimpleFluid::Meshes::OrthogonalCylindrial3D;
using ::SimpleFluid::Meshes::OrthogonalCylindrical3D;
using ::SimpleFluid::Meshes::SemiStructuredXY_Z;
using ::SimpleFluid::Meshes::SemiStructuredXYZ3D;
using ::SimpleFluid::Meshes::SemiStructuredXY_Z3D;
using ::SimpleFluid::Meshes::UnstructuredMeshIndexTypes;
using ::SimpleFluid::Meshes::UnstructuredMesh;
using ::SimpleFluid::Meshes::UnstructuredMesh3D;
using ::SimpleFluid::Meshes::PartitionedMesh;
using ::SimpleFluid::Meshes::PartitionedMeshClass;
using ::SimpleFluid::Meshes::STKMeshAdapter;
using ::SimpleFluid::Meshes::cartesian_product_2d;
using ::SimpleFluid::Meshes::cartesian_product_3d;
}
