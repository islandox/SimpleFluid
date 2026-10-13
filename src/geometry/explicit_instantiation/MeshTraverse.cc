/** @file MeshTraverse.cc
 * @brief Explicit instantiations of traversal setup for supported mesh families.
 */
#include "geometry/MeshTraverse.tcc"
#include "geometry/SolidSubdomain.hh"

namespace SimpleFluid
{

template class MeshTraversePlan<MeshHandle<DefaultTpetraTypes>>;
template class MeshTraversePlan<Mesh<DefaultTpetraTypes>>;
template class MeshTraversePlan<SolidSubdomain<DefaultTpetraTypes>>;
template class MeshTraversePlan<Meshes::OrthogonalCartesian3D>;
template class MeshTraversePlan<Meshes::OrthogonalCylindrial3D>;
template class MeshTraversePlan<Meshes::SemiStructuredXY_Z>;
template class MeshTraversePlan<Meshes::UnstructuredMesh>;
template class MeshTraversePlan<Meshes::MultiRegionMesh>;
template class MeshTraversePlan<Meshes::STKMeshAdapter<DefaultTpetraTypes>>;
template class MeshTraversePlan<Meshes::PartitionedMesh<Meshes::OrthogonalCartesian3D, DefaultTpetraTypes>>;
template class MeshTraversePlan<Meshes::PartitionedMesh<Meshes::UnstructuredMesh, DefaultTpetraTypes>>;

namespace detail
{

using Handle = MeshHandle<DefaultTpetraTypes>;
using Legacy = Mesh<DefaultTpetraTypes>;
using Solid = SolidSubdomain<DefaultTpetraTypes>;
using Cartesian = Meshes::OrthogonalCartesian3D;
using Cylindrical = Meshes::OrthogonalCylindrial3D;
using Layered = Meshes::SemiStructuredXY_Z;
using Unstructured = Meshes::UnstructuredMesh;
using Composite = Meshes::MultiRegionMesh;
using STK = Meshes::STKMeshAdapter<DefaultTpetraTypes>;
using PartitionedCartesian = Meshes::PartitionedMesh<Cartesian, DefaultTpetraTypes>;
using PartitionedUnstructured = Meshes::PartitionedMesh<Unstructured, DefaultTpetraTypes>;

template std::optional<StructuredTraverseLayout> TraverseAccess<Handle, Cartesian>::structured_layout() const;
template std::shared_ptr<const LayeredTraverseLayout> TraverseAccess<Handle, Cartesian>::layered_layout() const;

template std::optional<StructuredTraverseLayout> TraverseAccess<Handle, Cylindrical>::structured_layout() const;
template std::shared_ptr<const LayeredTraverseLayout> TraverseAccess<Handle, Cylindrical>::layered_layout() const;

template std::optional<StructuredTraverseLayout> TraverseAccess<Handle, Layered>::structured_layout() const;
template std::shared_ptr<const LayeredTraverseLayout> TraverseAccess<Handle, Layered>::layered_layout() const;

template std::optional<StructuredTraverseLayout> TraverseAccess<Handle, Unstructured>::structured_layout() const;
template std::shared_ptr<const LayeredTraverseLayout> TraverseAccess<Handle, Unstructured>::layered_layout() const;

template std::optional<StructuredTraverseLayout> TraverseAccess<Handle, STK>::structured_layout() const;
template std::shared_ptr<const LayeredTraverseLayout> TraverseAccess<Handle, STK>::layered_layout() const;

template std::optional<StructuredTraverseLayout> TraverseAccess<Handle, Composite>::structured_layout() const;
template std::shared_ptr<const LayeredTraverseLayout> TraverseAccess<Handle, Composite>::layered_layout() const;

template std::optional<StructuredTraverseLayout> TraverseAccess<Legacy, Legacy>::structured_layout() const;
template std::shared_ptr<const LayeredTraverseLayout> TraverseAccess<Legacy, Legacy>::layered_layout() const;

template std::optional<StructuredTraverseLayout> TraverseAccess<Solid, Solid>::structured_layout() const;
template std::shared_ptr<const LayeredTraverseLayout> TraverseAccess<Solid, Solid>::layered_layout() const;

template std::optional<StructuredTraverseLayout> TraverseAccess<Cartesian, Cartesian>::structured_layout() const;
template std::shared_ptr<const LayeredTraverseLayout> TraverseAccess<Cartesian, Cartesian>::layered_layout() const;

template std::optional<StructuredTraverseLayout> TraverseAccess<Cylindrical, Cylindrical>::structured_layout() const;
template std::shared_ptr<const LayeredTraverseLayout> TraverseAccess<Cylindrical, Cylindrical>::layered_layout() const;

template std::optional<StructuredTraverseLayout> TraverseAccess<Layered, Layered>::structured_layout() const;
template std::shared_ptr<const LayeredTraverseLayout> TraverseAccess<Layered, Layered>::layered_layout() const;

template std::optional<StructuredTraverseLayout> TraverseAccess<Unstructured, Unstructured>::structured_layout() const;
template std::shared_ptr<const LayeredTraverseLayout>
TraverseAccess<Unstructured, Unstructured>::layered_layout() const;

template std::optional<StructuredTraverseLayout> TraverseAccess<Composite, Composite>::structured_layout() const;
template std::shared_ptr<const LayeredTraverseLayout> TraverseAccess<Composite, Composite>::layered_layout() const;

template std::optional<StructuredTraverseLayout> TraverseAccess<STK, STK>::structured_layout() const;
template std::shared_ptr<const LayeredTraverseLayout> TraverseAccess<STK, STK>::layered_layout() const;

template std::optional<StructuredTraverseLayout>
TraverseAccess<PartitionedCartesian, Cartesian>::structured_layout() const;
template std::shared_ptr<const LayeredTraverseLayout>
TraverseAccess<PartitionedCartesian, Cartesian>::layered_layout() const;

template std::optional<StructuredTraverseLayout>
TraverseAccess<PartitionedUnstructured, Unstructured>::structured_layout() const;
template std::shared_ptr<const LayeredTraverseLayout>
TraverseAccess<PartitionedUnstructured, Unstructured>::layered_layout() const;

} // namespace detail
} // namespace SimpleFluid
