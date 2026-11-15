/** @file testFrameworkConcepts.cc
 * @brief Compile-time structural contracts and their public API boundaries.
 */
#include "fields/FieldConcepts.hh"
#include "geometry/MeshConcepts.hh"
#include "geometry/mesh/RegionProviderConcepts.hh"

#include "equations/Equation.hh"
#include "fields/FieldStored.tcc"
#include "geometry/MeshHandle.hh"
#include "geometry/SolidSubdomain.hh"
#include "geometry/mesh/ExplicitRegionProvider.hh"
#include "geometry/mesh/ExtrudedRegionProviders.hh"
#include "geometry/mesh/SweptRZRegionProviders.hh"
#include "problems/Problem.hh"
#include "utils/testing_environment.hh"

#include <gtest/gtest.h>

#include <span>
#include <string>
#include <type_traits>
#include <vector>

namespace
{
using namespace SimpleFluid;
using Pack = DefaultTpetraTypes;
using Variant = FVM::OperatorVariant<Pack>;
using ScalarEquation = Equation<ScalarCellFieldStored<Pack>>;

// These independent declarations have no library base class or implementation.
struct QueryMesh
{
    using scalar_type = double;
    using local_ordinal_type = int;
    using global_ordinal_type = long long;
    size_t num_owned_cells() const;
    size_t num_local_cells() const;
    size_t num_faces() const;
    std::span<const int> faces(int) const;
    int owner_cell(int) const;
    int neighbor_cell(int) const;
    double cell_volume(int) const;
    const MeshUtils::Vec3& cell_centroid(int) const;
    double face_area(int) const;
    MeshUtils::Vec3 face_centroid(int) const;
    MeshUtils::Vec3 face_area_vector(int) const;
};
struct WrongMeshRange : QueryMesh
{
    std::vector<std::string> faces(int) const;
};
struct NonRangeMesh : QueryMesh
{
    int faces(int) const;
};
struct WrongMeshMetric : QueryMesh
{
    std::string cell_volume(int) const;
};
struct MutableMeshQuery : QueryMesh
{
    double face_area(int);
};
struct Incomplete;

static_assert(MeshType<QueryMesh>);
static_assert(MeshType<MeshHandle<Pack>>);
static_assert(MeshType<Meshes::PartitionedMesh<Meshes::OrthogonalCartesian3D, Pack>>);
static_assert(MeshType<SolidSubdomain<Pack>>);
static_assert(!MeshType<int> && !MeshType<Incomplete>);
static_assert(!MeshType<WrongMeshRange> && !MeshType<WrongMeshMetric> && !MeshType<MutableMeshQuery>);
static_assert(!MeshType<NonRangeMesh>);

struct RegionTopology
{
    using index_type_pack = Meshes::UnstructuredMeshIndexTypes;
    Meshes::RegionLayout layout() const;
    const void* storage_identity() const;
    std::span<const size_t> cell_faces(size_t) const;
    std::span<const size_t> cell_nodes(size_t) const;
    std::span<const size_t> face_nodes(size_t) const;
    size_t owner_cell(size_t) const;
    size_t neighbor_cell(size_t) const;
    MeshUtils::CellType cell_type(size_t) const;
};
struct RegionGeometry
{
    Meshes::RegionLayout layout() const;
    const void* storage_identity() const;
    double cell_volume(size_t) const;
    MeshUtils::Vec3 cell_centroid(size_t) const;
    double face_area(size_t) const;
    MeshUtils::Vec3 face_centroid(size_t) const;
    MeshUtils::Vec3 face_area_vector(size_t) const;
    MeshUtils::Vec3 node_coordinates(size_t) const;
    std::uint64_t geometry_epoch() const;
};
struct WrongTopologyRange : RegionTopology
{
    std::vector<std::string> cell_faces(size_t) const;
};
struct NonRangeTopology : RegionTopology
{
    int cell_faces(size_t) const;
};
struct WrongTopologyPack : RegionTopology
{
    using index_type_pack = int;
};
struct MismatchedOrdinalRange
{
    std::vector<std::string>::const_iterator begin() const;
    std::vector<std::string>::const_iterator end() const;
    size_t size() const;
    size_t operator[](size_t) const;
};
struct MismatchedTopologyRange : RegionTopology
{
    MismatchedOrdinalRange cell_faces(size_t) const;
};
using UnsizedOrdinalRange = std::ranges::subrange<const size_t*, std::unreachable_sentinel_t>;
struct UnsizedTopologyRange : RegionTopology
{
    UnsizedOrdinalRange cell_faces(size_t) const;
};
struct WrongTopologyLayout : RegionTopology
{
    int layout() const;
};
struct WrongGeometryEpoch : RegionGeometry
{
    std::string geometry_epoch() const;
};
struct MutableGeometryQuery : RegionGeometry
{
    double cell_volume(size_t);
};

static_assert(Meshes::TopologyProvider<RegionTopology>);
static_assert(Meshes::GeometryProvider<RegionGeometry>);
static_assert(Meshes::TopologyProvider<Meshes::RectilinearTopology>);
static_assert(Meshes::GeometryProvider<Meshes::RectilinearGeometry>);
static_assert(Meshes::TopologyProvider<Meshes::ExplicitRegionProvider>);
static_assert(Meshes::GeometryProvider<Meshes::ExplicitRegionProvider>);
static_assert(!Meshes::TopologyProvider<int> && !Meshes::GeometryProvider<int>);
static_assert(!Meshes::TopologyProvider<WrongTopologyRange> && !Meshes::TopologyProvider<WrongTopologyLayout>);
static_assert(!Meshes::TopologyProvider<NonRangeTopology> && !Meshes::TopologyProvider<WrongTopologyPack>);
static_assert(std::ranges::random_access_range<MismatchedOrdinalRange>);
static_assert(std::ranges::random_access_range<UnsizedOrdinalRange> && !std::ranges::sized_range<UnsizedOrdinalRange>);
static_assert(!Meshes::TopologyProvider<MismatchedTopologyRange> && !Meshes::TopologyProvider<UnsizedTopologyRange>);
static_assert(!Meshes::GeometryProvider<WrongGeometryEpoch> && !Meshes::GeometryProvider<MutableGeometryQuery>);

struct Descriptor
{
    using value_type = double;
    using location_type = CellLocation;
    // Deliberately neither default- nor string-constructible.
    Descriptor(size_t identity, std::string name) : identity(identity), label(std::move(name)) {}
    size_t id() const { return identity; }
    const std::string& name() const { return label; }
    size_t identity;
    std::string label;
};
struct WrongFieldName : Descriptor
{
    std::string name() const;
};
struct MutableFieldName : Descriptor
{
    std::string& name() const;
};
struct DerivedString : std::string
{
};
struct DerivedFieldName : Descriptor
{
    const DerivedString& name() const;
};
struct WrongFieldLocation : Descriptor
{
    using location_type = int;
};
struct MissingFieldValue
{
    using location_type = CellLocation;
};

static_assert(FieldType<Descriptor>);
static_assert(FieldType<MutableFieldName> && FieldType<DerivedFieldName>);
static_assert(FieldType<ScalarCellFieldDescriptor<>> && FieldType<VectorCellFieldDescriptor<>>);
static_assert(FieldType<TensorCellFieldDescriptor<>> && FieldType<ScalarFaceFieldDescriptor<>>);
static_assert(FieldType<VectorFaceFieldDescriptor<>> && FieldType<ScalarBoundaryFaceFieldDescriptor<>>);
static_assert(FieldType<VectorBoundaryFaceFieldDescriptor<>>);
static_assert(!FieldType<int> && !FieldType<MissingFieldValue>);
static_assert(!FieldType<WrongFieldName> && !FieldType<WrongFieldLocation>);

template<class D, class M>
concept CanBindField = requires { typename FieldStored<D, Pack, M>; };
static_assert(CanBindField<Descriptor, MeshHandle<Pack>>);
static_assert(!CanBindField<WrongFieldName, MeshHandle<Pack>>);
static_assert(!CanBindField<Descriptor, WrongMeshMetric>);

// Public stored-field boundaries reject incomplete metadata before their bodies instantiate.
template<class F>
concept CanFormEquation = requires { typename Equation<F, Pack>; };
template<class F>
concept CanFormAssembledEquation = requires { typename AssembledEquation<F, Pack>; };
template<class F>
concept CanLookupProblemField = requires(Problem<Pack>& problem) { problem.template field<F>(std::string{}); };
static_assert(CanFormEquation<ScalarCellFieldStored<Pack>>);
static_assert(CanFormAssembledEquation<ScalarCellFieldStored<Pack>>);
static_assert(CanLookupProblemField<ScalarCellFieldStored<Pack>>);
static_assert(!CanFormEquation<int> && !CanFormEquation<MissingFieldValue>);
static_assert(!CanFormAssembledEquation<int> && !CanFormAssembledEquation<MissingFieldValue>);
static_assert(!CanLookupProblemField<int> && !CanLookupProblemField<MissingFieldValue>);

struct OtherPack : Pack
{
};
struct SourceAdapter
{
    operator Variant() const
    {
        return FVM::SourceOperator<Pack>{[](int, size_t) { return 2.0; }};
    }
};
struct ExplicitSourceAdapter
{
    explicit operator Variant() const;
};
struct LvalueSourceAdapter
{
    operator Variant() & { return SourceAdapter{}; }
};
struct LookalikeDiffusion
{
    double diffusivity;
};

static_assert(FVM::EquationTermType<FVM::TransientOperator<Pack>>);
static_assert(FVM::EquationTermType<FVM::ConvectionOperator<Pack>>);
static_assert(FVM::EquationTermType<FVM::DiffusionOperator<Pack>>);
static_assert(FVM::EquationTermType<FVM::SourceOperator<Pack>>);
static_assert(FVM::EquationTermType<FVM::GradientOperator> && FVM::EquationTermType<FVM::DivergenceOperator>);
static_assert(FVM::EquationTermType<Variant> && FVM::EquationTermType<const Variant&>);
static_assert(FVM::EquationTermType<SourceAdapter>);
static_assert(FVM::EquationTermType<LvalueSourceAdapter&> && !FVM::EquationTermType<LvalueSourceAdapter>);
static_assert(!FVM::EquationTermType<int> && !FVM::EquationTermType<FVM::SourceOperator<OtherPack>>);
static_assert(!FVM::EquationTermType<ExplicitSourceAdapter> && !FVM::EquationTermType<LookalikeDiffusion>);
static_assert(!FVM::EquationTermType<FVM::SourceOperator<Pack>, int>);

template<class Term>
concept CanAddTerm = requires(ScalarEquation& equation, Term&& term) {
    { equation.add_lhs(std::forward<Term>(term)) } -> std::same_as<ScalarEquation&>;
    { equation.add_rhs(std::forward<Term>(term)) } -> std::same_as<ScalarEquation&>;
};
static_assert(CanAddTerm<SourceAdapter> && CanAddTerm<const FVM::DiffusionOperator<Pack>&>);
static_assert(CanAddTerm<LvalueSourceAdapter&> && !CanAddTerm<LvalueSourceAdapter>);
static_assert(!CanAddTerm<int> && !CanAddTerm<ExplicitSourceAdapter>);
static_assert(!CanAddTerm<FVM::SourceOperator<OtherPack>>);

testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);

SP<const MeshHandle<Pack>> make_mesh()
{
    auto native = std::make_shared<Meshes::OrthogonalCartesian3D>(Vec3D<ArrReal>{{{0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0}}});
    return std::make_shared<MeshHandle<Pack>>(native);
}

TEST(FrameworkConceptsTest, StructuralDescriptorBindsWithoutLibraryInheritance)
{
    const auto mesh = make_mesh();
    FieldStored<Descriptor> field(Descriptor{73, "custom"}, mesh, 4.0);
    EXPECT_EQ(field.id(), 73u);
    EXPECT_EQ(field.name(), "custom");
    EXPECT_EQ(field.mesh_ptr(), mesh);
    EXPECT_DOUBLE_EQ(field.value(0), 4.0);
}

TEST(FrameworkConceptsTest, ConstrainedTermsRetainConversionsAndRuntimeValidation)
{
    auto field = std::make_shared<ScalarCellFieldStored<Pack>>(make_mesh(), 0.0, "unknown");
    ScalarEquation equation(field);
    const FVM::DiffusionOperator<Pack> diffusion{3.0};
    equation.add_lhs(diffusion).add_rhs(SourceAdapter{});
    EXPECT_DOUBLE_EQ(std::get<FVM::DiffusionOperator<Pack>>(equation.lhs_terms().at(0)).diffusivity, 3.0);
    EXPECT_DOUBLE_EQ(std::get<FVM::SourceOperator<Pack>>(equation.rhs_terms().at(0)).value(0, 0), 2.0);

    Variant invalid = FVM::DiffusionOperator<Pack>{-1.0};
    EXPECT_THROW(equation.add_lhs(invalid), std::invalid_argument);
    EXPECT_THROW(equation.add_lhs(FVM::DiffusionOperator<Pack>{-1.0}), std::invalid_argument);
    EXPECT_EQ(equation.lhs_terms().size(), 1u);
    // Keep the by-value overload addressable and usable with braced variants.
    auto add = static_cast<ScalarEquation& (ScalarEquation::*)(Variant)>(&ScalarEquation::add_rhs);
    (equation.*add)(Variant{FVM::SourceOperator<Pack>{[](int, size_t) { return 1.0; }}});
    equation.add_rhs({FVM::SourceOperator<Pack>{[](int, size_t) { return 1.0; }}});
    LvalueSourceAdapter adapter;
    equation.add_rhs(adapter);
    EXPECT_EQ(equation.rhs_terms().size(), 4u);
}
} // namespace
