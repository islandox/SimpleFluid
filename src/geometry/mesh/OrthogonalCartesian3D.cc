/**
 * @file OrthogonalCartesian3D.cc
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief OrthogonalCartesian3D implementation.
 * @version 0.1
 * @date 2026-07-21
 *
 * @copyright Copyright (c) 2026
 *
 */

#include "geometry/mesh/OrthogonalCartesian3D.hh"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace SimpleFluid::Meshes
{
namespace
{

/**
 * @brief Validate one Cartesian coordinate-edge array.
 * @param edges Edge coordinates to validate.
 * @param axis_name Axis label used in diagnostics.
 * @throws std::invalid_argument If the array is too short, non-finite, or not
 *         strictly increasing.
 * @throws std::overflow_error If its cell count exceeds the mesh ID type.
 */
void validate_edges(const Arr<real_t>& edges, const char* axis_name)
{
    if (edges.size() < 2)
    {
        throw std::invalid_argument(
            std::string("OrthogonalCartesian3D axis ") + axis_name
            + " requires at least two edge coordinates.");
    }
    if (edges.size() - 1
        > static_cast<size_t>(std::numeric_limits<int>::max()))
    {
        throw std::overflow_error(
            std::string("OrthogonalCartesian3D axis ") + axis_name
            + " has too many cells for its ID type.");
    }

    for (size_t edge = 0; edge < edges.size(); ++edge)
    {
        if (!std::isfinite(edges[edge]))
        {
            throw std::invalid_argument(
                std::string("OrthogonalCartesian3D axis ") + axis_name
                + " contains a non-finite edge coordinate.");
        }
        if (edge > 0 && edges[edge] <= edges[edge - 1])
        {
            throw std::invalid_argument(
                std::string("OrthogonalCartesian3D axis ") + axis_name
                + " edge coordinates must be strictly increasing.");
        }
    }
}

/**
 * @brief Narrow a size-based coordinate count to the mesh index type.
 * @param index Value to convert.
 * @return Converted non-negative index.
 * @throws std::overflow_error If @p index exceeds the integer ID type.
 */
int checked_index(size_t index)
{
    if (index > static_cast<size_t>(std::numeric_limits<int>::max()))
    {
        throw std::overflow_error(
            "OrthogonalCartesian3D local index exceeds its ID type.");
    }
    return static_cast<int>(index);
}

} // namespace

/**
 * @brief Construct and precompute Cartesian geometry and topology.
 * @throws std::invalid_argument If any coordinate array is invalid.
 * @throws std::overflow_error If entity counts exceed supported ID ranges.
 */
OrthogonalCartesian3D::OrthogonalCartesian3D(const Vec3D<Arr<real_t>>& cell_edges)
    : OrthogonalCartesian3D(cell_edges, {}) {}

OrthogonalCartesian3D::OrthogonalCartesian3D(
    const Vec3D<Arr<real_t>>& cell_edges, Vec3D<bool> periodic_dimensions)
    : d_cell_edges(cell_edges)
{
    validate_edges(d_cell_edges[X], "x");
    validate_edges(d_cell_edges[Y], "y");
    validate_edges(d_cell_edges[Z], "z");

    const auto nx = d_cell_edges[X].size() - 1;
    const auto ny = d_cell_edges[Y].size() - 1;
    const auto nz = d_cell_edges[Z].size() - 1;
    CHECK_PRODUCT_OVERFLOW(nx + 1, ny + 1, nz + 1);

    for (size_t axis=0; axis<3; ++axis)
        if (periodic_dimensions[axis] && d_cell_edges[axis].size()<3)
            throw std::invalid_argument("Periodic Cartesian axes require at least two cells.");
    d_indexer = Indexer(checked_index(nx), checked_index(ny), checked_index(nz),
                        periodic_dimensions[0], periodic_dimensions[1], periodic_dimensions[2]);

    for (size_t dim = 0; dim < 3; ++dim)
    {
        d_cell_centroids[dim] = MeshUtils::consec_mid(d_cell_edges[dim]);
        d_cell_widths[dim] = MeshUtils::consec_diff(d_cell_edges[dim]);
    }

    Base::d_num_cells = nx * ny * nz;

    Base::d_num_local_cells = Base::d_num_cells;
    Base::d_num_owned_cells = Base::d_num_cells;

    CHECK_SUM_OVERFLOW(d_indexer.num_faces_per_orientation[0], d_indexer.num_faces_per_orientation[1],
                       d_indexer.num_faces_per_orientation[2]);
    Base::d_num_faces = d_indexer.total_faces();
    Base::d_num_owned_faces = Base::d_num_faces;
    Base::d_num_nodes = d_indexer.total_nodes();

    d_topology = OrthoMeshTopo(
        d_indexer,
        {{"xmin", "xmax", "ymin", "ymax", "zmin", "zmax"}});
}

/**
 * @brief Replace one coordinate direction without changing structured topology.
 * @param axis Cartesian axis index.
 * @param edges Strictly increasing replacement edges with the existing size.
 * @throws std::invalid_argument If the axis or replacement layout is invalid.
 */
void OrthogonalCartesian3D::replace_axis_edges_fixed_topology(size_t axis, Arr<real_t> edges)
{
    require_geometry_writable();
    if (d_geometry_state.epoch == std::numeric_limits<std::uint64_t>::max())
    {
        throw std::overflow_error("OrthogonalCartesian3D geometry epoch overflow.");
    }
    if (axis >= d_cell_edges.size())
    {
        throw std::invalid_argument("OrthogonalCartesian3D fixed-topology update axis is invalid.");
    }
    if (edges.size() != d_cell_edges[axis].size())
    {
        throw std::invalid_argument("OrthogonalCartesian3D fixed-topology update cannot change "
                                    "the edge count.");
    }

    constexpr std::array<const char*, 3> axis_names{{"x", "y", "z"}};
    validate_edges(edges, axis_names[axis]);
    auto centroids = MeshUtils::consec_mid(edges);
    auto widths = MeshUtils::consec_diff(edges);

    d_cell_edges[axis] = std::move(edges);
    d_cell_centroids[axis] = std::move(centroids);
    d_cell_widths[axis] = std::move(widths);
    ++d_geometry_state.epoch;
}

/**
 * @brief Return a physical boundary batch name.
 * @param batch_id Boundary batch identifier.
 * @return Configured lower or upper axis name.
 * @throws std::out_of_range If @p batch_id is invalid.
 */
const std::string&
OrthogonalCartesian3D::boundary_batch_name_impl(int batch_id) const
{
    return d_topology.boundary_batch_name(batch_id);
}

} // namespace SimpleFluid::Meshes

#include "geometry/mesh/details/NativeLegacyCartesianAccess.hh"
#include <cstring>

namespace SimpleFluid::Meshes
{
OrthogonalCartesian3D::OrthogonalCartesian3D(AdmittedStorageTag, AdmittedStorage&& storage)
    : d_storage(std::move(storage.receipt)), d_cell_edges(std::move(storage.edges)),
      d_cell_centroids(std::move(storage.centroids)), d_cell_widths(std::move(storage.widths)),
      d_indexer(storage.indexer), d_topology(std::move(storage.topology))
{
    // Fresh GeometryExecutionGuard and GeometryEpochState; no source motion lease.
    Base::d_num_cells = d_indexer.total_cells();
    Base::d_num_local_cells = Base::d_num_cells;
    Base::d_num_owned_cells = Base::d_num_cells;
    Base::d_num_faces = d_indexer.total_faces();
    Base::d_num_owned_faces = Base::d_num_faces;
    Base::d_num_nodes = d_indexer.total_nodes();
}
} // namespace SimpleFluid::Meshes
namespace SimpleFluid::Meshes::details
{
namespace legacy_cartesian_detail
{
using Access = NativeLegacyCartesianAccess;
OrthogonalIndexer checked_indexer(Vec3D<size_t> counts, Vec3D<bool> periodic)
{
    const auto [nx, ny, nz] = counts;
#ifdef DEBUG_CHECK_ENABLED
    // The legacy macro builds its diagnostic string eagerly. Preserve its
    // original failure path while keeping successful native preflight free of
    // unadmitted diagnostic allocations.
    if (utils::detail::multiplication_overflows(nx + 1, ny + 1) ||
        utils::detail::multiplication_overflows((nx + 1) * (ny + 1), nz + 1))
        CHECK_PRODUCT_OVERFLOW(nx + 1, ny + 1, nz + 1);
#endif
    OrthogonalIndexer index(
        static_cast<int>(nx), static_cast<int>(ny), static_cast<int>(nz), periodic[0], periodic[1], periodic[2]);
#ifdef DEBUG_CHECK_ENABLED
    const auto a = index.num_faces_per_orientation[0], b = index.num_faces_per_orientation[1],
               c = index.num_faces_per_orientation[2];
    if (utils::detail::addition_overflows(a, b) || utils::detail::addition_overflows(a + b, c))
        CHECK_SUM_OVERFLOW(
            index.num_faces_per_orientation[0], index.num_faces_per_orientation[1], index.num_faces_per_orientation[2]);
#endif
    return index;
}
OrthogonalIndexer checked_indexer(const Access::Axes& axes, Vec3D<bool> periodic)
{
    constexpr std::array<const char*, 3> names{"x", "y", "z"};
    for (size_t axis = 0; axis < 3; ++axis)
    {
        const auto edges = axes[axis];
        if (edges.size() < 2)
            throw std::invalid_argument(
                std::string("OrthogonalCartesian3D axis ") + names[axis] + " requires at least two edge coordinates.");
        if (edges.size() - 1 > static_cast<size_t>(std::numeric_limits<int>::max()))
            throw std::overflow_error("OrthogonalCartesian3D axis has too many cells for its ID type.");
        for (size_t i = 0; i < edges.size(); ++i)
        {
            if (!std::isfinite(edges[i]))
                throw std::invalid_argument("OrthogonalCartesian3D contains a non-finite edge coordinate.");
            if (i && edges[i] <= edges[i - 1])
                throw std::invalid_argument("OrthogonalCartesian3D edges must be strictly increasing.");
        }
        if (periodic[axis] && edges.size() < 3)
            throw std::invalid_argument("Periodic Cartesian axes require at least two cells.");
        // num_nodes_per_dim is the same signed ordinal as num_cells_per_dim.
        if (!periodic[axis] && edges.size() - 1 == static_cast<size_t>(std::numeric_limits<int>::max()))
            throw std::overflow_error("OrthogonalCartesian3D node extent exceeds its ID type.");
    }
    return checked_indexer(Vec3D<size_t>{axes[0].size() - 1, axes[1].size() - 1, axes[2].size() - 1}, periodic);
}
struct Cursor
{
    std::span<const char> bytes;
    template<class T> T value()
    {
        if (bytes.size() < sizeof(T))
            throw std::invalid_argument("Truncated legacy Cartesian descriptor.");
        T result;
        std::memcpy(&result, bytes.data(), sizeof(T));
        bytes = bytes.subspan(sizeof(T));
        return result;
    }
    std::span<const char> array(size_t width)
    {
        const auto count = value<uint64_t>();
        if (!width || count > bytes.size() / width)
            throw std::invalid_argument("Legacy Cartesian array exceeds packet.");
        const auto size = static_cast<size_t>(count) * width;
        const auto result = bytes.first(size);
        bytes = bytes.subspan(size);
        return result;
    }
    std::string_view text()
    {
        const auto value = array(1);
        return {value.data(), value.size()};
    }
};
struct Plan
{
    std::string_view name;
    RegionLayout layout{};
    std::array<std::span<const char>, 3> axes;
    Access::Names names;
    OrthogonalIndexer index;
};
Plan preflight(std::span<const char> packet)
{
    Cursor cursor{packet};
    Plan result;
    if (cursor.value<uint32_t>() != 1)
        throw std::invalid_argument("Legacy Cartesian decoder requires tag1.");
    result.name = cursor.text();
    if (result.name.empty())
        throw std::invalid_argument("IsoRegion requires a name.");
    result.layout.family = cursor.value<RegionLayout::Family>();
    result.layout.cells = cursor.value<uint64_t>();
    result.layout.faces = cursor.value<uint64_t>();
    result.layout.nodes = cursor.value<uint64_t>();
    for (auto& extent : result.layout.extents)
        extent = cursor.value<uint64_t>();
    for (auto& periodic : result.layout.periodic)
    {
        const auto value = cursor.value<uint8_t>();
        if (value > 1)
            throw std::invalid_argument("Bad legacy Cartesian periodic flag.");
        periodic = value;
    }
    // Validate unaligned scalar buffers directly, without any vector allocation.
    Vec3D<size_t> counts{};
    for (size_t axis = 0; axis < 3; ++axis)
    {
        result.axes[axis] = cursor.array(sizeof(real_t));
        const auto size = result.axes[axis].size() / sizeof(real_t);
        if (size < 2 || size - 1 > static_cast<size_t>(std::numeric_limits<int>::max()) ||
            (!result.layout.periodic[axis] && size - 1 == static_cast<size_t>(std::numeric_limits<int>::max())))
            throw std::invalid_argument("Invalid legacy Cartesian axis count.");
        real_t previous = 0;
        for (size_t i = 0; i < size; ++i)
        {
            real_t value;
            std::memcpy(&value, result.axes[axis].data() + i * sizeof(real_t), sizeof(real_t));
            if (!std::isfinite(value) || (i && value <= previous))
                throw std::invalid_argument("Invalid legacy Cartesian edge coordinate.");
            previous = value;
        }
        if (result.layout.periodic[axis] && size < 3)
            throw std::invalid_argument("Periodic Cartesian axes require at least two cells.");
        counts[axis] = size - 1;
    }
    for (auto& name : result.names)
        name = cursor.text();
    // Existing tag1 encoding omits names for periodic batches. The legacy
    // native decoder retains the constructor's default unused axis names.
    for (size_t axis = 0; axis < 3; ++axis)
        if (result.layout.periodic[axis])
            for (size_t side = 0; side < 2; ++side)
            {
                const auto boundary = 2 * axis + side;
                if (!result.names[boundary].empty())
                    throw std::invalid_argument("Noncanonical periodic Cartesian boundary name.");
                result.names[boundary] = Access::default_names[boundary];
            }
    if (!cursor.bytes.empty())
        throw std::invalid_argument("Trailing legacy Cartesian descriptor bytes.");
    result.index = checked_indexer(counts, result.layout.periodic);
    if (result.layout.family != RegionLayout::Family::Rectilinear ||
        result.layout.cells != result.index.total_cells() || result.layout.faces != result.index.total_faces() ||
        result.layout.nodes != result.index.total_nodes())
        throw std::invalid_argument("Legacy Cartesian declared layout differs from its axes.");
    for (size_t axis = 0; axis < 3; ++axis)
        if (result.layout.extents[axis] != counts[axis])
            throw std::invalid_argument("Legacy Cartesian declared extent differs from its axis.");
    return result;
}
} // namespace legacy_cartesian_detail

OrthoMeshTopo NativeLegacyCartesianAccess::copy_topology(const OrthoMeshTopo& source, const Admission& admission)
{
    OrthoMeshTopo result;
    result.d_storage = geometry_detail::native_constructor_storage(admission);
    result.d_indexer = source.d_indexer;
    for (size_t boundary = 0; boundary < 6; ++boundary)
        result.d_storage.owner->string(result.d_boundary_names[boundary], source.d_boundary_names[boundary]);
    for (size_t axis = 0; axis < 3; ++axis)
    {
        if (source.d_face_cells_per_dim[axis].size() != static_cast<size_t>(source.d_indexer.num_nodes_per_dim[axis]) ||
            source.d_neighbors_per_dim[axis].size() != static_cast<size_t>(source.d_indexer.num_cells_per_dim[axis]))
            throw std::invalid_argument("Legacy Cartesian source adjacency extent changed.");
        result.d_storage.owner->copy(result.d_face_cells_per_dim[axis], std::span(source.d_face_cells_per_dim[axis]));
        result.d_storage.owner->copy(result.d_neighbors_per_dim[axis], std::span(source.d_neighbors_per_dim[axis]));
    }
    return result;
}
std::shared_ptr<const NativeLegacyCartesianAccess::Native> NativeLegacyCartesianAccess::publish(
    Native::AdmittedStorage&& data, const Admission& admission)
{
    auto tracked = geometry_detail::NativeAllocationEnvelopeAccess::make<Native>(
        admission, Native::AdmittedStorageTag{}, std::move(data));
    tracked.owner->d_allocation_envelope = tracked.envelope;
    return std::move(tracked.owner);
}
std::shared_ptr<const NativeLegacyCartesianAccess::Native> NativeLegacyCartesianAccess::make(
    const Axes& axes, const Names& names, Vec3D<bool> periodic, const Admission& admission)
{
    geometry_detail::NativeConstructorStorage::require_profile();
    Native::AdmittedStorage data;
    data.indexer = legacy_cartesian_detail::checked_indexer(axes, periodic);
    data.receipt = geometry_detail::native_constructor_storage(admission);
    for (size_t axis = 0; axis < 3; ++axis)
        data.receipt.owner->copy(data.edges[axis], axes[axis]);
    // Same axis order and original consec_mid/consec_diff expressions; no midpoint substitution.
    for (size_t axis = 0; axis < 3; ++axis)
    {
        const auto count = data.edges[axis].size() - 1;
        data.receipt.owner->reserve(data.centroids[axis], count);
        data.receipt.owner->reserve(data.widths[axis], count);
        for (size_t i = 0; i < count; ++i)
            data.centroids[axis].push_back((data.edges[axis][i] + data.edges[axis][i + 1]) / static_cast<real_t>(2));
        for (size_t i = 0; i < count; ++i)
            data.widths[axis].push_back(data.edges[axis][i + 1] - data.edges[axis][i]);
    }
    auto storage = geometry_detail::native_constructor_storage(admission);
    OrthoMeshTopo::BoundaryNames owned_names;
    for (size_t i = 0; i < 6; ++i)
        storage.owner->string(owned_names[i], names[i]);
    data.topology =
        OrthoMeshTopo(OrthoMeshTopo::AdmittedStorageTag{}, data.indexer, std::move(owned_names), std::move(storage));
    return publish(std::move(data), admission);
}
std::shared_ptr<const NativeLegacyCartesianAccess::Native> NativeLegacyCartesianAccess::detached_copy(
    const Native& source, const Admission& admission)
{
    return detached_copy(source, source.d_topology, admission);
}
std::shared_ptr<const NativeLegacyCartesianAccess::Native> NativeLegacyCartesianAccess::detached_copy(
    const Native& source, const OrthoMeshTopo& topology, const Admission& admission)
{
    geometry_detail::NativeConstructorStorage::require_profile();
    const auto native_lease = source.acquire_geometry_read();
    const auto topology_lease = topology.acquire_geometry_read();
    const auto epoch = source.geometry_epoch();
    if (source.d_indexer.num_cells_per_dim != topology.d_indexer.num_cells_per_dim ||
        source.d_indexer.periodic_dimensions != topology.d_indexer.periodic_dimensions ||
        source.num_cells() != source.d_indexer.total_cells() || source.num_faces() != source.d_indexer.total_faces() ||
        source.num_nodes() != source.d_indexer.total_nodes() || source.num_owned_cells() != source.num_cells() ||
        source.num_owned_faces() != source.num_faces())
        throw std::invalid_argument("Legacy Cartesian source layout changed.");
    Native::AdmittedStorage data;
    data.indexer = source.d_indexer;
    data.receipt = geometry_detail::native_constructor_storage(admission);
    for (size_t axis = 0; axis < 3; ++axis)
    {
        const auto count = static_cast<size_t>(source.d_indexer.num_cells_per_dim[axis]);
        if (source.d_cell_edges[axis].size() != count + 1 || source.d_cell_centroids[axis].size() != count ||
            source.d_cell_widths[axis].size() != count)
            throw std::invalid_argument("Legacy Cartesian source metric extent changed.");
        data.receipt.owner->copy(data.edges[axis], std::span(source.d_cell_edges[axis]));
        data.receipt.owner->copy(data.centroids[axis], std::span(source.d_cell_centroids[axis]));
        data.receipt.owner->copy(data.widths[axis], std::span(source.d_cell_widths[axis]));
    }
    data.topology = copy_topology(topology, admission);
    if (source.geometry_epoch() != epoch)
        throw std::invalid_argument("Legacy Cartesian source epoch changed during detach.");
    return publish(std::move(data), admission);
}
std::shared_ptr<const NativeLegacyCartesianAccess::Provider> NativeLegacyCartesianAccess::wrap(
    std::shared_ptr<const Native> native, const Admission& admission)
{
    if (!native)
        throw std::invalid_argument("Null admitted legacy Cartesian native owner.");
    Sink::require_envelope(native->d_allocation_envelope, native);
    auto tracked = geometry_detail::NativeAllocationEnvelopeAccess::make<Provider>(admission, std::move(native));
    tracked.owner->d_native_allocation = tracked.envelope;
    return std::move(tracked.owner);
}
NativeLegacyCartesianRegion NativeLegacyCartesianAccess::make_region(
    std::string_view name, const Axes& axes, const Names& names, Vec3D<bool> periodic, const Admission& admission)
{
    auto metadata = geometry_detail::native_constructor_storage(admission);
    auto native = make(axes, names, periodic, admission);
    auto provider = wrap(std::move(native), admission);
    std::string owned_name;
    metadata.owner->string(owned_name, name);
    return {std::move(metadata), Region(std::move(owned_name), provider, provider)};
}
NativeLegacyCartesianRegion NativeLegacyCartesianAccess::detach_region(const Region& source, const Admission& admission)
{
    const auto topology_read = source.topology().native().acquire_geometry_read();
    source.validate_static();
    auto metadata = geometry_detail::native_constructor_storage(admission);
    auto native = detached_copy(source.geometry().native(), source.topology().native().topology(), admission);
    auto provider = wrap(std::move(native), admission);
    std::string name;
    metadata.owner->string(name, source.name());
    source.validate_static();
    return {std::move(metadata), Region(std::move(name), provider, provider)};
}
NativeLegacyCartesianRegion NativeLegacyCartesianAccess::decode(
    std::span<const char> packet, const Admission& admission, std::string_view expected_name)
{
    geometry_detail::NativeConstructorStorage::require_profile();
    const auto plan = legacy_cartesian_detail::preflight(packet);
    if (!expected_name.empty() && plan.name != expected_name)
        throw std::invalid_argument("Legacy Cartesian descriptor names differ.");
    auto metadata = geometry_detail::native_constructor_storage(admission);
    Native::AdmittedStorage data;
    data.indexer = plan.index;
    data.receipt = geometry_detail::native_constructor_storage(admission);
    for (size_t axis = 0; axis < 3; ++axis)
    {
        const auto count = plan.axes[axis].size() / sizeof(real_t);
        data.receipt.owner->reserve(data.edges[axis], count);
        for (size_t i = 0; i < count; ++i)
        {
            real_t value;
            std::memcpy(&value, plan.axes[axis].data() + i * sizeof(real_t), sizeof(real_t));
            data.edges[axis].push_back(value);
        }
        data.receipt.owner->reserve(data.centroids[axis], count - 1);
        data.receipt.owner->reserve(data.widths[axis], count - 1);
        for (size_t i = 0; i + 1 < count; ++i)
            data.centroids[axis].push_back((data.edges[axis][i] + data.edges[axis][i + 1]) / static_cast<real_t>(2));
        for (size_t i = 0; i + 1 < count; ++i)
            data.widths[axis].push_back(data.edges[axis][i + 1] - data.edges[axis][i]);
    }
    auto storage = geometry_detail::native_constructor_storage(admission);
    OrthoMeshTopo::BoundaryNames names;
    for (size_t i = 0; i < 6; ++i)
        storage.owner->string(names[i], plan.names[i]);
    data.topology =
        OrthoMeshTopo(OrthoMeshTopo::AdmittedStorageTag{}, data.indexer, std::move(names), std::move(storage));
    auto provider = wrap(publish(std::move(data), admission), admission);
    std::string name;
    metadata.owner->string(name, plan.name);
    return {std::move(metadata), Region(std::move(name), provider, provider)};
}
void NativeLegacyCartesianAccess::visit(const std::shared_ptr<const Native>& owner, const Sink& sink)
{
    if (!owner)
        return;
    Sink::require_envelope(owner->d_allocation_envelope, owner);
    if (!owner->d_storage.owner || !owner->d_topology.d_storage.owner || owner->d_topology.d_allocation_envelope)
        throw std::invalid_argument("Legacy Cartesian requires admitted inline topology and metric storage.");
    if (!sink.semantic(owner, geometry_detail::NativeConstructorOwnerKind::NativeMesh, owner->d_allocation_envelope))
        return;
    sink.fixed(owner->d_allocation_envelope);
    sink.storage(owner->d_storage);
    // Topology is inline, not a fabricated shared object. Its object bytes are
    // already inside the native fixed allocation; only real storage edges follow.
    sink.storage(owner->d_topology.d_storage);
}
void NativeLegacyCartesianAccess::visit_provider(const std::shared_ptr<const Provider>& owner, const Sink& sink)
{
    if (!owner)
        return;
    Sink::require_envelope(owner->d_native_allocation, owner);
    if (!sink.semantic(owner, geometry_detail::NativeConstructorOwnerKind::Geometry, owner->d_native_allocation))
        return;
    sink.fixed(owner->d_native_allocation);
    visit(owner->d_mesh, sink);
}
void NativeLegacyCartesianAccess::visit_region(const Region& region, const Sink& sink)
{
    region.validate_static();
    visit_provider(region.d_topology, sink);
    visit_provider(region.d_geometry, sink);
}
} // namespace SimpleFluid::Meshes::details
