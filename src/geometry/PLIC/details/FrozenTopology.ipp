/** Construction helpers; included inside the PLIC details namespace. */
template<class Value> void reserve_topology(std::vector<Value>& values, std::size_t count)
{
    (void) topology_product(count, sizeof(Value));
    if (count > values.max_size())
        throw GeometryError(GeometryErrorCode::ResourceLimit, "Frozen topology vector size overflows.");
    active_work->charge(count, sizeof(Value));
    ordinary_kernel::NativeSplitCppProfile::request(count, sizeof(Value));
    values.reserve(count);
    if (values.capacity() > count)
        active_work->charge(values.capacity() - count, sizeof(Value));
}
template<ArithmeticMode Mode>
std::shared_ptr<GeometryStorage<Mode>> freeze_geometry(GeometryDraft<Mode>&& draft, bool force_wide = false)
{
    TopologyCounts counts{draft.vertices.size(), draft.faces.size(), 0, 0, 0};
    if (draft.faces.size() != draft.surfaces.size())
        throw GeometryError(GeometryErrorCode::InvalidTopology, "Frozen shell surface count differs from faces.");
    std::map<const StoredPlane<Mode>*, std::size_t> slots;
    const auto admit = [&](const StoredPlanePtr<Mode>& plane)
    {
        if (!plane)
            throw GeometryError(GeometryErrorCode::InvalidTopology, "Frozen shell has an absent support.");
        if (slots.contains(plane.get()))
            return;
        // Identity is the actual plane object, never numerical coefficients or a source authority.
        active_work->charge(1, sizeof(typename decltype(slots)::value_type) + 4 * sizeof(void*));
        ordinary_kernel::NativeSplitCppProfile::template node<typename decltype(slots)::value_type>();
        slots.emplace(plane.get(), slots.size());
    };
    for (const auto& vertex : draft.vertices)
    {
        counts.incident_slots = topology_sum(counts.incident_slots, vertex.incident.size());
        for (const auto& plane : vertex.incident)
            admit(plane);
    }
    for (const auto& face : draft.faces)
    {
        counts.face_slots = topology_sum(counts.face_slots, face.size());
        for (auto index : face)
            if (index >= counts.vertices)
                throw GeometryError(GeometryErrorCode::InvalidTopology, "Frozen shell face index is invalid.");
    }
    for (const auto& surface : draft.surfaces)
    {
        if (!surface.plane || !slots.contains(surface.plane.get()) ||
            (surface.orientation != 1 && surface.orientation != -1))
            throw GeometryError(GeometryErrorCode::InvalidTopology, "Frozen shell surface lacks incident support.");
    }
    counts.planes = slots.size();
    const bool compact = compact_topology(counts, force_wide);
    active_work->charge(1, sizeof(FrozenTopology<Mode>));
    ordinary_kernel::NativeSplitCppProfile::template combined<FrozenTopology<Mode>>();
    auto topology = std::make_shared<FrozenTopology<Mode>>();
    reserve_topology(topology->planes, counts.planes);
    topology->planes.resize(counts.planes);
    for (const auto& vertex : draft.vertices)
        for (const auto& plane : vertex.incident)
            topology->planes[slots.at(plane.get())] = plane;
    const auto populate = [&](auto& a)
    {
        using Index = typename std::remove_cvref_t<decltype(a.face_offsets)>::value_type;
        reserve_topology(a.face_offsets, topology_sum(counts.faces, 1));
        reserve_topology(a.face_vertices, counts.face_slots);
        reserve_topology(a.incident_offsets, topology_sum(counts.vertices, 1));
        reserve_topology(a.incident_planes, counts.incident_slots);
        reserve_topology(a.surface_planes, counts.faces);
        reserve_topology(a.surface_orientations, counts.faces);
        a.face_offsets.push_back(0);
        for (const auto& face : draft.faces)
        {
            for (auto index : face)
                a.face_vertices.push_back(static_cast<Index>(index));
            a.face_offsets.push_back(static_cast<Index>(a.face_vertices.size()));
        }
        a.incident_offsets.push_back(0);
        for (const auto& vertex : draft.vertices)
        {
            for (const auto& plane : vertex.incident)
                a.incident_planes.push_back(static_cast<Index>(slots.at(plane.get())));
            a.incident_offsets.push_back(static_cast<Index>(a.incident_planes.size()));
        }
        for (const auto& surface : draft.surfaces)
        {
            a.surface_planes.push_back(static_cast<Index>(slots.at(surface.plane.get())));
            a.surface_orientations.push_back(surface.orientation);
        }
    };
    if (compact)
        populate(std::get<0>(topology->arrays));
    else
    {
        topology->arrays.template emplace<1>();
        populate(std::get<1>(topology->arrays));
    }
    // Keep draft connectivity alive until all frozen allocation/admission succeeds.
    std::vector<FrozenVertex<Mode>> vertices;
    reserve_topology(vertices, counts.vertices);
    for (std::size_t i = 0; i < counts.vertices; ++i)
    {
        auto vertex = FrozenVertex<Mode>(draft.vertices[i]);
        vertex.incident = topology->incidence(i);
        vertices.push_back(std::move(vertex));
    }
    active_work->charge(1, sizeof(GeometryStorage<Mode>));
    ordinary_kernel::NativeSplitCppProfile::template combined<GeometryStorage<Mode>>();
    auto result = std::make_shared<GeometryStorage<Mode>>(
        std::move(static_cast<GeometryMetadata<Mode>&>(draft)), std::move(topology), std::move(vertices));
    return result;
}
template<ArithmeticMode Mode> std::vector<StoredVertex<Mode>> draft_vertices(const GeometryStorage<Mode>& source)
{
    std::vector<StoredVertex<Mode>> vertices;
    reserve_topology(vertices, source.vertices.size());
    for (const auto& vertex : source.vertices)
    {
        StoredVertex<Mode> copy;
        copy.coordinates = vertex.coordinates;
        copy.authorities = vertex.authorities;
        copy.source_identity = vertex.source_identity;
        copy.numerical_only = vertex.numerical_only;
        reserve_topology(copy.incident, vertex.incident.size());
        copy.incident.assign(vertex.incident.begin(), vertex.incident.end());
        vertices.push_back(std::move(copy));
    }
    return vertices;
}
template<ArithmeticMode Mode> std::shared_ptr<GeometryDraft<Mode>> draft_geometry(const GeometryStorage<Mode>& source)
{
    active_work->charge(1, sizeof(GeometryDraft<Mode>));
    auto result = std::make_shared<GeometryDraft<Mode>>(static_cast<const GeometryMetadata<Mode>&>(source));
    result->vertices = draft_vertices(source);
    reserve_topology(result->faces, source.faces.size());
    reserve_topology(result->surfaces, source.surfaces.size());
    for (const auto face : source.faces)
    {
        Face row;
        reserve_topology(row, face.size());
        row.assign(face.begin(), face.end());
        result->faces.push_back(std::move(row));
    }
    for (const auto surface : source.surfaces)
        result->surfaces.push_back(surface);
    result->buffers_prepaid = true;
    return result;
}
/** Rebind exactly the same slots to a new immutable owner table. */
template<ArithmeticMode Mode>
void bind_topology(GeometryStorage<Mode>& geometry, std::shared_ptr<const FrozenTopology<Mode>> topology)
{
    geometry.topology = std::move(topology);
    geometry.faces = geometry.topology->faces();
    geometry.surfaces = SurfaceRange<Mode>(geometry.topology.get());
    for (std::size_t i = 0; i < geometry.vertices.size(); ++i)
        geometry.vertices[i].incident = geometry.topology->incidence(i);
}

template<ArithmeticMode Mode> std::size_t geometry_metadata_copy_bytes(const GeometryStorage<Mode>& source)
{
    return topology_sum(
        sizeof(GeometryStorage<Mode>), topology_product(source.vertices.capacity(), sizeof(FrozenVertex<Mode>)));
}

/** Exact fresh payload clone: shared geometry/provenance owners remain borrowed. */
template<ArithmeticMode Mode>
std::shared_ptr<GeometryStorage<Mode>> clone_geometry_metadata(const GeometryStorage<Mode>& source)
{
    std::vector<FrozenVertex<Mode>> vertices;
    reserve_topology(vertices, source.vertices.size());
    vertices.assign(source.vertices.begin(), source.vertices.end());
    active_work->charge(1, sizeof(GeometryStorage<Mode>));
    return std::make_shared<GeometryStorage<Mode>>(
        GeometryMetadata<Mode>(source), source.topology, std::move(vertices));
}
template<ArithmeticMode Mode> std::shared_ptr<FrozenTopology<Mode>> clone_topology(const FrozenTopology<Mode>& source)
{
    active_work->charge(1, sizeof(FrozenTopology<Mode>));
    auto result = std::make_shared<FrozenTopology<Mode>>();
    reserve_topology(result->planes, source.planes.size());
    result->planes.assign(source.planes.begin(), source.planes.end());
    std::visit(
        [&](const auto& original)
        {
            using Arrays = std::remove_cvref_t<decltype(original)>;
            auto& copy = result->arrays.template emplace<Arrays>();
            const auto copy_buffer = [](auto& destination, const auto& values)
            {
                reserve_topology(destination, values.size());
                destination.assign(values.begin(), values.end());
            };
            copy_buffer(copy.face_offsets, original.face_offsets);
            copy_buffer(copy.face_vertices, original.face_vertices);
            copy_buffer(copy.incident_offsets, original.incident_offsets);
            copy_buffer(copy.incident_planes, original.incident_planes);
            copy_buffer(copy.surface_planes, original.surface_planes);
            copy_buffer(copy.surface_orientations, original.surface_orientations);
        },
        source.arrays);
    return result;
}
template<ArithmeticMode Mode> std::shared_ptr<StoredPlane<Mode>> clone_plane(const StoredPlane<Mode>& source)
{
    active_work->charge(1, sizeof(StoredPlane<Mode>));
    active_work->charge(source.source_vertices.size(), sizeof(StoredPointRef<Mode>));
    auto result = std::make_shared<StoredPlane<Mode>>(source);
    if (result->source_vertices.capacity() > source.source_vertices.size())
        active_work->charge(
            result->source_vertices.capacity() - source.source_vertices.size(), sizeof(StoredPointRef<Mode>));
    return result;
}
