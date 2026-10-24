BoundedClipResult BoundedConvexGeometry::clip_impl(
    const std::shared_ptr<const Impl>& source, std::size_t plane_id, bool measure_section) const
{
    const auto support = source->surfaces.at(plane_id);
    const auto& plane = support.plane;
    std::vector<Interval> distances;
    std::vector<int> signs;
    bool negative = false, positive = false;
    for (const auto& vertex : impl_->vertices)
    {
        auto d = distance(vertex, impl_->frame, plane) * Interval(support.orientation);
        const auto s = sign(d);
        distances.push_back(d);
        signs.push_back(s);
        negative = negative || s < 0;
        positive = positive || s > 0;
    }
    BoundedClipResult result;
    if (!negative || !positive)
    {
        // A support face/edge/point is a proven zero-volume contact. Preserve
        // its section dimension rather than manufacturing a positive piece.
        std::vector<std::size_t> contact;
        std::map<std::pair<std::size_t, std::size_t>, int> contact_edges;
        for (const auto& face : impl_->faces)
            if (std::all_of(face.begin(), face.end(), [&](auto i) { return signs[i] == 0; }))
                for (std::size_t i = 0; i < face.size(); ++i)
                {
                    const auto a = face[i], b = face[(i + 1) % face.size()];
                    contact_edges[std::minmax(a, b)] += a < b ? 1 : -1;
                }
        std::map<std::size_t, std::set<std::size_t>> boundary;
        for (const auto& [edge, coefficient] : contact_edges)
            if (coefficient)
            {
                if (std::abs(coefficient) != 1)
                    reject("support section has repeated oriented boundary coverage.");
                boundary[edge.first].insert(edge.second);
                boundary[edge.second].insert(edge.first);
            }
        if (!boundary.empty())
        {
            for (const auto& [vertex, neighbors] : boundary)
                if (neighbors.size() != 2)
                    reject("support section is not a closed boundary.");
            const auto first = boundary.begin()->first;
            auto previous = first, current = *boundary.at(first).begin();
            contact.push_back(first);
            while (current != first)
            {
                if (contact.size() >= boundary.size())
                    reject("support section boundary repeats a vertex.");
                contact.push_back(current);
                const auto& next = boundary.at(current);
                const auto following = *next.begin() == previous ? *next.rbegin() : *next.begin();
                previous = current;
                current = following;
            }
            if (contact.size() != boundary.size())
                reject("support section has disconnected atomic boundaries.");
        }
        if (contact.empty())
            for (std::size_t i = 0; i < signs.size(); ++i)
                if (signs[i] == 0)
                    contact.push_back(i);
        if (measure_section)
            result.section = section(*impl_, impl_->vertices, contact);
        auto self = BoundedConvexGeometryPtr(new BoundedConvexGeometry(impl_));
        if (!positive)
            result.inside = self;
        if (!negative)
            result.outside = self;
        // A valid positive-volume shell cannot lie entirely on this plane.
        if (!negative && !positive)
            reject("positive-volume shell has entirely unresolved support.");
        return result;
    }

    std::vector<Vertex> vertices = impl_->vertices;
    // Authenticate every zero-distance input incidence on the new shared plane.
    for (std::size_t i = 0; i < signs.size(); ++i)
        if (signs[i] == 0)
        {
            if (!incident(vertices[i], plane))
                vertices[i].incident.push_back(plane);
            for (const auto& key : plane->authorities)
                if (std::find(vertices[i].authorities.begin(), vertices[i].authorities.end(), key) ==
                    vertices[i].authorities.end())
                    vertices[i].authorities.push_back(key);
        }
    std::map<std::pair<std::size_t, std::size_t>, std::size_t> crossings;
    const auto crossing = [&](std::size_t a, std::size_t b)
    {
        const std::pair<std::size_t, std::size_t> key{std::min(a, b), std::max(a, b)};
        if (auto found = crossings.find(key); found != crossings.end())
            return found->second;
        a = key.first;
        b = key.second;
        const auto fraction = distances[a] / (distances[a] - distances[b]);
        if (!(fraction.lo > 0 && fraction.hi < 1))
            reject("edge intersection parameter is unresolved.");
        Vertex vertex;
        const auto denominator = distances[a] - distances[b];
        for (std::size_t axis = 0; axis < 3; ++axis)
        {
            // Delay the common division. A nonbinary edge parameter can
            // produce an exactly representable coordinate; rounding the
            // parameter first would destroy that cancellation and its support.
            vertex.coordinates[axis] =
                (vertices[b].coordinates[axis] * distances[a] - vertices[a].coordinates[axis] * distances[b]) /
                denominator;
        }
        for (const auto& p : vertices[a].incident)
            if (incident(vertices[b], p))
                vertex.incident.push_back(p);
        if (!incident(vertex, plane))
            vertex.incident.push_back(plane);
        for (const auto& key : vertices[a].authorities)
            if (std::find(vertices[b].authorities.begin(), vertices[b].authorities.end(), key) !=
                vertices[b].authorities.end())
                vertex.authorities.push_back(key);
        for (const auto& key : plane->authorities)
            if (std::find(vertex.authorities.begin(), vertex.authorities.end(), key) == vertex.authorities.end())
                vertex.authorities.push_back(key);
        if (vertices.size() >= impl_->limits.maximum_vertices)
            reject("intersection vertex cap exhausted.");
        const auto index = vertices.size();
        vertices.push_back(std::move(vertex));
        crossings.emplace(key, index);
        return index;
    };
    std::array<std::vector<Face>, 2> output_faces;
    std::array<std::vector<Surface>, 2> output_surfaces;
    std::set<std::pair<std::size_t, std::size_t>> cap_edges;
    for (std::size_t f = 0; f < impl_->faces.size(); ++f)
    {
        const auto& face = impl_->faces[f];
        for (unsigned outside = 0; outside < 2; ++outside)
        {
            Face loop;
            for (std::size_t i = 0; i < face.size(); ++i)
            {
                const auto a = face[i], b = face[(i + 1) % face.size()];
                if (outside ? signs[a] >= 0 : signs[a] <= 0)
                    loop.push_back(a);
                if (signs[a] * signs[b] < 0)
                    loop.push_back(crossing(a, b));
            }
            if (loop.size() >= 3 &&
                !std::all_of(loop.begin(), loop.end(), [&](auto i) { return incident(vertices[i], plane); }))
            {
                // A source face can meet the cut in an edge containing three
                // or more propagated collinear vertices. It is not a positive
                // area output face; the retained adjacent face contributes the
                // same authenticated edge chain to the cap.
                output_faces[outside].push_back(loop);
                output_surfaces[outside].push_back(impl_->surfaces[f]);
                if (!outside)
                    for (std::size_t i = 0; i < loop.size(); ++i)
                    {
                        const auto a = loop[i], b = loop[(i + 1) % loop.size()];
                        if (a != b && incident(vertices[a], plane) && incident(vertices[b], plane))
                            cap_edges.insert(std::minmax(a, b));
                    }
            }
        }
    }
    std::map<std::size_t, std::set<std::size_t>> adjacency;
    for (const auto& [a, b] : cap_edges)
    {
        adjacency[a].insert(b);
        adjacency[b].insert(a);
    }
    if (adjacency.size() < 3)
        reject("positive complementary cut has no closed section.");
    for (const auto& [vertex, neighbors] : adjacency)
        if (neighbors.size() != 2)
            reject("section edge incidence is not one closed cycle.");
    Face cap;
    const auto first = adjacency.begin()->first;
    std::size_t previous = first, current = *adjacency.at(first).begin();
    cap.push_back(first);
    while (current != first)
    {
        if (cap.size() >= adjacency.size())
            reject("section cycle repeats a vertex.");
        cap.push_back(current);
        const auto& neighbors = adjacency.at(current);
        const auto next = *neighbors.begin() == previous ? *neighbors.rbegin() : *neighbors.begin();
        previous = current;
        current = next;
    }
    if (cap.size() != adjacency.size())
        reject("section has disconnected boundaries.");
    Point cap_normal;
    for (std::size_t i = 1; i + 1 < cap.size(); ++i)
        cap_normal = cap_normal + cross(vertices[cap[i]].coordinates - vertices[cap[0]].coordinates,
                                      vertices[cap[i + 1]].coordinates - vertices[cap[0]].coordinates);
    const auto orientation = sign(dot(cap_normal, plane->normal) * Interval(support.orientation));
    if (!orientation)
        reject("section orientation is degenerate.");
    if (orientation < 0)
        std::reverse(cap.begin(), cap.end());
    if (measure_section)
        result.section = section(*impl_, vertices, cap);
    output_faces[0].push_back(cap);
    output_surfaces[0].push_back(support);
    std::reverse(cap.begin(), cap.end());
    output_faces[1].push_back(cap);
    output_surfaces[1].push_back({plane, -support.orientation});
    auto split = std::make_shared<Split>();
    split->parent = impl_->ancestry;
    const auto child = [&](unsigned outside)
    {
        auto poly = std::make_shared<Impl>();
        poly->frame = impl_->frame;
        poly->limits = impl_->limits;
        poly->ancestry = std::make_shared<Ancestry>(Ancestry{impl_->ancestry, split, outside != 0});
        std::set<std::size_t> used;
        for (const auto& face : output_faces[outside])
            used.insert(face.begin(), face.end());
        std::map<std::size_t, std::size_t> remap;
        for (const auto i : used)
        {
            remap.emplace(i, poly->vertices.size());
            poly->vertices.push_back(vertices[i]);
        }
        poly->faces = std::move(output_faces[outside]);
        poly->surfaces = std::move(output_surfaces[outside]);
        for (auto& face : poly->faces)
            for (auto& i : face)
                i = remap.at(i);
        finish(*poly);
        return BoundedConvexGeometryPtr(new BoundedConvexGeometry(std::move(poly)));
    };
    result.inside = child(0);
    result.outside = child(1);
    return result;
}
