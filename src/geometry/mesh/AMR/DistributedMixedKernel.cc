/** @file DistributedMixedKernel.cc
 * @brief Persistent support rewriting with exact physical remainder preservation.
 */
#include "geometry/mesh/AMR/DistributedMixedKernel.hh"
#include "geometry/PLIC/PolyhedralComplex.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveIdentityBudget.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveWire.hh"
#include "geometry/mesh/ExplicitRegionProvider.hh"
#include <algorithm>
#include <map>
#include <set>
namespace SimpleFluid::Meshes::AMR
{
namespace
{
using adaptive_detail::Budget;
using adaptive_detail::Reader;
using adaptive_detail::Writer;
AdaptiveBytes canonical_face(const PLIC::PlanarFacePtr<ArithmeticMode::Ordinary>& face)
{
    if (!face)
        throw std::invalid_argument("Join alias lacks original-root geometry.");
    const auto forward = PLIC::canonical_face_key(*face), backward = PLIC::canonical_face_key(*face->reversed());
    return std::min(forward, backward);
}
AdaptiveBytes rule_identity(const AdaptiveJoinRule& rule)
{
    Writer w;
    w.bytes(rule.original_support);
    w.bytes(rule.canonical_support);
    w.bytes(canonical_face(rule.original_root_selection));
    return std::move(w.data);
}
} // namespace
AdaptiveBytes encode_join_capsule(const AdaptiveJoinCapsule& capsule, std::size_t limit)
{
    Writer w;
    Budget budget{limit};
    budget.add(40);
    w.word(0x53464a4f494e3631ULL);
    w.word(2);
    w.word(capsule.route);
    w.word(capsule.rules.size());
    std::set<AdaptiveBytes> unique;
    for (const auto& rule : capsule.rules)
    {
        if (rule.original_support.empty() || rule.canonical_support.empty())
            throw std::invalid_argument("Empty mixed support provenance.");
        auto identity = rule_identity(rule);
        budget.add(identity.size() + sizeof(AdaptiveBytes));
        if (!unique.insert(identity).second)
            throw std::invalid_argument("Duplicate original-root join alias.");
        w.bytes(rule.original_support);
        w.bytes(rule.canonical_support);
        w.bytes(canonical_face(rule.original_root_selection));
    }
    w.word(capsule.boundaries.size());
    std::set<AdaptiveBytes> physical;
    for (const auto& rule : capsule.boundaries)
    {
        if (rule.original_support.empty())
            throw std::invalid_argument("Empty physical boundary support.");
        const auto shape = canonical_face(rule.original_root_selection);
        Writer identity;
        identity.bytes(rule.original_support);
        identity.bytes(shape);
        identity.bytes(std::span(rule.boundary));
        budget.add(identity.data.size() + sizeof(AdaptiveBoundaryRule));
        if (!physical.insert(identity.data).second)
            throw std::invalid_argument("Duplicate physical boundary alias.");
        w.bytes(rule.original_support);
        w.bytes(shape);
        w.bytes(std::span(rule.boundary));
    }
    return std::move(w.data);
}
AdaptiveJoinCapsule decode_join_capsule(std::span<const char> bytes, std::size_t limit)
{
    if (bytes.size() > limit)
        throw std::invalid_argument("Mixed join capsule exceeds byte budget.");
    Reader r{bytes};
    if (r.word() != 0x53464a4f494e3631ULL || r.word() != 2)
        throw std::invalid_argument("Unknown mixed join capsule version.");
    AdaptiveJoinCapsule capsule;
    capsule.route = r.word();
    auto count = r.word();
    if (count > (bytes.size() - r.offset) / 24)
        throw std::invalid_argument("Invalid mixed join alias count.");
    for (std::uint64_t i = 0; i < count; ++i)
    {
        AdaptiveJoinRule rule;
        rule.original_support = r.bytes();
        rule.canonical_support = r.bytes();
        rule.original_root_selection = PLIC::PlanarFace<ArithmeticMode::Ordinary>::decode(r.bytes());
        capsule.rules.push_back(std::move(rule));
    }
    count = r.word();
    if (count > (bytes.size() - r.offset) / 24)
        throw std::invalid_argument("Invalid physical boundary alias count.");
    for (std::uint64_t i = 0; i < count; ++i)
    {
        AdaptiveBoundaryRule rule;
        rule.original_support = r.bytes();
        rule.original_root_selection = PLIC::PlanarFace<ArithmeticMode::Ordinary>::decode(r.bytes());
        auto name = r.bytes();
        rule.boundary.assign(name.begin(), name.end());
        capsule.boundaries.push_back(std::move(rule));
    }
    r.done();
    (void) encode_join_capsule(capsule, limit);
    return capsule;
}
namespace
{
PLIC::PlanarFacePtr<ArithmeticMode::Ordinary> native_certificate(
    const MultiRegionMesh::Region& region, std::uint64_t face)
{
    return std::visit(
        [&](const auto& r) -> PLIC::PlanarFacePtr<ArithmeticMode::Ordinary>
        {
            if constexpr (requires { r.geometry().certified_face_geometry(face); })
                if (auto certificate = r.geometry().certified_face_geometry(face))
                    return certificate;
            if (r.layout().family == RegionLayout::Family::Cylindrical)
                throw std::invalid_argument(
                    "Analytic stationary geometry requires whole-model FrozenNative migration.");
            std::vector<PLIC::GeometryPoint> vertices;
            for (auto n : r.topology().face_nodes(face))
            {
                const auto p = r.geometry().node_coordinates(n);
                vertices.push_back({p.x, p.y, p.z});
            }
            const auto normal = r.geometry().face_area_vector(face);
            return PLIC::PlanarFace<ArithmeticMode::Ordinary>::make(vertices)->oriented_like(
                {normal.x, normal.y, normal.z});
        },
        region);
}
MultiRegionMesh::Region selected_native(const MultiRegionMesh::Region& region, std::span<const std::uint64_t> selected)
{
    return std::visit(
        [&](const auto& r) -> MultiRegionMesh::Region
        {
            if (r.layout().family != RegionLayout::Family::Explicit)
                return r;
            using Provider = ExplicitRegionProvider;
            std::set<std::uint64_t> full(selected.begin(), selected.end()), cells = full, faces, nodes;
            for (auto c : full)
                for (auto f : r.topology().cell_faces(c))
                {
                    faces.insert(f);
                    cells.insert(r.topology().owner_cell(f));
                    const auto neighbor = r.topology().neighbor_cell(f);
                    if (neighbor != UnstructuredMesh::invalid_ordinal)
                        cells.insert(neighbor);
                    for (auto n : r.topology().face_nodes(f))
                        nodes.insert(n);
                }
            std::vector<Provider::Cell> cc;
            std::vector<Provider::Face> ff;
            std::vector<Provider::Node> nn;
            std::map<int, std::string> boundaries;
            for (auto b : r.topology().boundary_batch_ids())
                boundaries.emplace(b, r.topology().boundary_batch_name(b));
            for (auto c : cells)
            {
                Provider::Cell row;
                row.id = c;
                row.type = r.topology().cell_type(c);
                row.volume = r.geometry().cell_volume(c);
                row.centroid = r.geometry().cell_centroid(c);
                if (full.contains(c))
                {
                    for (auto f : r.topology().cell_faces(c))
                        row.faces.push_back(f);
                    for (auto n : r.topology().cell_nodes(c))
                        row.nodes.push_back(n);
                }
                cc.push_back(std::move(row));
            }
            for (auto f : faces)
            {
                Provider::Face row;
                row.id = f;
                row.owner = r.topology().owner_cell(f);
                row.neighbor = r.topology().neighbor_cell(f);
                for (auto n : r.topology().face_nodes(f))
                    row.nodes.push_back(n);
                row.centroid = r.geometry().face_centroid(f);
                row.area_vector = r.geometry().face_area_vector(f);
                row.boundary = r.topology().boundary_id(f);
                ff.push_back(std::move(row));
            }
            for (auto n : nodes)
                nn.push_back({n, r.geometry().node_coordinates(n)});
            std::shared_ptr<const Provider> provider;
            if constexpr (requires { r.geometry().construction(); })
            {
                using Construction = std::remove_cvref_t<decltype(*r.geometry().construction())>;
                if constexpr (std::is_same_v<Construction, PLIC::PolyhedralComplex<ArithmeticMode::Ordinary>>)
                    if (r.geometry().construction())
                    {
                        auto construction = PLIC::PolyhedralComplex<ArithmeticMode::Ordinary>::decode(
                            r.geometry().construction()->encode(selected));
                        provider = std::make_shared<const Provider>(r.layout(), std::move(cc), std::move(ff),
                            std::move(nn), std::move(boundaries), std::move(construction));
                    }
            }
            if (!provider)
            {
                bool exact = false;
                if constexpr (requires { r.geometry().has_construction(); })
                    exact = r.geometry().has_construction();
                if (exact)
                {
                    std::map<std::uint64_t, PLIC::ConvexGeometryPtr<ArithmeticMode::Ordinary>> cell_proofs;
                    std::map<std::uint64_t, PLIC::PlanarFacePtr<ArithmeticMode::Ordinary>> face_proofs;
                    std::map<std::uint64_t, PLIC::PointIdentity<ArithmeticMode::Ordinary>> node_proofs;
                    for (auto c : full)
                    {
                        if constexpr (requires { r.geometry().certified_cell_geometry(c); })
                            cell_proofs.emplace(c, r.geometry().certified_cell_geometry(c));
                    }
                    for (auto f : faces)
                    {
                        auto proof = native_certificate(region, f);
                        auto identities = proof->vertex_identities();
                        for (auto n : r.topology().face_nodes(f))
                        {
                            const auto p = r.geometry().node_coordinates(n);
                            std::optional<PLIC::PointIdentity<ArithmeticMode::Ordinary>> match;
                            for (const auto& identity : identities)
                            {
                                const auto display = PLIC::point_display<ArithmeticMode::Ordinary>(identity);
                                if (display == PLIC::GeometryPoint{p.x, p.y, p.z})
                                {
                                    if (match && *match != identity)
                                        throw std::invalid_argument("Distinct exact stationary vertices have "
                                                                    "indistinguishable native displays.");
                                    match = identity;
                                }
                            }
                            if (!match)
                                throw std::invalid_argument("Stationary exact node lacks its native face certificate.");
                            auto [it, inserted] = node_proofs.emplace(n, *match);
                            if (!inserted && it->second != *match)
                                throw std::invalid_argument("Stationary exact node provenance disagrees across faces.");
                        }
                        face_proofs.emplace(f, std::move(proof));
                    }
                    provider = Provider::from_certified_records(r.layout(), std::move(cc), std::move(ff), std::move(nn),
                        std::move(boundaries), std::move(cell_proofs), std::move(face_proofs), node_proofs);
                }
                else
                    provider = std::make_shared<const Provider>(
                        r.layout(), std::move(cc), std::move(ff), std::move(nn), std::move(boundaries));
            }
            return DistributedExplicitRegion(r.name(), provider, provider);
        },
        region);
}
class StaticKernel final : public AdaptiveProviderKernel
{
    std::uint64_t root_, region_, material_;

public:
    StaticKernel(std::uint64_t root, std::uint64_t region, std::uint64_t material)
        : root_(root), region_(region), material_(material)
    {
    }
    AdaptiveBytes descriptor() const override
    {
        Writer w;
        w.word(0x5346535441543631ULL);
        w.word(root_);
        w.word(region_);
        w.word(material_);
        return std::move(w.data);
    }
    void validate(const AdaptiveLeaf& leaf) const override
    {
        const auto* key = std::get_if<StaticCellAncestry>(&leaf.key.ancestry);
        if (leaf.provider != AdaptiveProviderKind::Static || !key || key->provider_root != root_ ||
            leaf.key.region != region_ || leaf.material != material_ || key->native_cell != leaf.native_cell ||
            !leaf.native_recipe)
            throw std::invalid_argument("Stationary native root membership differs.");
        AdaptiveProviderKernel::validate(leaf);
    }
    std::vector<AdaptiveSurface> surfaces(const AdaptiveLeaf& leaf) const override
    {
        if (!leaf.native_recipe)
            throw std::invalid_argument("Missing stationary native recipe.");
        NativeFragmentCache cache;
        auto region = materialize_native_fragment(*leaf.native_recipe, cache);
        std::vector<AdaptiveSurface> faces;
        std::visit(
            [&](const auto& r)
            {
                if (leaf.native_cell >= r.layout().cells)
                    throw std::invalid_argument("Stationary native ordinal exceeds original root.");
                for (auto face : r.topology().cell_faces(leaf.native_cell))
                {
                    AdaptiveSurface use;
                    Writer w;
                    w.word(0x5346534641434536ULL);
                    w.word(root_);
                    w.word(face);
                    use.key = std::move(w.data);
                    use.geometry = native_certificate(region, face);
                    use.orientation = r.topology().owner_cell(face) == leaf.native_cell ? 1 : -1;
                    if (use.orientation < 0)
                        use.geometry = use.geometry->reversed();
                    use.exterior = r.topology().neighbor_cell(face) == UnstructuredMesh::invalid_ordinal;
                    if (use.exterior && r.topology().boundary_id(face) != UnstructuredMesh::invalid_boundary_id)
                        use.boundary = r.topology().boundary_batch_name(r.topology().boundary_id(face));
                    use.native_face = face;
                    use.require_native_whole_tile = true;
                    use.native_whole_geometry = use.geometry;
                    faces.push_back(std::move(use));
                }
            },
            region);
        return faces;
    }
    PLIC::ConvexGeometryPtr<ArithmeticMode::Ordinary> geometry(const AdaptiveLeaf& leaf) const override
    {
        std::vector<PLIC::PlanarFacePtr<ArithmeticMode::Ordinary>> faces;
        for (const auto& side : surfaces(leaf))
            faces.push_back(side.geometry);
        auto shape = PLIC::PlanarFace<ArithmeticMode::Ordinary>::cell_shell(faces);
        if (leaf.geometry && !shape->equivalent(*leaf.geometry))
            throw std::invalid_argument("Stationary detached native recipe changed its certified root cell.");
        return shape;
    }
    AdaptiveCellMetrics metrics(const AdaptiveLeaf& leaf) const override
    {
        NativeFragmentCache cache;
        auto region = materialize_native_fragment(*leaf.native_recipe, cache);
        return std::visit(
            [&](const auto& r)
            {
                const auto center = r.geometry().cell_centroid(leaf.native_cell);
                return AdaptiveCellMetrics{r.geometry().cell_volume(leaf.native_cell), {center.x, center.y, center.z}};
            },
            region);
    }
    AdaptiveSplitResult split(const AdaptiveLeaf&, AdaptiveSplit) const override
    {
        throw std::invalid_argument("Stationary native participants do not admit topology adaptation.");
    }
    std::vector<AdaptiveLeaf> repack(std::span<const AdaptiveLeaf> leaves) const override
    {
        std::map<AdaptiveBytes, std::vector<const AdaptiveLeaf*>> groups;
        for (const auto& leaf : leaves)
            groups[encode_native_fragment_descriptor(*leaf.native_recipe)].push_back(&leaf);
        NativeFragmentCache cache;
        std::vector<AdaptiveLeaf> result;
        for (const auto& [bytes, group] : groups)
        {
            auto input = decode_native_fragment_descriptor(bytes, cache);
            auto region = materialize_native_fragment(*input, cache);
            std::vector<std::uint64_t> cells;
            for (auto leaf : group)
                cells.push_back(leaf->native_cell);
            auto selected = selected_native(region, cells);
            auto recipe = extract_native_fragment(selected, cache);
            for (auto source : group)
            {
                auto leaf = *source;
                leaf.native_recipe = recipe;
                result.push_back(std::move(leaf));
            }
        }
        return result;
    }
};
class MixedKernel final : public AdaptiveProviderKernel
{
    std::map<std::uint64_t, DistributedMixedRoute> routes_;
    std::size_t budget_;
    AdaptiveBytes descriptor_;
    const DistributedMixedRoute& route(const AdaptiveLeaf& leaf) const
    {
        if (!leaf.join_recipe)
            throw std::invalid_argument("Detached mixed leaf lacks its local join capsule.");
        const auto capsule = decode_join_capsule(*leaf.join_recipe, budget_);
        const auto it = routes_.find(capsule.route);
        if (it == routes_.end() || it->second.provider != leaf.provider || it->second.region != leaf.key.region)
            throw std::invalid_argument("Mixed leaf route differs from typed root membership.");
        return it->second;
    }
    AdaptiveLeaf original(const AdaptiveLeaf& leaf) const
    {
        auto raw = leaf;
        raw.key.region = route(leaf).original_region;
        raw.join_recipe.reset();
        return raw;
    }
    AdaptiveCellKey global(const DistributedMixedRoute& route, AdaptiveCellKey key) const
    {
        if (key.region != route.original_region)
            throw std::invalid_argument("Pure provider returned a foreign root region.");
        key.region = route.region;
        return key;
    }
    AdaptiveBytes namespaced(const DistributedMixedRoute& route, std::span<const char> support) const
    {
        Writer w;
        w.word(0x53464d5355503631ULL);
        w.word(route.identity);
        w.bytes(support);
        return std::move(w.data);
    }
    std::vector<AdaptiveSurface> mapped(const AdaptiveLeaf& leaf) const
    {
        const auto& provider = route(leaf);
        const auto capsule = decode_join_capsule(*leaf.join_recipe, budget_);
        const auto raw = original(leaf);
        std::vector<AdaptiveSurface> result;
        Budget allocation{budget_};
        adaptive_detail::IdentityBudget identities;
        for (const auto& source : provider.kernel->surfaces(raw))
        {
            std::vector<PLIC::PlanarFacePtr<ArithmeticMode::Ordinary>> covered;
            std::vector<std::pair<AdaptiveBytes, PLIC::PlanarFacePtr<ArithmeticMode::Ordinary>>> joined;
            for (const auto& alias : capsule.rules)
                if (alias.original_support == source.key)
                {
                    if (PLIC::planar_support<ArithmeticMode::Ordinary>(*source.geometry) !=
                        PLIC::planar_support<ArithmeticMode::Ordinary>(*alias.original_root_selection))
                        throw std::invalid_argument("Original-root alias changed its exact support plane.");
                    auto piece = source.geometry->intersection(*alias.original_root_selection);
                    if (!piece)
                        continue;
                    bool duplicate = false;
                    for (const auto& [key, previous] : joined)
                        if (previous->intersection(*piece))
                        {
                            if (key == alias.canonical_support && previous->equivalent(*piece))
                            {
                                duplicate = true;
                                break;
                            }
                            throw std::invalid_argument("Mixed original-root join selections overlap inconsistently.");
                        }
                    if (duplicate)
                        continue;
                    identities.face(allocation, *piece);
                    allocation.add(alias.canonical_support.size());
                    covered.push_back(piece);
                    joined.emplace_back(alias.canonical_support, piece);
                    auto use = source;
                    use.geometry = std::move(piece);
                    use.key = alias.canonical_support;
                    use.orientation = PLIC::planar_orientation<ArithmeticMode::Ordinary>(*use.geometry);
                    use.exterior = false;
                    use.boundary.clear();
                    result.push_back(std::move(use));
                }
            auto remaining = covered.empty()
                                 ? std::vector<PLIC::PlanarFacePtr<ArithmeticMode::Ordinary>>{source.geometry}
                                 : PLIC::subtract_planar_faces<ArithmeticMode::Ordinary>(*source.geometry, covered);
            for (auto remainder : remaining)
            {
                if (!source.exterior)
                {
                    auto use = source;
                    use.geometry = remainder;
                    use.key = namespaced(provider, source.key);
                    allocation.add(use.key.size() + sizeof(AdaptiveSurface));
                    result.push_back(std::move(use));
                    continue;
                }
                std::vector<PLIC::PlanarFacePtr<ArithmeticMode::Ordinary>> physical;
                for (const auto& alias : capsule.boundaries)
                    if (alias.original_support == source.key)
                        if (auto piece = remainder->intersection(*alias.original_root_selection))
                        {
                            for (const auto& previous : physical)
                                if (previous->intersection(*piece))
                                    throw std::invalid_argument("Accepted physical boundary selections overlap.");
                            auto use = source;
                            use.geometry = piece;
                            use.boundary = alias.boundary;
                            Writer identity;
                            identity.bytes(namespaced(provider, source.key));
                            identity.bytes(canonical_face(alias.original_root_selection));
                            identity.bytes(std::span(alias.boundary));
                            use.key = std::move(identity.data);
                            identities.face(allocation, *piece);
                            allocation.add(use.key.size() + use.boundary.size() + sizeof(AdaptiveSurface));
                            physical.push_back(piece);
                            result.push_back(std::move(use));
                        }
                try
                {
                    remainder->certify_partition(physical);
                }
                catch (const std::invalid_argument& error)
                {
                    const auto point = remainder->centroid();
                    throw std::invalid_argument(
                        "Mixed physical boundary coverage for route " + std::to_string(provider.identity) + " region " +
                        std::to_string(leaf.key.region) + " ancestry " + std::to_string(leaf.key.ancestry.index()) +
                        " at (" + std::to_string(point.x) + "," + std::to_string(point.y) + "," +
                        std::to_string(point.z) + ") raw name '" + source.boundary + "', pieces " +
                        std::to_string(physical.size()) + ": " + error.what());
                }
            }
        }
        return result;
    }

public:
    MixedKernel(std::vector<DistributedMixedRoute> routes, std::size_t budget) : budget_(budget)
    {
        if (!budget || routes.empty())
            throw std::invalid_argument("Mixed dispatch requires root routes and a finite budget.");
        Writer w;
        w.word(0x53464d4958454436ULL);
        w.word(1);
        w.word(routes.size());
        Budget allocation{budget};
        for (auto& route : routes)
        {
            if (!route.kernel || route.periodic)
                throw std::invalid_argument("Mixed periodic contact frames are not qualified.");
            auto descriptor = route.kernel->descriptor();
            allocation.add(descriptor.size() + route.compact_root_schema.size() + sizeof(DistributedMixedRoute));
            w.word(route.identity);
            w.word(route.original_region);
            w.word(route.region);
            w.word(std::uint64_t(route.provider));
            w.bytes(descriptor);
            w.bytes(route.compact_root_schema);
            if (!routes_.emplace(route.identity, std::move(route)).second)
                throw std::invalid_argument("Duplicate immutable mixed participant route.");
        }
        descriptor_ = std::move(w.data);
    }
    AdaptiveBytes descriptor() const override { return descriptor_; }
    void validate(const AdaptiveLeaf& leaf) const override
    {
        const auto& provider = route(leaf);
        const auto raw = original(leaf);
        provider.kernel->validate(raw);
        std::vector<PLIC::PlanarFacePtr<ArithmeticMode::Ordinary>> faces;
        for (const auto& use : mapped(leaf))
            faces.push_back(use.geometry);
        if (!PLIC::PlanarFace<ArithmeticMode::Ordinary>::cell_shell(faces)->equivalent(*provider.kernel->geometry(raw)))
            throw std::invalid_argument("Mixed join aliases do not preserve the actual cell shell.");
    }
    PLIC::ConvexGeometryPtr<ArithmeticMode::Ordinary> geometry(const AdaptiveLeaf& leaf) const override
    {
        std::vector<PLIC::PlanarFacePtr<ArithmeticMode::Ordinary>> faces;
        for (const auto& use : mapped(leaf))
            faces.push_back(use.geometry);
        return PLIC::PlanarFace<ArithmeticMode::Ordinary>::cell_shell(faces);
    }
    AdaptiveCellMetrics metrics(const AdaptiveLeaf& leaf) const override
    {
        return route(leaf).kernel->metrics(original(leaf));
    }
    std::vector<AdaptiveSurface> surfaces(const AdaptiveLeaf& leaf) const override { return mapped(leaf); }
    AdaptiveTrace traces(const AdaptiveLeaf& leaf) const override
    {
        const auto& provider = route(leaf);
        auto trace = provider.kernel->traces(original(leaf));
        trace.key = leaf.key;
        for (auto& face : trace.faces)
            face.stem = namespaced(provider, face.stem);
        return trace;
    }
    AdaptiveSplit closure_split(const AdaptiveLeaf& leaf, std::size_t axis) const override
    {
        return route(leaf).kernel->closure_split(original(leaf), axis);
    }
    std::vector<AdaptiveCellKey> matching_partners(const AdaptiveLeaf& leaf) const override
    {
        const auto& provider = route(leaf);
        auto keys = provider.kernel->matching_partners(original(leaf));
        for (auto& key : keys)
            key = global(provider, std::move(key));
        return keys;
    }
    std::vector<AdaptiveCellKey> refinement_group(const AdaptiveCellKey& key, AdaptiveSplit split) const override
    {
        // Region/provider root namespaces are validated at bootstrap. A root
        // route remains valid after all current native fragments are discarded.
        const DistributedMixedRoute* found = nullptr;
        for (const auto& [id, route] : routes_)
            if (route.region == key.region && ((route.provider == AdaptiveProviderKind::Cartesian &&
                                                   std::holds_alternative<CartesianCellAncestry>(key.ancestry)) ||
                                                  (route.provider == AdaptiveProviderKind::Product &&
                                                      std::holds_alternative<ProductCellAncestry>(key.ancestry)) ||
                                                  (route.provider == AdaptiveProviderKind::Polyhedral &&
                                                      std::holds_alternative<ConvexCellAncestry>(key.ancestry)) ||
                                                  (route.provider == AdaptiveProviderKind::Static &&
                                                      std::holds_alternative<StaticCellAncestry>(key.ancestry))))
            {
                if (found)
                    throw std::invalid_argument("Mixed original-root namespaces are ambiguous.");
                found = &route;
            }
        if (!found)
            throw std::invalid_argument("No immutable provider root for mixed refinement key.");
        auto raw = key;
        raw.region = found->original_region;
        auto keys = found->kernel->refinement_group(raw, split);
        for (auto& k : keys)
            k = global(*found, std::move(k));
        return keys;
    }
    std::vector<AdaptiveBytes> coarsening_group(const AdaptiveFamily& family) const override
    {
        const auto& provider = route(family.parent);
        auto raw = family;
        raw.parent = original(family.parent);
        raw.key = adaptive_family_key(raw.parent.key, raw.split);
        std::vector<AdaptiveBytes> result;
        if (provider.provider == AdaptiveProviderKind::Cartesian)
            for (const auto& parent : provider.kernel->refinement_group(raw.parent.key, AdaptiveSplit::Isotropic))
                result.push_back(adaptive_family_key(global(provider, parent), AdaptiveSplit::Isotropic));
        else
            result.push_back(family.key);
        return result;
    }
    AdaptiveSplitResult split(const AdaptiveLeaf& parent, AdaptiveSplit split) const override
    {
        const auto& provider = route(parent);
        auto result = provider.kernel->split(original(parent), split);
        result.family.parent = parent;
        result.family.key = adaptive_family_key(parent.key, split);
        result.family.children.clear();
        for (auto& child : result.children)
        {
            child.key = global(provider, std::move(child.key));
            child.join_recipe = parent.join_recipe;
            child.parent_family = result.family.key;
            result.family.children.push_back(child.key);
        }
        result.family.anchor =
            encode_adaptive_key(*std::min_element(result.family.children.begin(), result.family.children.end()));
        return result;
    }
    std::vector<AdaptiveFamily> ancestry_families(const AdaptiveLeaf& leaf) const override
    {
        const auto& provider = route(leaf);
        auto families = provider.kernel->ancestry_families(original(leaf));
        for (auto& family : families)
        {
            family.parent.key = global(provider, std::move(family.parent.key));
            family.parent.join_recipe = leaf.join_recipe;
            family.key = adaptive_family_key(family.parent.key, family.split);
            for (auto& child : family.children)
                child = global(provider, std::move(child));
            family.anchor = encode_adaptive_key(leaf.key);
        }
        return families;
    }
    std::vector<AdaptiveLeaf> repack(std::span<const AdaptiveLeaf> leaves) const override
    {
        std::map<std::uint64_t, std::vector<AdaptiveLeaf>> groups;
        std::map<AdaptiveCellKey, const AdaptiveLeaf*> originals;
        for (const auto& leaf : leaves)
        {
            const auto& provider = route(leaf);
            groups[provider.identity].push_back(original(leaf));
            if (!originals.emplace(leaf.key, &leaf).second)
                throw std::invalid_argument("Duplicate mixed repack leaf.");
        }
        std::vector<AdaptiveLeaf> result;
        for (auto& [id, group] : groups)
        {
            const auto& provider = routes_.at(id);
            for (auto leaf : provider.kernel->repack(group))
            {
                leaf.key = global(provider, std::move(leaf.key));
                const auto& old = *originals.at(leaf.key);
                leaf.join_recipe = old.join_recipe;
                leaf.parent_family = old.parent_family;
                result.push_back(std::move(leaf));
            }
        }
        return result;
    }
};
} // namespace
AdaptiveBootstrap detach_static_participant(const MultiRegionMesh::Region& region, std::uint64_t root,
    std::uint64_t physical_region, std::uint64_t material, std::span<const std::uint64_t> owned, AdaptiveBytes domain)
{
    AdaptiveBootstrap result;
    result.kernel = std::make_shared<const StaticKernel>(root, physical_region, material);
    if (owned.empty())
        return result;
    auto selected = selected_native(region, owned);
    NativeFragmentCache cache;
    auto recipe = extract_native_fragment(selected, cache);
    std::set<std::uint64_t> unique;
    for (auto cell : owned)
    {
        if (!unique.insert(cell).second)
            throw std::invalid_argument("Repeated stationary native cell.");
        AdaptiveLeaf leaf;
        leaf.key = {physical_region, StaticCellAncestry{root, cell}};
        leaf.domain = domain;
        leaf.material = material;
        leaf.provider = AdaptiveProviderKind::Static;
        leaf.native_cell = cell;
        leaf.native_recipe = recipe;
        leaf.geometry = result.kernel->geometry(leaf);
        leaf.surfaces = result.kernel->surfaces(leaf);
        result.kernel->validate(leaf);
        result.support_roots.push_back(leaf);
        result.owned.push_back(std::move(leaf));
    }
    return result;
}
std::shared_ptr<const AdaptiveProviderKernel> mixed_adaptive_kernel(
    std::vector<DistributedMixedRoute> routes, std::size_t byte_budget)
{
    return std::make_shared<const MixedKernel>(std::move(routes), byte_budget);
}
} // namespace SimpleFluid::Meshes::AMR
