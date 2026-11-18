/** @file DistributedMixedKernel.cc
 * @brief Persistent support rewriting with exact physical remainder preservation.
 */
#include "geometry/mesh/AMR/DistributedMixedKernel.hh"
#include "geometry/PLIC/PolyhedralComplex.hh"
#include "geometry/PLIC/details/ExtrudedConstructionAccess.hh"
#include "geometry/PLIC/details/GeometryIdentityAccess.hh"
#include "geometry/PLIC/details/NativePlanarOwnerAccess.hh"
#include "geometry/VoFWireProtocol.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveIdentityBudget.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveWire.hh"
#include "geometry/mesh/AMR/NativeKernelAuthority.hh"
#include "geometry/mesh/AMR/NativeAdaptiveKeyCodec.hh"
#include "geometry/mesh/ExplicitRegionProvider.hh"
#include "geometry/mesh/NativeFragmentAccess.hh"
#include "geometry/mesh/NativeSparseAccess.hh"
#include <algorithm>
#include <cstring>
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
AdaptiveBytes face_certificate(const PLIC::PlanarFacePtr<ArithmeticMode::Ordinary>& face, std::size_t limit)
{
    if (!face)
        throw std::invalid_argument("Join alias lacks original-root geometry.");
    const auto reversed = face->reversed();
    const auto& selected = PLIC::canonical_face_key(*reversed) < PLIC::canonical_face_key(*face) ? reversed : face;
    return selected->encode(limit);
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
    budget.add(packet_codec::header_bytes<packet_codec::Record> + 3 * sizeof(std::uint64_t));
    packet_codec::write_header(w, packet_codec::Family::MixedJoin, 6, packet_codec::Record::Value);
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
        w.bytes(face_certificate(rule.original_root_selection, limit));
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
        w.bytes(face_certificate(rule.original_root_selection, limit));
        w.bytes(std::span(rule.boundary));
    }
    if (w.data.size() > limit)
        throw std::invalid_argument("Mixed join capsule exceeds byte budget.");
    return std::move(w.data);
}
AdaptiveJoinCapsule decode_join_capsule(std::span<const char> bytes, std::size_t limit)
{
    if (bytes.size() > limit)
        throw std::invalid_argument("Mixed join capsule exceeds byte budget.");
    Reader r{bytes};
    packet_codec::read_header(r, packet_codec::Family::MixedJoin, 6, packet_codec::Record::Value);
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
        rule.original_root_selection = PLIC::PlanarFace<ArithmeticMode::Ordinary>::decode(r.bytes(), limit);
        capsule.rules.push_back(std::move(rule));
    }
    count = r.word();
    if (count > (bytes.size() - r.offset) / 24)
        throw std::invalid_argument("Invalid physical boundary alias count.");
    for (std::uint64_t i = 0; i < count; ++i)
    {
        AdaptiveBoundaryRule rule;
        rule.original_support = r.bytes();
        rule.original_root_selection = PLIC::PlanarFace<ArithmeticMode::Ordinary>::decode(r.bytes(), limit);
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
using Planar = PLIC::details::NativePlanarOwnerAccess;
using Admission = geometry_detail::NativeConstructionAdmission;
using Face = PLIC::PlanarFacePtr<ArithmeticMode::Ordinary>;
using SourceFace = adaptive_detail::NativeKernelSourceFace;

constexpr std::uint64_t static_face_tag = 0x5346534641434536ULL;

bool static_source_recipe(const NativeFragmentDescriptor& recipe) noexcept
{
    using Kind = NativeFragmentDescriptor::Kind;
    if (recipe.kind == Kind::Cartesian || recipe.kind == Kind::ConstructedProduct)
        return recipe.parts.size() == 4 &&
               std::all_of(recipe.parts.begin(), recipe.parts.end(), [](const auto& part) { return bool(part); });
    if (recipe.kind != Kind::LegacyProcedural || recipe.parts.size() != 1 || !recipe.parts[0] ||
        recipe.parts[0]->size() < sizeof(std::uint32_t))
        return false;
    std::uint32_t tag;
    std::memcpy(&tag, recipe.parts[0]->data(), sizeof(tag));
    return tag == 1 || tag == 2 || tag == 5 || tag == 8;
}

bool same_static_recipe(const NativeFragmentDescriptor& original, const NativeFragmentDescriptor& current)
{
    return &original == &current ||
           (original.kind == current.kind && original.view == current.view &&
               original.parts.size() == current.parts.size() &&
               std::equal(original.parts.begin(), original.parts.end(), current.parts.begin(),
                   [](const auto& first, const auto& second)
                   { return first == second || (first && second && *first == *second); }));
}

bool same_outward_face(const Face& first, const Face& second)
{
    if (!first || !second || first->oppositely_oriented(*second) ||
        PLIC::planar_support(*first) != PLIC::planar_support(*second))
        return false;
    const auto& a = first->points();
    const auto& b = second->points();
    return std::is_permutation(a.begin(), a.end(), b.begin(), b.end(), [](const auto& p, const auto& q)
        {
            using Identity = PLIC::details::GeometryIdentityAccess;
            return Identity::point(p) == Identity::point(q);
        });
}

template<class R> Face admitted_outward_face(const R& region, std::uint64_t face, int sign,
    const geometry_detail::NativeConstructorStorageOwner& scratch)
{
    const auto& geometry = region.geometry();
    const auto& admission = scratch.owner->admission();
    Face certificate;
    using Region = std::decay_t<R>;
    if constexpr (std::is_same_v<Region, ExtrudedRegion>)
    {
        if (geometry.construction())
            certificate = PLIC::details::ExtrudedConstructionAccess::face(*geometry.construction(), face, admission);
    }
    else if constexpr (std::is_same_v<Region, DistributedExplicitRegion>)
    {
        if (auto original = adaptive_detail::NativeSparseAccess::source_provider_geometry(geometry))
            certificate = Planar::cell_face(*original, face, admission);
    }
    if (certificate)
        return sign > 0 ? certificate : Planar::reversed(*certificate, admission);
    const auto nodes = region.topology().face_nodes(face);
    if (nodes.size() < 3 || nodes.size() > PLIC::GeometryLimits{}.maximum_vertices)
        throw std::invalid_argument("Stationary native face exceeds the planar vertex bounds.");
    std::vector<PLIC::GeometryPoint> points;
    scratch.owner->reserve(points, nodes.size());
    for (auto node : nodes)
    {
        const auto point = geometry.node_coordinates(node);
        points.push_back({point.x, point.y, point.z});
    }
    const auto area = geometry.face_area_vector(face);
    return Planar::oriented_like(Planar::make(points, admission), {sign * area.x, sign * area.y, sign * area.z},
        admission);
}

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
    std::shared_ptr<const NativeFragmentDescriptor> root_recipe_;

public:
    AdaptiveContactCapabilities contact_capabilities(const AdaptiveLeaf&) const override
    {
        return {AdaptiveContactNormalization::PhysicalIdentity, false, false};
    }
    void visit_native_publication_children(const geometry_detail::NativeConstructorOwnerSink& sink,
        adaptive_detail::NativeKernelAuthority::IdentityVisit identity,
        adaptive_detail::NativeKernelAuthority::GeometryVisit geometry) const
    {
        NativeFragmentAccess::visit_descriptor(root_recipe_, sink, identity, geometry);
    }
    StaticKernel(std::uint64_t root, std::uint64_t region, std::uint64_t material,
        std::shared_ptr<const NativeFragmentDescriptor> recipe)
        : root_(root), region_(region), material_(material), root_recipe_(std::move(recipe))
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
    static constexpr bool native_source_publication_ready = true;
    static constexpr bool native_source_explicit_body_intake = true;
    bool supports_native_source(const AdaptiveLeaf& leaf) const noexcept
    {
        return root_recipe_ && static_source_recipe(*root_recipe_) && leaf.native_recipe &&
               static_source_recipe(*leaf.native_recipe) && leaf.provider == AdaptiveProviderKind::Static &&
               std::holds_alternative<StaticCellAncestry>(leaf.key.ancestry) && leaf.geometry &&
               !leaf.factor_recipe && !leaf.join_recipe;
    }
    std::size_t native_source_face_count(const AdaptiveLeaf& leaf) const { return leaf.geometry->num_faces(); }
    AdaptiveCellMetrics validate_native_source(const AdaptiveLeaf& leaf, const NativeFragmentRecord& fragment,
        const Admission& admission, std::span<SourceFace> routes) const
    {
        const auto& key = std::get<StaticCellAncestry>(leaf.key.ancestry);
        if (!supports_native_source(leaf) || key.provider_root != root_ || leaf.key.region != region_ ||
            leaf.material != material_ || key.native_cell != leaf.native_cell || leaf.domain.empty() ||
            !leaf.parent_family.empty() || !same_static_recipe(*root_recipe_, *leaf.native_recipe) ||
            routes.size() != leaf.surfaces.size() || routes.size() != leaf.geometry->num_faces())
            throw std::invalid_argument("Stationary native source differs from its exact detached root membership.");
        const auto scratch = geometry_detail::native_constructor_storage(admission);
        std::vector<Face> shell;
        scratch.owner->reserve(shell, routes.size());
        for (std::size_t i = 0; i < routes.size(); ++i)
            shell.push_back(Planar::cell_face(*leaf.geometry, i, admission));
        return std::visit(
            [&](const auto& region)
            {
                const auto& topology = region.topology();
                if (leaf.native_cell >= region.layout().cells)
                    throw std::invalid_argument("Stationary native ordinal exceeds the exact detached root.");
                const auto faces = topology.cell_faces(leaf.native_cell);
                if (faces.size() != routes.size())
                    throw std::invalid_argument("Stationary native source has an incomplete original shell.");
                for (std::size_t surface = 0; surface < routes.size(); ++surface)
                {
                    const auto face = faces[surface];
                    const auto owner = topology.owner_cell(face);
                    const auto neighbor = topology.neighbor_cell(face);
                    if (owner != leaf.native_cell && neighbor != leaf.native_cell)
                        throw std::invalid_argument("Stationary source has a foreign native incidence.");
                    const int sign = owner == leaf.native_cell ? 1 : -1;
                    const bool exterior = neighbor == UnstructuredMesh::invalid_ordinal;
                    const auto boundary = topology.boundary_id(face);
                    const std::string_view name = exterior && boundary != UnstructuredMesh::invalid_boundary_id
                        ? topology.boundary_batch_name(boundary) : std::string_view{};
                    const auto& use = leaf.surfaces[surface];
                    Reader support{use.key};
                    if (support.word() != static_face_tag || support.word() != root_ || support.word() != face)
                        throw std::invalid_argument("Stationary source has a foreign original face support.");
                    support.done();
                    const auto original = admitted_outward_face(region, face, sign, scratch);
                    if (use.native_face != face || use.orientation != sign || use.exterior != exterior ||
                        use.boundary != name || !use.authority_image.is_zero() || !use.require_native_whole_tile ||
                        !same_outward_face(original, use.geometry) || !same_outward_face(original, use.native_whole_geometry))
                        throw std::invalid_argument("Stationary surface differs from its original native face.");
                    const auto match = std::find_if(shell.begin(), shell.end(),
                        [&](const auto& side) { return same_outward_face(original, side); });
                    if (match == shell.end())
                        throw std::invalid_argument("Stationary source has a foreign exact convex shell.");
                    match->reset();
                    routes[surface] = {surface, face, sign};
                }
                const auto center = region.geometry().cell_centroid(leaf.native_cell);
                return AdaptiveCellMetrics{region.geometry().cell_volume(leaf.native_cell), {center.x, center.y, center.z}};
            },
            fragment.region());
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
                    w.word(static_face_tag);
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
    bool supports_leaf_local_repack() const noexcept override { return true; }
    std::vector<AdaptiveLeaf> repack(std::span<const AdaptiveLeaf> leaves) const override
    {
        if (root_recipe_ && static_source_recipe(*root_recipe_))
        {
            for (const auto& leaf : leaves)
                if (!leaf.native_recipe || !same_static_recipe(*root_recipe_, *leaf.native_recipe))
                    throw std::invalid_argument("Stationary repack changed the detached root recipe.");
            return {leaves.begin(), leaves.end()};
        }
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
} // namespace

class MixedKernel final : public AdaptiveProviderKernel
{
    geometry_detail::NativeConstructorStorageOwner native_routes_storage_;
    using Routes = std::map<std::uint64_t, DistributedMixedRoute, std::less<std::uint64_t>,
        geometry_detail::NativeStorageAllocator<std::pair<const std::uint64_t, DistributedMixedRoute>>>;
    Routes routes_;
    std::size_t budget_;
    AdaptiveBytes descriptor_;
    std::shared_ptr<const geometry_detail::NativeSchemaOwner> native_schema_;
    using Authority = adaptive_detail::NativeKernelAuthority;

    const DistributedMixedRoute& checked_route(const AdaptiveLeaf& leaf, std::uint64_t identity) const
    {
        const auto it = routes_.find(identity);
        if (it == routes_.end() || it->second.provider != leaf.provider || it->second.region != leaf.key.region)
            throw std::invalid_argument("Mixed leaf route differs from typed root membership.");
        return it->second;
    }
    Reader native_capsule(const AdaptiveLeaf& leaf) const
    {
        if (!leaf.join_recipe || leaf.join_recipe->size() > budget_)
            throw std::invalid_argument("Detached mixed leaf lacks its bounded local join capsule.");
        Reader reader{*leaf.join_recipe};
        packet_codec::read_header(reader, packet_codec::Family::MixedJoin, 6, packet_codec::Record::Value);
        return reader;
    }
    const DistributedMixedRoute& route(const AdaptiveLeaf& leaf) const
    {
        if (!leaf.join_recipe)
            throw std::invalid_argument("Detached mixed leaf lacks its local join capsule.");
        const auto capsule = decode_join_capsule(*leaf.join_recipe, budget_);
        return checked_route(leaf, capsule.route);
    }
    AdaptiveLeaf original(const AdaptiveLeaf& leaf) const
    {
        auto raw = leaf;
        raw.key.region = route(leaf).original_region;
        raw.join_recipe.reset();
        return raw;
    }
    static AdaptiveLeaf native_original(const AdaptiveLeaf& leaf, const DistributedMixedRoute& route,
        const geometry_detail::NativeConstructorStorageOwner& storage)
    {
        using adaptive_detail::NativeAdaptiveKeyCodec;
        AdaptiveLeaf raw;
        raw.key = NativeAdaptiveKeyCodec::copy(leaf.key, storage);
        raw.key.region = route.original_region;
        storage.owner->copy(raw.domain, std::span<const char>(leaf.domain));
        raw.material = leaf.material;
        raw.revision = leaf.revision;
        raw.provider = leaf.provider;
        raw.geometry = leaf.geometry;
        raw.native_recipe = leaf.native_recipe;
        raw.native_cell = leaf.native_cell;
        raw.factor_recipe = leaf.factor_recipe;
        raw.work = leaf.work;
        storage.owner->reserve(raw.surfaces, leaf.surfaces.size());
        for (const auto& surface : leaf.surfaces)
        {
            auto& copy = raw.surfaces.emplace_back();
            storage.owner->copy(copy.key, std::span<const char>(surface.key));
            storage.owner->string(copy.boundary, surface.boundary);
            copy.geometry = surface.geometry;
            copy.orientation = surface.orientation;
            copy.exterior = surface.exterior;
            copy.native_face = surface.native_face;
            copy.authority_image = surface.authority_image;
            copy.require_native_whole_tile = surface.require_native_whole_tile;
            copy.native_whole_geometry = surface.native_whole_geometry;
        }
        if (!leaf.parent_family.empty())
        {
            Reader family{leaf.parent_family};
            const auto tag = family.word();
            const auto split = family.word();
            auto parent = NativeAdaptiveKeyCodec::decode(family.bytes_view(), storage);
            family.done();
            if (tag != 0x534646414d363031ULL || parent.region != route.region ||
                split > static_cast<std::uint64_t>(AdaptiveSplit::Axial))
                throw std::invalid_argument("Mixed original parent family differs from its typed route.");
            parent.region = route.original_region;
            Writer encoded;
            storage.owner->reserve(encoded.data, leaf.parent_family.size());
            encoded.word(tag);
            encoded.word(split);
            NativeAdaptiveKeyCodec::write(encoded, parent);
            raw.parent_family = std::move(encoded.data);
        }
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

    PLIC::ConvexGeometryPtr<ArithmeticMode::Ordinary> certified_cartesian_shape(
        const AdaptiveLeaf& leaf, const AdaptiveLeaf& raw, const AdaptiveProviderKernel& provider) const
    {
        AdaptiveLeaf physical;
        physical.geometry = provider.geometry(raw);
        physical.surfaces = mapped(leaf);
        adaptive_detail::validate_adaptive_leaf_boundary(physical, budget_);
        return std::move(physical.geometry);
    }

public:
    static constexpr bool native_source_publication_ready = true;
    static constexpr bool native_source_explicit_body_intake = true;
    bool supports_native_source(const AdaptiveLeaf& leaf) const noexcept
    {
        try
        {
            auto reader = native_capsule(leaf);
            const auto& provider = checked_route(leaf, reader.word());
            if (!Authority::supports_original_source_validation(provider.kernel) || provider.periodic)
                return false;
            switch (leaf.provider)
            {
            case AdaptiveProviderKind::Cartesian:
                return leaf.native_recipe && !leaf.geometry && leaf.surfaces.empty() && !leaf.factor_recipe;
            case AdaptiveProviderKind::Product:
                return leaf.native_recipe && leaf.factor_recipe && !leaf.geometry && leaf.surfaces.empty();
            case AdaptiveProviderKind::Polyhedral:
                return leaf.geometry && !leaf.native_recipe && !leaf.factor_recipe;
            case AdaptiveProviderKind::Static:
                return leaf.geometry && leaf.native_recipe && !leaf.factor_recipe &&
                       static_source_recipe(*leaf.native_recipe);
            }
        }
        catch (const std::invalid_argument&)
        {
            return false;
        }
        return false;
    }
    AdaptiveCellMetrics prepare_native_source(const AdaptiveLeaf& leaf, const Admission& admission) const
    {
        auto reader = native_capsule(leaf);
        const auto& provider = checked_route(leaf, reader.word());
        const auto scratch = geometry_detail::native_constructor_storage(admission);
        const auto raw = native_original(leaf, provider, scratch);
        return Authority::prepare_source(provider.kernel, raw, admission);
    }
    AdaptiveCellMetrics collect_native_source(const AdaptiveLeaf& leaf, const NativeFragmentRecord& fragment,
        const Admission& admission, const geometry_detail::NativeConstructorStorageOwner& output,
        std::vector<SourceFace>& routes) const
    {
        auto reader = native_capsule(leaf);
        const auto& provider = checked_route(leaf, reader.word());
        const auto scratch = geometry_detail::native_constructor_storage(admission);
        const auto raw = native_original(leaf, provider, scratch);
        const auto count = Authority::source_face_count(provider.kernel, raw, fragment);
        struct OriginalSupport
        {
            AdaptiveBytes key;
            bool exterior = false;
        };
        struct OriginalFaces
        {
            const geometry_detail::NativeConstructorStorageOwner& storage;
            std::vector<OriginalSupport> supports;
        } originals{scratch, {}};
        scratch.owner->reserve(originals.supports, count);
        originals.supports.resize(count);
        std::vector<SourceFace> native_routes;
        scratch.owner->reserve(native_routes, count);
        native_routes.resize(count);
        const auto metrics = Authority::validate_source(provider.kernel, raw, fragment, admission, native_routes,
            &originals, [](void* context, std::size_t surface, std::span<const char> key, bool exterior)
            {
                auto& values = *static_cast<OriginalFaces*>(context);
                if (surface >= values.supports.size() || key.empty() || !values.supports[surface].key.empty())
                    throw std::invalid_argument("Mixed child supplied a missing or repeated original support.");
                auto& support = values.supports[surface];
                values.storage.owner->copy(support.key, key);
                support.exterior = exterior;
            });
        struct Selection
        {
            std::span<const char> original, destination;
            Face geometry;
        };
        const auto read_selections = [&](bool joined)
        {
            const auto size = reader.word();
            if (size > (reader.data.size() - reader.offset) / (3 * sizeof(std::uint64_t)))
                throw std::invalid_argument("Mixed native source has an invalid root selection count.");
            geometry_detail::NativeConstructorVector<Selection> selections(admission.allocator<Selection>());
            for (std::size_t i = 0; i < size; ++i)
            {
                const auto original = reader.bytes_view();
                const auto second = reader.bytes_view();
                const auto third = reader.bytes_view();
                if (original.empty() || (joined && second.empty()))
                    throw std::invalid_argument("Mixed native source lacks original-root selection provenance.");
                if (std::none_of(originals.supports.begin(), originals.supports.end(),
                        [&](const auto& support) { return std::ranges::equal(support.key, original); }))
                    continue; // This active descendant does not use that original root support.
                const auto destination = joined ? second : third;
                auto geometry = Planar::decode(joined ? third : second, budget_, admission);
                if (std::any_of(selections.begin(), selections.end(), [&](const auto& other)
                        {
                            return std::ranges::equal(other.original, original) &&
                                   std::ranges::equal(other.destination, destination) &&
                                   other.geometry->equivalent(*geometry);
                        }))
                    throw std::invalid_argument("Mixed native source repeats an original-root selection.");
                selections.push_back({original, destination, std::move(geometry)});
            }
            return selections;
        };
        const auto joined = read_selections(true);
        const auto boundary = read_selections(false);
        reader.done();
        geometry_detail::NativeConstructorVector<SourceFace> mapped_routes(admission.allocator<SourceFace>());
        for (std::size_t index = 0; index < count; ++index)
        {
            const auto& route = native_routes[index];
            const auto& support = originals.supports[index];
            if (route.surface != index || (route.outward_sign != 1 && route.outward_sign != -1) || support.key.empty())
                throw std::invalid_argument("Mixed child did not certify the original surface route.");
            const auto phase = geometry_detail::native_constructor_storage(admission);
            const auto source = !raw.surfaces.empty() ? raw.surfaces[index].geometry :
                std::visit([&](const auto& region)
                    { return admitted_outward_face(region, route.native_face, route.outward_sign, phase); }, fragment.region());
            if (!source)
                throw std::invalid_argument("Mixed child original support lacks its exact physical loop.");
            geometry_detail::NativeConstructorVector<Face> covered(admission.allocator<Face>());
            geometry_detail::NativeConstructorVector<std::span<const char>> destinations(
                admission.allocator<std::span<const char>>());
            geometry_detail::NativeConstructorVector<Face> complete(admission.allocator<Face>());
            const auto append = [&](const Face& face)
            {
                if (mapped_routes.size() == PLIC::GeometryLimits{}.maximum_faces)
                    throw std::invalid_argument("Mixed native source exceeds the supported mapped face bound.");
                mapped_routes.push_back({mapped_routes.size(), route.native_face, route.outward_sign});
                complete.push_back(face);
            };
            for (const auto& alias : joined)
            {
                if (!std::ranges::equal(alias.original, support.key))
                    continue;
                if (PLIC::planar_support(*source) != PLIC::planar_support(*alias.geometry))
                    throw std::invalid_argument("Mixed native join selection changed its original support plane.");
                const auto piece = Planar::intersection(*source, *alias.geometry, admission);
                if (!piece)
                    continue;
                bool duplicate = false;
                for (std::size_t previous = 0; previous < covered.size(); ++previous)
                    if (Planar::intersection(*covered[previous], *piece, admission))
                    {
                        if (std::ranges::equal(alias.destination, destinations[previous]) &&
                            covered[previous]->equivalent(*piece))
                        {
                            duplicate = true;
                            break;
                        }
                        throw std::invalid_argument("Mixed native join selections overlap inconsistently.");
                    }
                if (!duplicate)
                {
                    covered.push_back(piece);
                    destinations.push_back(alias.destination);
                    append(piece);
                }
            }
            const auto visit_remainder = [&](const Face& remainder)
            {
                if (!support.exterior)
                {
                    append(remainder);
                    return;
                }
                const auto begin = complete.size();
                for (const auto& alias : boundary)
                    if (std::ranges::equal(alias.original, support.key))
                        if (const auto piece = Planar::intersection(*remainder, *alias.geometry, admission))
                            append(piece);
                Planar::certify_partition(*remainder, std::span<const Face>(complete).subspan(begin), admission);
            };
            if (covered.empty())
                visit_remainder(source);
            else
                for (const auto& remainder : Planar::subtract(*source, covered, admission))
                    visit_remainder(remainder);
            Planar::certify_partition(*source, complete, admission);
        }
        output.owner->copy(routes, std::span<const SourceFace>(mapped_routes));
        return metrics;
    }
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
    AdaptiveBytes descriptor() const override { return native_schema_ ? native_schema_->descriptor() : descriptor_; }
    std::shared_ptr<const geometry_detail::NativeSchemaOwner> seal_native_publication_schema(
        const geometry_detail::NativeConstructionAdmission& admission)
    {
        if (native_schema_)
            throw std::logic_error("Mixed kernel source schema was already sealed.");
        auto storage = geometry_detail::native_constructor_storage(admission);
        Routes admitted(std::less<std::uint64_t>{}, Routes::allocator_type(storage.owner));
        for (const auto& [identity, previous] : routes_)
        {
            DistributedMixedRoute route;
            route.identity = previous.identity;
            route.original_region = previous.original_region;
            route.region = previous.region;
            route.provider = previous.provider;
            route.kernel = previous.kernel;
            route.periodic = previous.periodic;
            storage.owner->copy(route.compact_root_schema, std::span<const char>(previous.compact_root_schema));
            admitted.emplace(identity, std::move(route));
        }
        routes_.swap(admitted);
        native_routes_storage_ = std::move(storage);
        native_schema_ = geometry_detail::NativeSchemaOwnerAccess::adopt(std::move(descriptor_), admission);
        return native_schema_;
    }
    void visit_native_publication_children(const geometry_detail::NativeConstructorOwnerSink& sink,
        adaptive_detail::NativeKernelAuthority::IdentityVisit identity,
        adaptive_detail::NativeKernelAuthority::GeometryVisit geometry) const
    {
        if (!native_routes_storage_.owner || !native_schema_ || descriptor_.capacity())
            throw std::logic_error("Mixed source route storage is not sealed.");
        sink.storage(native_routes_storage_);
        // The routes map allocator carries one additional real strong edge.
        sink.storage_reference(native_routes_storage_.owner, native_routes_storage_.envelope);
        sink.schema(native_schema_);
        for (const auto& [id, route] : routes_)
            adaptive_detail::NativeKernelAuthority::visit(route.kernel, sink, identity, geometry);
    }
    void validate(const AdaptiveLeaf& leaf) const override
    {
        const auto& provider = route(leaf);
        const auto raw = original(leaf);
        provider.kernel->validate(raw);
        if (provider.kernel->has_native_cartesian_geometry())
        {
            // Each mapped piece belongs to one source face. Exact outward
            // partitions preserve the certified box without re-importing all
            // boundary-name subdivisions as a new convex cell.
            (void) certified_cartesian_shape(leaf, raw, *provider.kernel);
            return;
        }
        std::vector<PLIC::PlanarFacePtr<ArithmeticMode::Ordinary>> faces;
        for (const auto& use : mapped(leaf))
            faces.push_back(use.geometry);
        if (!PLIC::PlanarFace<ArithmeticMode::Ordinary>::cell_shell(faces)->equivalent(*provider.kernel->geometry(raw)))
            throw std::invalid_argument("Mixed join aliases do not preserve the actual cell shell.");
    }
    PLIC::ConvexGeometryPtr<ArithmeticMode::Ordinary> geometry(const AdaptiveLeaf& leaf) const override
    {
        const auto& provider = route(leaf);
        if (provider.kernel->has_native_cartesian_geometry())
            return certified_cartesian_shape(leaf, original(leaf), *provider.kernel);
        std::vector<PLIC::PlanarFacePtr<ArithmeticMode::Ordinary>> faces;
        for (const auto& use : mapped(leaf))
            faces.push_back(use.geometry);
        return PLIC::PlanarFace<ArithmeticMode::Ordinary>::cell_shell(faces);
    }
    AdaptiveCellMetrics metrics(const AdaptiveLeaf& leaf) const override
    {
        return route(leaf).kernel->metrics(original(leaf));
    }
    AdaptiveContactCapabilities contact_capabilities(const AdaptiveLeaf& leaf) const override
    {
        return route(leaf).kernel->contact_capabilities(original(leaf));
    }
    std::vector<AdaptiveContactImage> closed_contact_images(const AdaptiveLeaf& leaf) const override
    {
        // Mixed construction already rejects periodic routes. Each pure route
        // retains its complete logical frame; canonical physical coordinates
        // allow contact between distinct routes in the joined domain.
        return route(leaf).kernel->closed_contact_images(original(leaf));
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
    void validate_profiled_source_parent(const AdaptiveLeaf& leaf, const Admission& admission) const
    {
        auto reader = native_capsule(leaf);
        const auto& provider = checked_route(leaf, reader.word());
        const auto storage = geometry_detail::native_constructor_storage(admission);
        const auto raw = native_original(leaf, provider, storage);
        (void) Authority::prepare_source(provider.kernel, raw, admission);
    }
    void profiled_source_split(const AdaptiveLeaf& leaf, AdaptiveSplit split, const Admission& admission,
        void* context, Authority::SplitVisit visit) const
    {
        auto reader = native_capsule(leaf);
        const auto& provider = checked_route(leaf, reader.word());
        const auto storage = geometry_detail::native_constructor_storage(admission);
        const auto raw = native_original(leaf, provider, storage);
        struct Route
        {
            const AdaptiveLeaf& source;
            const AdaptiveLeaf& original;
            AdaptiveSplit split;
            const geometry_detail::NativeConstructorStorageOwner& storage;
            void* context;
            Authority::SplitVisit visit;
        } route{leaf, raw, split, storage, context, visit};
        Authority::profiled_source_split(provider.kernel, raw, split, admission, &route,
            [](void* context, AdaptiveSplitResult&& child)
            {
                using adaptive_detail::NativeAdaptiveKeyCodec;
                auto& binding = *static_cast<Route*>(context);
                const auto& original = binding.original;
                const auto& source = binding.source;
                const auto replace = [](AdaptiveBytes& field, std::span<const char> value)
                {
                    if (field.size() != value.size())
                        throw std::invalid_argument("Mixed source split changed its canonical key extent.");
                    std::copy(value.begin(), value.end(), field.begin());
                };
                if (child.family.parent.key != original.key || child.children.size() != child.family.children.size())
                    throw std::invalid_argument("Mixed source split has foreign original child membership.");
                child.family.parent.key.region = source.key.region;
                child.family.parent.join_recipe = source.join_recipe;
                replace(child.family.parent.parent_family, source.parent_family);
                Writer key;
                binding.storage.owner->reserve(key.data, child.family.key.size());
                key.word(0x534646414d363031ULL);
                key.word(static_cast<std::uint64_t>(binding.split));
                NativeAdaptiveKeyCodec::write(key, source.key);
                replace(child.family.key, key.data);
                for (std::size_t i = 0; i < child.children.size(); ++i)
                {
                    auto& row = child.children[i];
                    auto& saved = child.family.children[i];
                    if (row.key.region != original.key.region || row.key != saved)
                        throw std::invalid_argument("Mixed source split changed an original child key.");
                    saved.region = row.key.region = source.key.region;
                    row.join_recipe = source.join_recipe;
                    replace(row.parent_family, key.data);
                }
                if (child.family.children.empty())
                    throw std::invalid_argument("Mixed source split returned no original children.");
                const auto& anchor = *std::min_element(child.family.children.begin(), child.family.children.end());
                const auto encoded = NativeAdaptiveKeyCodec::encode(anchor, binding.storage);
                replace(child.family.anchor, encoded);
                binding.visit(binding.context, std::move(child));
            });
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
    std::vector<AdaptiveProductFamilySelector> ancestry_family_selectors(const AdaptiveLeaf& leaf) const override
    {
        const auto& provider = route(leaf);
        auto rows = provider.kernel->ancestry_family_selectors(original(leaf));
        for (auto& row : rows)
            row.parent = global(provider, std::move(row.parent));
        return rows;
    }
    bool product_family_proof_revision_invariant(const AdaptiveLeaf* parent) const override
    {
        return parent && route(*parent).kernel->product_family_proof_revision_invariant();
    }

    AdaptiveFamily ancestry_family(
        const AdaptiveLeaf& leaf, const AdaptiveProductFamilySelector& selector) const override
    {
        const auto& provider = route(leaf);
        if (selector.parent.region != leaf.key.region)
            throw std::invalid_argument("Foreign mixed factor parent selector.");
        auto raw = selector;
        raw.parent.region = provider.original_region;
        auto family = provider.kernel->ancestry_family(original(leaf), raw);
        family.parent.key = global(provider, std::move(family.parent.key));
        family.parent.join_recipe = leaf.join_recipe;
        family.key = adaptive_family_key(family.parent.key, family.split);
        for (auto& child : family.children)
            child = global(provider, std::move(child));
        family.anchor = encode_adaptive_key(leaf.key);
        return family;
    }
    bool supports_leaf_local_repack() const noexcept override
    {
        return std::all_of(routes_.begin(), routes_.end(),
            [](const auto& entry) { return entry.second.kernel->supports_leaf_local_repack(); });
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
AdaptiveBootstrap detach_static_participant(const MultiRegionMesh::Region& region, std::uint64_t root,
    std::uint64_t physical_region, std::uint64_t material, std::span<const std::uint64_t> owned, AdaptiveBytes domain)
{
    AdaptiveBootstrap result;
    NativeFragmentCache cache;
    std::shared_ptr<const NativeFragmentDescriptor> recipe;
    const bool procedural = std::holds_alternative<CartesianRegion>(region) ||
        std::holds_alternative<NativeIsoRegion<OrthogonalCartesian3D>>(region) ||
        std::holds_alternative<NativeIsoRegion<SemiStructuredXY_Z>>(region) ||
        std::holds_alternative<ExtrudedRegion>(region);
    // An initially empty participant can later receive original ordinals
    // during migration. Its closed root is the same detached procedural
    // descriptor from which nonempty ranks construct their leaves.
    if (procedural)
        recipe = extract_native_fragment(region, cache);
    else if (!owned.empty())
        recipe = extract_native_fragment(selected_native(region, owned), cache);
    result.kernel =
        adaptive_detail::NativeKernelAuthority::create<StaticKernel>(root, physical_region, material, recipe);
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
    const bool qualified = std::all_of(routes.begin(), routes.end(),
        [](const auto& route) { return adaptive_detail::NativeKernelAuthority::qualified(route.kernel); });
    if (qualified)
        return adaptive_detail::NativeKernelAuthority::create<MixedKernel>(std::move(routes), byte_budget);
    return std::make_shared<const MixedKernel>(std::move(routes), byte_budget);
}
} // namespace SimpleFluid::Meshes::AMR
