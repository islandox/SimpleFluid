/** @file AdaptiveProviderKernel.cc
 * @brief Actual-cell convex refinement and exact saved-parent inverse.
 */
#include "geometry/mesh/AMR/AdaptiveProviderKernel.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveWire.hh"
#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>
#include <set>
namespace SimpleFluid::Meshes::AMR
{
AdaptiveBytes adaptive_family_key(const AdaptiveCellKey& parent, AdaptiveSplit split)
{
    adaptive_detail::Writer w;
    w.word(0x534646414d363031ULL);
    w.word(std::uint64_t(split));
    w.bytes(encode_adaptive_key(parent));
    return std::move(w.data);
}
AdaptiveCellMetrics AdaptiveProviderKernel::metrics(const AdaptiveLeaf& leaf) const
{
    const auto shape = geometry(leaf);
    return {shape->moments().volume.as_double(), shape->centroid()};
}
void AdaptiveProviderKernel::validate(const AdaptiveLeaf& leaf) const
{
    if (leaf.domain.empty())
        throw std::invalid_argument("Missing connected-domain provenance.");
    const auto shape = geometry(leaf);
    const auto sides = surfaces(leaf);
    if (!shape || sides.size() != shape->num_faces())
        throw std::invalid_argument("Provider shell incidence is incomplete.");
    std::vector<ExactPlanarFacePtr> outward;
    for (const auto& side : sides)
    {
        if (side.key.empty() || !side.geometry || (side.orientation != 1 && side.orientation != -1) ||
            (!side.exterior && !side.boundary.empty()))
            throw std::invalid_argument("Invalid provider surface provenance.");
        outward.push_back(side.geometry);
    }
    if (!ExactPlanarFace::cell_shell(outward)->equivalent(*shape))
        throw std::invalid_argument("Provider surfaces differ from exact cell.");
}
AdaptiveBytes AdaptiveProviderKernel::contact_point(const AdaptiveLeaf&, std::span<const char> point) const
{
    return {point.begin(), point.end()};
}
std::vector<AdaptiveCellKey> AdaptiveProviderKernel::matching_partners(const AdaptiveLeaf&) const
{
    return {};
}
std::vector<AdaptiveCellKey> AdaptiveProviderKernel::refinement_group(const AdaptiveCellKey& key, AdaptiveSplit) const
{
    return {key};
}
std::vector<AdaptiveBytes> AdaptiveProviderKernel::coarsening_group(const AdaptiveFamily& family) const
{
    return {family.key};
}
AdaptiveTrace AdaptiveProviderKernel::traces(const AdaptiveLeaf& leaf) const
{
    return {leaf.key, {}};
}
AdaptiveSplit AdaptiveProviderKernel::closure_split(const AdaptiveLeaf&, std::size_t axis) const
{
    if (axis >= 3)
        throw std::out_of_range("Invalid closure axis.");
    return AdaptiveSplit::Isotropic;
}
std::vector<AdaptiveLeaf> AdaptiveProviderKernel::repack(std::span<const AdaptiveLeaf> leaves) const
{
    return {leaves.begin(), leaves.end()};
}
AdaptiveLeaf AdaptiveProviderKernel::coarsen(const AdaptiveFamily& family, std::span<const AdaptiveLeaf> children) const
{
    validate(family.parent);
    std::set<AdaptiveCellKey> expected(family.children.begin(), family.children.end()), seen;
    std::vector<ConvexGeometryPtr> shapes;
    if (expected.size() != family.children.size() || children.size() != expected.size())
        throw std::invalid_argument("Coarsen needs complete immediate family.");
    for (const auto& child : children)
    {
        if (!expected.contains(child.key) || !seen.insert(child.key).second ||
            child.material != family.parent.material || child.domain != family.parent.domain ||
            child.provider != family.parent.provider)
            throw std::invalid_argument("Foreign, repeated or changed coarsen child.");
        validate(child);
        shapes.push_back(geometry(child));
    }
    geometry(family.parent)->certify_partition(shapes);
    auto parent = family.parent;
    if (parent.revision == std::numeric_limits<std::uint64_t>::max())
        throw std::overflow_error("Adaptive leaf revision exhausted.");
    ++parent.revision;
    return parent;
}
namespace
{
class ConvexKernel final : public AdaptiveProviderKernel
{
    unsigned depth_;
    double quality_;

public:
    ConvexKernel(unsigned depth, double quality) : depth_(depth), quality_(quality)
    {
        if (!depth || depth > 64 || !std::isfinite(quality) || quality < 0)
            throw std::invalid_argument("Invalid convex kernel limits.");
    }
    AdaptiveBytes descriptor() const override
    {
        adaptive_detail::Writer w;
        w.word(0x5346434f4e563631ULL);
        w.word(depth_);
        w.word(std::bit_cast<std::uint64_t>(quality_));
        return std::move(w.data);
    }
    void validate(const AdaptiveLeaf& leaf) const override
    {
        const auto* key = std::get_if<ConvexCellAncestry>(&leaf.key.ancestry);
        if (leaf.provider != AdaptiveProviderKind::Polyhedral || !key || key->path.size() > depth_ ||
            leaf.native_recipe || leaf.factor_recipe ||
            std::any_of(key->path.begin(), key->path.end(), [](auto child) { return child > 7; }))
            throw std::invalid_argument("Convex provider/ancestry binding differs.");
        AdaptiveProviderKernel::validate(leaf);
    }
    ConvexGeometryPtr geometry(const AdaptiveLeaf& leaf) const override
    {
        if (!leaf.geometry)
            throw std::invalid_argument("Convex leaf needs actual retained geometry.");
        return leaf.geometry;
    }
    std::vector<AdaptiveSurface> surfaces(const AdaptiveLeaf& leaf) const override
    {
        if (leaf.surfaces.size() != geometry(leaf)->num_faces())
            throw std::invalid_argument("Convex leaf surface lineage incomplete.");
        return leaf.surfaces;
    }
    AdaptiveSplitResult split(const AdaptiveLeaf& parent, AdaptiveSplit kind) const override
    {
        const auto* ancestry = std::get_if<ConvexCellAncestry>(&parent.key.ancestry);
        if (kind != AdaptiveSplit::Isotropic || parent.provider != AdaptiveProviderKind::Polyhedral || !ancestry ||
            ancestry->path.size() >= depth_ ||
            std::any_of(ancestry->path.begin(), ancestry->path.end(), [](auto child) { return child > 7; }) ||
            parent.revision == std::numeric_limits<std::uint64_t>::max())
            throw std::invalid_argument("Unsupported or depth-limited convex split.");
        const auto parent_surfaces = surfaces(parent);
        AdaptiveSplitResult result;
        result.family.key = adaptive_family_key(parent.key, kind);
        result.family.parent = parent;
        result.family.split = kind;
        result.family.revision = parent.revision;
        for (const auto& piece : geometry(parent)->split_aabb_midpoints(quality_))
        {
            AdaptiveLeaf child;
            child.domain = parent.domain;
            child.key = parent.key;
            std::get<ConvexCellAncestry>(child.key.ancestry).path.push_back(piece.octant);
            child.provider = parent.provider;
            child.material = parent.material;
            child.geometry = piece.geometry;
            child.revision = parent.revision + 1;
            child.parent_family = result.family.key;
            child.work = parent.work;
            for (std::size_t f = 0; f < piece.face_origins.size(); ++f)
            {
                const auto origin = piece.face_origins[f];
                AdaptiveSurface use;
                if (origin.kind == ConvexFaceOrigin::Kind::ParentFace)
                    use = parent_surfaces.at(origin.index);
                else
                {
                    adaptive_detail::Writer w;
                    w.word(0x5346435554363031ULL);
                    w.bytes(encode_adaptive_key(parent.key));
                    w.word(origin.index);
                    use.key = std::move(w.data);
                    use.orientation = (piece.octant & (1U << origin.index)) ? -1 : 1;
                }
                use.geometry = piece.geometry->face_geometry(f);
                child.surfaces.push_back(std::move(use));
            }
            result.family.children.push_back(child.key);
            result.children.push_back(std::move(child));
        }
        if (result.children.empty())
            throw std::invalid_argument("Actual convex split has no positive child.");
        result.family.anchor =
            encode_adaptive_key(*std::min_element(result.family.children.begin(), result.family.children.end()));
        return result;
    }
};
} // namespace
std::shared_ptr<const AdaptiveProviderKernel> convex_adaptive_kernel(unsigned depth, double quality)
{
    return std::make_shared<const ConvexKernel>(depth, quality);
}
} // namespace SimpleFluid::Meshes::AMR
