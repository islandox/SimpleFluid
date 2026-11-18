/** @file AdaptiveProviderKernel.cc
 * @brief Actual-cell convex refinement and exact saved-parent inverse.
 */
#include "geometry/mesh/AMR/AdaptiveProviderKernel.hh"
#include "geometry/PLIC/PlanarGeometryDeclarations.hh"
#include "geometry/PLIC/details/NativeCellCodecAccess.hh"
#include "geometry/PLIC/details/NativeCellGeometrySourceAccess.hh"
#include "geometry/PLIC/details/NativePlanarOwnerAccess.hh"
#include "geometry/mesh/AMR/AdaptiveGeometrySnapshot.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveWire.hh"
#include "geometry/mesh/AMR/NativeAdaptiveKeyCodec.hh"
#include "geometry/mesh/AMR/NativeKernelAuthority.hh"
#include "geometry/mesh/NativeSparseAccess.hh"
#include "parallel/details/NativeTpetraMapAdmission.hh"
#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>
#include <set>
namespace SimpleFluid::Meshes::AMR
{
namespace
{
using CellCodec = PLIC::details::NativeCellCodecAccess;

void profile_convex_surface_copy(const AdaptiveSurface& value)
{
    CellCodec::charge_split_profile_request(value.key.size(), sizeof(char));
    if (value.boundary.size() > std::string{}.capacity())
        CellCodec::charge_split_profile_request(value.boundary.size() + 1, sizeof(char));
}
} // namespace
AdaptiveLeaf AdaptiveProviderKernel::compact_native_materialize(const AdaptiveLeaf&) const
{
    throw std::invalid_argument("Provider has no bounded compact native materialization.");
}
AdaptiveBytes adaptive_family_key(const AdaptiveCellKey& parent, AdaptiveSplit split)
{
    adaptive_detail::Writer w;
    if (CellCodec::split_profile_active())
        CellCodec::reserve_split_profile(w.data, 3 * sizeof(std::uint64_t) + encoded_adaptive_key_size(parent));
    w.word(0x534646414d363031ULL);
    w.word(std::uint64_t(split));
    CellCodec::charge_split_profile_request(encoded_adaptive_key_size(parent), sizeof(char));
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
    std::vector<PLIC::PlanarFacePtr<ArithmeticMode::Ordinary>> outward;
    for (const auto& side : sides)
    {
        if (side.key.empty() || !side.geometry || (side.orientation != 1 && side.orientation != -1) ||
            (!side.exterior && !side.boundary.empty()))
            throw std::invalid_argument("Invalid provider surface provenance.");
        outward.push_back(side.geometry);
    }
    if (!PLIC::PlanarFace<ArithmeticMode::Ordinary>::cell_shell(outward)->equivalent(*shape))
        throw std::invalid_argument("Provider surfaces differ from exact cell.");
}
PLIC::PointIdentity<ArithmeticMode::Ordinary> AdaptiveProviderKernel::contact_point(
    const AdaptiveLeaf&, const PLIC::PointIdentity<ArithmeticMode::Ordinary>& point) const
{
    return point;
}
std::vector<AdaptiveContactImage> AdaptiveProviderKernel::closed_contact_images(const AdaptiveLeaf& leaf) const
{
    if (contact_capabilities(leaf).normalization != AdaptiveContactNormalization::PhysicalIdentity)
        throw std::invalid_argument("Provider has no qualified closed-contact spatial images.");
    const auto bounds = geometry(leaf)->closed_contact_bounds();
    AdaptiveContactImage image;
    for (std::size_t axis = 0; axis < 3; ++axis)
    {
        image.bounds.lower[axis] = bounds[axis][0];
        image.bounds.upper[axis] = bounds[axis][1];
    }
    return {std::move(image)};
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
std::vector<AdaptiveProductFamilySelector> AdaptiveProviderKernel::ancestry_family_selectors(
    const AdaptiveLeaf& leaf) const
{
    std::vector<AdaptiveProductFamilySelector> result;
    for (auto& family : ancestry_families(leaf))
        result.push_back({std::move(family.parent.key), family.split});
    return result;
}
AdaptiveFamily AdaptiveProviderKernel::ancestry_family(
    const AdaptiveLeaf& leaf, const AdaptiveProductFamilySelector& selector) const
{
    for (auto& family : ancestry_families(leaf))
        if (family.parent.key == selector.parent && family.split == selector.split)
            return std::move(family);
    throw std::invalid_argument("Unknown exact product family selector.");
}
std::vector<AdaptiveLeaf> AdaptiveProviderKernel::repack(std::span<const AdaptiveLeaf> leaves) const
{
    return {leaves.begin(), leaves.end()};
}
AdaptiveLeaf AdaptiveProviderKernel::coarsen(const AdaptiveFamily& family, std::span<const AdaptiveLeaf> children) const
{
    validate(family.parent);
    std::set<AdaptiveCellKey> expected(family.children.begin(), family.children.end()), seen;
    std::vector<PLIC::ConvexGeometryPtr<ArithmeticMode::Ordinary>> shapes;
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
    static constexpr bool native_source_publication_ready = true;
    static constexpr bool native_source_explicit_body_intake = true;
    ConvexKernel(unsigned depth, double quality) : depth_(depth), quality_(quality)
    {
        if (!depth || depth > 64 || !std::isfinite(quality) || quality < 0)
            throw std::invalid_argument("Invalid convex kernel limits.");
    }
    void visit_native_publication_children(const geometry_detail::NativeConstructorOwnerSink&,
        adaptive_detail::NativeKernelAuthority::IdentityVisit,
        adaptive_detail::NativeKernelAuthority::GeometryVisit) const {}
    bool supports_leaf_local_repack() const noexcept override { return true; }
    AdaptiveContactCapabilities contact_capabilities(const AdaptiveLeaf&) const override
    {
        return {AdaptiveContactNormalization::PhysicalIdentity, false, false};
    }
    AdaptiveBytes descriptor() const override
    {
        adaptive_detail::Writer w;
        w.word(0x5346434f4e563631ULL);
        w.word(depth_);
        w.word(std::bit_cast<std::uint64_t>(quality_));
        return std::move(w.data);
    }
    bool supports_native_source(const AdaptiveLeaf& leaf) const noexcept
    {
        return leaf.provider == AdaptiveProviderKind::Polyhedral &&
            std::holds_alternative<ConvexCellAncestry>(leaf.key.ancestry) && leaf.geometry &&
            !leaf.native_recipe && !leaf.factor_recipe && !leaf.join_recipe && leaf.native_cell == 0;
    }
    std::size_t native_source_face_count(const AdaptiveLeaf& leaf) const { return leaf.geometry->num_faces(); }
    AdaptiveCellMetrics prepare_native_source(const AdaptiveLeaf& leaf,
        const geometry_detail::NativeConstructionAdmission& admission) const
    {
        using adaptive_detail::NativeAdaptiveKeyCodec;
        using adaptive_detail::Reader;
        using Planar = PLIC::details::NativePlanarOwnerAccess;
        if (!supports_native_source(leaf) || leaf.domain.empty())
            throw std::invalid_argument("Native convex source requires its original cell and typed ancestry.");
        const auto& ancestry = std::get<ConvexCellAncestry>(leaf.key.ancestry);
        if (ancestry.path.size() > depth_ ||
            std::any_of(ancestry.path.begin(), ancestry.path.end(), [](auto child) { return child > 7; }))
            throw std::invalid_argument("Invalid native convex source ancestry.");
        if (ancestry.path.empty())
        {
            if (!leaf.parent_family.empty())
                throw std::invalid_argument("Native convex root cannot have a parent family.");
        }
        else
        {
            Reader family{leaf.parent_family};
            if (family.word() != 0x534646414d363031ULL ||
                family.word() != static_cast<std::uint64_t>(AdaptiveSplit::Isotropic))
                throw std::invalid_argument("Native convex source has foreign parent-family lineage.");
            const auto storage = geometry_detail::native_constructor_storage(admission);
            const auto parent = NativeAdaptiveKeyCodec::decode(family.bytes_view(), storage);
            family.done();
            const auto* source = std::get_if<ConvexCellAncestry>(&parent.ancestry);
            if (!source || parent.region != leaf.key.region || source->root != ancestry.root ||
                source->path.size() + 1 != ancestry.path.size() ||
                !std::equal(source->path.begin(), source->path.end(), ancestry.path.begin()))
                throw std::invalid_argument("Native convex source ancestry differs from its exact parent key.");
        }
        const auto count = native_source_face_count(leaf);
        if (count < 4 || count > PLIC::GeometryLimits{}.maximum_faces || leaf.surfaces.size() != count)
            throw std::invalid_argument("Native convex source has an incomplete original boundary.");
        for (std::size_t face = 0; face < count; ++face)
        {
            const auto& side = leaf.surfaces[face];
            if (side.key.empty() || !side.geometry || (side.orientation != 1 && side.orientation != -1) ||
                (!side.exterior && !side.boundary.empty()) ||
                (side.require_native_whole_tile && !side.native_whole_geometry))
                throw std::invalid_argument("Native convex source has invalid original surface provenance.");
            const auto original = Planar::cell_face(*leaf.geometry, face, admission);
            const auto same_outward_face = [&](const auto& candidate)
            {
                return PLIC::planar_support(*original) == PLIC::planar_support(*candidate) &&
                    original->contains(*candidate) && candidate->contains(*original) &&
                    !original->oppositely_oriented(*candidate);
            };
            if (!same_outward_face(side.geometry) ||
                (side.native_whole_geometry && !same_outward_face(side.native_whole_geometry)))
                throw std::invalid_argument("Native convex source support differs from its original outward face.");
        }
        const auto metrics = PLIC::details::NativeCellGeometrySourceAccess::original_convex_metrics(*leaf.geometry);
        if (!(metrics.volume > 0) || !std::isfinite(metrics.volume))
            throw std::invalid_argument("Native convex source has invalid original volume.");
        return {metrics.volume, metrics.centroid};
    }
    AdaptiveCellMetrics validate_native_source(const AdaptiveLeaf& leaf, const NativeFragmentRecord& fragment,
        const geometry_detail::NativeConstructionAdmission&, std::span<adaptive_detail::NativeKernelSourceFace> routes)
        const
    {
        const auto* region = std::get_if<DistributedExplicitRegion>(&fragment.region());
        if (!region || region->layout().cells != 1 || region->layout().faces != routes.size() ||
            routes.size() != native_source_face_count(leaf) || &region->geometry() != &region->topology())
            throw std::invalid_argument("Native convex source has a foreign one-cell fragment.");
        const auto original = adaptive_detail::NativeSparseAccess::source_provider_geometry(region->geometry());
        if (original.get() != leaf.geometry.get() || original.owner_before(leaf.geometry) ||
            leaf.geometry.owner_before(original))
            throw std::invalid_argument("Native convex fragment has foreign original geometry ownership.");
        for (std::size_t face = 0; face < routes.size(); ++face)
        {
            if (region->topology().owner_cell(face) != 0 ||
                region->topology().neighbor_cell(face) != MultiRegionMesh::invalid_cell_id())
                throw std::invalid_argument("Native convex fragment has a foreign original face route.");
            routes[face] = {face, face, 1};
        }
        const auto center = region->geometry().cell_centroid(0);
        return {region->geometry().cell_volume(0), {center.x, center.y, center.z}};
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
    PLIC::ConvexGeometryPtr<ArithmeticMode::Ordinary> geometry(const AdaptiveLeaf& leaf) const override
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
        if (CellCodec::split_profile_active())
        {
            CellCodec::charge_split_profile_request(parent.surfaces.size(), sizeof(AdaptiveSurface));
            for (const auto& side : parent.surfaces)
                profile_convex_surface_copy(side);
        }
        const auto parent_surfaces = surfaces(parent);
        AdaptiveSplitResult result;
        result.family.key = adaptive_family_key(parent.key, kind);
        if (CellCodec::split_profile_active())
        {
            CellCodec::charge_split_profile_request(ancestry->path.size(), sizeof(std::uint64_t));
            CellCodec::charge_split_profile_request(parent.domain.size() + parent.parent_family.size(), sizeof(char));
            CellCodec::charge_split_profile_request(parent.surfaces.size(), sizeof(AdaptiveSurface));
            for (const auto& side : parent.surfaces)
                profile_convex_surface_copy(side);
        }
        result.family.parent = parent;
        result.family.split = kind;
        result.family.revision = parent.revision;
        const auto original = geometry(parent);
        const auto pieces = CellCodec::split_profile_active()
            ? CellCodec::profiled_octants(*original, quality_)
            : original->split_aabb_midpoints(quality_);
        if (CellCodec::split_profile_active())
        {
            CellCodec::reserve_split_profile(result.children, pieces.size());
            CellCodec::reserve_split_profile(result.family.children, pieces.size());
        }
        for (const auto& piece : pieces)
        {
            AdaptiveLeaf child;
            CellCodec::charge_split_profile_request(parent.domain.size(), sizeof(char));
            child.domain = parent.domain;
            CellCodec::charge_split_profile_request(ancestry->path.size(), sizeof(std::uint64_t));
            child.key = parent.key;
            auto& path = std::get<ConvexCellAncestry>(child.key.ancestry).path;
            CellCodec::reserve_next_split_profile(path);
            path.push_back(piece.octant);
            child.provider = parent.provider;
            child.material = parent.material;
            child.geometry = piece.geometry;
            child.revision = parent.revision + 1;
            CellCodec::charge_split_profile_request(result.family.key.size(), sizeof(char));
            child.parent_family = result.family.key;
            child.work = parent.work;
            if (CellCodec::split_profile_active())
                CellCodec::reserve_split_profile(child.surfaces, piece.face_origins.size());
            for (std::size_t f = 0; f < piece.face_origins.size(); ++f)
            {
                const auto origin = piece.face_origins[f];
                auto use = [&]
                {
                    if (origin.kind == PLIC::ConvexFaceOrigin::Kind::ParentFace)
                    {
                        const auto& inherited = parent_surfaces.at(origin.index);
                        if (CellCodec::split_profile_active())
                            profile_convex_surface_copy(inherited);
                        return AdaptiveSurface(inherited);
                    }
                    AdaptiveSurface cut;
                    adaptive_detail::Writer w;
                    if (CellCodec::split_profile_active())
                        CellCodec::reserve_split_profile(w.data, 3 * sizeof(std::uint64_t) + encoded_adaptive_key_size(parent.key));
                    w.word(0x5346435554363031ULL);
                    CellCodec::charge_split_profile_request(encoded_adaptive_key_size(parent.key), sizeof(char));
                    w.bytes(encode_adaptive_key(parent.key));
                    w.word(origin.index);
                    cut.key = std::move(w.data);
                    cut.orientation = (piece.octant & (1U << origin.index)) ? -1 : 1;
                    return cut;
                }();
                use.geometry = CellCodec::split_profile_active()
                    ? PLIC::details::NativePlanarOwnerAccess::cell_face(*piece.geometry, f, CellCodec::split_profile_admission())
                    : piece.geometry->face_geometry(f);
                child.surfaces.push_back(std::move(use));
            }
            CellCodec::reserve_next_split_profile(result.family.children);
            CellCodec::charge_split_profile_request(path.size(), sizeof(std::uint64_t));
            result.family.children.push_back(child.key);
            CellCodec::reserve_next_split_profile(result.children);
            result.children.push_back(std::move(child));
        }
        if (result.children.empty())
            throw std::invalid_argument("Actual convex split has no positive child.");
        const auto& anchor = *std::min_element(result.family.children.begin(), result.family.children.end());
        CellCodec::charge_split_profile_request(encoded_adaptive_key_size(anchor), sizeof(char));
        result.family.anchor = encode_adaptive_key(anchor);
        return result;
    }
    void profiled_source_split(const AdaptiveLeaf& leaf, AdaptiveSplit split,
        const geometry_detail::NativeConstructionAdmission& admission, void* context,
        adaptive_detail::NativeKernelAuthority::SplitVisit visit) const
    {
        struct Build
        {
            const ConvexKernel& kernel;
            const AdaptiveLeaf& leaf;
            AdaptiveSplit split;
            void* context;
            adaptive_detail::NativeKernelAuthority::SplitVisit visit;
        } build{*this, leaf, split, context, visit};
        CellCodec::with_split_profile(admission, &build, [](void* raw)
        {
            auto& values = *static_cast<Build*>(raw);
            auto subdivided = values.kernel.split(values.leaf, values.split);
            values.visit(values.context, std::move(subdivided));
        });
    }
    void validate_profiled_source_parent(
        const AdaptiveLeaf& leaf, const geometry_detail::NativeConstructionAdmission& admission) const
    {
        (void) prepare_native_source(leaf, admission);
    }
};
} // namespace
std::shared_ptr<const AdaptiveProviderKernel> convex_adaptive_kernel(unsigned depth, double quality)
{
    return adaptive_detail::NativeKernelAuthority::create<ConvexKernel>(depth, quality);
}
} // namespace SimpleFluid::Meshes::AMR

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
bool NativeKernelAuthority::source_profile_supported() noexcept
{
    return geometry_detail::native_constructor_capacity_profile_supported() &&
        geometry_construction_controls().supported && geometry_detail::NativeTpetraMapAdmission::profile_supported();
}
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
