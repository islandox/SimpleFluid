#include "geometry/mesh/AMR/MixedAMR.hh"
#include "geometry/PLIC/PlanarGeometry.hh"
#include "geometry/mesh/AMR/AdaptiveOverlap.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveWire.hh"
#include "geometry/mesh/AMR/DistributedCartesianKernel.hh"
#include "geometry/mesh/AMR/DistributedMixedKernel.hh"
#include "geometry/mesh/AMR/DistributedProductKernel.hh"
#include <Teuchos_CommHelpers.hpp>
#include <Teuchos_DefaultSerialComm.hpp>
#include <Tpetra_Core.hpp>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <exception>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <tuple>
namespace SimpleFluid::Meshes::AMR
{
namespace
{
using ID = std::uint64_t;
template<class F> void collective(F&& f)
{
    std::exception_ptr error;
    try
    {
        f();
    }
    catch (...)
    {
        error = std::current_exception();
    }
    int bad = bool(error), any = 0;
    Teuchos::reduceAll(*Tpetra::getDefaultComm(), Teuchos::REDUCE_MAX, 1, &bad, &any);
    if (any)
    {
        if (error)
            std::rethrow_exception(error);
        throw std::invalid_argument("Mixed AMR validation failed on another rank.");
    }
}
void same(std::span<const char> local)
{
    auto comm = Tpetra::getDefaultComm();
    collective(
        [&]
        {
            if (local.size() > std::size_t(std::numeric_limits<int>::max()))
                throw std::length_error("Mixed descriptor exceeds packet bound.");
        });
    int n = comm->getRank() == 0 ? int(local.size()) : 0;
    Teuchos::broadcast(*comm, 0, 1, &n);
    std::vector<char> root(n);
    if (comm->getRank() == 0)
        std::copy(local.begin(), local.end(), root.begin());
    Teuchos::broadcast(*comm, 0, n, root.data());
    collective(
        [&]
        {
            if (!std::equal(local.begin(), local.end(), root.begin(), root.end()))
                throw std::invalid_argument("Mixed AMR descriptions disagree.");
        });
}
size_t product_count(const MixedAMRGrid& g)
{
    return size_t(bool(g.product)) + g.local_products.size();
}
const std::shared_ptr<const LocalProductAMR>& product_at(const MixedAMRGrid& g, size_t p)
{
    if (g.product)
    {
        if (!p)
            return g.product;
        --p;
    }
    return g.local_products.at(p);
}
std::shared_ptr<const LocalProductAMR>& product_at(MixedAMRGrid& g, size_t p)
{
    if (g.product)
    {
        if (!p)
            return g.product;
        --p;
    }
    return g.local_products.at(p);
}
struct Patch
{
    size_t region;
    int boundary;
    std::vector<ID> faces;
    std::vector<PLIC::PlanarFacePtr<ArithmeticMode::Ordinary>> certificates;
};
std::vector<MultiRegionMesh::Interface> close_patches(std::vector<Patch> patches)
{
    // Large native patches are considered first. No synthetic intersection tile
    // is created: every retained tile is a whole native face of a declared patch.
    std::vector<size_t> order(patches.size());
    std::iota(order.begin(), order.end(), 0);
    auto area = [&](size_t i)
    {
        long double a = 0;
        for (const auto& f : patches[i].certificates)
            a = std::max(a, f->area().value);
        return a;
    };
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return area(a) > area(b); });
    std::vector<bool> used(patches.size());
    std::vector<MultiRegionMesh::Interface> result;
    for (auto i : order)
        if (!used[i])
        {
            const auto& coarse = patches[i];
            MultiProviderNonconformingInterface descriptor;
            descriptor.coarse_region = coarse.region;
            descriptor.coarse_boundary = coarse.boundary;
            descriptor.coarse_face_count = coarse.faces.size();
            std::vector<std::vector<RegionFace>> tiles(coarse.faces.size());
            std::vector<std::vector<PLIC::PlanarFacePtr<ArithmeticMode::Ordinary>>> certs(coarse.faces.size());
            std::vector<size_t> participants;
            for (auto j : order)
                if (j != i && !used[j] && patches[j].region != coarse.region)
                {
                    const auto& fine = patches[j];
                    std::vector<size_t> owner;
                    bool contained = true;
                    for (const auto& f : fine.certificates)
                    {
                        size_t k = 0;
                        for (; k < coarse.certificates.size(); ++k)
                            if (coarse.certificates[k]->oppositely_oriented(*f) && coarse.certificates[k]->contains(*f))
                                break;
                        if (k == coarse.certificates.size())
                        {
                            contained = false;
                            break;
                        }
                        owner.push_back(k);
                    }
                    if (!contained)
                        continue;
                    participants.push_back(j);
                    auto patch = std::find_if(descriptor.fine_patches.begin(), descriptor.fine_patches.end(),
                        [&](const auto& p) { return p.region == fine.region; });
                    if (patch == descriptor.fine_patches.end())
                        descriptor.fine_patches.push_back({fine.region, fine.boundary, fine.faces.size()});
                    else
                    {
                        patch->patches.push_back({fine.boundary, fine.faces.size(), {}});
                    }
                    for (size_t k = 0; k < owner.size(); ++k)
                    {
                        tiles[owner[k]].push_back({fine.region, fine.faces[k]});
                        certs[owner[k]].push_back(fine.certificates[k]);
                    }
                }
            if (participants.empty())
                continue;
            bool complete = true;
            for (size_t k = 0; k < tiles.size(); ++k)
            {
                if (tiles[k].empty())
                {
                    complete = false;
                    break;
                }
                try
                {
                    coarse.certificates[k]->certify_partition(certs[k]);
                }
                catch (const std::invalid_argument&)
                {
                    complete = false;
                    break;
                }
                descriptor.faces.push_back({coarse.faces[k], std::move(tiles[k])});
            }
            if (!complete)
                continue;
            used[i] = true;
            for (auto j : participants)
                used[j] = true;
            result.emplace_back(std::move(descriptor));
        }
    if (std::find(used.begin(), used.end(), false) != used.end())
        throw std::invalid_argument("Mixed crossed or unmatched seam needs compatible whole native tiles.");
    return result;
}
std::vector<MultiRegionMesh::Interface> seams(const LocalProductAMR& product, ID budget)
{
    std::vector<Patch> patches;
    ID count = 0;
    for (size_t r = 0; r < product.runs().size(); ++r)
    {
        const auto& run = product.runs()[r];
        const auto& top = run.region.topology();
        for (const auto& boundary : run.boundaries)
            if (!boundary.physical)
            {
                Patch p{r, boundary.boundary, {}, {}};
                for (ID f = 0; f < top.layout().faces; ++f)
                    if (top.boundary_id(f) == boundary.boundary)
                    {
                        if (count == budget)
                            throw std::length_error("Mixed interface tile budget exhausted.");
                        ++count;
                        auto certificate = run.region.geometry().certified_face_geometry(f);
                        if (!certificate)
                            throw std::invalid_argument("Mixed run lacks authoritative seam geometry.");
                        p.faces.push_back(f);
                        p.certificates.push_back(std::move(certificate));
                    }
                if (p.faces.empty())
                    throw std::invalid_argument("Mixed run seam is empty.");
                patches.push_back(std::move(p));
            }
    }
    return close_patches(std::move(patches));
}
MultiRegionMesh::Interface shifted(MultiRegionMesh::Interface interface, size_t offset)
{
    std::visit(
        [&](auto& n)
        {
            using T = std::remove_cvref_t<decltype(n)>;
            if constexpr (std::same_as<T, StructuredPatchInterface>)
            {
                n.first.region += offset;
                n.second.region += offset;
            }
            else if constexpr (std::same_as<T, ExplicitConformingInterface>)
            {
                n.first_region += offset;
                n.second_region += offset;
            }
            else if constexpr (std::same_as<T, NonconformingInterface>)
            {
                n.coarse_region += offset;
                n.fine_region += offset;
            }
            else
            {
                n.coarse_region += offset;
                for (auto& p : n.fine_patches)
                    p.region += offset;
                for (auto& f : n.faces)
                    for (auto& p : f.fine_faces)
                        p.region += offset;
            }
        },
        interface);
    return interface;
}
PLIC::PlanarFacePtr<ArithmeticMode::Ordinary> certificate(const MultiRegionMesh::Region& region, ID face)
{
    return std::visit(
        [&](const auto& r) -> PLIC::PlanarFacePtr<ArithmeticMode::Ordinary>
        {
            if constexpr (requires { r.geometry().certified_face_geometry(face); })
                if (auto c = r.geometry().certified_face_geometry(face))
                    return c;
            if (r.layout().family == RegionLayout::Family::Cylindrical)
                throw std::invalid_argument("Analytic curved participants cannot use an exact planar join.");
            std::vector<PLIC::GeometryPoint> points;
            for (auto n : r.topology().face_nodes(face))
            {
                auto p = r.geometry().node_coordinates(n);
                points.push_back({p.x, p.y, p.z});
            }
            auto area = r.geometry().face_area_vector(face);
            return PLIC::PlanarFace<ArithmeticMode::Ordinary>::make(points)->oriented_like({area.x, area.y, area.z});
        },
        region);
}
std::vector<Patch> named_patches(
    const std::vector<MultiRegionMesh::Region>& regions, size_t begin, size_t end, const std::string& name, ID& budget)
{
    std::vector<Patch> result;
    for (size_t r = begin; r < end; ++r)
        std::visit(
            [&](const auto& region)
            {
                const auto& top = region.topology();
                for (auto b : top.boundary_batch_ids())
                    if (top.boundary_batch_name(b) == name)
                    {
                        Patch patch{r, b, {}, {}};
                        for (ID f = 0; f < region.layout().faces; ++f)
                            if (top.boundary_id(f) == b)
                            {
                                if (!budget--)
                                    throw std::length_error("Mixed interface tile budget exhausted.");
                                patch.faces.push_back(f);
                                patch.certificates.push_back(certificate(regions[r], f));
                            }
                        if (!patch.faces.empty())
                            result.push_back(std::move(patch));
                    }
            },
            regions[r]);
    if (result.empty())
        throw std::invalid_argument("Mixed persistent join patch no longer exists.");
    return result;
}
template<class K> bool prefix(const K& a, const K& b)
{
    return a.root == b.root && a.path.size() <= b.path.size() &&
           std::equal(a.path.begin(), a.path.end(), b.path.begin());
}
} // namespace
struct MixedAMR::Data
{
    MixedAMRGrid grid;
    SP<const MultiRegionMesh> geometry;
    std::shared_ptr<const CompositePartition<>> partition;
    std::vector<ID> offsets;
    std::vector<size_t> participant_regions, participant_cells;
    struct SourceObservation
    {
        HierarchyObservation observation;
        SP<const MultiRegionMesh> canonical;
        ID canonical_epoch = 0;
    };
    std::vector<SourceObservation> adaptive_sources;
    ID generation = 0, partition_generation = 0;
    std::shared_ptr<const void> token = std::make_shared<const char>(0), topology = token, partition_token = token;
};
namespace
{
bool provider_unchanged(const std::shared_ptr<const AdaptiveHierarchy>& source) noexcept
{
    if (const auto h = std::dynamic_pointer_cast<const CartesianAMR>(source))
        return h->geometry_unchanged();
    if (const auto h = std::dynamic_pointer_cast<const ExtrudedAMR>(source))
        return h->geometry_unchanged();
    if (const auto h = std::dynamic_pointer_cast<const PolyhedralAMR>(source))
        return h->geometry_unchanged();
    return false;
}
bool adaptive_sources_unchanged(const MixedAMR::Data& data) noexcept
{
    try
    {
        if (data.adaptive_sources.size() != data.grid.adaptive.size())
            return false;
        for (size_t i = 0; i < data.grid.adaptive.size(); ++i)
        {
            const auto& source = data.grid.adaptive[i];
            const auto& retained = data.adaptive_sources[i];
            if (!provider_unchanged(source) || source->observation() != retained.observation ||
                source->native_geometry().get() != retained.canonical.get() ||
                retained.canonical->geometry_epoch() != retained.canonical_epoch)
                return false;
        }
        return true;
    }
    catch (...)
    {
        return false;
    }
}
AdaptiveCellKey data_key(const MixedAMR::Data& d, ID gid, size_t p)
{
    if (p < product_count(d.grid))
    {
        const auto r = size_t(std::upper_bound(d.offsets.begin(), d.offsets.end(), gid) - d.offsets.begin() - 1);
        auto key = product_at(d.grid, p)->cell_key(r - d.participant_regions[p], gid - d.offsets[r]);
        if (!p && d.grid.product)
            key.region = d.grid.physical_region;
        return key;
    }
    const auto a = p - product_count(d.grid);
    if (a < d.grid.adaptive.size())
        return d.grid.adaptive[a]->cell_key(gid - d.participant_cells[p]);
    const auto& stationary = d.grid.stationary.at(a - d.grid.adaptive.size());
    return {stationary.physical_region, StaticCellAncestry{stationary.provider_root, gid - d.participant_cells[p]}};
}
std::shared_ptr<MixedAMR::Data> build_once(MixedAMRGrid grid)
{
    auto d = std::make_shared<MixedAMR::Data>();
    d->grid = std::move(grid);
    collective(
        [&]
        {
            if ((!product_count(d->grid) && d->grid.adaptive.empty() && d->grid.stationary.empty()) ||
                !d->grid.maximum_cells || !d->grid.maximum_interface_tiles ||
                !std::isfinite(d->grid.maximum_span_ratio) || d->grid.maximum_span_ratio < 1 ||
                !d->grid.maximum_closure_rounds)
                throw std::invalid_argument("Mixed assembly requires product and positive budgets.");
        });
    std::vector<MultiRegionMesh::Region> regions;
    std::vector<MultiRegionMesh::Interface> interfaces;
    collective(
        [&]
        {
            d->offsets.push_back(0);
            auto append = [&](const auto& region)
            {
                const auto n = std::visit(
                    [](const auto& r) { return r.topology().layout().cells; }, MultiRegionMesh::Region(region));
                if (n > d->grid.maximum_cells - d->offsets.back())
                    throw std::length_error("Mixed cell budget exhausted.");
                regions.emplace_back(region);
                d->offsets.push_back(d->offsets.back() + n);
            };
            for (size_t p = 0; p < product_count(d->grid); ++p)
            {
                const auto& product = product_at(d->grid, p);
                if (!product)
                    throw std::invalid_argument("Null local product participant.");
                const auto violations = product->contact_violations(
                    d->grid.maximum_span_ratio, 1, static_cast<size_t>(d->grid.maximum_interface_tiles));
                if (!violations.empty())
                    throw std::invalid_argument(
                        "Mixed local product face/edge/vertex gradation exceeds supported bounds (dimension " +
                        std::to_string(violations.front().dimension) + ").");
                const auto offset = regions.size();
                d->participant_regions.push_back(offset);
                d->participant_cells.push_back(d->offsets.back());
                for (const auto& run : product->runs())
                    append(ExtrudedRegion("local_" + std::to_string(p) + "_" + run.region.name(),
                        std::shared_ptr<const ExtrudedTopology>(product, &run.region.topology()),
                        std::shared_ptr<const ExtrudedGeometry>(product, &run.region.geometry())));
                for (const auto& interface : seams(*product, d->grid.maximum_interface_tiles))
                    interfaces.push_back(shifted(interface, offset));
            }
            for (const auto& participant : d->grid.adaptive)
            {
                if (!participant)
                    throw std::invalid_argument("Null adaptive mixed participant.");
                if (!provider_unchanged(participant))
                    throw std::invalid_argument(
                        "Mixed adaptive source is unsupported or its canonical/resident geometry has moved.");
                const auto canonical = participant->native_geometry();
                if (!canonical)
                    throw std::invalid_argument("Mixed adaptive source has no canonical geometry.");
                d->adaptive_sources.push_back({participant->observation(), canonical, canonical->geometry_epoch()});
                if (const auto product = std::dynamic_pointer_cast<const ExtrudedAMR>(participant))
                    if (!product
                            ->contact_violations(
                                d->grid.maximum_span_ratio, 1, static_cast<size_t>(d->grid.maximum_interface_tiles))
                            .empty())
                        throw std::invalid_argument(
                            "Mixed whole product face/edge/vertex gradation exceeds supported bounds.");
                const auto offset = regions.size();
                d->participant_regions.push_back(offset);
                d->participant_cells.push_back(d->offsets.back());
                for (const auto& region : participant->native_geometry()->regions())
                    std::visit(
                        [&](const auto& native)
                        {
                            using R = std::remove_cvref_t<decltype(native)>;
                            using T = std::remove_cvref_t<decltype(native.topology())>;
                            using G = std::remove_cvref_t<decltype(native.geometry())>;
                            append(
                                R("adaptive_" + std::to_string(d->participant_regions.size() - 1) + "_" + native.name(),
                                    std::shared_ptr<const T>(participant, &native.topology()),
                                    std::shared_ptr<const G>(participant, &native.geometry())));
                        },
                        region);
                for (const auto& interface : participant->native_geometry()->interfaces())
                    interfaces.push_back(shifted(interface, offset));
            }
            for (const auto& participant : d->grid.stationary)
            {
                d->participant_regions.push_back(regions.size());
                d->participant_cells.push_back(d->offsets.back());
                append(participant.region);
            }
            d->participant_regions.push_back(regions.size());
            d->participant_cells.push_back(d->offsets.back());
            ID tile_budget = d->grid.maximum_interface_tiles;
            for (const auto& join : d->grid.joins)
            {
                if (join.first_participant + 1 >= d->participant_regions.size() ||
                    join.second_participant + 1 >= d->participant_regions.size() ||
                    join.first_participant == join.second_participant)
                    throw std::invalid_argument("Invalid persistent mixed join participants.");
                auto patches = named_patches(regions, d->participant_regions[join.first_participant],
                    d->participant_regions[join.first_participant + 1], join.first_patch, tile_budget);
                auto other = named_patches(regions, d->participant_regions[join.second_participant],
                    d->participant_regions[join.second_participant + 1], join.second_patch, tile_budget);
                patches.insert(
                    patches.end(), std::make_move_iterator(other.begin()), std::make_move_iterator(other.end()));
                auto joined = close_patches(std::move(patches));
                interfaces.insert(
                    interfaces.end(), std::make_move_iterator(joined.begin()), std::make_move_iterator(joined.end()));
            }
            interfaces.insert(interfaces.end(), d->grid.interfaces.begin(), d->grid.interfaces.end());
            d->geometry = std::make_shared<MultiRegionMesh>(
                std::move(regions), std::move(interfaces), Teuchos::rcp(new Teuchos::SerialComm<int>()));
        });
    std::vector<char> packet;
    collective(
        [&]
        {
            std::vector<ID> cells(d->offsets.back());
            std::iota(cells.begin(), cells.end(), ID(0));
            packet = d->geometry->serialize_partition(cells);
        });
    same(packet);
    std::ostringstream membership;
    membership << d->grid.physical_region << ' ' << product_count(d->grid) << ' ' << d->grid.maximum_cells << ' '
               << d->grid.maximum_interface_tiles << ' ' << std::hexfloat << d->grid.maximum_span_ratio << ' '
               << d->grid.maximum_closure_rounds;
    for (const auto& p : d->grid.stationary)
        membership << ' ' << p.physical_region << ' ' << p.material << ' ' << p.provider_root;
    for (size_t p = 0; p < product_count(d->grid); ++p)
    {
        const auto& product = product_at(d->grid, p);
        membership << ' ' << product->material_id();
        for (const auto& run : product->runs())
        {
            membership << " B" << run.base_keys.size() << ':';
            for (const auto& k : run.base_keys)
            {
                membership << k.root << ',' << k.root_template << '[';
                for (auto v : k.path)
                    membership << v << ',';
                membership << ']';
            }
            membership << " Z" << run.intervals.size() << ':';
            for (const auto& k : run.intervals)
            {
                membership << k.root << '[';
                for (auto v : k.path)
                    membership << v << ',';
                membership << ']';
            }
        }
        if (!product->runs().empty())
            membership << " R" << product->cell_key(0, 0).region;
    }
    for (const auto& participant : d->grid.adaptive)
        for (ID cell = 0; cell < participant->native_geometry()->num_cells(); ++cell)
        {
            const auto key = participant->cell_key(cell);
            membership << ' ' << key.region << ':' << participant->cell_material(cell) << ':' << key.ancestry.index()
                       << ':';
            std::visit(
                [&](const auto& k)
                {
                    using K = std::remove_cvref_t<decltype(k)>;
                    if constexpr (std::same_as<K, CartesianCellAncestry>)
                    {
                        membership << k.level;
                        for (auto i : k.index)
                            membership << ',' << i;
                    }
                    else if constexpr (std::same_as<K, ConvexCellAncestry>)
                    {
                        membership << k.root;
                        for (auto i : k.path)
                            membership << ',' << i;
                    }
                    else if constexpr (std::same_as<K, ProductCellAncestry>)
                    {
                        membership << k.base.root << ',' << k.base.root_template << '/';
                        for (auto i : k.base.path)
                            membership << i << ',';
                        membership << '/' << k.axial.root << '/';
                        for (auto i : k.axial.path)
                            membership << i << ',';
                    }
                    else
                        membership << k.provider_root << ',' << k.native_cell;
                },
                key.ancestry);
        }
    const auto text = membership.str();
    same(std::span<const char>(text.data(), text.size()));
    collective(
        [&]
        {
            std::set<AdaptiveCellKey> keys;
            for (size_t p = 0; p + 1 < d->participant_cells.size(); ++p)
                for (ID c = d->participant_cells[p]; c < d->participant_cells[p + 1]; ++c)
                    if (!keys.insert(data_key(*d, c, p)).second)
                        throw std::invalid_argument("Mixed participants repeat persistent cell provenance; provide "
                                                    "distinct physical/root identities.");
        });
    collective(
        [&]
        {
            if (!adaptive_sources_unchanged(*d))
                throw std::invalid_argument("Mixed adaptive source observation changed during assembly.");
        });
    return d;
}
void preflight_refinement(const MixedAMRGrid& grid)
{
    collective(
        [&]
        {
            ID cells = 0;
            for (size_t p = 0; p < product_count(grid); ++p)
            {
                const auto n = product_at(grid, p)->storage_counts().cells;
                if (n > grid.maximum_cells - cells)
                    throw std::length_error("Mixed source cell budget exhausted.");
                cells += n;
            }
            if (cells > grid.maximum_cells)
                throw std::length_error("Mixed source cell budget exhausted.");
            for (const auto& p : grid.adaptive)
            {
                const auto n = p->native_geometry()->num_cells();
                if (n > grid.maximum_cells - cells)
                    throw std::length_error("Mixed source cell budget exhausted.");
                cells += n;
            }
            for (const auto& p : grid.stationary)
            {
                const auto n = std::visit([](const auto& r) { return r.layout().cells; }, p.region);
                if (n > grid.maximum_cells - cells)
                    throw std::length_error("Mixed source cell budget exhausted.");
                cells += n;
            }
            // One supported closure round can split every current cell into at
            // most eight children. This intentionally conservative bootstrap bound
            // is checked before any provider allocates its candidate construction.
            if (cells > grid.maximum_cells / 8 || cells > grid.maximum_interface_tiles / 48)
                throw std::length_error("Mixed candidate expansion budget exhausted.");
        });
}
MixedAMRGrid refine_grid(MixedAMRGrid grid, const std::map<size_t, std::set<AdaptiveCellKey>>& requests)
{
    preflight_refinement(grid);
    for (const auto& [p, keys] : requests)
    {
        if (p < product_count(grid))
        {
            std::vector<ProductCellAncestry> marks;
            for (const auto& k : keys)
            {
                const auto* product = std::get_if<ProductCellAncestry>(&k.ancestry);
                if (!product)
                    throw std::invalid_argument("Mixed product mark has incompatible ancestry.");
                marks.push_back(*product);
            }
            product_at(grid, p) = product_at(grid, p)->adapt(marks, marks);
            continue;
        }
        collective(
            [&]
            {
                if (p >= product_count(grid) + grid.adaptive.size())
                    throw std::invalid_argument("Mixed physical closure requests unsupported static refinement.");
            });
        const auto& source = grid.adaptive[p - product_count(grid)];
        std::vector<size_t> cells;
        collective(
            [&]
            {
                for (const auto& key : keys)
                {
                    bool found = false;
                    for (ID gid = 0; gid < source->native_geometry()->num_cells(); ++gid)
                        if (source->cell_key(gid) == key)
                        {
                            cells.push_back(gid);
                            found = true;
                            break;
                        }
                    if (!found)
                        throw std::invalid_argument("Mixed mark does not identify a source leaf.");
                }
            });
        if (const auto h = std::dynamic_pointer_cast<const PolyhedralAMR>(source))
            grid.adaptive[p - product_count(grid)] = h->adapt(cells);
        else if (const auto h = std::dynamic_pointer_cast<const CartesianAMR>(source))
        {
            std::set<size_t> blocks;
            for (auto gid : cells)
                blocks.insert(h->cell_fragment(gid));
            grid.adaptive[p - product_count(grid)] = h->adapt(std::vector<size_t>(blocks.begin(), blocks.end()));
        }
        else if (const auto h = std::dynamic_pointer_cast<const ExtrudedAMR>(source))
        {
            std::set<size_t> base, axial;
            for (auto gid : cells)
            {
                const auto k = std::get<ProductCellAncestry>(h->cell_key(gid).ancestry);
                base.insert(std::find(h->base_keys().begin(), h->base_keys().end(), k.base) - h->base_keys().begin());
                axial.insert(std::find(h->interval_keys().begin(), h->interval_keys().end(), k.axial) -
                             h->interval_keys().begin());
            }
            grid.adaptive[p - product_count(grid)] = h->adapt(
                std::vector<size_t>(base.begin(), base.end()), std::vector<size_t>(axial.begin(), axial.end()));
        }
        else
            collective([&] { throw std::invalid_argument("Mixed participant has no qualified refinement adapter."); });
    }
    return grid;
}
void validate_native_contact_envelopes(
    const MixedAMR::Data& data, const std::vector<std::tuple<ID, size_t, size_t>>& joined)
{
    const auto adaptive = product_count(data.grid) + data.grid.adaptive.size();
    ID work = 0;
    auto spend = [&]
    {
        if (work == data.grid.maximum_interface_tiles)
            throw std::length_error("Mixed contact proof work budget exhausted.");
        ++work;
    };
    auto participant = [&](ID c)
    {
        return size_t(std::upper_bound(data.participant_cells.begin(), data.participant_cells.end(), c) -
                      data.participant_cells.begin() - 1);
    };
    auto geometry = [&](ID c)
    {
        const auto p = participant(c);
        if (p < adaptive)
            return PLIC::ConvexCellGeometry<ArithmeticMode::Ordinary>::from_native_cell(*data.geometry, c);
        const auto [region, local] = data.geometry->native_cell(c);
        const auto& native = data.geometry->regions()[region];
        const bool exact = std::visit(
            [](const auto& r)
            {
                if constexpr (requires { r.geometry().has_construction(); })
                    return r.geometry().has_construction();
                else
                    return false;
            },
            native);
        if (exact)
            return PLIC::ConvexCellGeometry<ArithmeticMode::Ordinary>::from_native_cell(*data.geometry, c);
        if (data.geometry->region_layout(region).family == RegionLayout::Family::Cylindrical)
            throw std::invalid_argument("Mixed curved static contact has no qualified exact span proof.");
        PLIC::GeometryPoint lo{INFINITY, INFINITY, INFINITY}, hi{-INFINITY, -INFINITY, -INFINITY};
        for (auto node : data.geometry->cell_nodes(c))
        {
            const auto v = data.geometry->node_coordinates(node);
            const PLIC::GeometryPoint point{v.x, v.y, v.z};
            for (size_t a = 0; a < 3; ++a)
            {
                lo[a] = std::min(lo[a], point[a]);
                hi[a] = std::max(hi[a], point[a]);
            }
        }
        // A represented static cell's AABB is only a physical-span proof. It is
        // never exported as an actual-cell/PLIC/remap certificate.
        return PLIC::ConvexCellGeometry<ArithmeticMode::Ordinary>::box(lo, hi);
    };
    std::map<ID, std::vector<ID>> nodes;
    for (size_t p = product_count(data.grid); p < adaptive; ++p)
        for (ID c = data.participant_cells[p]; c < data.participant_cells[p + 1]; ++c)
            for (auto node : data.geometry->cell_nodes(c))
            {
                spend();
                nodes[node].push_back(c);
            }
    std::set<std::pair<ID, ID>> checked;
    for (const auto& [node, cells] : nodes)
        for (size_t i = 0; i < cells.size(); ++i)
            for (size_t j = i + 1; j < cells.size(); ++j)
            {
                const auto a = std::min(cells[i], cells[j]), b = std::max(cells[i], cells[j]);
                if (participant(a) != participant(b) || checked.contains({a, b}))
                    continue;
                spend();
                checked.emplace(a, b);
                const auto ga = geometry(a), gb = geometry(b);
                for (size_t axis = 0; axis < 3; ++axis)
                    if (ga->axis_span_exceeds(*gb, axis, data.grid.maximum_span_ratio) ||
                        gb->axis_span_exceeds(*ga, axis, data.grid.maximum_span_ratio))
                        throw std::invalid_argument(
                            "Mixed internal adaptive edge/vertex gradation needs unsupported local closure.");
            }
    struct Envelope
    {
        std::array<PLIC::ConvexGeometryPtr<ArithmeticMode::Ordinary>, 3> minimum, maximum;
    };
    // Closed AABB overlap with each real canonical join face is a conservative
    // seam-local contact proof. Unrelated interior cells are not enrolled, and
    // no exact donor intersection search or expanded certificate cache is built.
    for (const auto& [face, a, b] : joined)
    {
        spend();
        const auto surface = PLIC::PlanarFace<ArithmeticMode::Ordinary>::from_native_face(*data.geometry, face);
        std::array<Envelope, 2> envelopes;
        size_t side = 0;
        for (auto p : {a, b})
        {
            auto& envelope = envelopes[side++];
            for (ID c = data.participant_cells[p]; c < data.participant_cells[p + 1]; ++c)
            {
                spend();
                const auto cell = geometry(c);
                if (!surface->cell_bounds_overlap(*cell))
                    continue;
                for (size_t axis = 0; axis < 3; ++axis)
                {
                    if (!envelope.minimum[axis] || envelope.minimum[axis]->axis_span_exceeds(*cell, axis, 1))
                        envelope.minimum[axis] = cell;
                    if (!envelope.maximum[axis] || cell->axis_span_exceeds(*envelope.maximum[axis], axis, 1))
                        envelope.maximum[axis] = cell;
                }
            }
        }
        if (!envelopes[0].minimum[0] || !envelopes[1].minimum[0])
            throw std::invalid_argument("Mixed join lacks a qualified closed seam neighborhood proof.");
        for (size_t axis = 0; axis < 3; ++axis)
        {
            spend();
            if (envelopes[0].maximum[axis]->axis_span_exceeds(
                    *envelopes[1].minimum[axis], axis, data.grid.maximum_span_ratio) ||
                envelopes[1].maximum[axis]->axis_span_exceeds(
                    *envelopes[0].minimum[axis], axis, data.grid.maximum_span_ratio))
                throw std::invalid_argument(
                    "Mixed joined-face edge/vertex envelope cannot certify configured gradation.");
        }
    }
}
std::shared_ptr<MixedAMR::Data> build(MixedAMRGrid grid)
{
    for (unsigned round = 0;; ++round)
    {
        auto d = build_once(std::move(grid));
        std::map<size_t, std::set<AdaptiveCellKey>> requests;
        collective(
            [&]
            {
                auto participant = [&](ID c)
                {
                    return size_t(std::upper_bound(d->participant_cells.begin(), d->participant_cells.end(), c) -
                                  d->participant_cells.begin() - 1);
                };
                auto key = [&](ID c, size_t p) { return data_key(*d, c, p); };
                std::vector<std::tuple<ID, size_t, size_t>> joined;
                for (ID f = 0; f < d->geometry->num_faces(); ++f)
                {
                    if (d->geometry->is_exterior_face(f))
                        continue;
                    const auto a = d->geometry->owner_cell(f), b = d->geometry->neighbor_cell(f);
                    const auto pa = participant(a), pb = participant(b);
                    const bool internal_adaptive = pa == pb && pa < product_count(d->grid) + d->grid.adaptive.size();
                    const auto adaptive_count = product_count(d->grid) + d->grid.adaptive.size();
                    if (pa >= adaptive_count && pb >= adaptive_count)
                        continue;
                    if (pa != pb)
                    {
                        if (joined.size() == d->grid.maximum_interface_tiles)
                            throw std::length_error("Mixed contact face budget exhausted.");
                        joined.emplace_back(f, pa, pb);
                    }
                    const auto ca =
                        pa < product_count(d->grid) + d->grid.adaptive.size()
                            ? PLIC::ConvexCellGeometry<ArithmeticMode::Ordinary>::from_native_cell(*d->geometry, a)
                            : PLIC::ConvexGeometryPtr<ArithmeticMode::Ordinary>{};
                    const auto cb =
                        pb < product_count(d->grid) + d->grid.adaptive.size()
                            ? PLIC::ConvexCellGeometry<ArithmeticMode::Ordinary>::from_native_cell(*d->geometry, b)
                            : PLIC::ConvexGeometryPtr<ArithmeticMode::Ordinary>{};
                    auto static_span = [&](ID c, size_t axis) -> long double
                    {
                        double lower = std::numeric_limits<double>::infinity(), upper = -lower;
                        for (auto node : d->geometry->cell_nodes(c))
                        {
                            const auto v = d->geometry->node_coordinates(node);
                            const double value = axis == 0 ? v.x : axis == 1 ? v.y : v.z;
                            lower = std::min(lower, value);
                            upper = std::max(upper, value);
                        }
                        const auto span = static_cast<long double>(upper) - lower;
                        if (!std::isfinite(span) || !(span > 0))
                            throw std::invalid_argument("Mixed static provider has an invalid physical span.");
                        return span;
                    };
                    for (size_t axis = 0; axis < 3; ++axis)
                    {
                        if (ca && cb)
                        {
                            const bool coarse_a = ca->axis_span_exceeds(*cb, axis, d->grid.maximum_span_ratio),
                                       coarse_b = cb->axis_span_exceeds(*ca, axis, d->grid.maximum_span_ratio);
                            if (internal_adaptive && (coarse_a || coarse_b))
                                throw std::invalid_argument("Mixed internal adaptive face gradation requires "
                                                            "unsupported provider-local closure.");
                            if (coarse_a)
                                requests[pa].insert(key(a, pa));
                            if (coarse_b)
                                requests[pb].insert(key(b, pb));
                        }
                        else
                        {
                            // Static represented planar providers retain their native shape;
                            // bounds are a closure metric, never a convex remap certificate.
                            const auto aw = static_span(a, axis), bw = static_span(b, axis);
                            const long double ratio = d->grid.maximum_span_ratio;
                            if (aw > ratio * bw)
                                requests[pa].insert(key(a, pa));
                            if (bw > ratio * aw)
                                requests[pb].insert(key(b, pb));
                        }
                    }
                }
                if (requests.empty())
                    validate_native_contact_envelopes(*d, joined);
                if (!requests.empty() && round >= d->grid.maximum_closure_rounds)
                    throw std::length_error("Mixed physical-size closure budget exhausted.");
            });
        if (requests.empty())
            return d;
        grid = refine_grid(d->grid, requests);
    }
}

} // namespace
MixedAMR::MixedAMR(MixedAMRGrid grid) : data_(build(std::move(grid))) {}
const SP<const MultiRegionMesh>& MixedAMR::native_geometry() const noexcept
{
    return data_->geometry;
}
const std::shared_ptr<const CompositePartition<>>& MixedAMR::native_partition() const noexcept
{
    return data_->partition;
}
HierarchyObservation MixedAMR::observation() const noexcept
{
    const auto* g = data_->partition ? data_->partition->geometry().get() : data_->geometry.get();
    ID epoch = std::numeric_limits<ID>::max();
    try
    {
        if (adaptive_sources_unchanged(*data_))
            epoch = g->geometry_epoch();
    }
    catch (...)
    {
    }
    return {data_->token, data_->topology, data_->partition_token, data_->generation, data_->partition_generation,
        epoch, g, Tpetra::getDefaultComm().getRawPtr()};
}
bool MixedAMR::geometry_unchanged() const noexcept
{
    try
    {
        return adaptive_sources_unchanged(*data_) && data_->geometry->geometry_epoch() == 0 &&
               (!data_->partition || data_->partition->geometry_unchanged());
    }
    catch (...)
    {
        return false;
    }
}
size_t MixedAMR::cell_fragment(ID gid) const
{
    if (gid >= data_->offsets.back())
        throw std::out_of_range("Mixed cell ordinal.");
    return std::upper_bound(data_->offsets.begin(), data_->offsets.end(), gid) - data_->offsets.begin() - 1;
}
AdaptiveCellKey MixedAMR::cell_key(ID gid) const
{
    cell_fragment(gid);
    const size_t p = std::upper_bound(data_->participant_cells.begin(), data_->participant_cells.end(), gid) -
                     data_->participant_cells.begin() - 1;
    return data_key(*data_, gid, p);
}
ID MixedAMR::cell_material(ID gid) const
{
    cell_fragment(gid);
    const size_t p = std::upper_bound(data_->participant_cells.begin(), data_->participant_cells.end(), gid) -
                     data_->participant_cells.begin() - 1;
    if (p < product_count(data_->grid))
        return product_at(data_->grid, p)->material_id();
    const auto a = p - product_count(data_->grid);
    if (a < data_->grid.adaptive.size())
        return data_->grid.adaptive[a]->cell_material(gid - data_->participant_cells[p]);
    return data_->grid.stationary.at(a - data_->grid.adaptive.size()).material;
}
PLIC::ConvexGeometryPtr<ArithmeticMode::Ordinary> MixedAMR::cell_geometry(ID gid) const
{
    if (!geometry_unchanged())
        throw std::invalid_argument("Mixed source observations are stale.");
    return PLIC::ConvexCellGeometry<ArithmeticMode::Ordinary>::from_native_cell(*data_->geometry, gid);
}
std::shared_ptr<const MixedAMR> MixedAMR::with_partition(std::shared_ptr<const CompositePartition<>> p) const
{
    collective(
        [&]
        {
            const auto comm = Tpetra::getDefaultComm();
            if (!geometry_unchanged() || !p || !p->geometry_unchanged() ||
                p->plan().communicator().getRawPtr() != comm.getRawPtr() ||
                (comm->getRank() == p->plan().source_rank() && !p->plan().matches_source(data_->geometry)))
                throw std::invalid_argument("Mixed partition source/revision/communicator mismatch.");
        });
    const auto id = std::to_string(p->plan().collective_identity());
    same(std::span<const char>(id.data(), id.size()));
    const std::string replacement = data_->partition == p ? "unchanged" : "replace";
    same(std::span<const char>(replacement.data(), replacement.size()));
    if (data_->partition == p)
        return std::shared_ptr<const MixedAMR>(new MixedAMR(data_));
    auto d = std::make_shared<Data>(*data_);
    collective(
        [&]
        {
            if (d->partition_generation == std::numeric_limits<ID>::max())
                throw std::overflow_error("Mixed partition generation exhausted.");
        });
    d->partition = std::move(p);
    ++d->partition_generation;
    d->token = std::make_shared<const char>(0);
    d->partition_token = d->token;
    return std::shared_ptr<const MixedAMR>(new MixedAMR(d));
}
std::shared_ptr<const MixedAMR> MixedAMR::adapt(const std::vector<ProductCellAncestry>& xy,
    const std::vector<ProductCellAncestry>& z, const std::vector<ProductCellAncestry>& cxy,
    const std::vector<ProductCellAncestry>& cz, const std::vector<AdaptiveCellKey>& static_marks) const
{
    collective(
        [&]
        {
            if (!geometry_unchanged() || !static_marks.empty() ||
                ((!xy.empty() || !z.empty() || !cxy.empty() || !cz.empty()) && !data_->grid.product))
                throw std::invalid_argument("Mixed stale source or unsupported static split.");
            if (data_->generation == std::numeric_limits<ID>::max() ||
                data_->partition_generation == std::numeric_limits<ID>::max())
                throw std::overflow_error("Mixed generation exhausted.");
        });
    std::ostringstream marks;
    for (const auto* group : {&xy, &z, &cxy, &cz})
    {
        marks << '|';
        for (const auto& k : *group)
        {
            marks << k.base.root << ',' << k.base.root_template << ':';
            for (auto p : k.base.path)
                marks << p << ',';
            marks << '/' << k.axial.root << ':';
            for (auto p : k.axial.path)
                marks << p << ',';
            marks << ';';
        }
    }
    const auto text = marks.str();
    same(std::span<const char>(text.data(), text.size()));
    if (xy.empty() && z.empty() && cxy.empty() && cz.empty())
        return std::shared_ptr<const MixedAMR>(new MixedAMR(data_));
    auto grid = data_->grid;
    if (!xy.empty() || !z.empty())
        preflight_refinement(grid);
    grid.product = grid.product->adapt(xy, z, cxy, cz);
    auto d = build(std::move(grid));
    collective(
        [&]
        {
            if (data_->partition_generation == std::numeric_limits<ID>::max())
                throw std::overflow_error("Mixed partition generation exhausted.");
        });
    d->generation = data_->generation + 1;
    d->partition_generation = data_->partition_generation + 1;
    return std::shared_ptr<const MixedAMR>(new MixedAMR(d));
}
std::shared_ptr<const MixedAMR> MixedAMR::adapt_participants(const std::vector<MixedParticipantMarks>& marks) const
{
    std::ostringstream out;
    bool changed = false, refining = false;
    collective(
        [&]
        {
            if (!geometry_unchanged())
                throw std::invalid_argument("Mixed adaptation source is stale.");
            if (data_->generation == std::numeric_limits<ID>::max() ||
                data_->partition_generation == std::numeric_limits<ID>::max())
                throw std::overflow_error("Mixed generation exhausted.");
            std::set<size_t> seen;
            auto product_text = [&](const auto& k)
            {
                out << k.base.root << ',' << k.base.root_template << '[';
                for (auto v : k.base.path)
                    out << v << ',';
                out << "]" << k.axial.root << '[';
                for (auto v : k.axial.path)
                    out << v << ',';
                out << ']';
            };
            for (const auto& m : marks)
            {
                if (m.participant + 1 >= data_->participant_cells.size() || !seen.insert(m.participant).second)
                    throw std::invalid_argument("Invalid or repeated mixed participant marks.");
                out << m.participant << ':';
                for (const auto* group : {&m.refine_xy, &m.refine_z, &m.coarsen_xy, &m.coarsen_z})
                {
                    out << group->size() << '{';
                    for (const auto& k : *group)
                        product_text(k);
                    out << '}';
                    changed = changed || !group->empty();
                }
                for (const auto* group : {&m.refine, &m.coarsen_polyhedral})
                {
                    out << group->size() << '{';
                    for (const auto& key : *group)
                    {
                        out << key.region << ':' << key.ancestry.index() << ':';
                        std::visit(
                            [&](const auto& k)
                            {
                                using K = std::remove_cvref_t<decltype(k)>;
                                if constexpr (std::same_as<K, ProductCellAncestry>)
                                    product_text(k);
                                else if constexpr (std::same_as<K, ConvexCellAncestry>)
                                {
                                    out << k.root << '[';
                                    for (auto v : k.path)
                                        out << v << ',';
                                    out << ']';
                                }
                                else if constexpr (std::same_as<K, CartesianCellAncestry>)
                                {
                                    out << k.level;
                                    for (auto v : k.index)
                                        out << ',' << v;
                                }
                                else
                                    out << k.provider_root << ',' << k.native_cell;
                            },
                            key.ancestry);
                    }
                    out << '}';
                    changed = changed || !group->empty();
                }
                out << m.coarsen_cartesian.size() << '{';
                for (const auto& b : m.coarsen_cartesian)
                {
                    out << b.level;
                    for (auto v : b.index)
                        out << ',' << v;
                    out << ';';
                }
                out << '}';
                changed = changed || !m.coarsen_cartesian.empty();
                refining = refining || !m.refine.empty() || !m.refine_xy.empty() || !m.refine_z.empty();
                const bool factor =
                    !m.refine_xy.empty() || !m.refine_z.empty() || !m.coarsen_xy.empty() || !m.coarsen_z.empty();
                if (factor && (!m.refine.empty() || !m.coarsen_polyhedral.empty() || !m.coarsen_cartesian.empty()))
                    throw std::invalid_argument("Mixed mark forms cannot be combined for one participant.");
                if (!m.refine.empty() && (!m.coarsen_polyhedral.empty() || !m.coarsen_cartesian.empty()))
                    throw std::invalid_argument(
                        "Mixed leaf refinement and family coarsening require separate transactions.");
            }
        });
    const auto text = out.str();
    same(std::span<const char>(text.data(), text.size()));
    if (!changed)
        return std::shared_ptr<const MixedAMR>(new MixedAMR(data_));
    auto grid = data_->grid;
    if (refining)
        preflight_refinement(grid);
    std::map<size_t, std::set<AdaptiveCellKey>> generic;
    for (const auto& m : marks)
    {
        const bool factor =
            !m.refine_xy.empty() || !m.refine_z.empty() || !m.coarsen_xy.empty() || !m.coarsen_z.empty();
        if (m.participant < product_count(grid))
        {
            collective(
                [&]
                {
                    if (!m.refine.empty() || !m.coarsen_polyhedral.empty() || !m.coarsen_cartesian.empty())
                        throw std::invalid_argument("Local products require explicit factor marks.");
                });
            if (factor)
                product_at(grid, m.participant) =
                    product_at(grid, m.participant)->adapt(m.refine_xy, m.refine_z, m.coarsen_xy, m.coarsen_z);
            continue;
        }
        const auto p = m.participant - product_count(grid);
        collective(
            [&]
            {
                if (p >= grid.adaptive.size())
                    throw std::invalid_argument("Static participant adaptation is unsupported.");
            });
        const auto source = grid.adaptive[p];
        if (factor)
        {
            const auto h = std::dynamic_pointer_cast<const ExtrudedAMR>(source);
            collective(
                [&]
                {
                    if (!h)
                        throw std::invalid_argument("Factor marks require an extruded participant.");
                });
            std::set<size_t> xy, z;
            std::set<BaseCellKey> cxy;
            std::set<AxialIntervalKey> cz;
            collective(
                [&]
                {
                    for (const auto& key : m.refine_xy)
                    {
                        auto i = std::find(h->base_keys().begin(), h->base_keys().end(), key.base);
                        if (i == h->base_keys().end())
                            throw std::invalid_argument("Missing mixed base source leaf.");
                        xy.insert(i - h->base_keys().begin());
                    }
                    for (const auto& key : m.refine_z)
                    {
                        auto i = std::find(h->interval_keys().begin(), h->interval_keys().end(), key.axial);
                        if (i == h->interval_keys().end())
                            throw std::invalid_argument("Missing mixed axial source leaf.");
                        z.insert(i - h->interval_keys().begin());
                    }
                    for (const auto& key : m.coarsen_xy)
                        cxy.insert(key.base);
                    for (const auto& key : m.coarsen_z)
                        cz.insert(key.axial);
                });
            grid.adaptive[p] = h->adapt(std::vector<size_t>(xy.begin(), xy.end()),
                std::vector<size_t>(z.begin(), z.end()), std::vector<BaseCellKey>(cxy.begin(), cxy.end()),
                std::vector<AxialIntervalKey>(cz.begin(), cz.end()));
        }
        else if (!m.coarsen_polyhedral.empty())
        {
            const auto h = std::dynamic_pointer_cast<const PolyhedralAMR>(source);
            collective(
                [&]
                {
                    if (!h || !m.coarsen_cartesian.empty())
                        throw std::invalid_argument("Convex parent marks require a polyhedral participant.");
                });
            grid.adaptive[p] = h->adapt({}, m.coarsen_polyhedral);
        }
        else if (!m.coarsen_cartesian.empty())
        {
            const auto h = std::dynamic_pointer_cast<const CartesianAMR>(source);
            collective(
                [&]
                {
                    if (!h)
                        throw std::invalid_argument("Block parent marks require a Cartesian participant.");
                });
            grid.adaptive[p] = h->adapt({}, m.coarsen_cartesian);
        }
        else if (!m.refine.empty())
            generic[m.participant].insert(m.refine.begin(), m.refine.end());
    }
    auto d = build(generic.empty() ? std::move(grid) : refine_grid(std::move(grid), generic));
    collective(
        [&]
        {
            if (data_->generation == std::numeric_limits<ID>::max())
                throw std::overflow_error("Mixed generation exhausted.");
        });
    collective(
        [&]
        {
            if (data_->partition_generation == std::numeric_limits<ID>::max())
                throw std::overflow_error("Mixed partition generation exhausted.");
        });
    d->generation = data_->generation + 1;
    d->partition_generation = data_->partition_generation + 1;
    return std::shared_ptr<const MixedAMR>(new MixedAMR(d));
}
MixedAMR::AdaptationPlan MixedAMR::plan_adaptation(const std::vector<ProductCellAncestry>& xy,
    const std::vector<ProductCellAncestry>& z, const std::vector<ProductCellAncestry>& cxy,
    const std::vector<ProductCellAncestry>& cz) const
{
    AdaptationPlan p;
    p.source_ = observation();
    p.candidate_ = adapt(xy, z, cxy, cz);
    return p;
}
MixedAMR::AdaptationPlan MixedAMR::plan_participants(const std::vector<MixedParticipantMarks>& marks) const
{
    AdaptationPlan p;
    p.source_ = observation();
    p.candidate_ = adapt_participants(marks);
    return p;
}
std::shared_ptr<const MixedAMR> MixedAMR::apply_adaptation(const AdaptationPlan& p) const
{
    collective(
        [&]
        {
            if (!geometry_unchanged() || p.source_ != observation() || !p.candidate_ ||
                !p.candidate_->geometry_unchanged())
                throw std::invalid_argument("Mixed adaptation plan is foreign or stale.");
        });
    std::vector<char> packet;
    collective(
        [&]
        {
            std::vector<ID> ids(p.candidate_->geometry()->num_cells());
            std::iota(ids.begin(), ids.end(), ID(0));
            packet = p.candidate_->geometry()->serialize_partition(ids);
        });
    same(packet);
    return p.candidate_;
}
std::vector<AdaptiveAncestryRelation> MixedAMR::overlap_relations(const MixedAMR& target) const
{
    std::vector<ID> ids(target.geometry()->num_cells());
    std::iota(ids.begin(), ids.end(), ID(0));
    return overlap_relations(target, ids);
}
std::vector<AdaptiveAncestryRelation> MixedAMR::overlap_relations(
    const MixedAMR& target, const std::vector<ID>& targets) const
{
    if (!geometry_unchanged() || !target.geometry_unchanged())
        throw std::invalid_argument("Mixed overlap source or target is stale.");
    using Kind = AdaptiveAncestryRelation::Kind;
    using ProductIndex = std::tuple<ID, BaseCellKey, AxialIntervalKey>;
    std::map<AdaptiveCellKey, ID> exact;
    std::map<AdaptiveCellKey, std::vector<ID>> descendants;
    std::map<ProductIndex, std::vector<ID>> both_desc, base_desc, z_desc;
    const auto base_parents = [](BaseCellKey key)
    {
        std::vector<BaseCellKey> result;
        for (;;)
        {
            result.push_back(key);
            if (key.path.empty())
                break;
            key.path.pop_back();
        }
        return result;
    };
    const auto z_parents = [](AxialIntervalKey key)
    {
        std::vector<AxialIntervalKey> result;
        for (;;)
        {
            result.push_back(key);
            if (key.path.empty())
                break;
            key.path.pop_back();
        }
        return result;
    };
    const auto parents = [](AdaptiveCellKey key)
    {
        std::vector<AdaptiveCellKey> result;
        for (;;)
        {
            result.push_back(key);
            if (auto* c = std::get_if<ConvexCellAncestry>(&key.ancestry))
            {
                if (c->path.empty())
                    break;
                c->path.pop_back();
            }
            else if (auto* c = std::get_if<CartesianCellAncestry>(&key.ancestry))
            {
                if (!c->level)
                    break;
                *c = c->parent();
            }
            else
                break;
        }
        return result;
    };
    for (ID s = 0; s < geometry()->num_cells(); ++s)
    {
        const auto key = cell_key(s);
        if (!exact.emplace(key, s).second)
            throw std::invalid_argument("Mixed source persistent cell key is repeated.");
        if (const auto* p = std::get_if<ProductCellAncestry>(&key.ancestry))
        {
            const auto bases = base_parents(p->base);
            const auto axes = z_parents(p->axial);
            for (const auto& b : bases)
            {
                base_desc[{key.region, b, p->axial}].push_back(s);
                for (const auto& z : axes)
                    both_desc[{key.region, b, z}].push_back(s);
            }
            for (const auto& z : axes)
                z_desc[{key.region, p->base, z}].push_back(s);
        }
        else
            for (const auto& ancestor : parents(key))
                descendants[ancestor].push_back(s);
    }
    std::vector<AdaptiveAncestryRelation> result;
    std::set<ID> seen_targets;
    for (const auto t : targets)
    {
        if (t >= target.geometry()->num_cells() || !seen_targets.insert(t).second)
            throw std::invalid_argument("Mixed overlap target selection is invalid.");
        const auto b = target.cell_key(t);
        std::set<ID> candidates;
        auto append = [&](const auto& index, const auto& key)
        {
            auto it = index.find(key);
            if (it != index.end())
                candidates.insert(it->second.begin(), it->second.end());
        };
        if (const auto* p = std::get_if<ProductCellAncestry>(&b.ancestry))
        {
            const auto bases = base_parents(p->base);
            const auto axes = z_parents(p->axial);
            append(both_desc, ProductIndex{b.region, p->base, p->axial});
            for (const auto& base : bases)
            {
                append(z_desc, ProductIndex{b.region, base, p->axial});
                for (const auto& z : axes)
                {
                    auto it = exact.find(AdaptiveCellKey{b.region, ProductCellAncestry{base, z}});
                    if (it != exact.end())
                        candidates.insert(it->second);
                }
            }
            for (const auto& z : axes)
                append(base_desc, ProductIndex{b.region, p->base, z});
        }
        else
        {
            append(descendants, b);
            for (const auto& ancestor : parents(b))
            {
                auto it = exact.find(ancestor);
                if (it != exact.end())
                    candidates.insert(it->second);
            }
        }
        for (const auto s : candidates)
        {
            const auto a = cell_key(s);
            if (cell_material(s) != target.cell_material(t))
                continue;
            if (a == b)
            {
                result.push_back({Kind::Unchanged, s, t});
                continue;
            }
            if (const auto* x = std::get_if<ProductCellAncestry>(&a.ancestry))
            {
                const auto& y = std::get<ProductCellAncestry>(b.ancestry);
                const bool bf = prefix(x->base, y.base), bc = prefix(y.base, x->base), zf = prefix(x->axial, y.axial),
                           zc = prefix(y.axial, x->axial);
                if ((bf || bc) && (zf || zc))
                    result.push_back({bf && zf   ? Kind::Refinement
                                      : bc && zc ? Kind::Coarsening
                                                 : Kind::ProductIntersection,
                        s, t});
            }
            else if (const auto* x = std::get_if<ConvexCellAncestry>(&a.ancestry))
            {
                const auto& y = std::get<ConvexCellAncestry>(b.ancestry);
                result.push_back({prefix(*x, y) ? Kind::Refinement : Kind::Coarsening, s, t});
            }
            else if (const auto* x = std::get_if<CartesianCellAncestry>(&a.ancestry))
            {
                const auto& y = std::get<CartesianCellAncestry>(b.ancestry);
                result.push_back({x->level < y.level ? Kind::Refinement : Kind::Coarsening, s, t});
            }
        }
    }
    return result;
}
MixedAMR::StorageBreakdown MixedAMR::storage_breakdown() const
{
    StorageBreakdown result;
    result.native_assembly = data_->geometry->storage_report();
    if (data_->partition)
        result.resident_partition_geometry = data_->partition->geometry()->storage_report();
    auto& report = result.assembly_metadata;
    report.objects = sizeof(*this) + sizeof(Data) +
                     data_->adaptive_sources.capacity() * sizeof(Data::SourceObservation) +
                     data_->grid.local_products.capacity() * sizeof(std::shared_ptr<const LocalProductAMR>) +
                     data_->grid.adaptive.capacity() * sizeof(std::shared_ptr<const AdaptiveHierarchy>) +
                     data_->grid.stationary.capacity() * sizeof(MixedStaticParticipant);
    report.indexing = data_->offsets.capacity() * sizeof(ID) +
                      (data_->participant_regions.capacity() + data_->participant_cells.capacity()) * sizeof(size_t);
    report.interfaces = data_->grid.joins.capacity() * sizeof(MixedPatchJoin) +
                        data_->grid.interfaces.capacity() * sizeof(MultiRegionMesh::Interface);
    for (const auto& join : data_->grid.joins)
        report.interfaces += join.first_patch.capacity() + join.second_patch.capacity();
    for (const auto& participant : data_->grid.stationary)
        std::visit([&](const auto& region) { report.objects += region.name().capacity(); }, participant.region);
    for (const auto& interface : data_->grid.interfaces)
        std::visit(
            [&](const auto& n)
            {
                using T = std::remove_cvref_t<decltype(n)>;
                if constexpr (std::same_as<T, ExplicitConformingInterface>)
                    report.interfaces += n.faces.capacity() * sizeof(n.faces[0]);
                else if constexpr (std::same_as<T, NonconformingInterface> ||
                                   std::same_as<T, MultiProviderNonconformingInterface>)
                {
                    report.interfaces += n.faces.capacity() * sizeof(n.faces[0]);
                    for (const auto& group : n.faces)
                        report.interfaces += group.fine_faces.capacity() * sizeof(group.fine_faces[0]);
                    if constexpr (std::same_as<T, MultiProviderNonconformingInterface>)
                    {
                        report.interfaces += n.fine_patches.capacity() * sizeof(FineInterfacePatch);
                        for (const auto& patch : n.fine_patches)
                        {
                            report.interfaces += patch.patches.capacity() * sizeof(FineNativePatch);
                            for (const auto& native : patch.patches)
                                report.interfaces += native.faces.capacity() * sizeof(ID);
                        }
                    }
                }
            },
            interface);
    std::set<const void*> seen;
    for (size_t p = 0; p < product_count(data_->grid); ++p)
    {
        const auto& product = product_at(data_->grid, p);
        if (seen.insert(product.get()).second)
            result.retained_participants.push_back({product.get(), product->storage_report(), true});
    }
    for (const auto& participant : data_->grid.adaptive)
        if (seen.insert(participant.get()).second)
        {
            if (const auto product = std::dynamic_pointer_cast<const ExtrudedAMR>(participant))
                result.retained_participants.push_back({participant.get(), product->storage_report(), true});
            else
                result.retained_participants.push_back(
                    {participant.get(), participant->native_geometry()->storage_report(), false});
        }
    return result;
}
void MixedContactDomains::validate() const
{
    for (const auto& d : domains)
        if (!d || !d->geometry_unchanged())
            throw std::invalid_argument("Contact domain is null or stale.");
    for (const auto& c : contacts)
    {
        if (c.first_domain >= domains.size() || c.second_domain >= domains.size() || c.first_domain == c.second_domain)
            throw std::invalid_argument("Contact domains must be distinct.");
        auto contains = [&](size_t i, const std::string& name)
        {
            for (auto b : domains[i]->geometry()->boundary_batch_ids())
                if (domains[i]->geometry()->boundary_batch_name(b) == name)
                    return true;
            return false;
        };
        if (!contains(c.first_domain, c.first_patch) || !contains(c.second_domain, c.second_patch))
            throw std::invalid_argument("Unknown external contact patch.");
    }
}
AdaptiveBootstrap MixedAMR::detach_owned(
    std::span<const std::uint64_t> owned, AdaptiveBytes domain, std::size_t byte_budget) const
{
    AdaptiveBootstrap result;
    collective(
        [&]
        {
            if (!geometry_unchanged() || domain.empty() || !byte_budget)
                throw std::invalid_argument("Mixed detach needs an unchanged accepted source, domain and byte budget.");
            using adaptive_detail::Writer;
            using adaptive_detail::Budget;
            const auto count = data_->participant_cells.size() - 1;
            Budget allocation{byte_budget};
            allocation.add(count * (sizeof(AdaptiveBootstrap) + sizeof(DistributedMixedRoute)));
            std::vector<std::vector<ID>> selected(count);
            std::set<ID> unique;
            auto participant = [&](ID cell)
            {
                if (cell >= data_->participant_cells.back())
                    throw std::out_of_range("Mixed detached source cell.");
                return std::size_t(
                    std::upper_bound(data_->participant_cells.begin(), data_->participant_cells.end(), cell) -
                    data_->participant_cells.begin() - 1);
            };
            auto root_key = [](AdaptiveCellKey key)
            {
                std::visit(
                    [](auto& p)
                    {
                        using T = std::decay_t<decltype(p)>;
                        if constexpr (std::is_same_v<T, CartesianCellAncestry>)
                        {
                            while (p.level)
                                p = p.parent();
                        }
                        else if constexpr (std::is_same_v<T, ConvexCellAncestry>)
                            p.path.clear();
                        else if constexpr (std::is_same_v<T, ProductCellAncestry>)
                        {
                            p.base.path.clear();
                            p.axial.path.clear();
                        }
                    },
                    key.ancestry);
                return key;
            };
            for (auto cell : owned)
            {
                allocation.add(sizeof(ID));
                if (!unique.insert(cell).second)
                    throw std::invalid_argument("Repeated mixed detached source cell.");
                const auto p = participant(cell);
                selected[p].push_back(cell - data_->participant_cells[p]);
            }
            auto original_region = [&](std::size_t p)
            {
                if (p < product_count(data_->grid))
                    return product_at(data_->grid, p)->cell_key(0, 0).region;
                auto a = p - product_count(data_->grid);
                if (a < data_->grid.adaptive.size())
                    return data_->grid.adaptive[a]->cell_key(0).region;
                return data_->grid.stationary[a - data_->grid.adaptive.size()].physical_region;
            };
            auto bootstrap = [&](std::size_t p, std::span<const ID> cells) -> AdaptiveBootstrap
            {
                if (p < product_count(data_->grid))
                {
                    std::vector<ProductCellAncestry> keys;
                    for (auto local : cells)
                        keys.push_back(std::get<ProductCellAncestry>(
                            data_key(*data_, data_->participant_cells[p] + local, p).ancestry));
                    return product_at(data_->grid, p)->detach_owned(keys, domain);
                }
                const auto a = p - product_count(data_->grid);
                if (a < data_->grid.adaptive.size())
                {
                    const auto& source = data_->grid.adaptive[a];
                    if (auto h = std::dynamic_pointer_cast<const CartesianAMR>(source))
                        return h->detach_owned(cells, domain);
                    if (auto h = std::dynamic_pointer_cast<const ExtrudedAMR>(source))
                        return h->detach_owned(cells, domain);
                    if (auto h = std::dynamic_pointer_cast<const PolyhedralAMR>(source))
                        return h->detach_owned(cells, domain);
                    throw std::invalid_argument("Mixed participant has no detached pure provider kernel.");
                }
                const auto& stationary = data_->grid.stationary[a - data_->grid.adaptive.size()];
                return detach_static_participant(stationary.region, stationary.provider_root,
                    stationary.physical_region, stationary.material, cells, domain);
            };
            std::vector<AdaptiveBootstrap> parts(count);
            std::vector<DistributedMixedRoute> routes;
            for (std::size_t p = 0; p < count; ++p)
            {
                parts[p] = bootstrap(p, selected[p]);
                DistributedMixedRoute route;
                route.identity = p;
                route.original_region = original_region(p);
                route.region = data_key(*data_, data_->participant_cells[p], p).region;
                route.kernel = parts[p].kernel;
                if (p < product_count(data_->grid))
                {
                    route.provider = AdaptiveProviderKind::Product;
                    route.compact_root_schema = product_at(data_->grid, p)->root_descriptor();
                }
                else
                {
                    const auto a = p - product_count(data_->grid);
                    if (a < data_->grid.adaptive.size())
                    {
                        const auto& source = data_->grid.adaptive[a];
                        if (auto h = std::dynamic_pointer_cast<const CartesianAMR>(source))
                        {
                            route.provider = AdaptiveProviderKind::Cartesian;
                            route.periodic = std::any_of(h->config().periodic.begin(), h->config().periodic.end(),
                                [](bool value) { return value; });
                            route.compact_root_schema = parts[p].kernel->descriptor();
                        }
                        else if (auto h = std::dynamic_pointer_cast<const ExtrudedAMR>(source))
                        {
                            route.provider = AdaptiveProviderKind::Product;
                            route.compact_root_schema = h->root_descriptor();
                        }
                        else if (std::dynamic_pointer_cast<const PolyhedralAMR>(source))
                        {
                            route.provider = AdaptiveProviderKind::Polyhedral;
                            route.compact_root_schema = parts[p].kernel->descriptor();
                        }
                    }
                    else
                    {
                        route.provider = AdaptiveProviderKind::Static;
                        route.compact_root_schema = parts[p].kernel->descriptor();
                        const auto periodic = std::visit([](const auto& r) { return r.layout().periodic; },
                            data_->grid.stationary[a - data_->grid.adaptive.size()].region);
                        route.periodic = std::any_of(periodic.begin(), periodic.end(), [](bool b) { return b; });
                    }
                }
                if (route.periodic)
                    throw std::invalid_argument("Mixed periodic participant frames are not qualified; use standalone "
                                                "Cartesian periodic hierarchy.");
                // Whole physical joins are compact root declarations. Current face
                // membership vectors are deliberately absent from this descriptor.
                Writer schema;
                schema.bytes(route.compact_root_schema);
                schema.word(data_->grid.joins.size());
                for (const auto& join : data_->grid.joins)
                {
                    schema.word(join.first_participant);
                    schema.word(join.second_participant);
                    schema.bytes(std::span(join.first_patch));
                    schema.bytes(std::span(join.second_patch));
                }
                route.compact_root_schema = std::move(schema.data);
                allocation.add(route.compact_root_schema.size());
                routes.push_back(std::move(route));
            }
            using Root = std::pair<std::size_t, AdaptiveCellKey>;
            struct Prototype
            {
                AdaptiveLeaf leaf;
                std::vector<AdaptiveSurface> faces;
            };
            std::map<Root, Prototype> prototypes;
            auto global_key = [&](std::size_t p, AdaptiveCellKey key)
            {
                key.region = routes[p].region;
                return key;
            };
            auto prototype = [&](std::size_t p, ID current) -> Prototype&
            {
                const auto global_root = root_key(data_key(*data_, current, p));
                const Root identity{p, global_root};
                auto found = prototypes.find(identity);
                if (found != prototypes.end())
                    return found->second;
                const ID local = current - data_->participant_cells[p];
                auto temporary = bootstrap(p, std::span<const ID>(&local, 1));
                if (temporary.owned.size() != 1)
                    throw std::logic_error("One source cell did not produce one detached root branch.");
                auto leaf = temporary.owned[0];
                if (leaf.provider == AdaptiveProviderKind::Product)
                    leaf = product_support_root(leaf);
                else if (leaf.provider == AdaptiveProviderKind::Cartesian)
                {
                    const auto a = p - product_count(data_->grid);
                    const auto source = std::dynamic_pointer_cast<const CartesianAMR>(data_->grid.adaptive[a]);
                    auto key = root_key(leaf.key);
                    leaf = cartesian_adaptive_leaf(source->config(), std::move(key), domain);
                }
                else if (leaf.provider == AdaptiveProviderKind::Polyhedral)
                {
                    const auto key = root_key(leaf.key);
                    auto it = std::find_if(temporary.support_roots.begin(), temporary.support_roots.end(),
                        [&](const auto& root) { return root.key == key; });
                    if (it == temporary.support_roots.end())
                        throw std::invalid_argument("Needed original polyhedral support root was not detached.");
                    leaf = *it;
                }
                auto faces = routes[p].kernel->surfaces(leaf);
                for (const auto& face : faces)
                    allocation.add(face.geometry->storage_bytes() + face.key.size());
                auto [it, inserted] = prototypes.emplace(identity, Prototype{std::move(leaf), std::move(faces)});
                (void) inserted;
                return it->second;
            };
            std::set<Root> needed;
            for (auto cell : owned)
            {
                const auto p = participant(cell);
                const Root root{p, root_key(data_key(*data_, cell, p))};
                needed.insert(root);
                (void) prototype(p, cell);
            }
            struct RuleGroup
            {
                AdaptiveBytes original, canonical;
                PLIC::PlanarFacePtr<ArithmeticMode::Ordinary> common;
                std::vector<PLIC::PlanarFacePtr<ArithmeticMode::Ordinary>> selected;
            };
            std::map<Root, std::map<AdaptiveBytes, RuleGroup>> groups;
            for (ID face = 0; face < data_->geometry->num_faces(); ++face)
            {
                const auto owner = data_->geometry->owner_cell(face), neighbor = data_->geometry->neighbor_cell(face);
                if (neighbor == MultiRegionMesh::invalid_cell_id())
                    continue;
                const auto p = participant(owner), q = participant(neighbor);
                if (p == q)
                    continue;
                const Root left{p, root_key(data_key(*data_, owner, p))},
                    right{q, root_key(data_key(*data_, neighbor, q))};
                if (!needed.contains(left) && !needed.contains(right))
                    continue;
                auto selected_face =
                    PLIC::PlanarFace<ArithmeticMode::Ordinary>::from_native_face(*data_->geometry, face);
                auto& a = prototype(p, owner);
                auto& b = prototype(q, neighbor);
                auto support = [&](Prototype& root) -> const AdaptiveSurface&
                {
                    const AdaptiveSurface* found = nullptr;
                    for (const auto& candidate : root.faces)
                        if (candidate.geometry->contains(*selected_face))
                        {
                            if (found)
                                throw std::invalid_argument(
                                    "Mixed joined tile ambiguously spans original atomic root faces.");
                            found = &candidate;
                        }
                    if (!found)
                        throw std::invalid_argument("Mixed joined face lacks an immutable original-root support.");
                    return *found;
                };
                const auto& af = support(a);
                const auto& bf = support(b);
                auto common = af.geometry->intersection(*bf.geometry);
                if (!common || !common->contains(*selected_face))
                    throw std::invalid_argument("Mixed source join lies outside original root intersection.");
                Writer canonical;
                canonical.word(0x53464d4a4f494e36ULL);
                canonical.bytes(domain);
                if (p < q)
                {
                    canonical.word(p);
                    canonical.word(q);
                    canonical.bytes(encode_adaptive_key(left.second));
                    canonical.bytes(af.key);
                    canonical.bytes(encode_adaptive_key(right.second));
                    canonical.bytes(bf.key);
                }
                else
                {
                    canonical.word(q);
                    canonical.word(p);
                    canonical.bytes(encode_adaptive_key(right.second));
                    canonical.bytes(bf.key);
                    canonical.bytes(encode_adaptive_key(left.second));
                    canonical.bytes(af.key);
                }
                auto forward = PLIC::canonical_face_key(*common),
                     reverse = PLIC::canonical_face_key(*common->reversed());
                canonical.bytes(std::min(forward, reverse));
                auto id = std::move(canonical.data);
                auto add = [&](const Root& root, const AdaptiveSurface& source)
                {
                    if (!needed.contains(root))
                        return;
                    auto [it, inserted] = groups[root].try_emplace(id, RuleGroup{source.key, id, common, {}});
                    (void) inserted;
                    allocation.add(
                        selected_face->storage_bytes() + sizeof(PLIC::PlanarFacePtr<ArithmeticMode::Ordinary>));
                    it->second.selected.push_back(selected_face);
                };
                add(left, af);
                add(right, bf);
            }
            // Preserve accepted physical destinations independently of pure provider
            // naming. Only needed original-root faces are retained, including the
            // ancestor supports absent from an interior descendant's current shell.
            std::map<Root, std::vector<AdaptiveBoundaryRule>> physical;
            for (ID face = 0; face < data_->geometry->num_faces(); ++face)
            {
                if (!data_->geometry->is_exterior_face(face))
                    continue;
                const auto owner = data_->geometry->owner_cell(face);
                const auto p = participant(owner);
                const Root root{p, root_key(data_key(*data_, owner, p))};
                if (!needed.contains(root))
                    continue;
                auto shape = PLIC::PlanarFace<ArithmeticMode::Ordinary>::from_native_face(*data_->geometry, face);
                auto& original = prototype(p, owner);
                const AdaptiveSurface* support = nullptr;
                for (const auto& candidate : original.faces)
                    if (candidate.geometry->contains(*shape))
                    {
                        if (support)
                            throw std::invalid_argument("Ambiguous physical original-root support.");
                        support = &candidate;
                    }
                if (!support)
                    throw std::invalid_argument("Physical source face lacks original-root support.");
                std::string name;
                const auto boundary = data_->geometry->boundary_id(face);
                if (boundary != MultiRegionMesh::invalid_boundary_id)
                    name = data_->geometry->boundary_batch_name(boundary);
                allocation.add(
                    shape->storage_bytes() + support->key.size() + name.size() + sizeof(AdaptiveBoundaryRule));
                physical[root].push_back({support->key, std::move(shape), std::move(name)});
            }
            std::map<Root, std::shared_ptr<const AdaptiveBytes>> capsules;
            for (const auto& root : needed)
            {
                AdaptiveJoinCapsule capsule;
                capsule.route = root.first;
                for (auto& [id, group] : groups[root])
                {
                    std::vector<PLIC::PlanarFacePtr<ArithmeticMode::Ordinary>> unique;
                    for (const auto& face : group.selected)
                    {
                        bool duplicate = false;
                        for (const auto& previous : unique)
                            if (auto overlap = previous->intersection(*face))
                            {
                                if (previous->equivalent(*face))
                                {
                                    duplicate = true;
                                    break;
                                }
                                throw std::invalid_argument(
                                    "Accepted mixed source has overlapping canonical join tiles.");
                            }
                        if (!duplicate)
                            unique.push_back(face);
                    }
                    bool complete = true;
                    try
                    {
                        group.common->certify_partition(unique);
                    }
                    catch (const std::invalid_argument&)
                    {
                        complete = false;
                    }
                    if (complete)
                        capsule.rules.push_back({group.original, group.canonical, group.common});
                    else
                        for (auto& face : unique)
                            capsule.rules.push_back({group.original, group.canonical, std::move(face)});
                }
                // A single accepted destination can cover a complete original
                // support after subtracting joins. Collapse its source subdivisions
                // to that immutable root selection; distinct destinations stay exact.
                std::map<AdaptiveBytes, std::vector<AdaptiveBoundaryRule>> by_support;
                for (auto& rule : physical[root])
                    by_support[rule.original_support].push_back(std::move(rule));
                for (auto& [support, rules] : by_support)
                {
                    const bool one_name = std::all_of(rules.begin(), rules.end(),
                        [&](const auto& rule) { return rule.boundary == rules.front().boundary; });
                    const auto& original = prototypes.at(root).faces;
                    auto original_face = std::find_if(
                        original.begin(), original.end(), [&](const auto& face) { return face.key == support; });
                    if (original_face == original.end())
                        throw std::logic_error("Lost original physical support.");
                    std::vector<PLIC::PlanarFacePtr<ArithmeticMode::Ordinary>> coverage;
                    for (const auto& rule : rules)
                        coverage.push_back(rule.original_root_selection);
                    for (const auto& rule : capsule.rules)
                        if (rule.original_support == support)
                            coverage.push_back(rule.original_root_selection);
                    bool complete = one_name;
                    if (complete)
                        try
                        {
                            original_face->geometry->certify_partition(coverage);
                        }
                        catch (const std::invalid_argument&)
                        {
                            complete = false;
                        }
                    if (complete)
                        capsule.boundaries.push_back({support, original_face->geometry, rules.front().boundary});
                    else
                        for (auto& rule : rules)
                            capsule.boundaries.push_back(std::move(rule));
                }
                auto bytes = encode_join_capsule(capsule, byte_budget);
                allocation.add(bytes.size());
                capsules.emplace(root, std::make_shared<const AdaptiveBytes>(std::move(bytes)));
            }
            auto wrap = [&](std::size_t p, AdaptiveLeaf leaf)
            {
                leaf.key = global_key(p, std::move(leaf.key));
                leaf.join_recipe = capsules.at({p, root_key(leaf.key)});
                if (!leaf.parent_family.empty())
                {
                    adaptive_detail::Reader family{leaf.parent_family};
                    if (family.word() != 0x534646414d363031ULL)
                        throw std::invalid_argument("Unknown inherited mixed family identity.");
                    auto split = family.word();
                    auto key = decode_adaptive_key(family.bytes());
                    family.done();
                    leaf.parent_family = adaptive_family_key(global_key(p, std::move(key)), AdaptiveSplit(split));
                }
                return leaf;
            };
            for (std::size_t p = 0; p < count; ++p)
            {
                for (auto& leaf : parts[p].owned)
                    result.owned.push_back(wrap(p, std::move(leaf)));
                for (auto& family : parts[p].families)
                {
                    family.parent = wrap(p, std::move(family.parent));
                    family.key = adaptive_family_key(family.parent.key, family.split);
                    for (auto& child : family.children)
                        child = global_key(p, std::move(child));
                    family.anchor = encode_adaptive_key(global_key(p, decode_adaptive_key(family.anchor)));
                    result.families.push_back(std::move(family));
                }
            }
            for (auto cell : owned)
                result.source_cells.push_back({data_key(*data_, cell, participant(cell)), cell});
            result.kernel = mixed_adaptive_kernel(std::move(routes), byte_budget);
            for (const auto& leaf : result.owned)
                result.kernel->validate(leaf);
            for (const auto& family : result.families)
                result.kernel->validate(family.parent);
            if (!geometry_unchanged())
                throw std::invalid_argument("Mixed source moved while detaching its root aliases.");
        });
    return result;
}
} // namespace SimpleFluid::Meshes::AMR
