/** @file FetchedNativeSources.cc
 * @brief Collective authenticated compact source-only geometry facade.
 */
#include "FVM/GeneralVoF.hh"
#include "FVM/details/GeneralVoF/CellCatalog.hh"
#include "FVM/details/GeneralVoFInternal.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveNative.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveWire.hh"
#include <limits>
#include <memory>
#include <set>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>
namespace SimpleFluid::VoF
{
namespace
{
namespace AMR = Meshes::AMR;
namespace catalog = general_vof_detail::cell_catalog_detail;
using Budget = general_vof_detail::wire::Budget;
/** The fetched draft is constructed under the hierarchy's unchanged boundary.
 * Before allocating any new facade/catalog/gather storage, join its actual
 * surviving ownership to the caller's operation scope. Every registry row is
 * admitted before insertion; independently allocated equal owners stay distinct. */
class DraftAdmission
{
    enum class Kind
    {
        Face,
        FacePayload,
        Recipe,
        Bytes,
        Factor
    };
    struct Less
    {
        bool operator()(const std::pair<Kind, const void*>& a, const std::pair<Kind, const void*>& b) const noexcept
        {
            return a.first != b.first ? a.first < b.first : std::less<const void*>{}(a.second, b.second);
        }
    };
    Budget& budget_;
    PLIC::GeometryAllocationLedger& geometry_;
    std::set<std::pair<Kind, const void*>, Less> owners_;
    AMR::adaptive_detail::ImmutableSequenceLedger tile_metadata_;
    bool unique(Kind kind, const void* pointer)
    {
        if (!pointer || owners_.contains({kind, pointer}))
            return false;
        catalog::charge_items(budget_, 1, sizeof(typename decltype(owners_)::value_type) + 4 * sizeof(void*));
        owners_.emplace(kind, pointer);
        return true;
    }
    template<class T> void vector(const std::vector<T>& values)
    {
        catalog::charge_items(budget_, values.capacity(), sizeof(T));
    }
    void key(const AMR::AdaptiveCellKey& value)
    {
        std::visit(
            [&](const auto& ancestry)
            {
                if constexpr (requires { ancestry.path; })
                    vector(ancestry.path);
                else if constexpr (requires { ancestry.base.path; })
                {
                    vector(ancestry.base.path);
                    vector(ancestry.axial.path);
                }
            },
            value.ancestry);
    }
    void bytes(const std::shared_ptr<const AMR::AdaptiveBytes>& value)
    {
        if (value && unique(Kind::Bytes, value.get()))
        {
            budget_.add(sizeof(*value));
            vector(*value);
        }
    }
    void recipe(const std::shared_ptr<const Meshes::NativeFragmentDescriptor>& value)
    {
        if (value && unique(Kind::Recipe, value.get()))
        {
            budget_.add(sizeof(*value));
            vector(value->view);
            vector(value->parts);
            for (const auto& part : value->parts)
                bytes(part);
        }
    }
    void face(const PLIC::PlanarFacePtr<ArithmeticMode::Ordinary>& value)
    {
        if (!value)
            return;
        if (unique(Kind::Face, value.get()))
            budget_.add(sizeof(*value));
        if (unique(Kind::FacePayload, value->storage_identity()))
            budget_.add(value->exclusive_storage_bytes());
        geometry_.retain(PLIC::planar_support<ArithmeticMode::Ordinary>(*value), budget_.used, budget_.limit);
    }
    template<class Identity> void identity(const Identity& value)
    {
        if constexpr (Identity::shared_storage)
            geometry_.retain(value, budget_.used, budget_.limit);
    }
    void leaf(const AMR::AdaptiveLeaf& value)
    {
        key(value.key);
        vector(value.domain);
        vector(value.parent_family);
        vector(value.surfaces);
        if (value.geometry)
            geometry_.retain(value.geometry, budget_.used, budget_.limit);
        for (const auto& surface : value.surfaces)
        {
            vector(surface.key);
            // Match the established AMR admission's conservative SSO allowance.
            budget_.add(surface.boundary.capacity());
            face(surface.geometry);
            face(surface.native_whole_geometry);
            identity(surface.authority_image);
        }
        recipe(value.native_recipe);
        bytes(value.join_recipe);
        if (value.factor_recipe && unique(Kind::Factor, value.factor_recipe.get()))
        {
            budget_.add(sizeof(*value.factor_recipe));
            bytes(value.factor_recipe->root);
            bytes(value.factor_recipe->base);
            bytes(value.factor_recipe->axis);
        }
    }

public:
    DraftAdmission(Budget& budget, PLIC::GeometryAllocationLedger& geometry) : budget_(budget), geometry_(geometry)
    {
        budget_.add(sizeof(DraftAdmission));
    }
    void retain(const AMR::AdaptiveNativeDraft& draft)
    {
        vector(draft.owned);
        vector(draft.ghosts);
        vector(draft.tiles);
        vector(draft.nodes);
        vector(draft.cell_bindings);
        vector(draft.family_references);
        for (const auto* rows : {&draft.owned, &draft.ghosts})
            for (const auto& value : *rows)
                leaf(value);
        for (const auto& tile : draft.tiles)
        {
            tile_metadata_.retain(tile.domain, budget_);
            tile_metadata_.retain(tile.surface, budget_);
            tile_metadata_.retain(tile.node_gids, budget_);
            tile_metadata_.retain(tile.second_node_gids, budget_);
            vector(tile.identity);
            key(tile.first);
            if (tile.second)
                key(*tile.second);
            budget_.add(tile.boundary.capacity());
            face(tile.geometry);
            identity(tile.second_image_identity);
        }
        for (const auto& node : draft.nodes)
        {
            vector(node.identity);
            identity(node.coordinate_identity);
        }
        for (const auto& binding : draft.cell_bindings)
        {
            key(binding.key);
            recipe(binding.native_recipe);
        }
        for (const auto& family : draft.family_references)
            vector(family);
    }
};
} // namespace
struct GeneralVoFFetchedNativeSources::Storage
{
    Meshes::AMR::AdaptiveNativeDraft draft;
    std::shared_ptr<const PLIC::CellGeometryCatalog<ArithmeticMode::Ordinary>> catalog;
    std::size_t admitted = 0, fetch_limit = 0;
};
const Meshes::AMR::AdaptiveNativeDraft& GeneralVoFFetchedNativeSources::draft() const noexcept
{
    return storage_->draft;
}
PLIC::CellGeometryRange<ArithmeticMode::Ordinary> GeneralVoFFetchedNativeSources::geometry() const
{
    return PLIC::CellGeometryRange<ArithmeticMode::Ordinary>(storage_->catalog);
}
std::size_t GeneralVoFFetchedNativeSources::admitted_bytes() const noexcept
{
    return storage_->admitted;
}
std::size_t GeneralVoFFetchedNativeSources::fetch_byte_budget() const noexcept
{
    return storage_->fetch_limit;
}
GeneralVoFFetchedNativeSources GeneralVoF::fetch_native_sources(
    std::span<const Key> requested, std::size_t operation_byte_budget) const
{
    namespace detail = general_vof_detail;
    namespace catalog = detail::cell_catalog_detail;
    const auto& hierarchy = *data_->hierarchy;
    const auto comm = hierarchy.communicator();
    detail::wire::Budget budget{operation_byte_budget};
    std::shared_ptr<GeneralVoFFetchedNativeSources::Storage> result;
    // No local allocation/validation may fail before peers enter this guard.
    detail::wire::collective(*comm,
        [&]
        {
            budget.add(sizeof(GeneralVoFFetchedNativeSources));
            budget.add(sizeof(GeneralVoFFetchedNativeSources::Storage));
            // Validate representable request bookkeeping before entering fetch. The
            // request span itself remains borrowed and is not copied here.
            (void) catalog::checked_product(requested.size(), sizeof(Key));
            switch (data_->options.arithmetic_mode)
            {
                case ArithmeticMode::Ordinary:
                case ArithmeticMode::BoundedCertification:
                case ArithmeticMode::Exact:
                    break;
                default:
                    throw std::invalid_argument("Unknown fetched native source arithmetic mode.");
            }
        });
    // Fetch owns its established hierarchy budget and collective sequence.
    // In particular, remote requested cells and empty owners participate.
    auto fetched = Meshes::AMR::adaptive_detail::fetch_compact_adaptive_native_draft(hierarchy, requested);
    detail::wire::collective(*comm,
        [&]
        {
            PLIC::GeometryAllocationLedger original_owners;
            DraftAdmission draft_admission(budget, original_owners);
            draft_admission.retain(fetched);
            result = std::make_shared<GeneralVoFFetchedNativeSources::Storage>();
            result->fetch_limit = hierarchy.options().byte_budget;
            result->draft = std::move(fetched);
            const auto& draft = result->draft;
            if (draft.owned.size() > std::numeric_limits<std::size_t>::max() - draft.ghosts.size())
                throw std::invalid_argument("Fetched source row count overflows.");
            std::vector<const Meshes::AMR::AdaptiveLeaf*> leaves;
            catalog::reserve_leaf_references(leaves, draft.owned.size() + draft.ghosts.size(), budget);
            for (const auto* rows : {&draft.owned, &draft.ghosts})
                for (const auto& leaf : *rows)
                    leaves.push_back(&leaf);
            detail::cell_catalog_detail::CatalogBuild built;
            switch (data_->options.arithmetic_mode)
            {
                case ArithmeticMode::Ordinary:
                    built = catalog::build<ArithmeticMode::Ordinary>(hierarchy, leaves, budget, nullptr,
                        catalog::CatalogPurpose::OriginalSources, &original_owners);
                    break;
                case ArithmeticMode::BoundedCertification:
                    built = catalog::build<ArithmeticMode::BoundedCertification>(hierarchy, leaves, budget, nullptr,
                        catalog::CatalogPurpose::OriginalSources, &original_owners);
                    break;
                case ArithmeticMode::Exact:
                    built = catalog::build<ArithmeticMode::Exact>(hierarchy, leaves, budget, nullptr,
                        catalog::CatalogPurpose::OriginalSources, &original_owners);
                    break;
                default:
                    throw std::invalid_argument("Unknown fetched native source arithmetic mode.");
            }
            result->catalog = std::move(built.ordinary);
            result->admitted = budget.used;
        });
    return GeneralVoFFetchedNativeSources(std::move(result));
}
} // namespace SimpleFluid::VoF
