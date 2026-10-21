/**
 * @file CartesianAMR.cc
 * @brief Collective dyadic hierarchy adaptation and canonical Cartesian seams.
 */
#include "geometry/mesh/AMR/CartesianAMR.hh"

#include <Teuchos_CommHelpers.hpp>
#include <Tpetra_Core.hpp>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <exception>
#include <iomanip>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>

namespace SimpleFluid::Meshes::AMR
{
namespace
{
using ID = std::uint64_t;
using LeafSet = std::set<AMRBlock>;

ID next_identity()
{
    static std::atomic<ID> next{0};
    const auto comm = Tpetra::getDefaultComm();
    ID value = 0;
    if (comm->getRank() == 0)
    {
        auto previous = next.load();
        while (previous != std::numeric_limits<ID>::max())
        {
            if (next.compare_exchange_weak(previous, previous + 1))
            {
                value = previous + 1;
                break;
            }
        }
    }
    Teuchos::broadcast(*comm, 0, 1, &value);
    if (value == 0)
        throw std::overflow_error("Cartesian AMR collective identity overflow.");
    return value;
}

ID product(ID a, ID b)
{
    if (b && a > std::numeric_limits<ID>::max() / b)
        throw std::overflow_error("Cartesian AMR index extent overflows.");
    return a * b;
}

ID per_block(const AMRGrid& config)
{
    return product(product(config.block_cells[0], config.block_cells[1]), config.block_cells[2]);
}

ID level_scale(unsigned level)
{
    if (level >= std::numeric_limits<ID>::digits - 1)
        throw std::invalid_argument("Cartesian AMR maximum level exceeds index precision.");
    return ID{1} << level;
}

void validate_config(const AMRGrid& config)
{
    const auto finest = level_scale(config.max_level);
    for (std::size_t a = 0; a < 3; ++a)
    {
        if (config.root_blocks[a] == 0 || config.block_cells[a] < 2 || config.block_cells[a] % 2 != 0 ||
            config.block_cells[a] >= std::numeric_limits<unsigned>::max())
            throw std::invalid_argument("Cartesian AMR requires positive root counts and even block cell counts >=2.");
        const auto count = product(product(config.root_blocks[a], config.block_cells[a]), finest);
        if (count > static_cast<ID>(std::numeric_limits<std::int64_t>::max() / 4))
            throw std::invalid_argument("Cartesian AMR finest-grid coordinates exceed signed index limits.");
        const double end = config.origin[a] + config.extent[a];
        const double h = config.extent[a] / static_cast<double>(count);
        if (!std::isfinite(config.origin[a]) || !std::isfinite(config.extent[a]) || config.extent[a] <= 0.0 ||
            !std::isfinite(end) || !std::isfinite(h) || !(h > 0.0) || !(config.origin[a] + h > config.origin[a]) ||
            !(end - h < end))
            throw std::invalid_argument("Cartesian AMR domain or finest spacing is not finite and representable.");
    }
    const auto roots = product(product(config.root_blocks[0], config.root_blocks[1]), config.root_blocks[2]);
    if (product(roots, per_block(config)) >= static_cast<ID>(std::numeric_limits<std::int64_t>::max()))
        throw std::invalid_argument("Cartesian AMR cell GIDs exceed signed global ordinal limits.");
}

void validate_block(const AMRGrid& config, const AMRBlock& block)
{
    if (block.level > config.max_level)
        throw std::invalid_argument("Cartesian AMR block exceeds max_level.");
    const auto scale = level_scale(block.level);
    for (std::size_t a = 0; a < 3; ++a)
        if (block.index[a] >= product(config.root_blocks[a], scale))
            throw std::invalid_argument("Cartesian AMR block index is outside the domain.");
}

template<class Action> void collective_check(Action&& action)
{
    std::exception_ptr error;
    try
    {
        action();
    }
    catch (...)
    {
        error = std::current_exception();
    }
    const auto comm = Tpetra::getDefaultComm();
    const int failed = error ? 1 : 0;
    int any_failed = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &failed, &any_failed);
    if (any_failed)
    {
        if (error)
            std::rethrow_exception(error);
        throw std::invalid_argument("Cartesian AMR request failed validation on another rank.");
    }
}

void same_description(const std::string& local)
{
    const auto comm = Tpetra::getDefaultComm();
    collective_check(
        [&]
        {
            if (local.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
                throw std::overflow_error("Cartesian AMR collective description exceeds message bounds.");
        });
    int size = comm->getRank() == 0 ? static_cast<int>(local.size()) : 0;
    Teuchos::broadcast(*comm, 0, 1, &size);
    std::string reference = comm->getRank() == 0 ? local : std::string(static_cast<std::size_t>(size), '\0');
    Teuchos::broadcast(*comm, 0, size, reference.data());
    collective_check(
        [&]
        {
            if (reference != local)
                throw std::invalid_argument("Cartesian AMR configurations or adaptation requests differ across ranks.");
        });
}

std::string description(const AMRGrid& config, const std::vector<AMRBlock>& blocks, ID generation)
{
    std::ostringstream out;
    out << std::hexfloat << generation << ' ' << config.max_level << ' ';
    for (std::size_t a = 0; a < 3; ++a)
        out << config.root_blocks[a] << ' ' << config.block_cells[a] << ' ' << config.origin[a] << ' '
            << config.extent[a] << ' ' << config.periodic[a] << ' ';
    out << config.region_id << ' ' << config.material_id << ' ' << blocks.size() << ' ';
    for (const auto& block : blocks)
        out << block.level << ' ' << block.index[0] << ' ' << block.index[1] << ' ' << block.index[2] << ' ';
    return out.str();
}

std::array<AMRBlock, 8> children(const AMRBlock& parent)
{
    std::array<AMRBlock, 8> result;
    for (std::size_t child = 0; child < 8; ++child)
    {
        result[child].level = parent.level + 1;
        for (std::size_t a = 0; a < 3; ++a)
            result[child].index[a] = 2 * parent.index[a] + ((child >> a) & 1U);
    }
    return result;
}

AMRBlock parent_of(AMRBlock block)
{
    --block.level;
    for (auto& index : block.index)
        index /= 2;
    return block;
}

LeafSet::const_iterator covering_leaf(const LeafSet& leaves, AMRBlock block)
{
    while (true)
    {
        const auto found = leaves.find(block);
        if (found != leaves.end() || block.level == 0)
            return found;
        block = parent_of(block);
    }
}

void split(LeafSet& leaves, const AMRBlock& block, const AMRGrid& config)
{
    if (block.level >= config.max_level)
        throw std::invalid_argument("Cartesian AMR refinement exceeds max_level.");
    if (leaves.erase(block) != 1)
        throw std::invalid_argument("Cartesian AMR refinement key is not a leaf.");
    for (const auto& child : children(block))
        leaves.insert(child);
}

void balance(LeafSet& leaves, const AMRGrid& config)
{
    // Fine leaves query the ancestors of all 26 neighboring logical boxes.
    // This avoids pairwise comparisons of an increasingly large leaf catalog.
    while (true)
    {
        LeafSet refine;
        for (const auto& block : leaves)
            for (int k = -1; k <= 1; ++k)
                for (int j = -1; j <= 1; ++j)
                    for (int i = -1; i <= 1; ++i)
                    {
                        const std::array<int, 3> offset{i, j, k};
                        const int nonzero = (i != 0) + (j != 0) + (k != 0);
                        if (nonzero == 0)
                            continue;
                        auto neighbor = block;
                        bool outside = false;
                        bool periodic_face = false;
                        for (std::size_t a = 0; a < 3; ++a)
                        {
                            const auto count =
                                static_cast<std::int64_t>(product(config.root_blocks[a], level_scale(block.level)));
                            auto index = static_cast<std::int64_t>(block.index[a]) + offset[a];
                            if (index < 0 || index >= count)
                            {
                                if (!config.periodic[a])
                                {
                                    outside = true;
                                    break;
                                }
                                index = (index + count) % count;
                                periodic_face = nonzero == 1;
                            }
                            neighbor.index[a] = static_cast<std::size_t>(index);
                        }
                        if (outside)
                            continue;
                        const auto found = covering_leaf(leaves, neighbor);
                        if (found != leaves.end() &&
                            (found->level + 1 < block.level || (periodic_face && found->level < block.level)))
                            refine.insert(*found);
                    }
        if (refine.empty())
            return;
        for (const auto& block : refine)
            split(leaves, block, config);
    }
}
} // namespace

struct CartesianAMR::LeafDirectory
{
    std::map<AMRBlock, std::size_t> ordinals;
};

CartesianAMR::CartesianAMR(AMRGrid config) : config_(std::move(config))
{
    collective_check(
        [&]
        {
            validate_config(config_);
            for (std::size_t k = 0; k < config_.root_blocks[2]; ++k)
                for (std::size_t j = 0; j < config_.root_blocks[1]; ++j)
                    for (std::size_t i = 0; i < config_.root_blocks[0]; ++i)
                        blocks_.push_back({0, {i, j, k}});
            std::sort(blocks_.begin(), blocks_.end());
        });
    same_description(description(config_, blocks_, generation_));
    build_geometry();
    collective_identity_ = next_identity();
}

CartesianAMR::CartesianAMR(AMRGrid config, std::vector<AMRBlock> blocks, ID generation)
    : config_(std::move(config)), blocks_(std::move(blocks)), generation_(generation), partition_generation_(generation)
{
    collective_check(
        [&]
        {
            validate_config(config_);
            if (blocks_.empty() || !std::is_sorted(blocks_.begin(), blocks_.end()) ||
                std::adjacent_find(blocks_.begin(), blocks_.end()) != blocks_.end())
                throw std::invalid_argument("Cartesian AMR leaf catalog is empty, unsorted, or duplicated.");
            for (const auto& block : blocks_)
                validate_block(config_, block);
            if (product(blocks_.size(), per_block(config_)) >=
                static_cast<ID>(std::numeric_limits<std::int64_t>::max()))
                throw std::overflow_error("Cartesian AMR adapted cell GIDs exceed signed global ordinals.");
        });
    same_description(description(config_, blocks_, generation_));
    build_geometry();
    collective_identity_ = next_identity();
}

const AMRGrid& CartesianAMR::config() const noexcept
{
    return config_;
}
const std::vector<AMRBlock>& CartesianAMR::blocks() const noexcept
{
    return blocks_;
}
const SP<const MultiRegionMesh>& CartesianAMR::geometry() const noexcept
{
    return geometry_;
}
ID CartesianAMR::topology_generation() const noexcept
{
    return generation_;
}
ID CartesianAMR::partition_generation() const noexcept
{
    return partition_generation_;
}

const SP<const MultiRegionMesh>& CartesianAMR::native_geometry() const noexcept { return geometry_; }

bool CartesianAMR::geometry_unchanged() const noexcept
{
    try
    {
        return geometry_->geometry_epoch() == geometry_epoch_ &&
            (!partition_ || partition_->geometry_unchanged());
    }
    catch (...) { return false; }
}

HierarchyObservation CartesianAMR::observation() const noexcept
{
    const auto& native = partition_ ? partition_->geometry() : geometry_;
    ID epoch = std::numeric_limits<ID>::max();
    try { epoch = native->geometry_epoch(); }
    catch (...) { /* Immutable consumers reject through geometry_unchanged(). */ }
    return {identity_, topology_identity_, partition_identity_, generation_, partition_generation_,
        epoch, native.get(), Tpetra::getDefaultComm().getRawPtr()};
}

AdaptiveCellKey CartesianAMR::cell_key(ID cell) const
{
    const auto leaf = cell_block(cell);
    const auto local = cell % per_block(config_);
    const std::array<ID, 3> index{local % config_.block_cells[0],
        (local / config_.block_cells[0]) % config_.block_cells[1],
        local / (config_.block_cells[0] * config_.block_cells[1])};
    CartesianCellAncestry ancestry{blocks_[leaf].level, {}};
    for (std::size_t a = 0; a < 3; ++a)
        ancestry.index[a] = blocks_[leaf].index[a] * config_.block_cells[a] + index[a];
    return {config_.region_id, ancestry};
}

ID CartesianAMR::cell_material(ID cell) const { (void)cell_block(cell); return config_.material_id; }
std::size_t CartesianAMR::cell_fragment(ID cell) const { return cell_block(cell); }
const std::shared_ptr<const CompositePartition<>>& CartesianAMR::native_partition() const noexcept { return partition_; }

std::shared_ptr<const CartesianAMR> CartesianAMR::with_partition(
    std::shared_ptr<const CompositePartition<>> partition) const
{
    collective_check([&] {
        const auto comm = Tpetra::getDefaultComm();
        if (!geometry_unchanged() || !partition || !partition->geometry_unchanged() ||
            partition->plan().communicator().getRawPtr() != comm.getRawPtr() ||
            (comm->getRank() == partition->plan().source_rank() && !partition->plan().matches_source(geometry_)) ||
            partition->plan().global_cells() != geometry_->num_cells())
            throw std::invalid_argument("Cartesian AMR partition has a foreign source, revision, or communicator.");
        if (partition_generation_ == std::numeric_limits<ID>::max())
            throw std::overflow_error("Cartesian AMR partition generation overflow.");
    });
    auto result = std::make_shared<CartesianAMR>(*this);
    same_description(std::to_string(collective_identity_) + " " + std::to_string(partition->plan().collective_identity()));
    same_description(partition_ == partition ? "unchanged" : "replace");
    if (partition_ == partition)
        return result;
    result->partition_ = std::move(partition);
    ++result->partition_generation_;
    result->identity_ = std::make_shared<const char>(0);
    result->partition_identity_ = result->identity_;
    result->collective_identity_ = next_identity();
    return result;
}

CartesianAMR::AdaptationPlan CartesianAMR::plan_adaptation(
    const std::vector<std::size_t>& refine, const std::vector<AMRBlock>& coarsen) const
{
    AdaptationPlan plan;
    plan.source_ = observation();
    plan.candidate_ = adapt(refine, coarsen);
    plan.collective_id_ = next_identity();
    return plan;
}

std::shared_ptr<const CartesianAMR> CartesianAMR::apply_adaptation(const AdaptationPlan& plan) const
{
    collective_check([&] {
        if (!geometry_unchanged() || !(plan.source_ == observation()) || !plan.candidate_)
            throw std::invalid_argument("Cartesian AMR adaptation plan is stale or belongs to a different source.");
    });
    same_description(std::to_string(plan.collective_id_));
    return plan.candidate_;
}

double CartesianAMR::coordinate(std::size_t axis, ID finest_cell_index) const
{
    const auto total =
        product(product(config_.root_blocks[axis], config_.block_cells[axis]), level_scale(config_.max_level));
    return config_.origin[axis] +
           config_.extent[axis] * (static_cast<double>(finest_cell_index) / static_cast<double>(total));
}

AMRBox CartesianAMR::block_box(std::size_t leaf) const
{
    return key_box(blocks_.at(leaf));
}

AMRBox CartesianAMR::key_box(const AMRBlock& block) const
{
    const auto scale = level_scale(config_.max_level - block.level);
    AMRBox box;
    for (std::size_t a = 0; a < 3; ++a)
    {
        const auto begin = product(product(block.index[a], config_.block_cells[a]), scale);
        box.lower[a] = coordinate(a, begin);
        box.upper[a] = coordinate(a, begin + product(config_.block_cells[a], scale));
    }
    return box;
}

std::array<double, 3> CartesianAMR::block_spacing(std::size_t leaf) const
{
    const auto& block = blocks_.at(leaf);
    std::array<double, 3> spacing{};
    for (std::size_t a = 0; a < 3; ++a)
        spacing[a] = config_.extent[a] /
                     static_cast<double>(
                         product(product(config_.root_blocks[a], config_.block_cells[a]), level_scale(block.level)));
    return spacing;
}

std::size_t CartesianAMR::cell_block(ID cell) const
{
    const auto count = per_block(config_);
    if (cell >= product(count, blocks_.size()))
        throw std::out_of_range("Cartesian AMR cell GID is outside the hierarchy.");
    return static_cast<std::size_t>(cell / count);
}

AMRBox CartesianAMR::cell_box(ID cell) const
{
    const auto leaf = cell_block(cell);
    const auto local = cell % per_block(config_);
    const std::array<ID, 3> index{local % config_.block_cells[0],
        (local / config_.block_cells[0]) % config_.block_cells[1],
        local / (config_.block_cells[0] * config_.block_cells[1])};
    const auto& block = blocks_[leaf];
    const auto scale = level_scale(config_.max_level - block.level);
    AMRBox box;
    for (std::size_t a = 0; a < 3; ++a)
    {
        const auto begin = product(product(block.index[a], config_.block_cells[a]) + index[a], scale);
        box.lower[a] = coordinate(a, begin);
        box.upper[a] = coordinate(a, begin + scale);
    }
    return box;
}

std::vector<AMROverlap> CartesianAMR::overlaps(const AMRBox& request) const
{
    auto query = request;
    bool empty = false;
    for (std::size_t a = 0; a < 3; ++a)
    {
        const double end = config_.origin[a] + config_.extent[a];
        const double tolerance = 16.0 * std::numeric_limits<double>::epsilon() *
                                 std::max({std::abs(config_.origin[a]), std::abs(end), config_.extent[a]});
        if (!std::isfinite(query.lower[a]) || !std::isfinite(query.upper[a]) || query.lower[a] > query.upper[a] ||
            query.lower[a] < config_.origin[a] - tolerance || query.upper[a] > end + tolerance)
            throw std::invalid_argument("Cartesian AMR overlap query must be finite, ordered, and domain-contained.");
        query.lower[a] = std::clamp(query.lower[a], config_.origin[a], end);
        query.upper[a] = std::clamp(query.upper[a], config_.origin[a], end);
        empty = empty || query.lower[a] == query.upper[a];
    }
    std::vector<AMROverlap> result;
    if (empty)
        return result;
    const auto count = per_block(config_);
    const auto collect_leaf = [&](std::size_t leaf, const AMRBox& box)
    {
        const auto spacing = block_spacing(leaf);
        std::array<std::size_t, 3> lo{}, hi{};
        bool touches = true;
        for (std::size_t a = 0; a < 3; ++a)
        {
            const double lower = std::max(query.lower[a], box.lower[a]);
            const double upper = std::min(query.upper[a], box.upper[a]);
            if (!(upper > lower))
            {
                touches = false;
                break;
            }
            // One-cell padding prevents arithmetic rounding from excluding a
            // genuine sliver. Exact shared-coordinate intersections remove it.
            const auto begin = static_cast<std::size_t>(std::max(0.0, std::floor((lower - box.lower[a]) / spacing[a])));
            lo[a] = begin == 0 ? 0 : begin - 1;
            hi[a] = static_cast<std::size_t>(std::min(
                static_cast<double>(config_.block_cells[a]), std::ceil((upper - box.lower[a]) / spacing[a]) + 1.0));
        }
        if (!touches)
            return;
        for (std::size_t k = lo[2]; k < hi[2]; ++k)
            for (std::size_t j = lo[1]; j < hi[1]; ++j)
                for (std::size_t i = lo[0]; i < hi[0]; ++i)
                {
                    const ID cell = leaf * count + i + config_.block_cells[0] * (j + config_.block_cells[1] * k);
                    auto intersection = cell_box(cell);
                    bool positive = true;
                    for (std::size_t a = 0; a < 3; ++a)
                    {
                        intersection.lower[a] = std::max(intersection.lower[a], query.lower[a]);
                        intersection.upper[a] = std::min(intersection.upper[a], query.upper[a]);
                        positive = positive && intersection.upper[a] > intersection.lower[a];
                    }
                    if (positive)
                        result.push_back({cell, intersection});
                }
    };
    const auto visit = [&](auto&& self, const AMRBlock& key) -> void
    {
        const auto box = key_box(key);
        for (std::size_t a = 0; a < 3; ++a)
            if (std::min(query.upper[a], box.upper[a]) <= std::max(query.lower[a], box.lower[a]))
                return;
        const auto found = leaf_directory_->ordinals.find(key);
        if (found != leaf_directory_->ordinals.end())
        {
            collect_leaf(found->second, box);
            return;
        }
        if (key.level >= config_.max_level)
            throw std::logic_error("Cartesian AMR overlap descent reached an uncovered finest block.");
        for (const auto& child : children(key))
            self(self, child);
    };
    std::array<std::size_t, 3> root_lo{}, root_hi{};
    for (std::size_t a = 0; a < 3; ++a)
    {
        const long double scale = static_cast<long double>(config_.root_blocks[a]) / config_.extent[a];
        const auto first = static_cast<std::size_t>(
            std::max(0.0L, std::floor((static_cast<long double>(query.lower[a]) - config_.origin[a]) * scale)));
        root_lo[a] = first == 0 ? 0 : first - 1;
        root_hi[a] = static_cast<std::size_t>(std::min(static_cast<long double>(config_.root_blocks[a]),
            std::ceil((static_cast<long double>(query.upper[a]) - config_.origin[a]) * scale) + 1.0L));
    }
    for (std::size_t k = root_lo[2]; k < root_hi[2]; ++k)
        for (std::size_t j = root_lo[1]; j < root_hi[1]; ++j)
            for (std::size_t i = root_lo[0]; i < root_hi[0]; ++i)
                visit(visit, AMRBlock{0, {i, j, k}});
    // Keep the established leaf-ordinal, x-fastest cell order independently of
    // the traversal order of roots or recursively visited children.
    std::sort(result.begin(), result.end(), [](const auto& a, const auto& b) { return a.cell < b.cell; });
    return result;
}

std::vector<std::size_t> CartesianAMR::intersecting_blocks(const AMRBox& request) const
{
    auto query = request;
    for (std::size_t a = 0; a < 3; ++a)
    {
        if (!std::isfinite(query.lower[a]) || !std::isfinite(query.upper[a]) || query.lower[a] > query.upper[a])
            throw std::invalid_argument("Cartesian AMR block query must be finite and ordered.");
    }
    for (std::size_t a = 0; a < 3; ++a)
    {
        const double end = config_.origin[a] + config_.extent[a];
        if (query.upper[a] < config_.origin[a] || query.lower[a] > end)
            return {};
        query.lower[a] = std::max(query.lower[a], config_.origin[a]);
        query.upper[a] = std::min(query.upper[a], end);
    }
    std::vector<std::size_t> result;
    const auto visit = [&](auto&& self, const AMRBlock& key) -> void
    {
        const auto box = key_box(key);
        for (std::size_t a = 0; a < 3; ++a)
            if (query.upper[a] < box.lower[a] || box.upper[a] < query.lower[a])
                return;
        const auto found = leaf_directory_->ordinals.find(key);
        if (found != leaf_directory_->ordinals.end())
        {
            result.push_back(found->second);
            return;
        }
        if (key.level >= config_.max_level)
            throw std::logic_error("Cartesian AMR block query reached an uncovered finest block.");
        for (const auto& child : children(key))
            self(self, child);
    };
    std::array<std::size_t, 3> lo{}, hi{};
    for (std::size_t a = 0; a < 3; ++a)
    {
        const long double scale = static_cast<long double>(config_.root_blocks[a]) / config_.extent[a];
        const auto first = static_cast<std::size_t>(
            std::max(0.0L, std::floor((static_cast<long double>(query.lower[a]) - config_.origin[a]) * scale)));
        lo[a] = first == 0 ? 0 : first - 1;
        hi[a] = static_cast<std::size_t>(std::min(static_cast<long double>(config_.root_blocks[a]),
            std::ceil((static_cast<long double>(query.upper[a]) - config_.origin[a]) * scale) + 1.0L));
    }
    for (std::size_t k = lo[2]; k < hi[2]; ++k)
        for (std::size_t j = lo[1]; j < hi[1]; ++j)
            for (std::size_t i = lo[0]; i < hi[0]; ++i)
                visit(visit, AMRBlock{0, {i, j, k}});
    std::sort(result.begin(), result.end());
    return result;
}

std::shared_ptr<const CartesianAMR> CartesianAMR::adapt(
    const std::vector<std::size_t>& refine, const std::vector<AMRBlock>& coarsen) const
{
    LeafSet requested_refine;
    LeafSet requested_coarsen;
    collective_check(
        [&]
        {
            if (!geometry_unchanged())
                throw std::invalid_argument("Cartesian AMR requires unchanged immutable geometry.");
            for (const auto leaf : refine)
            {
                const auto& block = blocks_.at(leaf);
                if (block.level >= config_.max_level)
                    throw std::invalid_argument("Cartesian AMR refinement exceeds max_level.");
                requested_refine.insert(block);
            }
            for (const auto& parent : coarsen)
            {
                validate_block(config_, parent);
                if (parent.level >= config_.max_level)
                    throw std::invalid_argument("Cartesian AMR coarsening parent has no admissible children.");
                requested_coarsen.insert(parent);
                for (const auto& child : children(parent))
                    if (!std::binary_search(blocks_.begin(), blocks_.end(), child) || requested_refine.contains(child))
                        throw std::invalid_argument(
                            "Cartesian AMR coarsening requires complete unrefined leaf siblings.");
            }
        });
    std::ostringstream request;
    request << description(config_, blocks_, generation_) << " refine ";
    for (const auto& block : requested_refine)
        request << block.level << ' ' << block.index[0] << ' ' << block.index[1] << ' ' << block.index[2] << ' ';
    request << " coarsen ";
    for (const auto& block : requested_coarsen)
        request << block.level << ' ' << block.index[0] << ' ' << block.index[1] << ' ' << block.index[2] << ' ';
    same_description(request.str());
    LeafSet leaves(blocks_.begin(), blocks_.end());
    collective_check(
        [&]
        {
            for (const auto& parent : requested_coarsen)
            {
                for (const auto& child : children(parent))
                    leaves.erase(child);
                leaves.insert(parent);
            }
            for (const auto& block : requested_refine)
                split(leaves, block, config_);
            balance(leaves, config_);
        });
    std::vector<AMRBlock> candidate(leaves.begin(), leaves.end());
    if (candidate == blocks_)
        return std::make_shared<const CartesianAMR>(*this);
    collective_check(
        [&]
        {
            if (generation_ == std::numeric_limits<ID>::max())
                throw std::overflow_error("Cartesian AMR generation counter overflow.");
        });
    auto result = std::shared_ptr<CartesianAMR>(new CartesianAMR(config_, std::move(candidate), generation_ + 1));
    collective_check([&] {
        if (partition_generation_ == std::numeric_limits<ID>::max())
            throw std::overflow_error("Cartesian AMR partition generation overflow.");
    });
    result->partition_generation_ = partition_generation_ + 1;
    return result;
}

void CartesianAMR::build_geometry()
{
    std::vector<MultiRegionMesh::Region> regions;
    std::vector<MultiRegionMesh::Interface> interfaces;
    collective_check(
        [&]
        {
            const auto& cells = config_.block_cells;
            auto topology = std::make_shared<const RectilinearTopology>(std::make_shared<const OrthoMeshTopo>(
                static_cast<unsigned>(cells[0]), static_cast<unsigned>(cells[1]), static_cast<unsigned>(cells[2]),
                false, false, false, OrthoMeshTopo::BoundaryNames{"xmin", "xmax", "ymin", "ymax", "zmin", "zmax"}));
            auto directory = std::make_shared<LeafDirectory>();
            auto& ordinals = directory->ordinals;
            LeafSet leaves;
            regions.reserve(blocks_.size());
            for (std::size_t leaf = 0; leaf < blocks_.size(); ++leaf)
            {
                const auto& block = blocks_[leaf];
                ordinals.emplace(block, leaf);
                leaves.insert(block);
                Vec3D<ArrReal> edges;
                const auto scale = level_scale(config_.max_level - block.level);
                for (std::size_t a = 0; a < 3; ++a)
                    for (std::size_t i = 0; i <= cells[a]; ++i)
                        edges[a].push_back(coordinate(a, product(product(block.index[a], cells[a]) + i, scale)));
                const auto name = "block_" + std::to_string(block.level) + "_" + std::to_string(block.index[0]) + "_" +
                                  std::to_string(block.index[1]) + "_" + std::to_string(block.index[2]);
                regions.emplace_back(
                    CartesianRegion(name, topology, std::make_shared<const RectilinearGeometry>(std::move(edges))));
            }
            leaf_directory_ = std::move(directory);
            auto native_face = [&](std::size_t axis, const std::array<std::size_t, 3>& p)
            {
                return topology->indexer().face_ordinal({static_cast<unsigned>(p[0]), static_cast<unsigned>(p[1]),
                    static_cast<unsigned>(p[2]), static_cast<std::uint8_t>(axis)});
            };
            auto coarse_fine = [&](std::size_t coarse, std::size_t fine, std::size_t axis, bool coarse_upper)
            {
                NonconformingInterface interface;
                interface.coarse_region = coarse;
                interface.fine_region = fine;
                interface.coarse_boundary = static_cast<int>(2 * axis + coarse_upper);
                interface.fine_boundary = static_cast<int>(2 * axis + !coarse_upper);
                interface.partial_coarse_patch = true;
                const auto b = (axis + 1) % 3;
                const auto c = (axis + 2) % 3;
                for (std::size_t v = 0; v < cells[c] / 2; ++v)
                    for (std::size_t u = 0; u < cells[b] / 2; ++u)
                    {
                        std::array<std::size_t, 3> p{};
                        p[axis] = coarse_upper ? cells[axis] : 0;
                        p[b] = (blocks_[fine].index[b] % 2) * (cells[b] / 2) + u;
                        p[c] = (blocks_[fine].index[c] % 2) * (cells[c] / 2) + v;
                        CoarseFineFaceMapping mapping{native_face(axis, p), {}};
                        for (std::size_t dv = 0; dv < 2; ++dv)
                            for (std::size_t du = 0; du < 2; ++du)
                            {
                                p[axis] = coarse_upper ? 0 : cells[axis];
                                p[b] = 2 * u + du;
                                p[c] = 2 * v + dv;
                                mapping.fine_faces.push_back(native_face(axis, p));
                            }
                        interface.faces.push_back(std::move(mapping));
                    }
                interfaces.emplace_back(std::move(interface));
            };
            for (std::size_t leaf = 0; leaf < blocks_.size(); ++leaf)
                for (std::size_t axis = 0; axis < 3; ++axis)
                {
                    const auto& block = blocks_[leaf];
                    auto neighbor = block;
                    ++neighbor.index[axis];
                    const auto count = product(config_.root_blocks[axis], level_scale(block.level));
                    const bool periodic = neighbor.index[axis] == count;
                    if (periodic)
                    {
                        if (!config_.periodic[axis])
                            continue;
                        neighbor.index[axis] = 0;
                    }
                    const auto found = covering_leaf(leaves, neighbor);
                    if (found != leaves.end())
                    {
                        const auto other = ordinals.at(*found);
                        if (found->level == block.level)
                        {
                            StructuredPatchInterface interface{
                                {leaf, static_cast<int>(2 * axis + 1)}, {other, static_cast<int>(2 * axis)}};
                            if (periodic)
                            {
                                MeshUtils::Vec3 translation{};
                                translation.component(axis) = -config_.extent[axis];
                                interface.periodic_translation = translation;
                            }
                            interfaces.emplace_back(std::move(interface));
                        }
                        else
                        {
                            if (periodic || found->level + 1 != block.level)
                                throw std::logic_error(
                                    "Cartesian AMR interface violates balance or periodic matching.");
                            coarse_fine(other, leaf, axis, false);
                        }
                    }
                    else
                    {
                        if (periodic)
                            throw std::logic_error("Cartesian AMR periodic partners require equal levels.");
                        const auto b = (axis + 1) % 3;
                        const auto c = (axis + 2) % 3;
                        for (std::size_t v = 0; v < 2; ++v)
                            for (std::size_t u = 0; u < 2; ++u)
                            {
                                AMRBlock fine{block.level + 1, {}};
                                for (std::size_t a = 0; a < 3; ++a)
                                    fine.index[a] = 2 * neighbor.index[a];
                                fine.index[b] += u;
                                fine.index[c] += v;
                                const auto other = ordinals.find(fine);
                                if (other == ordinals.end())
                                    throw std::logic_error(
                                        "Cartesian AMR leaf face lacks complete balanced neighbors.");
                                coarse_fine(leaf, other->second, axis, true);
                            }
                    }
                }
        });
    geometry_ = std::make_shared<const MultiRegionMesh>(std::move(regions), std::move(interfaces), InterfaceTolerance{},
        MultiRegionMesh::BoundaryNamePolicy::MergeMatchingNames);
    geometry_epoch_ = geometry_->geometry_epoch();
}

} // namespace SimpleFluid::Meshes::AMR
