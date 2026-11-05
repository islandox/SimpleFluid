#include "CartesianSourceManifests.hh"
#include "BudgetedOwnerExchange.hh"
#include "OwnedSupportMembership.hh"
#include "SupportRegistry.hh"
#include <algorithm>
#include <bit>
#include <cfenv>
#include <cmath>
#include <limits>
#include <tuple>
namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
namespace
{
using cartesian_detail::add;
using cartesian_detail::product;
using Allocation = cartesian_detail::Allocation;
constexpr std::uint64_t manifest_magic = 0x534643534d414e31ULL;
constexpr std::size_t manifest_bytes = 264;
class MetadataEnvironment
{
    std::fenv_t caller_{};

public:
    MetadataEnvironment()
    {
        if (std::feholdexcept(&caller_))
            throw std::invalid_argument("Cannot isolate source manifest metadata.");
    }
    ~MetadataEnvironment() { (void) std::fesetenv(&caller_); }
};
std::array<char, 152> grid_descriptor(const AMRGrid& grid)
{
    std::array<char, 152> result{};
    std::size_t at = 0;
    const auto word = [&](std::uint64_t value)
    {
        for (unsigned b = 0; b < 8; ++b)
            result[at++] = char(value >> (b * 8));
    };
    word(0x5346434152544b31ULL);
    for (unsigned axis = 0; axis < 3; ++axis)
    {
        word(grid.root_blocks[axis]);
        word(grid.block_cells[axis]);
        word(std::bit_cast<std::uint64_t>(grid.origin[axis]));
        word(std::bit_cast<std::uint64_t>(grid.extent[axis]));
        word(grid.periodic[axis]);
    }
    word(grid.max_level);
    word(grid.region_id);
    word(grid.material_id);
    return result;
}
std::size_t host_size(std::uint64_t value)
{
    if (value > std::numeric_limits<std::size_t>::max())
        throw std::invalid_argument("Source manifest count is unrepresentable.");
    return std::size_t(value);
}
std::uint64_t total(const AMRGrid& grid, unsigned level, unsigned axis)
{
    if (level > grid.max_level || level >= 63)
        throw std::invalid_argument("Invalid source manifest axis level.");
    return product(product(grid.root_blocks[axis], grid.block_cells[axis]), std::uint64_t(1) << level);
}
ContactHaloContext read_context(Reader& reader)
{
    return {reader.word(), reader.word(), reader.word(), reader.word(), reader.word()};
}
void write_context(Writer& writer, ContactHaloContext context)
{
    for (auto value : {context.epoch, context.topology, context.partition, context.round, context.transaction})
        writer.word(value);
}
} // namespace
CartesianIncidenceRange CartesianSourceManifestSet::Source::range(std::size_t index) const
{
    if (!active_ranges_.empty())
        return active_ranges_.at(index);
    if (index >= ranges_)
        throw std::out_of_range("Invalid source manifest range index.");
    Reader reader{std::span<const char>(identity_).subspan(ranges_offset_ + 56 * index, 56)};
    CartesianIncidenceRange value;
    value.level = unsigned(reader.word());
    for (auto& n : value.lower)
        n = reader.word();
    for (auto& n : value.upper)
        n = reader.word();
    return value;
}
double CartesianSourceManifestSet::Source::Axis::value(std::size_t index) const
{
    if (index >= size())
        throw std::out_of_range("Invalid source manifest axis index.");
    Reader reader{words.subspan(index * 8, 8)};
    return std::bit_cast<double>(reader.word());
}
CartesianSourceManifestSet::Source::Axis CartesianSourceManifestSet::Source::axis(std::size_t index) const
{
    if (index >= axes_)
        throw std::out_of_range("Invalid source manifest axis selector.");
    Reader reader{std::span<const char>(identity_).subspan(axes_offset_)};
    for (std::size_t i = 0;; ++i)
    {
        const auto level = reader.word(), direction = reader.word(), first = reader.word(),
                   count = host_size(reader.word());
        const auto bytes = product(count, 8);
        const auto values = reader.data.subspan(reader.offset, bytes);
        if (i == index)
            return {unsigned(level), unsigned(direction), first, values};
        reader.offset += bytes;
    }
}
std::uint64_t CartesianSourceManifestSet::Source::revision(const CartesianCellAncestry& cell) const
{
    bool present = false;
    for (std::size_t i = 0; i < range_count(); ++i)
    {
        const auto r = range(i);
        bool inside = r.level == cell.level;
        for (unsigned a = 0; a < 3; ++a)
            inside = inside && cell.index[a] >= r.lower[a] && cell.index[a] < r.upper[a];
        present = present || inside;
    }
    if (!present)
        throw std::invalid_argument("Foreign source manifest cell revision.");
    std::size_t lower = 0, upper = revisions_;
    while (lower < upper)
    {
        const auto middle = lower + (upper - lower) / 2;
        Reader reader{std::span<const char>(identity_).subspan(revisions_offset_ + 40 * middle, 40)};
        CartesianCellAncestry key;
        key.level = unsigned(reader.word());
        for (auto& v : key.index)
            v = reader.word();
        if (std::tie(key.level, key.index) < std::tie(cell.level, cell.index))
            lower = middle + 1;
        else
            upper = middle;
    }
    if (lower < revisions_)
    {
        Reader reader{std::span<const char>(identity_).subspan(revisions_offset_ + 40 * lower, 40)};
        CartesianCellAncestry key;
        key.level = unsigned(reader.word());
        for (auto& v : key.index)
            v = reader.word();
        const auto revision = reader.word();
        if (key.level == cell.level && key.index == cell.index)
            return revision;
    }
    return base_revision_;
}
void CartesianSourceManifestSet::Source::authenticate(
    const AMRGrid& grid, ContactHaloContext context, int owner, std::size_t cells, std::uint64_t version)
{
    owner_ = owner;
    cells_ = cells;
    version_ = version;
    const MetadataEnvironment environment;
    if (identity_.empty())
    {
        if (cells || version)
            throw std::invalid_argument("Nonempty source inventory has no immutable identity.");
        return;
    }
    Reader reader{identity_};
    if (reader.word() != 0x5346434953524331ULL)
        throw std::invalid_argument("Invalid source identity family.");
    evaluation_ = {reader.word(), reader.word(), reader.word(), reader.word()};
    const auto expected = grid_descriptor(grid);
    if (!std::ranges::equal(reader.bytes_view(), expected))
        throw std::invalid_argument("Source manifest differs from trusted grid contract.");
    const auto domain = reader.bytes_view();
    domain_offset_ = std::size_t(domain.data() - identity_.data());
    domain_size_ = domain.size();
    if (domain.empty() || reader.word() != std::uint64_t(owner) || reader.word() != version ||
        read_context(reader) != context)
        throw std::invalid_argument("Stale or foreign full source identity context.");
    base_revision_ = reader.word();
    if (host_size(reader.word()) != cells)
        throw std::invalid_argument("Source inventory count differs from immutable identity.");
    ranges_ = host_size(reader.word());
    ranges_offset_ = reader.offset;
    if (ranges_ > (reader.data.size() - reader.offset) / 56)
        throw std::invalid_argument("Truncated source manifest ranges.");
    std::size_t counted = 0;
    for (std::size_t i = 0; i < ranges_; ++i)
    {
        const auto level = reader.word();
        if (level > grid.max_level || level >= 63)
            throw std::invalid_argument("Invalid source range level.");
        std::array<std::uint64_t, 3> lower{}, upper{};
        for (auto& n : lower)
            n = reader.word();
        for (auto& n : upper)
            n = reader.word();
        std::size_t volume = 1;
        for (unsigned a = 0; a < 3; ++a)
        {
            if (lower[a] >= upper[a] || upper[a] > total(grid, unsigned(level), a))
                throw std::invalid_argument("Invalid source range bounds.");
            volume = product(volume, host_size(upper[a] - lower[a]));
        }
        for (std::size_t old = 0; old < i; ++old)
        {
            Reader previous{std::span<const char>(identity_).subspan(ranges_offset_ + 56 * old, 56)};
            const auto old_level = unsigned(previous.word());
            std::array<std::uint64_t, 3> old_lower{}, old_upper{};
            for (auto& n : old_lower)
                n = previous.word();
            for (auto& n : old_upper)
                n = previous.word();
            const bool current_is_fine = level >= old_level;
            const auto shift = current_is_fine ? unsigned(level - old_level) : unsigned(old_level - level);
            const auto& coarse_lower = current_is_fine ? old_lower : lower;
            const auto& coarse_upper = current_is_fine ? old_upper : upper;
            const auto& fine_lower = current_is_fine ? lower : old_lower;
            const auto& fine_upper = current_is_fine ? upper : old_upper;
            bool overlap = true;
            for (unsigned a = 0; a < 3; ++a)
                overlap = overlap && coarse_lower[a] < (((fine_upper[a] - 1) >> shift) + 1) &&
                          (fine_lower[a] >> shift) < coarse_upper[a];
            if (overlap)
                throw std::invalid_argument("Original source ranges are not a disjoint ancestry antichain.");
        }
        counted = add(counted, volume);
    }
    if (counted != cells)
        throw std::invalid_argument("Incomplete source range inventory.");
    axes_ = host_size(reader.word());
    axes_offset_ = reader.offset;
    if (axes_ > (reader.data.size() - reader.offset) / 32)
        throw std::invalid_argument("Truncated source manifest axes.");
    for (std::size_t i = 0; i < axes_; ++i)
    {
        const auto level = reader.word(), axis = reader.word(), first = reader.word(), count = host_size(reader.word());
        if (level > grid.max_level || axis >= 3 || !count || count > (reader.data.size() - reader.offset) / 8 ||
            first > total(grid, unsigned(level), unsigned(axis)) ||
            count - 1 > total(grid, unsigned(level), unsigned(axis)) - first)
            throw std::invalid_argument("Invalid occupied source axis interval.");
        double previous = 0;
        for (std::size_t j = 0; j < count; ++j)
        {
            const auto value = std::bit_cast<double>(reader.word());
            if (!std::isfinite(value) || (j && value <= previous))
                throw std::invalid_argument("Invalid actual rounded source axis facts.");
            previous = value;
        }
    }
    revisions_ = host_size(reader.word());
    revisions_offset_ = reader.offset;
    if (revisions_ > (reader.data.size() - reader.offset) / 40)
        throw std::invalid_argument("Truncated source revision overrides.");
    CartesianCellAncestry previous_revision;
    for (std::size_t i = 0; i < revisions_; ++i)
    {
        const auto level = reader.word();
        if (level > grid.max_level)
            throw std::invalid_argument("Invalid source revision level.");
        CartesianCellAncestry cell;
        cell.level = unsigned(level);
        for (unsigned a = 0; a < 3; ++a)
            if ((cell.index[a] = reader.word()) >= total(grid, unsigned(level), a))
                throw std::invalid_argument("Invalid source revision cell.");
        if (i && !(std::tie(previous_revision.level, previous_revision.index) < std::tie(cell.level, cell.index)))
            throw std::invalid_argument("Source revision overrides are not strictly ordered unique keys.");
        bool member = false;
        for (std::size_t r = 0; r < ranges_; ++r)
        {
            const auto box = range(r);
            bool inside = box.level == cell.level;
            for (unsigned a = 0; a < 3; ++a)
                inside = inside && box.lower[a] <= cell.index[a] && cell.index[a] < box.upper[a];
            member = member || inside;
        }
        if (!member)
            throw std::invalid_argument("Foreign source revision override outside original occupied ranges.");
        previous_revision = cell;
        (void) reader.word();
    }
    reader.done();
}
bool CartesianSourceManifestSet::has_explicit_support(std::span<const char> key) const noexcept
{
    for (const auto& cell : explicit_cells_)
        for (const auto& member : cell.members)
            if (std::ranges::equal(member.support, key))
                return true;
    return false;
}
bool CartesianSourceManifestSet::native_member(const RegistryMember& member, std::span<const char> domain) const
{
    const auto* ancestry = std::get_if<CartesianCellAncestry>(&member.cell.ancestry);
    if (!ancestry || member.cell.region != grid_.region_id || member.surface >= 6)
        return false;
    for (const auto& source : sources_)
        if (source.owner() == member.owner && source.cell_count() && std::ranges::equal(source.domain(), domain))
            for (std::size_t i = 0; i != source.range_count(); ++i)
            {
                const auto range = source.range(i);
                bool inside = range.level == ancestry->level;
                for (unsigned a = 0; a != 3; ++a)
                    inside = inside && range.lower[a] <= ancestry->index[a] && ancestry->index[a] < range.upper[a];
                if (inside)
                    return source.revision(*ancestry) == member.revision;
            }
    return false;
}
CartesianSourceManifestSet::~CartesianSourceManifestSet()
{
    decltype(sources_){}.swap(sources_);
    decltype(explicit_cells_){}.swap(explicit_cells_);
    local_.reset();
    decltype(leases_){}.swap(leases_);
    if (budget_)
        budget_->used -= charge_;
}
void CartesianSourceManifestSet::decode_inventory(std::vector<Source>& sources, const AMRGrid& grid,
    ContactHaloContext context, int owner, std::uint64_t version, std::size_t cells, std::span<const char> bytes,
    Allocation& memory)
{
    Reader reader{bytes};
    if (reader.word() != 0x534643494e563031ULL || reader.word() != 1 || read_context(reader) != context ||
        reader.word() != std::uint64_t(owner) || reader.word() != version || host_size(reader.word()) != cells)
        throw std::invalid_argument("Stale full source inventory binding.");
    const auto batches = host_size(reader.word());
    if (reader.word() || batches > (reader.data.size() - reader.offset) / 24)
        throw std::invalid_argument("Malformed source inventory batch count.");
    const auto first = sources.size();
    memory.reserve(sources, add(first, std::max(std::size_t(1), batches)));
    std::size_t counted = 0;
    for (std::size_t batch = 0; batch < batches; ++batch)
    {
        const auto identity = reader.bytes_view();
        const auto active_cells = host_size(reader.word()), range_count = host_size(reader.word());
        if (!active_cells || !range_count || range_count > (reader.data.size() - reader.offset) / 56)
            throw std::invalid_argument("Empty or truncated active source batch.");
        Reader original{identity};
        if (original.word() != 0x5346434953524331ULL)
            throw std::invalid_argument("Unknown native source in current inventory.");
        for (unsigned i = 0; i != 4; ++i)
            (void) original.word();
        (void) original.bytes_view();
        (void) original.bytes_view();
        if (original.word() != std::uint64_t(owner))
            throw std::invalid_argument("Foreign original source owner in inventory.");
        const auto source_version = original.word();
        const auto source_context = read_context(original);
        (void) original.word();
        const auto original_cells = host_size(original.word());
        sources.emplace_back();
        auto& source = sources.back();
        memory.copy(source.identity_, identity);
        source.authenticate(grid, source_context, owner, original_cells, source_version);
        memory.reserve(source.active_ranges_, range_count);
        std::size_t batch_cells = 0;
        for (std::size_t range_index = 0; range_index < range_count; ++range_index)
        {
            CartesianIncidenceRange range;
            const auto level = reader.word();
            if (level > grid.max_level || level >= 63)
                throw std::invalid_argument("Invalid active source selector level.");
            range.level = unsigned(level);
            for (auto& value : range.lower)
                value = reader.word();
            for (auto& value : range.upper)
                value = reader.word();
            std::size_t volume = 1, covered = 0;
            for (unsigned axis = 0; axis != 3; ++axis)
            {
                if (range.lower[axis] >= range.upper[axis] || range.upper[axis] > total(grid, range.level, axis))
                    throw std::invalid_argument("Invalid active source selector bounds.");
                volume = product(volume, host_size(range.upper[axis] - range.lower[axis]));
            }
            // Sum exact ancestry intersections. Coalescing may span several
            // adjacent original boxes; no geometry or neighbor inference occurs.
            for (std::size_t old = 0; old < source.ranges_; ++old)
            {
                Reader old_reader{
                    std::span<const char>(source.identity_).subspan(source.ranges_offset_ + 56 * old, 56)};
                CartesianIncidenceRange prior;
                prior.level = unsigned(old_reader.word());
                for (auto& value : prior.lower)
                    value = old_reader.word();
                for (auto& value : prior.upper)
                    value = old_reader.word();
                if (prior.level != range.level)
                    continue;
                std::size_t intersection = 1;
                for (unsigned axis = 0; axis != 3; ++axis)
                {
                    const auto lower = std::max(prior.lower[axis], range.lower[axis]);
                    const auto upper = std::min(prior.upper[axis], range.upper[axis]);
                    intersection = product(intersection, upper > lower ? host_size(upper - lower) : 0);
                }
                covered = add(covered, intersection);
            }
            if (covered != volume)
                throw std::invalid_argument("Active selector exceeds authenticated original source membership.");
            for (std::size_t prior_source = 0; prior_source < sources.size(); ++prior_source)
                for (const auto& prior : sources[prior_source].active_ranges_)
                {
                    const auto& coarse = prior.level <= range.level ? prior : range;
                    const auto& fine = prior.level <= range.level ? range : prior;
                    const auto shift = fine.level - coarse.level;
                    bool overlap = true;
                    for (unsigned axis = 0; axis != 3; ++axis)
                        overlap = overlap && coarse.lower[axis] < (((fine.upper[axis] - 1) >> shift) + 1) &&
                                  (fine.lower[axis] >> shift) < coarse.upper[axis];
                    if (overlap)
                        throw std::invalid_argument("Duplicate complete active source membership.");
                }
            source.active_ranges_.push_back(range);
            batch_cells = add(batch_cells, volume);
        }
        if (batch_cells != active_cells)
            throw std::invalid_argument("Incomplete active source batch selectors.");
        source.cells_ = active_cells;
        counted = add(counted, active_cells);
    }
    if (counted != cells)
        throw std::invalid_argument("Incomplete current source owner inventory.");
    if (!batches)
    {
        sources.emplace_back();
        sources.back().authenticate(grid, context, owner, 0, 0);
    }
    reader.done();
}
void CartesianSourceManifestSet::decode_wire(const AMRGrid& grid, ContactHaloContext context, std::uint64_t generation,
    std::span<const OwnerPacket> records, int ranks, std::vector<Source>& sources,
    std::vector<ExplicitCell>& explicit_cells, Allocation& memory)
{
    if (ranks <= 0 || records.size() != std::size_t(ranks) || !sources.empty())
        throw std::invalid_argument("Missing complete source owner receipt, including empty owners.");
    Allocation scratch(memory.budget);
    std::vector<unsigned char> seen;
    scratch.reserve(seen, std::size_t(ranks));
    seen.resize(ranks, 0);
    for (const auto& packet : records)
    {
        if (packet.peer < 0 || packet.peer >= ranks || seen[packet.peer]++)
            throw std::invalid_argument("Duplicate or foreign source owner receipt.");
        Reader reader{packet.bytes};
        if (reader.word() != manifest_magic)
            throw std::invalid_argument("Invalid source manifest family.");
        const auto schema = reader.word();
        if ((schema != 1 && schema != 2 && schema != 3) || read_context(reader) != context ||
            reader.word() != std::uint64_t(packet.peer))
            throw std::invalid_argument("Stale source manifest context or owner.");
        const auto cells = host_size(reader.word());
        const auto version = reader.word(), empty = reader.word();
        if (empty > 1 || bool(empty) != (cells == 0))
            throw std::invalid_argument("Invalid empty source completion.");
        if (reader.word() != generation)
            throw std::invalid_argument("Stale complete source generation.");
        if (schema >= 2 && (!version || version != generation))
            throw std::invalid_argument("Current source inventory version differs from its complete generation.");
        const auto descriptor = grid_descriptor(grid);
        if (!std::ranges::equal(reader.bytes_view(), descriptor))
            throw std::invalid_argument("Source completion differs from actual trusted provider grid.");
        const auto identity = reader.bytes_view();
        if (schema >= 2)
            decode_inventory(sources, grid, context, packet.peer, version, cells, identity, memory);
        else
        {
            memory.append(sources);
            sources.emplace_back();
            memory.copy(sources.back().identity_, identity);
            sources.back().authenticate(grid, context, packet.peer, cells, version);
        }
        if (schema == 3)
        {
            const auto count = host_size(reader.word());
            if (count > (reader.data.size() - reader.offset) / 24)
                throw std::invalid_argument("Truncated explicit source contributor completion.");
            memory.reserve(explicit_cells, add(explicit_cells.size(), count));
            for (std::size_t i = 0; i != count; ++i)
            {
                const auto encoded = reader.bytes_view();
                explicit_cells.emplace_back();
                auto& cell = explicit_cells.back();
                const auto before = memory.budget.used;
                try
                {
                    cell.key = decode_bounded_key(encoded, memory.budget);
                }
                catch (...)
                {
                    memory.bytes = add(memory.bytes, memory.budget.used - before);
                    throw;
                }
                memory.bytes = add(memory.bytes, memory.budget.used - before);
                cell.owner = packet.peer;
                cell.revision = reader.word();
                const auto members = host_size(reader.word());
                if (members != 6 || members > (reader.data.size() - reader.offset) / 16)
                    throw std::invalid_argument("Explicit Cartesian contributor has incomplete native face ordinals.");
                for (std::size_t previous = 0; previous + 1 < explicit_cells.size(); ++previous)
                    if (explicit_cells[previous].key == cell.key)
                        throw std::invalid_argument("Repeated explicit source cell contributor.");
                const auto* ancestry = std::get_if<CartesianCellAncestry>(&cell.key.ancestry);
                if (!ancestry || cell.key.region != grid.region_id || ancestry->level > grid.max_level)
                    throw std::invalid_argument("Foreign explicit Cartesian contributor key.");
                for (unsigned a = 0; a != 3; ++a)
                    if (ancestry->index[a] >= total(grid, ancestry->level, a))
                        throw std::invalid_argument("Explicit source cell lies outside the actual trusted grid.");
                for (std::size_t previous = 0; previous + 1 < explicit_cells.size(); ++previous)
                {
                    const auto* old = std::get_if<CartesianCellAncestry>(&explicit_cells[previous].key.ancestry);
                    if (!old)
                        continue;
                    const auto& coarse = old->level <= ancestry->level ? *old : *ancestry;
                    const auto& fine = old->level <= ancestry->level ? *ancestry : *old;
                    const auto shift = fine.level - coarse.level;
                    bool overlap = true;
                    for (unsigned a = 0; a != 3; ++a)
                        overlap = overlap && coarse.index[a] == (fine.index[a] >> shift);
                    if (overlap)
                        throw std::invalid_argument("Explicit source cells are not an ancestry antichain.");
                }

                for (const auto& source : sources)
                    if (source.cell_count())
                        for (std::size_t r = 0; r != source.range_count(); ++r)
                        {
                            const auto box = source.range(r);
                            const auto shift =
                                box.level > ancestry->level ? box.level - ancestry->level : ancestry->level - box.level;
                            bool overlap = true;
                            for (unsigned a = 0; a != 3; ++a)
                            {
                                if (box.level <= ancestry->level)
                                    overlap = overlap && box.lower[a] <= (ancestry->index[a] >> shift) &&
                                              (ancestry->index[a] >> shift) < box.upper[a];
                                else
                                    overlap = overlap && (box.lower[a] >> shift) <= ancestry->index[a] &&
                                              ancestry->index[a] < (((box.upper[a] - 1) >> shift) + 1);
                            }
                            if (overlap)
                                throw std::invalid_argument("Explicit/native source inventories overlap.");
                        }
                memory.reserve(cell.members, members);
                std::span<const char> domain;
                for (std::size_t ordinal = 0; ordinal != members; ++ordinal)
                {
                    const auto support = reader.bytes_view();
                    const auto encoded_size = host_size(reader.word());
                    Reader key{support};
                    if (!encoded_size || key.word() != 0x5346434152543631ULL)
                        throw std::invalid_argument("Invalid explicit Cartesian support header.");
                    const auto member_domain = key.bytes_view();
                    if (member_domain.empty() || (ordinal && !std::ranges::equal(domain, member_domain)) ||
                        key.word() != grid.region_id)
                        throw std::invalid_argument("Foreign explicit support domain or region.");
                    domain = member_domain;
                    const auto axis = key.word(), level = key.word(), plane = key.word();
                    if (axis >= 3 || level > ancestry->level)
                        throw std::invalid_argument("Invalid explicit support axis/level.");
                    bool valid_plane = false;
                    for (unsigned side = 0; side != 2; ++side)
                    {
                        auto boundary = ancestry->index[axis] + side;
                        if (grid.periodic[axis] && boundary == total(grid, ancestry->level, unsigned(axis)))
                            boundary = 0;
                        unsigned reduced = ancestry->level;
                        while (reduced && boundary % 2 == 0)
                        {
                            boundary /= 2;
                            --reduced;
                        }
                        valid_plane = valid_plane || (reduced == level && boundary == plane);
                    }
                    if (!valid_plane)
                        throw std::invalid_argument("Explicit support plane differs from its full cell key.");
                    for (unsigned a = 0; a != 3; ++a)
                        if (a != axis && key.word() != (ancestry->index[a] >> (ancestry->level - unsigned(level))))
                            throw std::invalid_argument("Explicit support transverse identity differs.");
                    key.done();
                    for (const auto& previous : cell.members)
                        if (std::ranges::equal(previous.support, support))
                            throw std::invalid_argument("Duplicate explicit face ordinal/key.");
                    cell.members.emplace_back();
                    memory.copy(cell.members.back().support, support);
                    cell.members.back().encoded_size = encoded_size;
                }
            }
        }
        reader.done();
    }
    for (const auto& cell : explicit_cells)
    {
        const auto& ancestry = std::get<CartesianCellAncestry>(cell.key.ancestry);
        for (const auto& source : sources)
            if (source.cell_count())
                for (std::size_t r = 0; r != source.range_count(); ++r)
                {
                    const auto box = source.range(r);
                    const auto shift =
                        box.level > ancestry.level ? box.level - ancestry.level : ancestry.level - box.level;
                    bool overlap = true;
                    for (unsigned a = 0; a != 3; ++a)
                        if (box.level <= ancestry.level)
                            overlap = overlap && box.lower[a] <= (ancestry.index[a] >> shift) &&
                                      (ancestry.index[a] >> shift) < box.upper[a];
                        else
                            overlap = overlap && (box.lower[a] >> shift) <= ancestry.index[a] &&
                                      ancestry.index[a] < (((box.upper[a] - 1) >> shift) + 1);
                    if (overlap)
                        throw std::invalid_argument("Complete native/explicit inventories overlap.");
                }
    }
}
void CartesianSourceManifestSet::validate_untrusted_wire(const AMRGrid& grid, ContactHaloContext context,
    std::uint64_t generation, std::span<const OwnerPacket> records, int ranks, Budget& budget)
{
    Allocation memory(budget);
    std::vector<Source> sources;
    std::vector<ExplicitCell> explicit_cells;
    decode_wire(grid, context, generation, records, ranks, sources, explicit_cells, memory);
}
std::shared_ptr<const CartesianSourceManifestSet> CartesianSourceManifestSet::exchange(
    const CartesianSourceManifestAdmission& admission, Teuchos::RCP<const Teuchos::Comm<int>> comm, Budget& budget,
    std::shared_ptr<Budget> keeper, std::shared_ptr<cartesian_detail::ArenaCredits> arena)
{
    if (comm.is_null())
        throw std::invalid_argument("Source manifests require a communicator.");
    cartesian_detail::check_keeper(budget, keeper);
    Allocation retained(budget), transport(budget);
    if (!arena || arena->budget != &budget)
        throw std::invalid_argument("Source manifests require their explicit arena.");
    cartesian_detail::CreditConstruction credits(arena);
    std::shared_ptr<CartesianSourceManifestSet> result;
    std::vector<OwnerPacket> outgoing;
    collective(*comm,
        [&]
        {
            if (admission.inventory_ && (admission.inventory_->owner() != comm->getRank() ||
                                            admission.inventory_->context() != admission.context_ ||
                                            admission.inventory_->cell_count() != admission.cells_ ||
                                            admission.inventory_->version() != admission.generation_))
                throw std::invalid_argument("Native source witness differs from closed owner inventory.");
            if (!admission.inventory_)
                throw std::invalid_argument("Missing native owner construction witness.");
            retained.retain(sizeof(CartesianSourceManifestSet) + 128);
            result = std::shared_ptr<CartesianSourceManifestSet>(new CartesianSourceManifestSet);
            result->grid_ = admission.grid_;
            result->context_ = admission.context_;
            result->generation_ = admission.generation_;
            result->local_ = admission.inventory_;
            result->owners_ = comm->getSize();
            if (result->local_)
                credits.retain(result->local_, result->local_->allocation_budget(), result->local_->metadata_bytes(),
                    result->local_->has_arena_keeper(), result->leases_, retained);
            result->local_->visit_dependencies(
                [&](const auto& owner, const Budget* origin, std::size_t bytes, bool keep)
                {
                    using E = typename std::remove_cvref_t<decltype(owner)>::element_type;
                    const std::shared_ptr<const std::remove_const_t<E>> fixed = owner;
                    credits.retain(fixed, origin, bytes, keep, result->leases_, retained);
                });
            transport.reserve(outgoing, std::size_t(comm->getSize()));
            const auto identity = std::span<const char>(admission.inventory_->identity());
            const auto version = admission.inventory_->version();
            const auto descriptor = grid_descriptor(admission.grid_);
            for (int peer = 0; peer < comm->getSize(); ++peer)
            {
                auto size = add(add(manifest_bytes, identity.size()), 8);
                std::size_t explicit_count = 0;
                if (admission.owned_)
                    for (const auto& [slot, cell] : admission.owned_->cells_)
                        if (!admission.inventory_->contains(admission.owned_->pools_->cells.get(slot)))
                        {
                            ++explicit_count;
                            size = add(size, add(24, key_wire_size(admission.owned_->pools_->cells.get(slot))));
                            for (const auto& member : cell->members)
                                size =
                                    add(size, add(16, admission.owned_->pools_->supports.get(member.support).size()));
                        }
                transport.retain(size);
                Writer writer;
                writer.data.reserve(size);
                if (writer.data.capacity() != size)
                    throw std::invalid_argument("Source manifest frame capacity differs.");
                writer.word(manifest_magic);
                writer.word(3);
                write_context(writer, admission.context_);
                writer.word(comm->getRank());
                writer.word(admission.cells_);
                writer.word(version);
                writer.word(admission.cells_ == 0);
                writer.word(admission.generation_);
                writer.bytes(descriptor);
                writer.bytes(identity);
                writer.word(explicit_count);
                if (admission.owned_)
                    for (const auto& [slot, cell] : admission.owned_->cells_)
                        if (!admission.inventory_->contains(admission.owned_->pools_->cells.get(slot)))
                        {
                            Allocation key_memory(budget);
                            const auto& key = admission.owned_->pools_->cells.get(slot);
                            key_memory.retain(cartesian_detail::insertion_growth(key_wire_size(key)));
                            const auto encoded = encode_adaptive_key(key);
                            writer.bytes(encoded);
                            writer.word(cell->revision);
                            writer.word(cell->members.size());
                            for (const auto& member : cell->members)
                            {
                                writer.bytes(admission.owned_->pools_->supports.get(member.support));
                                writer.word(member.encoded_size);
                            }
                        }
                if (writer.data.size() != size)
                    throw std::logic_error("Source manifest frame size differs.");
                outgoing.push_back({peer, std::move(writer.data)});
            }
        });
    OwnerExchange incoming;
    const auto before = budget.used;
    try
    {
        incoming = consume_budgeted_owner_packets(std::move(outgoing), comm, budget, transport.bytes);
    }
    catch (...)
    {
        if (budget.used >= before)
            transport.bytes = add(transport.bytes, budget.used - before);
        else
            transport.bytes -= before - budget.used;
        throw;
    }
    if (budget.used >= before)
        transport.bytes = add(transport.bytes, budget.used - before);
    else
        transport.bytes -= before - budget.used;
    collective(*comm,
        [&]
        {
            decode_wire(admission.grid_, admission.context_, admission.generation_, incoming.packets, comm->getSize(),
                result->sources_, result->explicit_cells_, retained);
        });
    result->budget_ = &budget;
    result->keeper_ = std::move(keeper);
    result->charge_ = std::exchange(retained.bytes, 0);
    credits.committed = true;
    return result;
}
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
