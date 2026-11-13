/** @file ProductFamilyFragmentPackets.ipp
 * @brief Shared dictionary frames for a single Product authority wave.
 *
 * Included after ClosureScratch is defined. Frame views never outlive the
 * changed records or received packets they borrow.
 */
struct ProductProofPartLess
{
    using is_transparent = void;
    bool operator()(const AdaptiveFamilyProofPart& a, const AdaptiveFamilyProofPart& b) const { return *a < *b; }
    bool operator()(const AdaptiveFamilyProofPart& a, std::span<const char> b) const
    {
        return std::lexicographical_compare(a->begin(), a->end(), b.begin(), b.end());
    }
    bool operator()(std::span<const char> a, const AdaptiveFamilyProofPart& b) const
    {
        return std::lexicographical_compare(a.begin(), a.end(), b->begin(), b->end());
    }
};

class ProductWaveDictionary
{
    std::set<AdaptiveFamilyProofPart, ProductProofPartLess> parts_;

public:
    static std::size_t body_bytes(const AdaptiveBytes& value)
    {
        return persistent_ordered_detail::checked_sum(sizeof(AdaptiveBytes) + 4 * sizeof(void*), value.capacity());
    }

    [[nodiscard]] AdaptiveFamilyProofPart retain(std::span<const char> bytes, ClosureScratch& work)
    {
        if (const auto at = parts_.find(bytes); at != parts_.end())
            return *at;
        work.add(persistent_ordered_detail::checked_sum(
            bytes.size(), sizeof(AdaptiveBytes) + sizeof(AdaptiveFamilyProofPart) + 8 * sizeof(void*)));
        auto owner = std::make_shared<const AdaptiveBytes>(bytes.begin(), bytes.end());
        if (owner->capacity() != bytes.size())
            throw std::invalid_argument("Product wave dictionary capacity differs from admission.");
        parts_.insert(owner);
        return owner;
    }
};

struct ProductProofFrame
{
    int peer;
    std::span<const char> family, anchor;
    std::uint64_t revision, maximum;
    AdaptiveFamilyProofView proof;
};
void validate_product_proposal_route(const ProductProofFrame& row, int shard, int ranks, bool duplicate)
{
    if (row.peer < 0 || row.peer >= ranks || row.family.empty() || authority(row.family, ranks) != shard || duplicate ||
        (row.anchor.empty() && (!row.proof.record.empty() || row.revision || row.maximum)) ||
        row.revision > row.maximum)
        throw std::invalid_argument("Invalid or duplicate Product proposal delta.");
}
void validate_product_anchor_receipt(
    std::span<const char> anchor, int owner, std::uint64_t revision, const AdaptiveOwner& actual)
{
    if (actual.owner != owner || actual.revision != revision || !std::ranges::equal(anchor, actual.key))
        throw std::invalid_argument("Product proposal lacks actual anchor ownership and revision.");
}

struct ProductProofContext
{
    std::uint64_t topology, partition, transaction;
};
constexpr std::uint64_t product_fragment_magic = 0x534650524f4f4631ULL;

void encode_product_frames(std::span<ProductProofFrame> frames, std::uint64_t phase, ProductProofContext context,
    ClosureScratch& memory, std::vector<OwnerPacket>& packets)
{
    using persistent_ordered_detail::checked_product;
    using persistent_ordered_detail::checked_sum;
    std::sort(frames.begin(), frames.end(),
        [](const auto& a, const auto& b)
        {
            return a.peer == b.peer ? std::lexicographical_compare(
                                          a.family.begin(), a.family.end(), b.family.begin(), b.family.end())
                                    : a.peer < b.peer;
        });
    for (std::size_t first = 0; first < frames.size();)
    {
        std::size_t last = first + 1;
        while (last < frames.size() && frames[last].peer == frames[first].peer)
            ++last;
        ClosureScratch scratch{memory.budget};
        using Dictionary = std::map<AdaptiveFamilyProofPart, std::size_t, ProductProofPartLess>;
        Dictionary dictionary;
        std::vector<AdaptiveFamilyProofPart> ordered;
        auto bytes = 7 * sizeof(std::uint64_t);
        for (const auto& row : frames.subspan(first, last - first))
        {
            bytes = checked_sum(bytes, 6 * sizeof(std::uint64_t));
            bytes = checked_sum(bytes, row.family.size());
            bytes = checked_sum(bytes, row.anchor.size());
            bytes = checked_sum(bytes, row.proof.record.size());
            bytes = checked_sum(bytes, checked_product(row.proof.parts.size(), sizeof(std::uint64_t)));
            for (const auto& part : row.proof.parts)
            {
                if (!part)
                    throw std::invalid_argument("Missing Product proof dictionary part.");
                if (dictionary.contains(part))
                    continue;
                scratch.add(sizeof(Dictionary::value_type) + 4 * sizeof(void*));
                scratch.append(ordered);
                dictionary.emplace(part, ordered.size());
                ordered.push_back(part);
                bytes = checked_sum(bytes, checked_sum(sizeof(std::uint64_t), part->size()));
            }
        }
        memory.append(packets);
        memory.add(bytes);
        Writer writer;
        writer.data.reserve(bytes);
        if (writer.data.capacity() != bytes)
            throw std::invalid_argument("Product fragment packet capacity differs from admission.");
        writer.word(product_fragment_magic);
        writer.word(phase);
        writer.word(context.topology);
        writer.word(context.partition);
        writer.word(context.transaction);
        writer.word(ordered.size());
        for (const auto& part : ordered)
            writer.bytes(*part);
        writer.word(last - first);
        for (const auto& row : frames.subspan(first, last - first))
        {
            writer.bytes(row.family);
            writer.bytes(row.anchor);
            writer.word(row.revision);
            writer.word(row.maximum);
            writer.bytes(row.proof.record);
            writer.word(row.proof.parts.size());
            for (const auto& part : row.proof.parts)
                writer.word(dictionary.at(part));
        }
        if (writer.data.size() != bytes)
            throw std::logic_error("Product fragment packet size differs from admission.");
        packets.push_back({frames[first].peer, std::move(writer.data)});
        first = last;
    }
}

template<class Receive>
void decode_product_frames(const OwnerPacket& packet, std::uint64_t phase, ProductProofContext context,
    ProductWaveDictionary& shared, ClosureScratch& memory, Receive&& receive,
    const ProductFamilyElectionState* accepted_elected = nullptr)
{
    using persistent_ordered_detail::checked_product;
    Reader reader{packet.bytes};
    if (reader.word() != product_fragment_magic || reader.word() != phase || reader.word() != context.topology ||
        reader.word() != context.partition || reader.word() != context.transaction)
        throw std::invalid_argument("Foreign or stale Product family proof frame.");
    ClosureScratch scratch{memory.budget};
    const auto count = reader.word();
    if (count > (reader.data.size() - reader.offset) / sizeof(std::uint64_t))
        throw std::invalid_argument("Product proof dictionary count exceeds its frame.");
    std::vector<AdaptiveFamilyProofPart> parts;
    scratch.add(checked_product(count, sizeof(AdaptiveFamilyProofPart)));
    parts.reserve(count);
    if (parts.capacity() != count)
        throw std::invalid_argument("Product proof directory capacity differs from admission.");
    std::set<const void*> unique;
    for (std::size_t i = 0; i < count; ++i)
    {
        const auto bytes = reader.bytes_view();
        auto part = accepted_elected ? accepted_elected->elected_part(bytes) : nullptr;
        if (!part)
            part = shared.retain(bytes, memory);
        scratch.add(sizeof(const void*) + 4 * sizeof(void*));
        if (!unique.insert(part.get()).second)
            throw std::invalid_argument("Duplicate Product proof dictionary value.");
        parts.push_back(std::move(part));
    }
    const auto records = reader.word();
    if (records > (reader.data.size() - reader.offset) / (6 * sizeof(std::uint64_t)))
        throw std::invalid_argument("Product proof record count exceeds its frame.");
    for (std::size_t i = 0; i < records; ++i)
    {
        const auto family = reader.bytes_view(), anchor = reader.bytes_view();
        const auto revision = reader.word(), maximum = reader.word();
        const auto record = reader.bytes_view();
        const auto selectors = reader.word();
        if (selectors > parts.size() || selectors > (reader.data.size() - reader.offset) / sizeof(std::uint64_t) ||
            (record.empty() && selectors))
            throw std::invalid_argument("Invalid Product proof selector count.");
        ClosureScratch row_memory{memory.budget};
        row_memory.add(checked_product(selectors, sizeof(AdaptiveFamilyProofPart)));
        std::vector<AdaptiveFamilyProofPart> local;
        local.reserve(selectors);
        if (local.capacity() != selectors)
            throw std::invalid_argument("Product local selector capacity differs from admission.");
        for (std::size_t j = 0; j < selectors; ++j)
        {
            const auto id = reader.word();
            if (id >= parts.size())
                throw std::invalid_argument("Foreign Product proof dictionary selector.");
            local.push_back(parts[id]);
        }
        receive(ProductProofFrame{packet.peer, family, anchor, revision, maximum, {record, local}});
    }
    reader.done();
}
