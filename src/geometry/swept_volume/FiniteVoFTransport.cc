#include "geometry/swept_volume/FiniteVoFTransport.tcc"
namespace SimpleFluid::geometry::swept_volume
{
void Work::retain_trajectories(const std::shared_ptr<const OrdinaryTrajectorySnapshot>& snapshot)
{
    if (!snapshot)
        throw std::invalid_argument("Finite VoF sweep has no trajectory snapshot.");
    for (const auto& owner : trajectory_owners)
        if (owner == snapshot)
            return;
    charge(snapshot->storage_bytes());
    snapshot->retain_provenance(allocations, bytes, limits.byte_budget);
    if (trajectory_owners.size() == trajectory_owners.capacity())
    {
        const auto count = trajectory_owners.capacity() ? 2 * trajectory_owners.capacity() : 1;
        if (count > std::numeric_limits<std::size_t>::max() / sizeof(snapshot))
            throw PLIC::GeometryError(PLIC::GeometryErrorCode::ResourceLimit,
                "Finite VoF trajectory owner capacity overflows.");
        charge(count * sizeof(snapshot));
        trajectory_owners.reserve(count);
        charge((trajectory_owners.capacity() - count) * sizeof(snapshot));
    }
    trajectory_owners.push_back(snapshot);
}
std::shared_ptr<const OrdinaryTrajectorySnapshot> Work::copy_face_trajectories(
    std::span<const std::uint64_t> nodes, const OrdinaryTrajectorySnapshot::TraceMap& traces)
{
    using Snapshot = OrdinaryTrajectorySnapshot;
    charge(sizeof(Snapshot));
    charge_items(nodes.size(), sizeof(Snapshot::Entry));
    const auto old_capacity = trajectory_owners.capacity();
    auto owner_capacity = old_capacity;
    if (trajectory_owners.size() == old_capacity)
    {
        if (old_capacity > std::numeric_limits<std::size_t>::max() / 2)
            throw PLIC::GeometryError(PLIC::GeometryErrorCode::ResourceLimit,
                "Finite VoF trajectory owner capacity overflows.");
        owner_capacity = old_capacity ? 2 * old_capacity : 1;
        charge_items(owner_capacity, sizeof(std::shared_ptr<const Snapshot>));
    }
    // Preserve duplicate -> lookup -> mode rejection in face order without
    // quadratic scans when the configured face/simplex limit is large.
    charge_items(nodes.size(), sizeof(std::uint64_t) + 4 * sizeof(void*));
    {
        std::set<std::uint64_t> unique;
        for (const auto node : nodes)
        {
            if (!unique.insert(node).second)
                throw std::invalid_argument("Finite VoF canonical face repeats a node.");
            const auto& trace = traces.at(node);
            if (trace.first.arithmetic_mode != arithmetic_mode || trace.second.arithmetic_mode != arithmetic_mode)
                throw std::invalid_argument("Finite VoF sweep trajectory arithmetic mode differs.");
        }
    }
    for (const auto node : nodes)
    {
        const auto& trace = traces.at(node);
        allocations.retain_point_provenance(trace.first, bytes, limits.byte_budget);
        allocations.retain_point_provenance(trace.second, bytes, limits.byte_budget);
    }
    auto result = std::shared_ptr<Snapshot>(new Snapshot(std::vector<Snapshot::Entry>{}));
    result->entries_.reserve(nodes.size());
    charge_items(result->entries_.capacity() - nodes.size(), sizeof(Snapshot::Entry));
    if (owner_capacity != old_capacity)
    {
        trajectory_owners.reserve(owner_capacity);
        charge_items(trajectory_owners.capacity() - owner_capacity, sizeof(std::shared_ptr<const Snapshot>));
    }
    static_assert(std::is_nothrow_copy_constructible_v<Snapshot::Entry>);
    static_assert(std::is_nothrow_swappable_v<Snapshot::Entry>);
    for (const auto node : nodes)
    {
        const auto& trace = traces.at(node);
        result->entries_.push_back({node, trace.first, trace.second});
    }
    std::sort(result->entries_.begin(), result->entries_.end(),
        [](const auto& a, const auto& b) noexcept { return a.node < b.node; });
    trajectory_owners.push_back(result);
    return result;
}
std::shared_ptr<const OrdinaryTrajectorySnapshot> OrdinaryTrajectorySnapshot::create(std::vector<Entry> entries)
{
    std::sort(entries.begin(), entries.end(), [](const auto& a, const auto& b) { return a.node < b.node; });
    for (std::size_t i = 0; i < entries.size(); ++i)
    {
        if (entries[i].current.arithmetic_mode != ArithmeticMode::Ordinary ||
            entries[i].departure.arithmetic_mode != ArithmeticMode::Ordinary)
            throw std::invalid_argument("Ordinary trajectory snapshot arithmetic mode differs.");
        if (i && entries[i - 1].node == entries[i].node)
            throw std::invalid_argument("Ordinary trajectory snapshot repeats a canonical node.");
    }
    return std::shared_ptr<const OrdinaryTrajectorySnapshot>(new OrdinaryTrajectorySnapshot(std::move(entries)));
}
std::shared_ptr<const OrdinaryTrajectorySnapshot> OrdinaryTrajectorySnapshot::freeze(
    TraceMap& traces, std::size_t& used, std::size_t limit)
{
    const auto charge = [&](std::size_t count, std::size_t width)
    {
        if (count > (limit - std::min(used, limit)) / width)
            throw PLIC::GeometryError(PLIC::GeometryErrorCode::ResourceLimit,
                "Finite VoF trajectory freeze exceeds its overlapping byte budget.");
        used += count * width;
    };
    charge(traces.size(), sizeof(TraceMap::value_type) + 4 * sizeof(void*));
    PLIC::GeometryAllocationLedger owners;
    for (const auto& [id, trace] : traces)
    {
        owners.retain_point_provenance(trace.first, used, limit);
        owners.retain_point_provenance(trace.second, used, limit);
    }
    charge(1, sizeof(OrdinaryTrajectorySnapshot));
    charge(traces.size(), sizeof(Entry));
    // Complete both object/control-block allocation and buffer allocation
    // before moving anything from the authenticated map. Their allocator
    // overhead is outside retained-payload accounting, as for geometry owners.
    auto result =
        std::shared_ptr<OrdinaryTrajectorySnapshot>(new OrdinaryTrajectorySnapshot(std::vector<Entry>{}));
    result->entries_.reserve(traces.size());
    charge(result->entries_.capacity() - traces.size(), sizeof(Entry));
    static_assert(std::is_nothrow_move_constructible_v<Entry>);
    // TraceMap already supplies canonical, unique identity order. With the
    // admitted capacity held, this transfer cannot allocate or throw.
    for (auto& [id, trace] : traces)
        result->entries_.push_back({id, std::move(trace.first), std::move(trace.second)});
    traces.clear();
    return result;
}
std::size_t OrdinaryTrajectorySnapshot::slot(std::uint64_t node) const
{
    const auto found = std::lower_bound(entries_.begin(), entries_.end(), node,
        [](const auto& entry, auto identity) { return entry.node < identity; });
    if (found == entries_.end() || found->node != node)
        throw std::out_of_range("Finite VoF trajectory is missing a canonical node.");
    return static_cast<std::size_t>(found - entries_.begin());
}
void OrdinaryTrajectorySnapshot::retain_provenance(
    PLIC::GeometryAllocationLedger& ledger, std::size_t& used, std::size_t limit) const
{
    for (const auto& entry : entries_)
    {
        ledger.retain_point_provenance(entry.current, used, limit);
        ledger.retain_point_provenance(entry.departure, used, limit);
    }
}
template Sweep<ArithmeticMode::Ordinary> prepare_sweep<ArithmeticMode::Ordinary>(std::uint64_t,
    std::span<const std::uint64_t>, std::shared_ptr<const OrdinaryTrajectorySnapshot>, Value, Work&, Work&,
    long double);
template Sweep<ArithmeticMode::Ordinary> prepare_sweep<ArithmeticMode::Ordinary>(std::uint64_t,
    std::span<const std::uint64_t>,
    const std::map<std::uint64_t,
        std::pair<PLIC::Point<ArithmeticMode::Ordinary>, PLIC::Point<ArithmeticMode::Ordinary>>>&,
    Value, Work&, long double);
template Sweep<ArithmeticMode::Ordinary> materialize_sweep<ArithmeticMode::Ordinary>(
    Sweep<ArithmeticMode::Ordinary>, std::span<const std::uint64_t>, Work&);
template Sweep<ArithmeticMode::Ordinary> make_sweep<ArithmeticMode::Ordinary>(std::uint64_t,
    std::span<const std::uint64_t>,
    const std::map<std::uint64_t,
        std::pair<PLIC::Point<ArithmeticMode::Ordinary>, PLIC::Point<ArithmeticMode::Ordinary>>>&,
    Value, Work&, long double);
template std::vector<Atom<ArithmeticMode::Ordinary>> recipient_partition<ArithmeticMode::Ordinary>(std::uint64_t,
    const std::map<std::uint64_t, GeometryPtr<ArithmeticMode::Ordinary>>&,
    const std::vector<std::pair<const Sweep<ArithmeticMode::Ordinary>*, int>>&, Work&, double);
template Sweep<ArithmeticMode::BoundedCertification> prepare_sweep<ArithmeticMode::BoundedCertification>(std::uint64_t,
    std::span<const std::uint64_t>,
    const std::map<std::uint64_t, std::pair<PLIC::Point<ArithmeticMode::BoundedCertification>,
                                      PLIC::Point<ArithmeticMode::BoundedCertification>>>&,
    Value, Work&, long double);
template Sweep<ArithmeticMode::BoundedCertification> materialize_sweep<ArithmeticMode::BoundedCertification>(
    Sweep<ArithmeticMode::BoundedCertification>, std::span<const std::uint64_t>, Work&);
template Sweep<ArithmeticMode::BoundedCertification> make_sweep<ArithmeticMode::BoundedCertification>(std::uint64_t,
    std::span<const std::uint64_t>,
    const std::map<std::uint64_t, std::pair<PLIC::Point<ArithmeticMode::BoundedCertification>,
                                      PLIC::Point<ArithmeticMode::BoundedCertification>>>&,
    Value, Work&, long double);
template std::vector<Atom<ArithmeticMode::BoundedCertification>>
recipient_partition<ArithmeticMode::BoundedCertification>(std::uint64_t,
    const std::map<std::uint64_t, GeometryPtr<ArithmeticMode::BoundedCertification>>&,
    const std::vector<std::pair<const Sweep<ArithmeticMode::BoundedCertification>*, int>>&, Work&, double);
} // namespace SimpleFluid::geometry::swept_volume
