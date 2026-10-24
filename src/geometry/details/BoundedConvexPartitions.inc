namespace
{
bool structural_cover(
    const std::shared_ptr<const Ancestry>& root, const std::vector<std::shared_ptr<const Ancestry>>& leaves)
{
    if (leaves.empty())
        return false;
    if (leaves.size() > std::numeric_limits<std::size_t>::max() / 2)
        reject("structural proof node count overflows.");
    // A complete binary complement tree has fewer than twice as many total
    // nodes as leaves. Charge the whole proof before allocating its worklist;
    // private split identity, not numerical equality, authenticates siblings.
    active_work->charge(leaves.size(),
        2 * sizeof(std::shared_ptr<const Ancestry>) + sizeof(const Split*) + sizeof(unsigned) + 4 * sizeof(void*));
    std::vector<std::shared_ptr<const Ancestry>> pending;
    pending.reserve(2 * leaves.size());
    pending.insert(pending.end(), leaves.begin(), leaves.end());
    std::map<const Split*, unsigned> siblings;
    unsigned roots = 0;
    for (std::size_t i = 0; i < pending.size(); ++i)
    {
        active_work->pair();
        const auto& node = pending[i];
        if (node == root)
        {
            if (++roots > 1)
                return false;
            continue;
        }
        if (!node || !node->parent || !node->split || node->split->parent != node->parent)
            return false;
        const auto split = node->split.get();
        const unsigned side = node->outside ? 2 : 1;
        auto& covered = siblings[split];
        if (covered & side)
            return false; // Duplicate/ancestor-overlapping supplied pieces.
        covered |= side;
        if (covered == 3)
        {
            siblings.erase(split);
            pending.push_back(node->parent);
        }
    }
    return roots == 1 && siblings.empty();
}
} // namespace
void BoundedConvexGeometry::certify_partition(
    std::span<const BoundedConvexGeometryPtr> pieces, double relative_budget) const
{
    Operation scope(impl_->limits);
    ++statistics.partition_checks;
    if (!std::isfinite(relative_budget) || relative_budget < 0)
        reject("invalid partition physical budget.");
    std::vector<std::shared_ptr<const Ancestry>> leaves;
    Interval total;
    for (const auto& piece : pieces)
    {
        if (!piece)
            reject("null partition piece.");
        leaves.push_back(piece->impl_->ancestry);
        total = total + piece->impl_->volume;
    }
    const bool structural = structural_cover(impl_->ancestry, leaves);
    if (structural)
        ++statistics.structural_partition_reuses;
    else
    {
        for (const auto& piece : pieces)
            for (const auto& surface : impl_->surfaces)
                for (const auto& vertex : piece->impl_->vertices)
                {
                    active_work->pair();
                    const auto d = distance(vertex, piece->impl_->frame, surface.plane) * Interval(surface.orientation);
                    if (sign(d) > 0)
                        reject("partition piece escapes its parent.");
                }
        for (std::size_t i = 0; i < pieces.size(); ++i)
            for (std::size_t j = 0; j < i; ++j)
            {
                active_work->pair();
                if (pieces[i]->intersect(*pieces[j]))
                    reject("partition pieces have overlapping interiors.");
            }
    }
    const auto residual = total - impl_->volume;
    const auto allowance = Interval(relative_budget) * Interval(impl_->volume.lo);
    if (std::max(std::abs(residual.lo), std::abs(residual.hi)) > allowance.lo)
        reject("partition closure exceeds its physical budget.");
}
