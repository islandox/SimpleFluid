/** Compiled finite geometry kernels. Included only by the owning implementation unit.
 * Namespace operators bind shared storage to the fixed arithmetic policy; Mode
 * templates retain the compiled implementation boundary across geometry units.
 */
namespace SimpleFluid::Meshes::bounded_geometry_detail
{
namespace SIMPLEFLUID_GEOMETRY_KERNEL SIMPLEFLUID_LOCAL
{
template<ArithmeticMode Mode>
bool structural_cover(
    const std::shared_ptr<const Ancestry>& root, const std::vector<std::shared_ptr<const Ancestry>>& leaves)
{
    static_assert(Mode == kernel_mode);
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

template<ArithmeticMode Mode>
void GeometryKernel<Mode>::certify_partition(
    std::span<const BoundedConvexGeometryPtr> pieces, double relative_budget) const
{
    static_assert(Mode == kernel_mode);
    Operation scope(impl_->limits);
    ++statistics.partition_checks;
    if (impl_->partition_proof_only)
        reject("private partition witness cannot be a measured partition parent.");
    if (!std::isfinite(relative_budget) || relative_budget < 0)
        reject("invalid partition physical budget.");
    active_work->charge(pieces.size(), sizeof(std::shared_ptr<const Ancestry>));
    std::vector<std::shared_ptr<const Ancestry>> leaves;
    leaves.reserve(pieces.size());
    for (const auto& piece : pieces)
    {
        if (!piece)
            reject("null partition piece.");
        require_mode(arithmetic_mode(), piece->arithmetic_mode());
        if (piece->impl_->partition_proof_only)
            reject("private partition witness cannot be a measured partition piece.");
        leaves.push_back(piece->impl_->ancestry);
    }
    if (!pieces.empty())
    {
        const auto& certificate = pieces.front()->impl_->partition_certificate;
        if (certificate && certificate->parent == impl_->ancestry && certificate->members.size() == pieces.size())
        {
            active_work->charge(pieces.size(), sizeof(unsigned char));
            std::vector<unsigned char> seen(pieces.size(), 0);
            bool complete = true;
            for (const auto& piece : pieces)
            {
                if (piece->impl_->partition_certificate != certificate)
                {
                    complete = false;
                    break;
                }
                std::size_t member = 0;
                for (; member < certificate->members.size(); ++member)
                {
                    active_work->pair();
                    if (certificate->members[member].lock() == piece->impl_->ancestry)
                        break;
                }
                if (member == seen.size() || seen[member]++)
                {
                    complete = false;
                    break;
                }
            }
            if (complete)
            {
                ++statistics.structural_partition_reuses;
                return;
            }
        }
    }
    const bool structural = structural_cover<Mode>(impl_->ancestry, leaves);
    if (structural)
    {
        ++statistics.structural_partition_reuses;
        // The private complete complement tree proves an exact set partition,
        // including the identity leaf. Its volume closure is identically zero;
        // independently subtracting cached interval measures would discard that
        // dependency. Individual piece measures keep their original uncertainty
        // for the caller's transfer, positivity and recipient-phase gates.
        return;
    }
    else
    {
        for (const auto& piece : pieces)
        {
            if (construction_contains<Mode>(impl_->ancestry, piece->impl_->ancestry))
                continue;
            for (const auto& surface : impl_->surfaces)
                for (const auto& vertex : piece->impl_->vertices)
                {
                    active_work->pair();
                    const auto d =
                        distance<Mode>(vertex, piece->impl_->frame, surface.plane) * Interval(surface.orientation);
                    if (sign<Mode>(d) > 0)
                        reject("partition piece escapes its parent.");
                }
        }
        for (std::size_t i = 0; i < pieces.size(); ++i)
            for (std::size_t j = 0; j < i; ++j)
            {
                active_work->pair();
                if (construction_disjoint<Mode>(pieces[i]->impl_->ancestry, pieces[j]->impl_->ancestry))
                    continue;
                if (pieces[i]->intersect(*pieces[j]))
                    reject("partition pieces have overlapping interiors.");
            }
    }
    Interval total;
    for (const auto& piece : pieces)
        total = total + piece->impl_->volume;
    const auto residual = total - impl_->volume;
    const auto allowance = Interval(relative_budget) * Interval(impl_->volume.lo);
    if (std::max(std::abs(residual.lo), std::abs(residual.hi)) > allowance.lo)
    {
        // Independent measure intervals forget the common parent expression.
        // After containment and disjointness, prove completeness by removing
        // each supplied convex piece from the parent. Every outside fragment
        // is retained; only the intersection with the piece is removed. This
        // is a coverage proof, never a priority rule for overlapping donors.
        active_work->charge(1, sizeof(BoundedConvexGeometry) + sizeof(BoundedConvexGeometryPtr));
        std::vector<BoundedConvexGeometryPtr> remainder{BoundedConvexGeometryPtr(new BoundedConvexGeometry(impl_))};
        for (const auto& piece : pieces)
        {
            std::vector<BoundedConvexGeometryPtr> next;
            for (auto fragment : remainder)
                for (std::size_t face = 0; fragment && face < piece->impl_->faces.size(); ++face)
                {
                    active_work->pair();
                    auto split = fragment->clip_impl(piece->impl_, face, false, true, true);
                    if (split.outside)
                    {
                        active_work->charge(1, 2 * sizeof(BoundedConvexGeometryPtr));
                        next.push_back(std::move(split.outside));
                    }
                    fragment = std::move(split.inside);
                }
            remainder = std::move(next);
            if (remainder.empty())
                return;
        }
        reject("partition closure exceeds its physical budget and has a nonempty geometric remainder.");
    }
}

template<ArithmeticMode Mode>
std::vector<BoundedConvexGeometryPtr> GeometryKernel<Mode>::validate_donors(
    std::span<const BoundedConvexGeometryPtr> donors, BoundedGeometryLimits limits)
{
    static_assert(Mode == kernel_mode);
    Operation scope(limits);
    struct Candidate
    {
        std::size_t index;
        CoordinateRange x;
    };
    active_work->charge(donors.size(), sizeof(BoundedConvexGeometryPtr) + sizeof(Candidate));
    std::vector<Candidate> ordered;
    ordered.reserve(donors.size());
    for (std::size_t i = 0; i < donors.size(); ++i)
    {
        if (!donors[i] || donors[i]->impl_->partition_proof_only)
            reject("donor partition requires measured immutable geometry.");
        require_mode(limits.arithmetic_mode, donors[i]->arithmetic_mode());
        const auto& geometry = *donors[i]->impl_;
        const auto bounds = transform<Mode>(geometry.bounds, geometry.frame, donors.front()->impl_->frame);
        ordered.push_back({i, bounds[0]});
    }
    std::sort(ordered.begin(), ordered.end(),
        [](const auto& a, const auto& b)
        {
            active_work->pair();
            return a.x.lo < b.x.lo;
        });
    // Bounds screen candidate pairs; overlap is proved or rejected once for
    // this immutable donor set, never hidden by later subtraction priority.
    for (std::size_t i = 0; i < ordered.size(); ++i)
        for (std::size_t j = i + 1; j < ordered.size(); ++j)
        {
            active_work->pair();
            if (ordered[j].x.lo >= ordered[i].x.hi)
                break;
            if (donors[ordered[i].index]->intersect(*donors[ordered[j].index]))
                reject("donor partition has overlapping interiors.");
        }
    return {donors.begin(), donors.end()};
}

template<ArithmeticMode Mode>
std::vector<BoundedPartitionPiece> PartitionKernel<Mode>::partition(
    const BoundedConvexGeometry& parent, std::size_t maximum_pieces) const
{
    static_assert(Mode == kernel_mode);
    auto limits = parent.impl_->limits;
    limits.maximum_vertices = std::min(limits.maximum_vertices, limits_.maximum_vertices);
    limits.maximum_faces = std::min(limits.maximum_faces, limits_.maximum_faces);
    limits.maximum_operations = std::min(Work(limits).operation_limit, Work(limits_).operation_limit);
    limits.maximum_pair_tests = std::min(limits.maximum_pair_tests, limits_.maximum_pair_tests);
    limits.byte_budget = std::min(limits.byte_budget, limits_.byte_budget);
    require_mode(limits_.arithmetic_mode, parent.arithmetic_mode());
    Operation scope(limits);
    if (!maximum_pieces || parent.impl_->partition_proof_only)
        reject("invalid measured partition parent or fragment cap.");
    if (parent.impl_->vertices.size() > limits.maximum_vertices || parent.impl_->faces.size() > limits.maximum_faces)
        reject("partition parent exceeds its vertex/face cap.");
    std::vector<BoundedPartitionPiece> result;
    std::vector<BoundedConvexGeometryPtr> remainder;
    const auto retain = [&](auto& list, auto value)
    {
        if (list.size() >= maximum_pieces)
            reject("constructed partition fragment cap exhausted.");
        if (list.size() == list.capacity())
        {
            const auto capacity = list.capacity();
            const auto next = capacity > maximum_pieces / 2 ? maximum_pieces : capacity ? 2 * capacity : 1;
            active_work->charge(next, sizeof(value));
            list.reserve(next);
        }
        list.push_back(std::move(value));
    };
    active_work->charge(1, sizeof(BoundedConvexGeometry));
    retain(remainder, BoundedConvexGeometryPtr(new BoundedConvexGeometry(parent.impl_)));
    for (std::size_t donor = 0; donor < donors_.size() && !remainder.empty(); ++donor)
    {
        const auto& cell = donors_[donor];
        std::vector<BoundedConvexGeometryPtr> next;
        for (auto fragment : remainder)
        {
            active_work->pair();
            if (separated_bounds<Mode>(*fragment->impl_, *cell->impl_))
            {
                retain(next, std::move(fragment));
                continue;
            }
            for (std::size_t face = 0; fragment && face < cell->impl_->faces.size(); ++face)
            {
                active_work->pair();
                if (separated_bounds<Mode>(*fragment->impl_, *cell->impl_))
                {
                    retain(next, std::move(fragment));
                    break;
                }
                auto split = fragment->clip_impl(cell->impl_, face, false, true, true);
                if (split.outside)
                    retain(next, std::move(split.outside));
                fragment = std::move(split.inside);
            }
            if (fragment)
            {
                // These are set-partition witnesses, not physical transfer
                // pieces. A donor's union will be measured as one convex
                // parent/donor intersection after the coverage proof succeeds.
                retain(result, BoundedPartitionPiece{donor, std::move(fragment)});
            }
        }
        remainder = std::move(next);
    }
    if (!remainder.empty())
        reject("constructed donor partition leaves an uncovered remainder.");
    std::vector<std::shared_ptr<const Ancestry>> leaves;
    active_work->charge(result.size(), sizeof(std::shared_ptr<const Ancestry>));
    for (const auto& piece : result)
        leaves.push_back(piece.geometry->impl_->ancestry);
    if (!structural_cover<Mode>(parent.impl_->ancestry, leaves))
        reject("constructed donor partition has an incomplete split certificate.");
    active_work->charge(donors_.size(), sizeof(unsigned char));
    std::vector<unsigned char> used(donors_.size(), 0);
    for (const auto& piece : result)
        used[piece.donor] = 1;
    result.clear();
    for (std::size_t donor = 0; donor < donors_.size(); ++donor)
        if (used[donor])
        {
            // Earlier subtractions removed only regions in other, validated
            // disjoint donors. Thus this donor's witness union equals parent
            // intersect donor up to shared zero-volume boundaries.
            auto geometry = parent.intersect(*donors_[donor]);
            if (!geometry)
                reject("measured donor intersection contradicts its construction certificate.");
            retain(result, BoundedPartitionPiece{donor, std::move(geometry)});
        }
    // Seal membership only after every measured physical output succeeds.
    active_work->charge(1, sizeof(PartitionCertificate));
    auto certificate = std::make_shared<PartitionCertificate>();
    certificate->parent = parent.impl_->ancestry;
    active_work->charge(result.size(), sizeof(std::weak_ptr<const Ancestry>));
    certificate->members.reserve(result.size());
    for (const auto& piece : result)
        certificate->members.push_back(piece.geometry->impl_->ancestry);
    for (auto& piece : result)
    {
        active_work->charge(piece.geometry->storage_bytes() + sizeof(BoundedConvexGeometry), 1);
        auto certified = std::make_shared<BoundedConvexGeometry::Impl>(*piece.geometry->impl_);
        certified->partition_certificate = certificate;
        piece.geometry = BoundedConvexGeometryPtr(new BoundedConvexGeometry(std::move(certified)));
    }
    ++statistics.structural_partition_reuses;
    return result;
}

template bool structural_cover<kernel_mode>(
    const std::shared_ptr<const Ancestry>& root, const std::vector<std::shared_ptr<const Ancestry>>& leaves);
template void GeometryKernel<kernel_mode>::certify_partition(
    std::span<const BoundedConvexGeometryPtr> pieces, double relative_budget) const;
template std::vector<BoundedConvexGeometryPtr> GeometryKernel<kernel_mode>::validate_donors(
    std::span<const BoundedConvexGeometryPtr>, BoundedGeometryLimits);
template std::vector<BoundedPartitionPiece> PartitionKernel<kernel_mode>::partition(
    const BoundedConvexGeometry& parent, std::size_t maximum_pieces) const;
} // namespace SIMPLEFLUID_GEOMETRY_KERNEL SIMPLEFLUID_LOCAL
} // namespace SimpleFluid::Meshes::bounded_geometry_detail
