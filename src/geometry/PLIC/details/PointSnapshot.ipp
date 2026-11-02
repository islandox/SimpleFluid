#pragma once
#include <cmath>

namespace SimpleFluid::PLIC
{
namespace snapshot_detail
{
inline std::uint32_t index(std::size_t value)
{
    if (value >= std::numeric_limits<std::uint32_t>::max())
        throw GeometryError(GeometryErrorCode::ResourceLimit, "Point snapshot index capacity exhausted.");
    return static_cast<std::uint32_t>(value);
}
template<class T> void push(std::vector<T>& values, T value, PointSnapshotAllocation allocation)
{
    if (values.size() == values.capacity())
    {
        if (values.capacity() > (std::numeric_limits<std::uint32_t>::max() - std::size_t(1)) / 2)
            throw GeometryError(GeometryErrorCode::ResourceLimit, "Point snapshot table capacity overflows.");
        const auto next = values.capacity() ? 2 * values.capacity() : 1;
        (void) index(next);
        allocation.add(next, sizeof(T));
        values.reserve(next);
    }
    values.push_back(std::move(value));
}
template<class F> decltype(auto) dispatch(const PointSnapshotContextBase* context, F&& function)
{
    if (!context)
        throw GeometryError(GeometryErrorCode::ProvenanceMismatch, "Missing point snapshot context.");
    if (context->mode == ArithmeticMode::Ordinary)
        return function(*static_cast<const PointSnapshotContext<ArithmeticMode::Ordinary>*>(context));
    if (context->mode == ArithmeticMode::BoundedCertification)
        return function(*static_cast<const PointSnapshotContext<ArithmeticMode::BoundedCertification>*>(context));
    throw GeometryError(GeometryErrorCode::ProvenanceMismatch, "Foreign point snapshot arithmetic mode.");
}
template<ArithmeticMode Mode>
const PointCoordinates<Mode>& coordinates(const PointSnapshotContext<Mode>& context, DynamicStoredPointRef ref)
{
    switch (ref.role)
    {
        case PointProvenanceRole::Native: {
            const auto& row = context.native_refs.at(ref.index);
            return context.native_fact_blocks.at(row.fact_block)->points.at(row.fact_index);
        }
        case PointProvenanceRole::Generated:
            return context.generated_points.at(ref.index);
        case PointProvenanceRole::Export:
            return context.export_points.at(ref.index);
        case PointProvenanceRole::Numerical:
            return context.numerical_points.at(ref.index);
    }
    throw GeometryError(GeometryErrorCode::ProvenanceMismatch, "Unknown point snapshot role.");
}
template<ArithmeticMode Mode>
PointCoordinates<Mode> coordinates(const std::array<long double, 3>& value, const std::array<long double, 3>& error)
{
    PointCoordinates<Mode> result;
    result.value = value;
    if constexpr (Mode == ArithmeticMode::BoundedCertification)
        result.absolute_error = error;
    return result;
}
} // namespace snapshot_detail

template<ArithmeticMode Mode> std::size_t PointSnapshotContext<Mode>::exclusive_storage_bytes() const
{
    std::size_t bytes = sizeof(*this) + 2 * sizeof(void*);
    const auto add = [&](std::size_t count, std::size_t width)
    {
        if (width && count > (std::numeric_limits<std::size_t>::max() - bytes) / width)
            throw GeometryError(GeometryErrorCode::ResourceLimit, "Point snapshot footprint overflows.");
        bytes += count * width;
    };
    add(dependencies.capacity(), sizeof(dependencies.front()));
    add(native_fact_blocks.capacity(), sizeof(native_fact_blocks.front()));
    add(native_refs.capacity(), sizeof(NativePointRefRow<Mode>));
    add(generated_recipes.capacity(), sizeof(GeneratedPointRecipe<Mode>));
    add(export_aliases.capacity(), sizeof(ExportPointAlias<Mode>));
    add(generated_points.capacity(), sizeof(PointCoordinates<Mode>));
    add(export_points.capacity(), sizeof(PointCoordinates<Mode>));
    add(numerical_points.capacity(), sizeof(PointCoordinates<Mode>));
    add(witnesses.capacity(), sizeof(PointErrorWitness));
    add(authority_table.capacity(), sizeof(PlaneAuthority));
    add(incidence.capacity(), sizeof(PrivateAuthorityRef<Mode>));
    add(edges.capacity(), sizeof(StoredPointEdge<Mode>));
    return bytes;
}

inline PointProvenanceView::PointProvenanceView(const PointSnapshotContextBase* context, DynamicStoredPointRef ref)
{
    if (!context || !ref)
        return;
    if (ref.dependency_slot)
    {
        context = context->dependencies.at(ref.dependency_slot - 1).get();
        ref.dependency_slot = 0;
    }
    snapshot_detail::dispatch(context, [&](const auto& typed) { (void) snapshot_detail::coordinates(typed, ref); });
    context_ = context;
    ref_ = ref;
}
template<ArithmeticMode Mode> TypedPointProvenanceRef<Mode>::TypedPointProvenanceRef(const PointProvenanceRef& source)
{
    if (!source)
        return;
    if (source.context_pin()->mode != Mode)
        throw GeometryError(GeometryErrorCode::ProvenanceMismatch, "Foreign typed point provenance mode.");
    owner_ = std::static_pointer_cast<const PointSnapshotContext<Mode>>(source.context_pin());
    const auto ref = source.local_ref();
    ref_ = StoredPointRef<Mode>{ref.dependency_slot, ref.index, ref.role};
}
template<ArithmeticMode Mode> PointProvenanceView TypedPointProvenanceRef<Mode>::view() const
{
    return resolve_point(owner_, ref_);
}
template<ArithmeticMode Mode> TypedPointProvenanceRef<Mode>::operator PointProvenanceRef() const
{
    return bind_point(owner_, ref_);
}
inline PointProvenanceRef PointProvenanceView::owning() const
{
    if (!*this)
        return {};
    return bind_point(context_->shared_from_this(), ref_);
}
inline ArithmeticMode PointProvenanceView::arithmetic_mode() const
{
    if (!*this)
        throw GeometryError(GeometryErrorCode::ProvenanceMismatch, "Empty point provenance has no arithmetic mode.");
    return context_->mode;
}
inline const std::array<long double, 3>& PointProvenanceView::value() const
{
    return snapshot_detail::dispatch(context_, [&](const auto& typed) -> const std::array<long double, 3>&
        { return snapshot_detail::coordinates(typed, ref_).value; });
}
inline const std::array<long double, 3>& PointProvenanceView::absolute_error() const
{
    static constexpr std::array<long double, 3> zero{};
    return snapshot_detail::dispatch(context_,
        [&](const auto& typed) -> const std::array<long double, 3>&
        {
            using Context = std::decay_t<decltype(typed)>;
            if constexpr (std::is_same_v<Context, PointSnapshotContext<ArithmeticMode::BoundedCertification>>)
                return snapshot_detail::coordinates(typed, ref_).absolute_error;
            else
            {
                if (ref_.role == PointProvenanceRole::Native)
                {
                    const auto& row = typed.native_refs.at(ref_.index);
                    for (const auto& witness : typed.native_fact_blocks.at(row.fact_block)->witnesses)
                        if (witness.index == row.fact_index)
                            return witness.error;
                }
                else
                    for (const auto& witness : typed.witnesses)
                        if (witness.role == ref_.role && witness.index == ref_.index)
                            return witness.error;
                return zero;
            }
        });
}
inline bool PointProvenanceView::represented_native_source() const
{
    return *this && ref_.role == PointProvenanceRole::Native &&
           snapshot_detail::dispatch(
               context_, [&](const auto& typed) { return typed.native_refs.at(ref_.index).represented; });
}
inline bool PointProvenanceView::constructed() const
{
    if (!*this || ref_.role == PointProvenanceRole::Native)
        return false;
    if (ref_.role == PointProvenanceRole::Generated)
        return snapshot_detail::dispatch(
            context_, [&](const auto& typed) { return typed.generated_recipes.at(ref_.index).constructed; });
    return ref_.role != PointProvenanceRole::Export || snapshot_detail::dispatch(context_, [&](const auto& typed)
                                                           { return typed.export_aliases.at(ref_.index).constructed; });
}
inline bool PointProvenanceView::numerical_only() const
{
    return *this && (ref_.role == PointProvenanceRole::Numerical ||
                        (ref_.role == PointProvenanceRole::Export &&
                            snapshot_detail::dispatch(context_, [&](const auto& typed)
                                { return typed.export_aliases.at(ref_.index).numerical_only; })));
}
inline PointAuthorityRange PointProvenanceView::supports() const
{
    if (!*this || numerical_only())
        return {};
    return snapshot_detail::dispatch(context_,
        [&](const auto& typed)
        {
            using Context = std::decay_t<decltype(typed)>;
            constexpr auto mode = std::is_same_v<Context, PointSnapshotContext<ArithmeticMode::Ordinary>>
                                      ? ArithmeticMode::Ordinary
                                      : ArithmeticMode::BoundedCertification;
            LocalAuthorityRange<mode> range;
            if (ref_.role == PointProvenanceRole::Native)
                range = typed.native_refs.at(ref_.index).incidence;
            else if (ref_.role == PointProvenanceRole::Generated)
                range = typed.generated_recipes.at(ref_.index).incidence;
            else
                range = typed.export_aliases.at(ref_.index).incidence;
            return resolve_authorities(&typed, range);
        });
}
inline PointProvenanceView PointProvenanceView::source_origin() const
{
    if (!*this || ref_.role != PointProvenanceRole::Export)
        return {};
    return snapshot_detail::dispatch(
        context_, [&](const auto& typed) { return resolve_point(&typed, typed.export_aliases.at(ref_.index).source); });
}
inline PointEdgeRange PointProvenanceView::source_edges() const
{
    if (!*this || ref_.role != PointProvenanceRole::Generated)
        return {};
    return {*this, snapshot_detail::dispatch(context_,
                       [&](const auto& typed) { return typed.generated_recipes.at(ref_.index).edges.count; })};
}
inline bool PointProvenanceView::proves_incidence(const PlaneAuthority& key) const
{
    const auto range = supports();
    return std::find(range.begin(), range.end(), key) != range.end();
}
inline std::array<PointProvenanceView, 2> PointEdgeRange::Iterator::operator*() const
{
    return snapshot_detail::dispatch(owner.context(),
        [&](const auto& typed)
        {
            const auto& recipe = typed.generated_recipes.at(owner.local_ref().index);
            const auto& pair = typed.edges.at(recipe.edges.first + index);
            return std::array{resolve_point(&typed, pair[0]), resolve_point(&typed, pair[1])};
        });
}

template<ArithmeticMode Mode>
PointSnapshotBuilder<Mode>::PointSnapshotBuilder(PointSnapshotAllocation allocation) : allocation_(allocation)
{
    allocation_.add(1, sizeof(Context) + 2 * sizeof(void*));
    context_ = std::make_shared<Context>();
}
template<ArithmeticMode Mode> template<class T> void PointSnapshotBuilder<Mode>::append(std::vector<T>& values, T value)
{
    snapshot_detail::push(values, std::move(value), allocation_);
}
template<ArithmeticMode Mode> void PointSnapshotBuilder<Mode>::require_draft() const
{
    if (context_->frozen || aborted_)
        throw GeometryError(GeometryErrorCode::ProvenanceMismatch, "Point snapshot builder is already frozen.");
}
template<ArithmeticMode Mode> StoredPointRef<Mode> PointSnapshotBuilder<Mode>::import(PointProvenanceView point)
{
    require_draft();
    Mutation mutation{*this};
    if (!point)
        return {};
    if (point.arithmetic_mode() != Mode)
        throw GeometryError(GeometryErrorCode::ProvenanceMismatch, "A point snapshot cannot import a foreign mode.");
    auto local = point.local_ref();
    if (point.context() == context_.get())
        return {local.dependency_slot, local.index, local.role};
    if (!point.context()->frozen)
        throw GeometryError(
            GeometryErrorCode::ProvenanceMismatch, "Only frozen parent point contexts may be imported.");
    for (std::size_t index = 0; index < context_->dependencies.size(); ++index)
        if (context_->dependencies[index].get() == point.context())
            return {snapshot_detail::index(index + 1), local.index, local.role};
    append(context_->dependencies, point.context()->shared_from_this());
    return {snapshot_detail::index(context_->dependencies.size()), local.index, local.role};
}
template<ArithmeticMode Mode> void PointSnapshotBuilder<Mode>::retain_context(const PointSnapshotContextBase* context)
{
    require_draft();
    Mutation mutation{*this};
    if (!context || context == context_.get())
        return;
    if (!context->frozen || context->mode != Mode)
        throw GeometryError(GeometryErrorCode::ProvenanceMismatch, "Invalid retained plane point context.");
    for (const auto& parent : context_->dependencies)
        if (parent.get() == context)
            return;
    append(context_->dependencies, context->shared_from_this());
}
template<ArithmeticMode Mode>
PrivateAuthorityRef<Mode> PointSnapshotBuilder<Mode>::register_authority(const PlaneAuthority& key)
{
    require_draft();
    Mutation mutation{*this};
    if (key.empty())
        throw GeometryError(GeometryErrorCode::ProvenanceMismatch, "An empty authority cannot enter a point snapshot.");
    details::OriginalSupportAuthorityAccess::validate_boundary(key);
    if (const auto found = authority_index_.find(key); found != authority_index_.end())
        return {found->second};
    const auto slot = snapshot_detail::index(context_->authority_table.size());
    allocation_.add(1, sizeof(typename decltype(authority_index_)::value_type) + 4 * sizeof(void*));
    append(context_->authority_table, key);
    authority_index_.emplace(key, slot);
    return {slot};
}
template<ArithmeticMode Mode>
LocalAuthorityRange<Mode> PointSnapshotBuilder<Mode>::append_incidence(PointAuthorityRange supports)
{
    const LocalAuthorityRange<Mode> result{
        snapshot_detail::index(context_->incidence.size()), snapshot_detail::index(supports.size())};
    // Context-bound ranges resolve offsets on every access, so growing this
    // same draft's tables never invalidates their iterators.
    for (const auto& support : supports)
        append(context_->incidence, register_authority(support));
    return result;
}
template<ArithmeticMode Mode>
LocalAuthorityRange<Mode> PointSnapshotBuilder<Mode>::append_authority(
    LocalAuthorityRange<Mode> previous, AdmittedPointIncidence<Mode> additional)
{
    require_draft();
    Mutation mutation{*this};
    const auto first = snapshot_detail::index(context_->incidence.size());
    const auto old = resolve_authorities(context_.get(), previous);
    for (const auto& key : old)
        append(context_->incidence, register_authority(key));
    for (const auto& key : additional.keys_)
        append(context_->incidence, register_authority(key));
    return {first, snapshot_detail::index(previous.size() + additional.keys_.size())};
}
template<ArithmeticMode Mode>
void PointSnapshotBuilder<Mode>::witness(
    PointProvenanceRole role, std::uint32_t index, const std::array<long double, 3>& error)
{
    if constexpr (Mode == ArithmeticMode::Ordinary)
        if (error != std::array<long double, 3>{})
            append(context_->witnesses, PointErrorWitness{role, index, error});
}
template<ArithmeticMode Mode>
PointProvenanceView PointSnapshotBuilder<Mode>::native_source(std::array<long double, 3> value,
    std::array<long double, 3> error, AdmittedPointIncidence<Mode> proof, PointProvenanceView facts)
{
    require_draft();
    Mutation mutation{*this};
    NativePointRefRow<Mode> row;
    if (facts)
    {
        if (!facts.context()->frozen && facts.context() != context_.get())
            throw GeometryError(GeometryErrorCode::ProvenanceMismatch, "Unfinished foreign native fact context.");
        if (facts.arithmetic_mode() != Mode || facts.local_ref().role != PointProvenanceRole::Native ||
            facts.value() != value || facts.absolute_error() != error)
            throw GeometryError(
                GeometryErrorCode::ProvenanceMismatch, "Native point fact binding differs from source.");
        const auto& source = *static_cast<const Context*>(facts.context());
        const auto& previous = source.native_refs.at(facts.local_ref().index);
        const auto& block = source.native_fact_blocks.at(previous.fact_block);
        const auto found = std::find(context_->native_fact_blocks.begin(), context_->native_fact_blocks.end(), block);
        if (found == context_->native_fact_blocks.end())
        {
            row.fact_block = snapshot_detail::index(context_->native_fact_blocks.size());
            append(context_->native_fact_blocks, block);
        }
        else
            row.fact_block = snapshot_detail::index(found - context_->native_fact_blocks.begin());
        row.fact_index = previous.fact_index;
    }
    else
    {
        if (!own_facts_)
        {
            allocation_.add(1, sizeof(NativePointFactBlock<Mode>) + 2 * sizeof(void*));
            own_facts_ = std::make_shared<NativePointFactBlock<Mode>>();
            append(context_->native_fact_blocks, std::shared_ptr<const NativePointFactBlock<Mode>>(own_facts_));
        }
        const auto found =
            std::find(context_->native_fact_blocks.begin(), context_->native_fact_blocks.end(), own_facts_);
        row.fact_block = snapshot_detail::index(found - context_->native_fact_blocks.begin());
        row.fact_index = snapshot_detail::index(own_facts_->points.size());
        append(own_facts_->points, snapshot_detail::coordinates<Mode>(value, error));
        if constexpr (Mode == ArithmeticMode::Ordinary)
            if (error != std::array<long double, 3>{})
                append(own_facts_->witnesses, PointErrorWitness{PointProvenanceRole::Native, row.fact_index, error});
    }
    row.incidence = append_incidence(proof.keys_);
    row.represented = error == std::array<long double, 3>{} &&
                      std::all_of(value.begin(), value.end(), [](auto x) { return std::isfinite(x); });
    const auto index = snapshot_detail::index(context_->native_refs.size());
    append(context_->native_refs, row);
    return resolve_point(context_.get(), StoredPointRef<Mode>{0, index, PointProvenanceRole::Native});
}
template<ArithmeticMode Mode>
PointProvenanceView PointSnapshotBuilder<Mode>::sealed(
    std::array<long double, 3> value, std::array<long double, 3> error, AdmittedPointIncidence<Mode> proof)
{
    require_draft();
    Mutation mutation{*this};
    const auto point = generated(value, error, {}, proof);
    context_->generated_recipes.at(point.local_ref().index).constructed = false;
    return point;
}
template<ArithmeticMode Mode>
PointProvenanceView PointSnapshotBuilder<Mode>::generated(std::array<long double, 3> value,
    std::array<long double, 3> error, std::span<const std::array<PointProvenanceView, 2>> edges,
    AdmittedPointIncidence<Mode> proof)
{
    require_draft();
    Mutation mutation{*this};
    const auto index = snapshot_detail::index(context_->generated_recipes.size());
    GeneratedPointRecipe<Mode> row{append_incidence(proof.keys_),
        {snapshot_detail::index(context_->edges.size()), snapshot_detail::index(edges.size())}};
    for (const auto& edge : edges)
        append(context_->edges, StoredPointEdge<Mode>{import(edge[0]), import(edge[1])});
    append(context_->generated_recipes, row);
    append(context_->generated_points, snapshot_detail::coordinates<Mode>(value, error));
    witness(PointProvenanceRole::Generated, index, error);
    return resolve_point(context_.get(), StoredPointRef<Mode>{0, index, PointProvenanceRole::Generated});
}
template<ArithmeticMode Mode>
PointProvenanceView PointSnapshotBuilder<Mode>::export_alias(std::array<long double, 3> value,
    std::array<long double, 3> error, PointProvenanceView source, AdmittedPointIncidence<Mode> proof,
    bool numerical_only)
{
    require_draft();
    Mutation mutation{*this};
    if (numerical_only || (source && source.numerical_only()))
        return numerical(value, error);
    const auto index = snapshot_detail::index(context_->export_aliases.size());
    ExportPointAlias<Mode> row{import(source), append_incidence(proof.keys_),
        !source || source.constructed() || source.value() != value || source.absolute_error() != error, false};
    append(context_->export_aliases, row);
    append(context_->export_points, snapshot_detail::coordinates<Mode>(value, error));
    witness(PointProvenanceRole::Export, index, error);
    return resolve_point(context_.get(), StoredPointRef<Mode>{0, index, PointProvenanceRole::Export});
}
template<ArithmeticMode Mode>
PointProvenanceView PointSnapshotBuilder<Mode>::numerical(
    std::array<long double, 3> value, std::array<long double, 3> error)
{
    require_draft();
    Mutation mutation{*this};
    const auto index = snapshot_detail::index(context_->numerical_points.size());
    append(context_->numerical_points, snapshot_detail::coordinates<Mode>(value, error));
    witness(PointProvenanceRole::Numerical, index, error);
    return resolve_point(context_.get(), StoredPointRef<Mode>{0, index, PointProvenanceRole::Numerical});
}
template<ArithmeticMode Mode> std::shared_ptr<const PointSnapshotContext<Mode>> PointSnapshotBuilder<Mode>::freeze()
{
    require_draft();
    Mutation mutation{*this};
    context_->frozen = true;
    own_facts_.reset();
    authority_index_.clear();
    return context_;
}
} // namespace SimpleFluid::PLIC
