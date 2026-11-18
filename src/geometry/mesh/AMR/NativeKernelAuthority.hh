/** @file NativeKernelAuthority.hh
 * @brief Factory-only immutable semantics authority for native publication.
 */
#pragma once
#include "geometry/details/NativeAllocationEnvelope.hh"
#include "geometry/details/NativeSchemaOwner.hh"
#include "geometry/details/NativeConstructorOwnerSink.hh"
#include "geometry/PLIC/details/ExtrudedConstructionOwnerAccess.hh"
#include "geometry/mesh/AMR/AdaptiveProviderKernel.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveWire.hh"
#include <memory>
#include <span>
#include <stdexcept>

namespace SimpleFluid::Meshes
{
class NativeFragmentRecord;
}
namespace SimpleFluid::Meshes::AMR
{
struct DistributedMixedRoute;
class MixedKernel;
namespace adaptive_detail
{
class NativePublicationSession;
class NativeCellSourceProof;
class NativePublicationTestAccess;

struct NativeKernelSourceFace
{
    std::uint64_t surface = 0, native_face = 0;
    int outward_sign = 0;
    bool physical_boundary = false;
};

/** This is a stable-semantics capability, never a source/native geometry proof.
 * Only exact final built-in factories can issue it. In particular a virtual
 * provider opt-in, descriptor, enum, or a caller-created subclass cannot do so.
 */
class SIMPLEFLUID_LOCAL NativeKernelAuthority final
{
    class ConstructionKey
    {
        friend class NativeKernelAuthority;
        ConstructionKey() = default;

    public:
        ConstructionKey(const ConstructionKey&) = default;
        ConstructionKey(ConstructionKey&&) = default;
    };

public:
    using Sink = geometry_detail::NativeConstructorOwnerSink;
    using IdentityVisit = PLIC::details::ExtrudedConstructionOwnerAccess::IdentityVisit;
    using GeometryVisit = void (*)(void*, const PLIC::ConvexGeometryPtr<ArithmeticMode::Ordinary>&);
    using ChildProjection = void (*)(const AdaptiveProviderKernel&, const Sink&, IdentityVisit, GeometryVisit);
    using SupportVisit = void (*)(void*, std::size_t, std::span<const char>, bool);
    using SplitVisit = void (*)(void*, AdaptiveSplitResult&&);
    NativeKernelAuthority(ConstructionKey, const std::shared_ptr<AdaptiveProviderKernel>& kernel,
        std::shared_ptr<const geometry_detail::NativeAllocationEnvelope> kernel_allocation,
        std::shared_ptr<const geometry_detail::NativeSchemaOwner> schema, ChildProjection children)
        : kernel_(kernel), kernel_allocation_(std::move(kernel_allocation)), schema_(std::move(schema)), children_(children)
    {
    }
    NativeKernelAuthority(const NativeKernelAuthority&) = delete;
    NativeKernelAuthority& operator=(const NativeKernelAuthority&) = delete;
    NativeKernelAuthority(NativeKernelAuthority&&) = delete;
    NativeKernelAuthority& operator=(NativeKernelAuthority&&) = delete;

private:
    friend auto ::SimpleFluid::Meshes::AMR::cartesian_adaptive_kernel(AMRGrid)
        -> std::shared_ptr<const AdaptiveProviderKernel>;
    friend auto ::SimpleFluid::Meshes::AMR::product_adaptive_kernel() -> std::shared_ptr<const AdaptiveProviderKernel>;
    friend auto ::SimpleFluid::Meshes::AMR::convex_adaptive_kernel(unsigned, double)
        -> std::shared_ptr<const AdaptiveProviderKernel>;
    friend auto ::SimpleFluid::Meshes::AMR::detach_static_participant(const MultiRegionMesh::Region&, std::uint64_t,
        std::uint64_t, std::uint64_t, std::span<const std::uint64_t>, AdaptiveBytes) -> AdaptiveBootstrap;
    friend auto ::SimpleFluid::Meshes::AMR::mixed_adaptive_kernel(std::vector<DistributedMixedRoute>, std::size_t)
        -> std::shared_ptr<const AdaptiveProviderKernel>;

    template<class Kernel, class... Args>
    [[nodiscard]] static std::shared_ptr<const AdaptiveProviderKernel> create(Args&&... args)
    {
        static_assert(std::is_final_v<Kernel> && std::is_base_of_v<AdaptiveProviderKernel, Kernel>);
        // Kernel factories construct caller input before a hierarchy operation
        // exists. Its detached scalar domain records exact fixed allocations;
        // hierarchy/source admission subsequently imports these actual owners.
        auto counter =
            geometry_detail::NativeConstructionCounterAccess::create<Budget>(std::numeric_limits<std::size_t>::max());
        geometry_detail::NativeConstructionAdmission admission(counter);
        auto kernel =
            geometry_detail::NativeAllocationEnvelopeAccess::make<Kernel>(admission, std::forward<Args>(args)...);
        std::shared_ptr<const geometry_detail::NativeSchemaOwner> schema;
        if constexpr (requires { kernel.owner->seal_native_publication_schema(admission); })
            schema = kernel.owner->seal_native_publication_schema(admission);
        else
            schema = geometry_detail::NativeSchemaOwnerAccess::adopt(kernel.owner->descriptor(), admission);
        std::shared_ptr<AdaptiveProviderKernel> base = kernel.owner;
        auto authority = geometry_detail::NativeAllocationEnvelopeAccess::make<NativeKernelAuthority>(
            admission, ConstructionKey{}, base, kernel.envelope, std::move(schema),
            [](const AdaptiveProviderKernel& value, const Sink& sink, IdentityVisit identity, GeometryVisit geometry)
            {
                static_cast<const Kernel&>(value).visit_native_publication_children(sink, identity, geometry);
            });
        authority.owner->allocation_ = authority.envelope;
        using Storage = geometry_detail::NativeConstructorStorageOwner;
        if constexpr (requires { Kernel::native_source_explicit_body_intake; })
            authority.owner->explicit_body_intake_ = Kernel::native_source_explicit_body_intake;
        if constexpr (requires(const AdaptiveLeaf& leaf, void* context, SplitVisit visit)
            {
                kernel.owner->validate_profiled_source_parent(leaf, admission);
                kernel.owner->profiled_source_split(leaf, AdaptiveSplit::Isotropic, admission, context, visit);
            })
            authority.owner->profiled_split_ = [](const AdaptiveProviderKernel& base, const AdaptiveLeaf& leaf,
                AdaptiveSplit kind, const geometry_detail::NativeConstructionAdmission& admission, void* context,
                SplitVisit visit, bool verify_parent)
            {
                const auto& kernel = static_cast<const Kernel&>(base);
                if (verify_parent)
                    kernel.validate_profiled_source_parent(leaf, admission);
                kernel.profiled_source_split(leaf, kind, admission, context, visit);
            };
        if constexpr (requires { Kernel::native_source_publication_ready; })
            authority.owner->source_publication_ready_ = Kernel::native_source_publication_ready;
        if constexpr (requires(const AdaptiveLeaf& leaf)
            { kernel.owner->supports_native_source(leaf); })
            authority.owner->source_eligible_ = [](const AdaptiveProviderKernel& base, const AdaptiveLeaf& leaf) noexcept
            { return static_cast<const Kernel&>(base).supports_native_source(leaf); };
        if constexpr (requires(const AdaptiveLeaf& leaf) { kernel.owner->prepare_native_source(leaf, admission); })
            authority.owner->prepare_source_ = [](const AdaptiveProviderKernel& base, const AdaptiveLeaf& leaf,
                                                   const geometry_detail::NativeConstructionAdmission& admission)
            { return static_cast<const Kernel&>(base).prepare_native_source(leaf, admission); };
        if constexpr (requires(const AdaptiveLeaf& leaf, const NativeFragmentRecord& fragment, const Storage& storage,
                          std::vector<NativeKernelSourceFace>& faces)
            { kernel.owner->collect_native_source(leaf, fragment, admission, storage, faces); })
            authority.owner->source_collector_ = [](const AdaptiveProviderKernel& base, const AdaptiveLeaf& leaf,
                const NativeFragmentRecord& fragment, const geometry_detail::NativeConstructionAdmission& admission,
                const Storage& storage, std::vector<NativeKernelSourceFace>& faces)
            { return static_cast<const Kernel&>(base).collect_native_source(leaf, fragment, admission, storage, faces); };
        if constexpr (requires(const AdaptiveLeaf& leaf, const NativeFragmentRecord& fragment,
                                  std::span<NativeKernelSourceFace> faces)
            {
                kernel.owner->validate_native_source(leaf, fragment, admission, faces);
            })
        {
            authority.owner->source_face_count_ = [](const AdaptiveProviderKernel& base, const AdaptiveLeaf& leaf,
                                                     const NativeFragmentRecord& fragment)
            {
                const auto& kernel = static_cast<const Kernel&>(base);
                if constexpr (requires { kernel.native_source_face_count(leaf, fragment); })
                    return kernel.native_source_face_count(leaf, fragment);
                else
                    return kernel.native_source_face_count(leaf);
            };
            authority.owner->source_validator_ = [](const AdaptiveProviderKernel& base, const AdaptiveLeaf& leaf,
                                                    const NativeFragmentRecord& fragment,
                                                    const geometry_detail::NativeConstructionAdmission& admission,
                                                std::span<NativeKernelSourceFace> faces, void* context,
                                                SupportVisit support)
            {
                const auto& kernel = static_cast<const Kernel&>(base);
                if constexpr (requires { kernel.validate_native_source(leaf, fragment, admission, faces,
                                             context, support); })
                    return kernel.validate_native_source(leaf, fragment, admission, faces, context, support);
                else
                {
                    const auto metrics = kernel.validate_native_source(leaf, fragment, admission, faces);
                    if (support)
                    {
                        if (leaf.surfaces.size() != faces.size())
                            throw std::invalid_argument("Native provider lacks its original support metadata.");
                        for (std::size_t i = 0; i < leaf.surfaces.size(); ++i)
                            support(context, i, leaf.surfaces[i].key, leaf.surfaces[i].exterior);
                    }
                    return metrics;
                }
            };
        }
        if constexpr (requires(const AdaptiveLeaf& leaf, const NativeFragmentRecord& fragment,
                          std::span<const NativeKernelSourceFace> previous, std::span<NativeKernelSourceFace> faces)
            { kernel.owner->rebind_native_source(leaf, fragment, previous, leaf, fragment, faces, admission); })
            authority.owner->source_rebind_ = [](const AdaptiveProviderKernel& base, const AdaptiveLeaf& previous,
                const NativeFragmentRecord& previous_fragment, std::span<const NativeKernelSourceFace> previous_faces,
                const AdaptiveLeaf& current,
                const NativeFragmentRecord& current_fragment, std::span<NativeKernelSourceFace> faces,
                const geometry_detail::NativeConstructionAdmission& admission)
            {
                return static_cast<const Kernel&>(base).rebind_native_source(
                    previous, previous_fragment, previous_faces, current, current_fragment, faces, admission);
            };
        base->native_authority_ = std::move(authority.owner);
        return base;
    }

public:
    /** Exact final-factory thunk; never a virtual provider opt-in. The source
     * semantic first/last transition gates dynamic children independently of
     * weak/fixed envelope observations. */
    static void visit(const std::shared_ptr<const AdaptiveProviderKernel>& kernel, const Sink& sink,
        IdentityVisit identity, GeometryVisit geometry)
    {
        if (!qualified(kernel))
            throw std::invalid_argument("Kernel lacks its exact final factory authority/control.");
        const auto& authority = kernel->native_authority_;
        if (!sink.semantic(kernel, geometry_detail::NativeConstructorOwnerKind::NativeMesh,
                authority->kernel_allocation_))
            return;
        if (sink.semantic(authority, geometry_detail::NativeConstructorOwnerKind::NativeMesh, authority->allocation_))
        {
            sink.fixed(authority->allocation_);
            sink.fixed(authority->kernel_allocation_);
            sink.schema(authority->schema_);
        }
        if (!authority->children_)
            throw std::logic_error("Kernel has no closed child projection thunk.");
        authority->children_(*kernel, sink, identity, geometry);
    }
    /** Read-only observations grant no authority installation or source proof. */
    [[nodiscard]] static bool qualified(const std::shared_ptr<const AdaptiveProviderKernel>& kernel) noexcept
    {
        if (!kernel || !kernel->native_authority_)
            return false;
        const auto& authority = kernel->native_authority_;
        const auto& fixed = authority->allocation_;
        const auto& source = authority->kernel_allocation_;
        const auto& schema = authority->schema_;
        return authority->matches(kernel) && fixed && fixed->authentic_owner(fixed) && fixed->matches(authority) &&
               source && source->authentic_owner(source) && source->shares_control(kernel) && schema &&
               schema->allocation() && schema->allocation()->authentic_owner(schema->allocation()) &&
               schema->allocation()->matches(schema);
    }
    [[nodiscard]] static bool source_profile_supported() noexcept;
    /** Separate from stable factory semantics: only factories with a complete
     * admitted first-source numerical verifier select private publication.
     */
    [[nodiscard]] static bool supports_source_validation(
        const std::shared_ptr<const AdaptiveProviderKernel>& kernel) noexcept
    {
        return has_numerical_verifier(kernel) && kernel->native_authority_->source_publication_ready_;
    }
    [[nodiscard]] static bool supports_source_validation(const AdaptiveProviderKernel& candidate) noexcept
    {
        const auto& authority = candidate.native_authority_;
        if (!authority)
            return false;
        const auto kernel = authority->kernel_.lock();
        return kernel.get() == &candidate && supports_source_validation(kernel);
    }
    [[nodiscard]] static bool supports_source_validation(
        const std::shared_ptr<const AdaptiveProviderKernel>& kernel, const AdaptiveLeaf& leaf) noexcept
    {
        return supports_source_validation(kernel) && kernel->native_authority_->source_eligible_(*kernel, leaf);
    }
    [[nodiscard]] static bool supports_source_validation(
        const AdaptiveProviderKernel& candidate, const AdaptiveLeaf& leaf) noexcept
    {
        const auto& authority = candidate.native_authority_;
        if (!authority)
            return false;
        const auto kernel = authority->kernel_.lock();
        return kernel.get() == &candidate && supports_source_validation(kernel, leaf);
    }
    [[nodiscard]] static bool admits_explicit_source_body(
        const std::shared_ptr<const AdaptiveProviderKernel>& kernel) noexcept
    {
        return supports_source_validation(kernel) && kernel->native_authority_->explicit_body_intake_;
    }
    [[nodiscard]] static bool declares_explicit_source_body(
        const std::shared_ptr<const AdaptiveProviderKernel>& kernel) noexcept
    {
        return qualified(kernel) && kernel->native_authority_->explicit_body_intake_;
    }
    [[nodiscard]] static bool admits_profiled_source_split(
        const std::shared_ptr<const AdaptiveProviderKernel>& kernel, const AdaptiveLeaf& leaf) noexcept
    {
        return leaf.provider == AdaptiveProviderKind::Polyhedral && supports_source_validation(kernel, leaf) &&
            kernel->native_authority_->profiled_split_;
    }
    static void profiled_source_split(const std::shared_ptr<const AdaptiveProviderKernel>& kernel,
        const AdaptiveLeaf& leaf, AdaptiveSplit kind, const geometry_detail::NativeConstructionAdmission& admission,
        void* context, SplitVisit visit, bool verify_parent = false)
    {
        if (!visit || !admits_profiled_source_split(kernel, leaf))
            throw std::invalid_argument("Native split lacks its actual factory and active source body.");
        kernel->native_authority_->profiled_split_(*kernel, leaf, kind, admission, context, visit, verify_parent);
    }
    [[nodiscard]] static std::weak_ptr<const NativeKernelAuthority> observation(
        const std::shared_ptr<const AdaptiveProviderKernel>& kernel) noexcept
    {
        return qualified(kernel) ? kernel->native_authority_ : std::weak_ptr<const NativeKernelAuthority>{};
    }
    [[nodiscard]] static std::shared_ptr<const geometry_detail::NativeSchemaOwner> schema_observation(
        const std::shared_ptr<const AdaptiveProviderKernel>& kernel) noexcept
    {
        return qualified(kernel) ? kernel->native_authority_->schema_ : nullptr;
    }
    [[nodiscard]] static std::shared_ptr<const geometry_detail::NativeAllocationEnvelope> allocation_observation(
        const std::shared_ptr<const AdaptiveProviderKernel>& kernel) noexcept
    {
        return qualified(kernel) ? kernel->native_authority_->allocation_ : nullptr;
    }
    [[nodiscard]] static std::shared_ptr<const geometry_detail::NativeAllocationEnvelope> kernel_allocation_observation(
        const std::shared_ptr<const AdaptiveProviderKernel>& kernel) noexcept
    {
        return qualified(kernel) ? kernel->native_authority_->kernel_allocation_ : nullptr;
    }

private:
    friend class NativePublicationSession;
    friend class ::SimpleFluid::Meshes::AMR::MixedKernel;
    // The numerical capability can be consumed by an isolated verifier.
    // Hierarchy publication also requires the actual provider funding profile.
    [[nodiscard]] static bool has_numerical_verifier(
        const std::shared_ptr<const AdaptiveProviderKernel>& kernel) noexcept
    {
        if (!source_profile_supported() || !qualified(kernel))
            return false;
        const auto& authority = *kernel->native_authority_;
        return authority.source_eligible_ && (authority.source_validator_ || authority.source_collector_);
    }
    [[nodiscard]] static bool has_numerical_verifier(
        const std::shared_ptr<const AdaptiveProviderKernel>& kernel, const AdaptiveLeaf& leaf) noexcept
    {
        return has_numerical_verifier(kernel) && kernel->native_authority_->source_eligible_(*kernel, leaf);
    }
    [[nodiscard]] static bool supports_original_source_validation(
        const std::shared_ptr<const AdaptiveProviderKernel>& kernel) noexcept
    {
        return supports_source_validation(kernel) && kernel->native_authority_->source_validator_;
    }
    [[nodiscard]] static std::size_t source_face_count(
        const std::shared_ptr<const AdaptiveProviderKernel>& kernel, const AdaptiveLeaf& leaf,
        const NativeFragmentRecord& fragment)
    {
        if (!has_numerical_verifier(kernel, leaf) || !kernel->native_authority_->source_face_count_)
            throw std::invalid_argument("Native source lacks its exact admitted factory verifier.");
        const auto count = kernel->native_authority_->source_face_count_(*kernel, leaf, fragment);
        if (count < 4 || count > PLIC::GeometryLimits{}.maximum_faces)
            throw std::invalid_argument("Native source face count exceeds the supported closed-boundary limits.");
        return count;
    }
    [[nodiscard]] static AdaptiveCellMetrics validate_source(
        const std::shared_ptr<const AdaptiveProviderKernel>& kernel, const AdaptiveLeaf& leaf,
        const NativeFragmentRecord& fragment, const geometry_detail::NativeConstructionAdmission& admission,
        std::span<NativeKernelSourceFace> faces, void* context = nullptr, SupportVisit support = nullptr)
    {
        if (!has_numerical_verifier(kernel, leaf) || !kernel->native_authority_->source_validator_)
            throw std::invalid_argument("Native source lacks its exact admitted factory verifier.");
        return kernel->native_authority_->source_validator_(*kernel, leaf, fragment, admission, faces, context, support);
    }
    [[nodiscard]] static AdaptiveCellMetrics collect_source(
        const std::shared_ptr<const AdaptiveProviderKernel>& kernel, const AdaptiveLeaf& leaf,
        const NativeFragmentRecord& fragment, const geometry_detail::NativeConstructionAdmission& admission,
        const geometry_detail::NativeConstructorStorageOwner& storage, std::vector<NativeKernelSourceFace>& faces)
    {
        if (!has_numerical_verifier(kernel, leaf))
            throw std::invalid_argument("Native source lacks its exact admitted factory verifier.");
        if (const auto collect = kernel->native_authority_->source_collector_)
        {
            const auto metrics = collect(*kernel, leaf, fragment, admission, storage, faces);
            if (faces.size() < 4 || faces.size() > PLIC::GeometryLimits{}.maximum_faces)
                throw std::invalid_argument("Native mapped face count exceeds the supported closed-boundary limits.");
            return metrics;
        }
        const auto count = source_face_count(kernel, leaf, fragment);
        storage.owner->reserve(faces, count);
        faces.resize(count);
        return validate_source(kernel, leaf, fragment, admission, faces);
    }
    [[nodiscard]] static AdaptiveCellMetrics prepare_source(
        const std::shared_ptr<const AdaptiveProviderKernel>& kernel, const AdaptiveLeaf& leaf,
        const geometry_detail::NativeConstructionAdmission& admission)
    {
        if (!has_numerical_verifier(kernel, leaf) || !kernel->native_authority_->prepare_source_)
            throw std::invalid_argument("Native cell fragment requires authenticated original source metrics.");
        return kernel->native_authority_->prepare_source_(*kernel, leaf, admission);
    }
    [[nodiscard]] static bool can_rebind_source(const std::shared_ptr<const AdaptiveProviderKernel>& kernel,
        const AdaptiveLeaf& previous, const AdaptiveLeaf& current) noexcept
    {
        return has_numerical_verifier(kernel, previous) && has_numerical_verifier(kernel, current) &&
            kernel->native_authority_->source_rebind_;
    }
    [[nodiscard]] static bool rebind_source(const std::shared_ptr<const AdaptiveProviderKernel>& kernel,
        const AdaptiveLeaf& previous, const NativeFragmentRecord& previous_fragment,
        std::span<const NativeKernelSourceFace> previous_faces, const AdaptiveLeaf& current,
        const NativeFragmentRecord& current_fragment, std::span<NativeKernelSourceFace> faces,
        const geometry_detail::NativeConstructionAdmission& admission)
    {
        return can_rebind_source(kernel, previous, current) && kernel->native_authority_->source_rebind_(
            *kernel, previous, previous_fragment, previous_faces, current, current_fragment, faces, admission);
    }
    [[nodiscard]] bool matches(const std::shared_ptr<const AdaptiveProviderKernel>& kernel) const noexcept
    {
        const auto actual = kernel_.lock();
        return actual && actual.get() == kernel.get() && !actual.owner_before(kernel) && !kernel.owner_before(actual);
    }
    const std::weak_ptr<const AdaptiveProviderKernel> kernel_;
    const std::shared_ptr<const geometry_detail::NativeAllocationEnvelope> kernel_allocation_;
    const std::shared_ptr<const geometry_detail::NativeSchemaOwner> schema_;
    std::shared_ptr<const geometry_detail::NativeAllocationEnvelope> allocation_;
    AdaptiveCellMetrics (*prepare_source_)(const AdaptiveProviderKernel&, const AdaptiveLeaf&,
        const geometry_detail::NativeConstructionAdmission&) = nullptr;
    std::size_t (*source_face_count_)(const AdaptiveProviderKernel&, const AdaptiveLeaf&,
        const NativeFragmentRecord&) = nullptr;
    AdaptiveCellMetrics (*source_validator_)(const AdaptiveProviderKernel&, const AdaptiveLeaf&,
        const NativeFragmentRecord&, const geometry_detail::NativeConstructionAdmission&,
        std::span<NativeKernelSourceFace>, void*, SupportVisit) = nullptr;
    AdaptiveCellMetrics (*source_collector_)(const AdaptiveProviderKernel&, const AdaptiveLeaf&,
        const NativeFragmentRecord&, const geometry_detail::NativeConstructionAdmission&,
        const geometry_detail::NativeConstructorStorageOwner&, std::vector<NativeKernelSourceFace>&) = nullptr;
    bool (*source_rebind_)(const AdaptiveProviderKernel&, const AdaptiveLeaf&, const NativeFragmentRecord&,
        std::span<const NativeKernelSourceFace>, const AdaptiveLeaf&, const NativeFragmentRecord&,
        std::span<NativeKernelSourceFace>,
        const geometry_detail::NativeConstructionAdmission&) = nullptr;
    bool (*source_eligible_)(const AdaptiveProviderKernel&, const AdaptiveLeaf&) noexcept = nullptr;
    bool source_publication_ready_ = false;
    bool explicit_body_intake_ = false;
    void (*profiled_split_)(const AdaptiveProviderKernel&, const AdaptiveLeaf&, AdaptiveSplit,
        const geometry_detail::NativeConstructionAdmission&, void*, SplitVisit, bool) = nullptr;
    const ChildProjection children_;
};
} // namespace adaptive_detail
} // namespace SimpleFluid::Meshes::AMR
