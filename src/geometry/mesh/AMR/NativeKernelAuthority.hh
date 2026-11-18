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
namespace adaptive_detail
{
class NativePublicationSession;
class NativeCellSourceProof;
class NativePublicationTestAccess;

struct NativeKernelSourceFace
{
    std::uint64_t surface = 0, native_face = 0;
    int outward_sign = 0;
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
        if constexpr (requires(const AdaptiveLeaf& leaf, const NativeFragmentRecord& fragment,
                                  std::span<NativeKernelSourceFace> faces)
            {
                kernel.owner->native_source_face_count(leaf);
                kernel.owner->validate_native_source(leaf, fragment, admission, faces);
            })
        {
            if constexpr (requires { Kernel::native_source_publication_ready; })
                authority.owner->source_publication_ready_ = Kernel::native_source_publication_ready;
            authority.owner->source_face_count_ = [](const AdaptiveProviderKernel& base, const AdaptiveLeaf& leaf)
            { return static_cast<const Kernel&>(base).native_source_face_count(leaf); };
            authority.owner->source_validator_ = [](const AdaptiveProviderKernel& base, const AdaptiveLeaf& leaf,
                                                    const NativeFragmentRecord& fragment,
                                                    const geometry_detail::NativeConstructionAdmission& admission,
                                                    std::span<NativeKernelSourceFace> faces)
            { return static_cast<const Kernel&>(base).validate_native_source(leaf, fragment, admission, faces); };
            authority.owner->source_eligible_ = [](const AdaptiveProviderKernel& base, const AdaptiveLeaf& leaf) noexcept
            { return static_cast<const Kernel&>(base).supports_native_source(leaf); };
            if constexpr (requires(const AdaptiveLeaf& leaf) { kernel.owner->prepare_native_source(leaf, admission); })
                authority.owner->prepare_source_ = [](const AdaptiveProviderKernel& base, const AdaptiveLeaf& leaf,
                                                       const geometry_detail::NativeConstructionAdmission& admission)
                { return static_cast<const Kernel&>(base).prepare_native_source(leaf, admission); };
        }
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
    // The numerical capability can be consumed by an isolated verifier.
    // Hierarchy publication also requires the actual provider funding profile.
    [[nodiscard]] static bool has_numerical_verifier(
        const std::shared_ptr<const AdaptiveProviderKernel>& kernel) noexcept
    {
        return source_profile_supported() && qualified(kernel) && kernel->native_authority_->source_validator_;
    }
    [[nodiscard]] static bool has_numerical_verifier(
        const std::shared_ptr<const AdaptiveProviderKernel>& kernel, const AdaptiveLeaf& leaf) noexcept
    {
        return has_numerical_verifier(kernel) && kernel->native_authority_->source_eligible_(*kernel, leaf);
    }
    [[nodiscard]] static std::size_t source_face_count(
        const std::shared_ptr<const AdaptiveProviderKernel>& kernel, const AdaptiveLeaf& leaf)
    {
        if (!has_numerical_verifier(kernel, leaf))
            throw std::invalid_argument("Native source lacks its exact admitted factory verifier.");
        const auto count = kernel->native_authority_->source_face_count_(*kernel, leaf);
        if (count < 4 || count > PLIC::GeometryLimits{}.maximum_faces)
            throw std::invalid_argument("Native source face count exceeds the supported closed-boundary limits.");
        return count;
    }
    [[nodiscard]] static AdaptiveCellMetrics validate_source(
        const std::shared_ptr<const AdaptiveProviderKernel>& kernel, const AdaptiveLeaf& leaf,
        const NativeFragmentRecord& fragment, const geometry_detail::NativeConstructionAdmission& admission,
        std::span<NativeKernelSourceFace> faces)
    {
        if (!has_numerical_verifier(kernel, leaf))
            throw std::invalid_argument("Native source lacks its exact admitted factory verifier.");
        return kernel->native_authority_->source_validator_(*kernel, leaf, fragment, admission, faces);
    }
    [[nodiscard]] static AdaptiveCellMetrics prepare_source(
        const std::shared_ptr<const AdaptiveProviderKernel>& kernel, const AdaptiveLeaf& leaf,
        const geometry_detail::NativeConstructionAdmission& admission)
    {
        if (!has_numerical_verifier(kernel, leaf) || !kernel->native_authority_->prepare_source_)
            throw std::invalid_argument("Native cell fragment requires authenticated original source metrics.");
        return kernel->native_authority_->prepare_source_(*kernel, leaf, admission);
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
    std::size_t (*source_face_count_)(const AdaptiveProviderKernel&, const AdaptiveLeaf&) = nullptr;
    AdaptiveCellMetrics (*source_validator_)(const AdaptiveProviderKernel&, const AdaptiveLeaf&,
        const NativeFragmentRecord&, const geometry_detail::NativeConstructionAdmission&,
        std::span<NativeKernelSourceFace>) = nullptr;
    bool (*source_eligible_)(const AdaptiveProviderKernel&, const AdaptiveLeaf&) noexcept = nullptr;
    bool source_publication_ready_ = false;
    const ChildProjection children_;
};
} // namespace adaptive_detail
} // namespace SimpleFluid::Meshes::AMR
