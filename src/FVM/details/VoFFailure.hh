/** @file VoFFailure.hh
 * @brief Bounded diagnostics for rank-local VoF work and collective rejection.
 */
#pragma once
#include "SimpleFluidExport.hh"
#include "geometry/PLIC/GeometryError.hh"
#include "utils/ArithmeticMode.hh"
#include <Teuchos_CommHelpers.hpp>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <limits>
#include <new>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <utility>

namespace SimpleFluid::detail::vof_failure
{
enum class Category
{
    Unknown,
    Allocation,
    ResourceLimit,
    Metadata,
    Arithmetic
};
enum class Field : std::uint64_t
{
    Cell = 1,
    Face = 2,
    Donor = 4,
    Recipient = 8,
    Construction = 16,
    Arithmetic = 32,
    Scale = 64,
    VolumeBudget = 128,
    ByteBudget = 256,
    BytesUsed = 512,
    PairBudget = 1024,
    PairTests = 2048,
    Retry = 4096,
    GeometryCode = 8192,
    Simplex = 16384,
    Control = 32768
};
/** Fixed-size wire record. A field is meaningful only when its presence bit is set.
 * Construction is an ordinal within the identified face, never a fabricated global ID.
 * Simplex stores four physical input vertices in xyz order, when available.
 */
struct Record
{
    std::uint64_t present = 0;
    int origin_rank = -1;
    Category category = Category::Unknown;
    int geometry_code = -1;
    ArithmeticMode arithmetic_mode = ArithmeticMode::Ordinary;
    std::uint64_t cell = 0, face = 0, donor = 0, recipient = 0, construction = 0;
    std::uint64_t byte_budget = 0, bytes_used = 0, pair_budget = 0, pair_tests = 0;
    unsigned retry = 0;
    long double scale = 0, volume_budget = 0, control = 0;
    std::array<long double, 12> simplex{};
    std::array<char, 80> stage{};
    std::array<char, 384> message{};
    [[nodiscard]] bool has(Field field) const noexcept { return (present & std::uint64_t(field)) != 0; }
    void mark(Field field) noexcept { present |= std::uint64_t(field); }
};
static_assert(std::is_trivially_copyable_v<Record>);
template<std::size_t N> void copy(std::array<char, N>& to, const char* from) noexcept
{
    if (!from)
        return;
    std::size_t i = 0;
    for (; i + 1 < N && from[i]; ++i)
        to[i] = from[i];
    to[i] = 0;
}
struct ContextState
{
    Record* active = nullptr;
    Record unwound;
    bool captured = false;
};
// One provider shares scoped context across FVM, Solvers, and calling DSOs.
SIMPLEFLUID_FVM_EXPORT ContextState& context_state() noexcept;
/** Stack-only inherited context. The innermost unwinding scope is retained. */
class Scope
{
public:
    explicit Scope(const char* stage = nullptr) noexcept
        : previous_(context_state().active), exceptions_(std::uncaught_exceptions())
    {
        // A fresh operation after a locally recovered trial starts new context.
        if (std::uncaught_exceptions() == 0)
            context_state().captured = false;
        if (previous_)
            record = *previous_;
        if (stage)
            copy(record.stage, stage);
        context_state().active = &record;
    }
    ~Scope()
    {
        if (std::uncaught_exceptions() > exceptions_ && !context_state().captured)
        {
            context_state().unwound = record;
            context_state().captured = true;
        }
        context_state().active = previous_;
    }
    Scope(const Scope&) = delete;
    Scope& operator=(const Scope&) = delete;
    Record record;

private:
    Record* previous_;
    int exceptions_;
};
inline const char* category_name(Category category) noexcept
{
    switch (category)
    {
        case Category::Allocation:
            return "allocation";
        case Category::ResourceLimit:
            return "resource_limit";
        case Category::Metadata:
            return "metadata";
        case Category::Arithmetic:
            return "arithmetic";
        default:
            return "unknown";
    }
}
/** Shared inspection interface, independent of the legacy exception base. */
class SIMPLEFLUID_PUBLIC_TYPE Failure
{
public:
    virtual ~Failure() = default;
    [[nodiscard]] virtual const Record& record() const noexcept = 0;
    [[nodiscard]] virtual std::exception_ptr local_cause() const noexcept = 0;
};
template<class Base> class SIMPLEFLUID_PUBLIC_TYPE CollectiveFailure final : public Base, public Failure
{
public:
    CollectiveFailure(const Record& record, std::exception_ptr cause) : Base(""), record_(record), cause_(cause)
    {
        std::snprintf(text_.data(), text_.size(),
            "VoF collective rejection: stage=%s category=%s origin_rank=%d cause=%s "
            "[context presence=0x%llx cell=%llu face=%llu donor=%llu recipient=%llu construction=%llu "
            "arithmetic=%d scale=%Lg volume_budget=%Lg byte_budget=%llu bytes_used=%llu retry=%u; "
            "unmarked fields are absent]",
            record.stage.data(), category_name(record.category), record.origin_rank, record.message.data(),
            static_cast<unsigned long long>(record.present), static_cast<unsigned long long>(record.cell),
            static_cast<unsigned long long>(record.face), static_cast<unsigned long long>(record.donor),
            static_cast<unsigned long long>(record.recipient), static_cast<unsigned long long>(record.construction),
            int(record.arithmetic_mode), record.scale, record.volume_budget,
            static_cast<unsigned long long>(record.byte_budget), static_cast<unsigned long long>(record.bytes_used),
            record.retry);
        if (record.has(Field::Simplex))
        {
            const auto used = std::char_traits<char>::length(text_.data());
            const auto& p = record.simplex;
            std::snprintf(text_.data() + used, text_.size() - used,
                " simplex=[(%.*Lg,%.*Lg,%.*Lg),(%.*Lg,%.*Lg,%.*Lg),(%.*Lg,%.*Lg,%.*Lg),(%.*Lg,%.*Lg,%.*Lg)]", 21, p[0],
                21, p[1], 21, p[2], 21, p[3], 21, p[4], 21, p[5], 21, p[6], 21, p[7], 21, p[8], 21, p[9], 21, p[10], 21,
                p[11]);
        }
    }
    [[nodiscard]] const char* what() const noexcept override { return text_.data(); }
    [[nodiscard]] const Record& record() const noexcept override { return record_; }
    [[nodiscard]] std::exception_ptr local_cause() const noexcept override { return cause_; }

private:
    Record record_;
    std::exception_ptr cause_;
    std::array<char, 1536> text_{};
};
inline void classify(Record& record, const std::exception_ptr& error) noexcept
{
    try
    {
        std::rethrow_exception(error);
    }
    catch (const Failure& failure)
    {
        record = failure.record();
    }
    catch (const PLIC::GeometryError& failure)
    {
        record.geometry_code = int(failure.code());
        record.mark(Field::GeometryCode);
        record.category = failure.code() == PLIC::GeometryErrorCode::ResourceLimit          ? Category::ResourceLimit
                          : failure.code() == PLIC::GeometryErrorCode::UnresolvedArithmetic ? Category::Arithmetic
                                                                                            : Category::Metadata;
        copy(record.message, failure.what());
    }
    catch (const std::bad_alloc& failure)
    {
        record.category = Category::Allocation;
        copy(record.message, failure.what());
    }
    catch (const std::length_error& failure)
    {
        record.category = Category::ResourceLimit;
        copy(record.message, failure.what());
    }
    catch (const std::invalid_argument& failure)
    {
        // Legacy ordinary geometry reports this represented metric through
        // invalid_argument. Classify its diagnostic without replacing the cause
        // or changing any geometry retry/fallback exception contract.
        record.category = std::string_view(failure.what())
                                  .starts_with("Ordinary FP geometry: computed signed volume is not positive (volume=")
                              ? Category::Arithmetic
                              : Category::Metadata;
        copy(record.message, failure.what());
    }
    catch (const std::exception& failure)
    {
        record.category = Category::Unknown;
        copy(record.message, failure.what());
    }
    catch (...)
    {
        record.category = Category::Unknown;
        copy(record.message, "non-standard local exception");
    }
}
/** action must contain only local work: no collective may be bypassed by an exception.
 * Consensus and bounded broadcast happen after catching every local exception.
 * The selected record is identical on every rank; each locally failing rank keeps
 * its own original cause, even when a lower rank supplied the selected record.
 */
template<class Base = std::invalid_argument, class Work>
void collective(const Teuchos::Comm<int>& comm, Work&& action, const char* stage = "general_vof")
{
    Scope scope(stage);
    const bool saved_captured = context_state().captured;
    const auto saved_context = context_state().unwound;
    context_state().captured = false;
    std::exception_ptr error;
    Record record;
    try
    {
        std::forward<Work>(action)();
    }
    catch (...)
    {
        error = std::current_exception();
        record = context_state().captured ? context_state().unwound : scope.record;
        record.origin_rank = comm.getRank();
        classify(record, error);
    }
    context_state().captured = saved_captured;
    context_state().unwound = saved_context;
    int local = error ? comm.getRank() : std::numeric_limits<int>::max(), origin = 0;
    Teuchos::reduceAll(comm, Teuchos::REDUCE_MIN, 1, &local, &origin);
    if (origin == std::numeric_limits<int>::max())
        return;
    // Fixed-size POD transfer never builds a dynamically sized diagnostic packet.
    Teuchos::broadcast(comm, origin, int(sizeof(Record)), reinterpret_cast<char*>(&record));
    throw CollectiveFailure<Base>(record, error);
}
} // namespace SimpleFluid::detail::vof_failure
