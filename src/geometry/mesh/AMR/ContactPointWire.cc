#include "geometry/mesh/AMR/ContactPointWire.hh"
#include "geometry/PLIC/details/GeometryIdentityAccess.hh"
#include <algorithm>
#include <array>
#include <cfenv>
#include <limits>
#include <stdexcept>

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
namespace
{
class Environment
{
public:
    Environment()
    {
        if (std::feholdexcept(&saved_) != 0)
            throw std::runtime_error("Cannot isolate contact-point floating-point environment.");
    }
    ~Environment() { if (active_) std::fesetenv(&saved_); }
    void restore()
    {
        if (std::fesetenv(&saved_) != 0)
            throw std::runtime_error("Cannot restore contact-point floating-point environment.");
        active_ = false;
    }
    Environment(const Environment&) = delete;
    Environment& operator=(const Environment&) = delete;
private:
    std::fenv_t saved_;
    bool active_ = true;
};
struct Stream
{
    PLIC::details::OrdinaryIdentityKey point;
    std::span<const char> domain;
    std::size_t length;
    explicit Stream(ContactPointKeyView key) : domain(key.domain)
    {
        if (!key.point || domain.size() > std::numeric_limits<std::size_t>::max() - 16 - point.bytes.size())
            throw std::invalid_argument("Invalid contact-point key view.");
        // Ordinary points have a statically bounded inline representation.
        PLIC::details::GeometryIdentityAccess::emit_canonical(*key.point, &point,
            [](void* context, std::span<const char> bytes)
            {
                auto& output = *static_cast<PLIC::details::OrdinaryIdentityKey*>(context);
                if (bytes.size() > output.bytes.size() - output.size)
                    throw std::invalid_argument("Contact-point canonical key exceeds inline bound.");
                std::copy(bytes.begin(), bytes.end(), output.bytes.begin() + output.size);
                output.size += bytes.size();
            });
        length = 16 + domain.size() + point.size;
    }
    std::size_t size() const noexcept { return length; }
    char operator[](std::size_t i) const noexcept
    {
        if (i < 8)
            return static_cast<char>(static_cast<std::uint64_t>(domain.size()) >> (8 * i));
        i -= 8;
        if (i < domain.size())
            return domain[i];
        i -= domain.size();
        if (i < 8)
            return static_cast<char>(static_cast<std::uint64_t>(point.size) >> (8 * i));
        return point.bytes[i - 8];
    }
};
template<class A, class B> bool less(const A& first, const B& second)
{
    const auto common = std::min(first.size(), second.size());
    for (std::size_t i = 0; i < common; ++i)
    {
        if (first[i] < second[i])
            return true;
        if (second[i] < first[i])
            return false;
    }
    return first.size() < second.size();
}
template<class A> std::uint64_t hash(const A& value) noexcept
{
    std::uint64_t result = 14695981039346656037ULL;
    for (std::size_t i = 0; i < value.size(); ++i)
    {
        result ^= static_cast<unsigned char>(value[i]);
        result *= 1099511628211ULL;
    }
    return result;
}
} // namespace
bool ContactPointWireLess::operator()(ContactPointKeyView first, ContactPointKeyView second) const
{
    Environment environment;
    const auto result = less(Stream(first), Stream(second));
    environment.restore();
    return result;
}
bool ContactPointWireLess::operator()(ContactPointKeyView first, std::span<const char> second) const
{
    Environment environment;
    const auto result = less(Stream(first), second);
    environment.restore();
    return result;
}
bool ContactPointWireLess::operator()(std::span<const char> first, ContactPointKeyView second) const
{
    Environment environment;
    const auto result = less(first, Stream(second));
    environment.restore();
    return result;
}
std::uint64_t ContactPointWireHasher::operator()(ContactPointKeyView value) const
{
    Environment environment;
    const auto result = hash(Stream(value));
    environment.restore();
    return result;
}
std::uint64_t ContactPointWireHasher::operator()(std::span<const char> value) const noexcept
{
    return hash(value);
}
std::size_t contact_point_key_size(ContactPointKeyView value)
{
    Environment environment;
    const auto result = Stream(value).size();
    environment.restore();
    return result;
}
std::size_t framed_contact_point_key_size(ContactPointKeyView value)
{
    const auto bytes = contact_point_key_size(value);
    if (bytes > std::numeric_limits<std::size_t>::max() - 8)
        throw std::invalid_argument("Framed contact key size overflows.");
    return bytes + 8;
}
void emit_framed_contact_point_key(
    ContactPointKeyView value, void* context, void (*append)(void*, std::span<const char>))
{
    if (!append) throw std::invalid_argument("Missing contact key emitter.");
    Environment environment;
    const Stream stream(value);
    const auto word = [&](std::size_t n)
    {
        std::array<char, 8> bytes;
        for (unsigned i = 0; i < 8; ++i) bytes[i] = static_cast<char>(std::uint64_t(n) >> (8 * i));
        append(context, bytes);
    };
    word(stream.size());
    word(value.domain.size());
    append(context, value.domain);
    word(stream.point.size);
    append(context, {stream.point.bytes.data(), stream.point.size});
    environment.restore();
}
void append_framed_contact_point_key(Writer& writer, ContactPointKeyView value)
{
    emit_framed_contact_point_key(value, &writer, [](void* context, std::span<const char> bytes)
        {
            auto& output = static_cast<Writer*>(context)->data;
            output.insert(output.end(), bytes.begin(), bytes.end());
        });
}
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
