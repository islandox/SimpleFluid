#include "FVM/details/FiniteGeometryAccess.hh"
#include "solvers/details/GeneralVoF/Execution.hh"
#include "solvers/details/GeneralVoF/FaceRates.hh"
#include <bit>
#include <cmath>
#include <limits>
#include <map>

namespace SimpleFluid::detail::general_vof
{
namespace
{
void append_word(std::vector<char>& bytes, std::uint64_t value)
{
    for (unsigned a = 0; a < 8; ++a)
        bytes.push_back(char(value >> (8 * a)));
}
} // namespace
using GO = DefaultTpetraTypes::global_ordinal_type;
using LO = DefaultTpetraTypes::local_ordinal_type;
template<ArithmeticMode Mode>
std::vector<PLIC::DynamicGeometryValue> face_rates(const MeshHandle<>& mesh, const VoF::GeneralVoF& phases,
    const GeneralVoFFiniteProjectionProvenance& projection, const GeneralVoFFlowWorkspace& flow)
{
    const Execution execution(mesh);
    using A = VoF::finite_detail::Arithmetic<Mode>;
    execution.require(phases.arithmetic_mode() == Mode && projection.arithmetic_mode == Mode &&
                          projection.face_rate.size() == mesh.num_owned_faces() &&
                          projection.face_radius.size() == mesh.num_owned_faces(),
        "Finite VoF projection rate authority is incomplete.");
    std::array<std::vector<double>, 6> wire;
    execution.collective(
        [&]
        {
            for (auto& v : wire)
                v.resize(mesh.num_faces());
            for (std::size_t f = 0; f < mesh.num_owned_faces(); ++f)
                for (unsigned k = 0; k < 2; ++k)
                {
                    const auto value = k == 0 ? projection.face_rate[f] : projection.face_radius[f];
                    int exponent = 0;
                    const auto mantissa = std::frexp(value, &exponent);
                    const auto high = static_cast<double>(mantissa), low = static_cast<double>(mantissa - high);
                    if (!std::isfinite(value) || (k && value < 0) || static_cast<long double>(high) + low != mantissa)
                        throw std::runtime_error("Finite VoF rate exceeds fixed wire precision.");
                    wire[3 * k][f] = high;
                    wire[3 * k + 1][f] = low;
                    wire[3 * k + 2][f] = exponent;
                }
        });
    for (auto& value : wire)
        value = flow.faces(value);
    std::vector<PLIC::DynamicGeometryValue> result(mesh.num_faces());
    execution.collective(
        [&]
        {
            for (std::size_t f = 0; f < result.size(); ++f)
            {
                long double values[2]{};
                for (unsigned k = 0; k < 2; ++k)
                {
                    const auto exponent = wire[3 * k + 2][f];
                    if (!std::isfinite(exponent) || exponent < std::numeric_limits<int>::min() ||
                        exponent > std::numeric_limits<int>::max() || exponent != std::trunc(exponent))
                        throw std::runtime_error("Finite VoF canonical wire exponent is malformed.");
                    values[k] =
                        std::scalbn(static_cast<long double>(wire[3 * k][f]) + wire[3 * k + 1][f], int(exponent));
                    if (!std::isfinite(values[k]) || (k && values[k] < 0))
                        throw std::runtime_error("Finite VoF canonical wire rate is nonfinite.");
                }
                result[f] = A::make_value(values[0], values[1]);
            }
        });
    return result;
}

template<ArithmeticMode Mode>
std::vector<VoF::GeneralVoFFaceVolume> face_volumes(const MeshHandle<>& mesh, const VoF::GeneralVoF& phases,
    const GeneralVoFFiniteProjectionProvenance& projection, const GeneralVoFFlowWorkspace& flow, double dt)
{
    const Execution execution(mesh);
    using A = VoF::finite_detail::Arithmetic<Mode>;
    const auto rates = face_rates<Mode>(mesh, phases, projection, flow);
    std::vector<VoF::GeneralVoFFaceVolume> result;
    execution.collective(
        [&]
        {
            for (const auto& tile : phases.tiles())
            {
                const auto local = mesh.overlap_face_map()->getLocalElement(GO(tile.face_gid));
                if (local < 0)
                    throw std::runtime_error("Finite VoF missing canonical face rate.");
                const auto integrated = A::multiply(rates[local], A::make_value(dt, 0));
                const auto diagnostic = static_cast<double>(integrated.value);
                if (!std::isfinite(diagnostic))
                    throw std::runtime_error("Finite VoF face diagnostic overflows.");
                result.push_back({tile.face_gid, diagnostic, std::nullopt, integrated});
            }
        });
    return result;
}

template<class Topology> Topology projection_topology(const MeshHandle<>& mesh, const VoF::GeneralVoF& phases)
{
    const Execution execution(mesh);
    Topology result;
    execution.collective(
        [&]
        {
            std::map<GO, const Meshes::AMR::AdaptiveTile*> tiles;
            for (const auto& tile : phases.tiles())
                tiles.emplace(GO(tile.face_gid), &tile);
            const auto lease = mesh.acquire_execution_view();
            for (std::size_t c = 0; c < mesh.num_owned_cells(); ++c)
                result.cells.push_back(
                    {mesh.cell_global_id(LO(c)), Meshes::AMR::encode_adaptive_key(phases.owned_order()[c]), {}});
            for (std::size_t f = 0; f < mesh.num_owned_faces(); ++f)
            {
                const auto face = LO(f);
                if (mesh.is_geometry_exterior_face(face))
                    continue;
                const auto& tile = *tiles.at(mesh.face_global_id(face));
                std::vector<char> identity;
                for (const auto* bytes : {&tile.domain, &tile.surface, &tile.identity})
                {
                    append_word(identity, bytes->size());
                    identity.insert(identity.end(), bytes->begin(), bytes->end());
                }
                result.faces.push_back({mesh.face_global_id(face), mesh.cell_global_id(mesh.owner_cell(face)),
                    mesh.cell_global_id(mesh.neighbor_cell(face)), std::move(identity), {}, {}});
            }
        });
    return result;
}
} // namespace SimpleFluid::detail::general_vof
