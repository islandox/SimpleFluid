/** @file testGeneralVoFPhiClosure.cc
 * @brief Arithmetic provenance, nonzero circulation, and bounded closure rollback.
 */
#include "FVM/AdaptiveVoF.hh"
#include "solvers/details/ConservativeVoFExactArithmetic.hh"
#include "solvers/details/GeneralVoFPhiClosure.hh"
#include "solvers/details/GeneralVoFProjectionArithmetic.hh"
#include "utils/testing_environment.hh"
#include <Teuchos_CommHelpers.hpp>
#include <cmath>
#include <gtest/gtest.h>
#include <limits>
namespace
{
using namespace SimpleFluid;
using LO = DefaultTpetraTypes::local_ordinal_type;
using GO = DefaultTpetraTypes::global_ordinal_type;
using M = Meshes::ExactGeometryMeasure;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);
TEST(GeneralVoFPhiClosureTest, ArithmeticTracksCancellationAndRejectsUncertainDenominator)
{
    detail::GeneralVoFProjectionArithmetic arithmetic(1024 * 1024);
    const auto third = arithmetic.divide(arithmetic.leaf(1), arithmetic.leaf(3));
    const auto difference = arithmetic.subtract(arithmetic.multiply(third, arithmetic.leaf(3)), arithmetic.leaf(1));
    EXPECT_EQ(difference.value, 0);
    EXPECT_GT(arithmetic.radius(difference).sign(), 0);
    EXPECT_EQ(difference.operations, 3u);
    EXPECT_THROW((void) arithmetic.divide(arithmetic.leaf(1), difference), std::runtime_error);
    const auto zero = arithmetic.multiply(arithmetic.leaf(0), arithmetic.leaf(3));
    EXPECT_EQ(arithmetic.radius(zero).sign(), 0);
}
TEST(GeneralVoFPhiClosureTest, InnerPressureScaleBoundsNormalizedRhsWithoutCarrierRangeLoss)
{
    constexpr auto cap = 1024 * 1024;
    auto huge = M::from_value(2);
    for (unsigned i = 0; i < 12; ++i)
        huge = huge.squared();
    const auto tiny = M::from_value(std::numeric_limits<double>::denorm_min()).squared();
    for (const auto& maximum : {M::from_value(1), M::from_value(3),
             M::from_value(1).divided_by(M::from_value(3)), huge, tiny})
    {
        const auto authority = maximum.encode();
        const auto scale = detail::general_vof_pressure_normalization_scale(maximum, cap);
        EXPECT_GT(scale.sign(), 0);
        EXPECT_GT(scale.compare(maximum), 0);
        EXPECT_LE(scale.compare(maximum.scaled(2)), 0);
        const auto normalized = maximum.divided_by(scale, cap);
        EXPECT_GE(normalized.certified().as_double(), .5);
        EXPECT_LT(normalized.certified().as_double(), 1);
        const auto recovered = normalized.divided_by(M::from_value(1).divided_by(scale, cap), cap);
        EXPECT_EQ(recovered.compare(maximum), 0);
        EXPECT_EQ(maximum.encode(), authority);
    }
    EXPECT_EQ(detail::general_vof_pressure_normalization_scale(M{}, cap).compare(M::from_value(1)), 0);
    EXPECT_THROW((void) detail::general_vof_pressure_normalization_scale(M::from_value(-1), cap),
        std::invalid_argument);
    EXPECT_THROW((void) detail::general_vof_pressure_normalization_scale(M::from_value(3), 1), std::length_error);
}
TEST(GeneralVoFPhiClosureTest, CirculationClosureBoundFailureAndRecovery)
{
    Meshes::AMR::AMRGrid grid;
    grid.root_blocks = {2, 1, 1};
    grid.block_cells = {2, 2, 2};
    grid.extent = {4, 2, 2};
    grid.periodic = {true, true, true};
    VoF::AdaptiveVoF geometry(std::make_shared<const Meshes::AMR::CartesianAMR>(grid));
    const auto mesh = geometry.mesh();
    // Fixed fixture identities use physical index/face location, independently of
    // native GID assignment. The production caller supplies provider certificates.
    const auto identity = [](const auto& x, int axis)
    {
        std::vector<char> key;
        for (double v : {x.x, x.y, x.z})
            key.push_back(char(std::lround(2 * v)));
        key.push_back(char(axis));
        return key;
    };
    const auto radius = M::from_value(std::ldexp(1.0, -45));
    std::vector<detail::GeneralVoFPhiCell> cells;
    for (std::size_t c = 0; c < mesh->num_owned_cells(); ++c)
        cells.push_back({mesh->cell_global_id(LO(c)), identity(mesh->cell_centroid(LO(c)), 0), radius.scaled(6)});
    std::vector<detail::GeneralVoFPhiFace> base;
    GO local_first = std::numeric_limits<GO>::max(), first = 0;
    for (std::size_t f = 0; f < mesh->num_owned_faces(); ++f)
    {
        const auto face = LO(f);
        if (mesh->is_geometry_exterior_face(face))
            continue;
        const auto normal = mesh->face_normal(face);
        const int axis = std::abs(normal.x) > .5 ? 0 : std::abs(normal.y) > .5 ? 1 : 2;
        const auto phi = M::from_value(axis == 0 ? normal.x : 0);
        base.push_back({mesh->face_global_id(face), mesh->cell_global_id(mesh->owner_cell(face)),
            mesh->cell_global_id(mesh->neighbor_cell(face)), identity(mesh->face_centroid(face), axis), phi, radius});
        local_first = std::min(local_first, mesh->face_global_id(face));
    }
    Teuchos::reduceAll(*mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_MIN, 1, &local_first, &first);
    for (int multiplier : {0, 1, 1024, 1})
    {
        auto faces = base;
        for (auto& f : faces)
            if (f.face == first)
                f.volume = f.volume + radius.scaled(multiplier / 8.0);
        if (multiplier == 1024)
        {
            EXPECT_FALSE(detail::try_close_general_vof_phi(mesh, cells, faces, 4 * 1024 * 1024));
            EXPECT_THROW((void) detail::close_general_vof_phi(mesh, cells, faces, 4 * 1024 * 1024),
                detail::GeneralVoFPhiInfeasible);
            continue;
        }
        const auto result = detail::close_general_vof_phi(mesh, cells, faces, 4 * 1024 * 1024);
        ASSERT_EQ(result.volume.size(), faces.size());
        for (std::size_t i = 0; i < faces.size(); ++i)
        {
            const auto magnitude =
                result.correction[i].sign() < 0 ? result.correction[i].scaled(-1) : result.correction[i];
            EXPECT_LE(magnitude.compare(radius), 0);
            EXPECT_EQ(result.volume[i].compare(faces[i].volume + result.correction[i]), 0);
        }
    }

    // A source-free circulation can carry thousands of historical binary bits.
    // Fixed-precision proposals spend only the independently supplied original
    // arithmetic capacities; original input authority remains unchanged.
    auto power = M::from_value(2);
    for (unsigned i = 0; i < 12; ++i)
        power = power.squared();
    power = power.scaled(std::ldexp(1.0, 512));
    const auto tail = M::from_value(1).divided_by(power);
    auto precise = base;
    for (auto& face : precise)
        face.volume = (face.volume + tail.scaled(face.volume.sign())).compacted(1024 * 1024);
    const auto bounded = detail::close_general_vof_bounded_rate(mesh, cells, precise, 4 * 1024 * 1024);
    for (std::size_t i = 0; i < precise.size(); ++i)
    {
        EXPECT_EQ(bounded.volume[i].compare(base[i].volume), 0);
        EXPECT_EQ((precise[i].volume + bounded.correction[i]).compare(bounded.volume[i]), 0);
        const auto magnitude = bounded.correction[i].sign() < 0 ? bounded.correction[i].scaled(-1)
                                                               : bounded.correction[i];
        EXPECT_LE(magnitude.compare(precise[i].arithmetic_radius), 0);
        if (base[i].volume.sign())
            EXPECT_LT(bounded.volume[i].storage_bytes(), precise[i].volume.storage_bytes());
    }
    // Scalar and packet boundaries differ with actual rank ownership. Vary only
    // authenticated key padding, within the SAME four-MiB cap, and require an
    // observed typed resource fallback at an admitted original boundary.
    bool found_resource_fallback = false;
    for (const auto key_bytes : {256, 512, 1024, 1536, 2048})
    {
        auto framed = precise;
        for (auto& face : framed)
            face.identity.resize(key_bytes, 'p');
        std::size_t rejected_budget = 0, admitted_budget = 4 * 1024 * 1024;
        try
        {
            (void) detail::close_general_vof_phi(mesh, cells, framed, admitted_budget);
        }
        catch (const std::exception&)
        {
            continue;
        }
        while (admitted_budget - rejected_budget > 1)
        {
            const auto budget = rejected_budget + (admitted_budget - rejected_budget) / 2;
            try
            {
                (void) detail::close_general_vof_phi(mesh, cells, framed, budget);
                admitted_budget = budget;
            }
            catch (const std::exception&)
            {
                rejected_budget = budget;
            }
        }
        try
        {
            const auto boundary = detail::close_general_vof_bounded_rate(mesh, cells, framed, admitted_budget);
            if (!boundary.bounded_resource_fallback)
                continue;
            found_resource_fallback = true;
            for (std::size_t i = 0; i < framed.size(); ++i)
            {
                EXPECT_EQ(boundary.volume[i].compare(framed[i].volume), 0);
                EXPECT_EQ(boundary.correction[i].sign(), 0);
            }
            break;
        }
        catch (const std::exception&)
        {
            continue;
        }
    }
    EXPECT_TRUE(found_resource_fallback);
    auto strict_cells = cells;
    for (auto& cell : strict_cells)
        cell.arithmetic_radius = {};
    const auto unchanged = detail::close_general_vof_bounded_rate(mesh, strict_cells, precise, 4 * 1024 * 1024);
    for (std::size_t i = 0; i < precise.size(); ++i)
    {
        // Positive face capacity cannot authorize spending a zero cell budget.
        EXPECT_EQ(unchanged.volume[i].compare(precise[i].volume), 0);
        EXPECT_EQ(unchanged.correction[i].sign(), 0);
    }
    auto tiny_faces = base;
    const auto below_range = M::from_value(std::numeric_limits<double>::denorm_min()).squared();
    for (auto& face : tiny_faces)
        face.volume = below_range.scaled(face.volume.sign());
    const auto tiny_closed = detail::close_general_vof_bounded_rate(mesh, cells, tiny_faces, 4 * 1024 * 1024);
    for (std::size_t i = 0; i < tiny_faces.size(); ++i)
        EXPECT_EQ(tiny_closed.volume[i].compare(tiny_faces[i].volume), 0);
    for (auto& face : precise)
        face.arithmetic_radius = {};
    const auto immutable = detail::close_general_vof_bounded_rate(mesh, cells, precise, 4 * 1024 * 1024);
    for (std::size_t i = 0; i < precise.size(); ++i)
    {
        EXPECT_EQ(immutable.volume[i].compare(precise[i].volume), 0);
        EXPECT_EQ(immutable.correction[i].sign(), 0);
    }

    auto invalid = base;
    for (auto& face : invalid)
        face.identity.clear();
    EXPECT_THROW((void) detail::try_close_general_vof_phi(mesh, cells, invalid, 4 * 1024 * 1024), std::runtime_error);
    EXPECT_THROW((void) detail::try_close_general_vof_phi(mesh, cells, base, 1), std::runtime_error);

    // Independent xy layers are connected physically by immutable z faces.
    // A forest built from all faces could select those zero-capacity links and
    // incorrectly reject a perfectly closable upper-layer xy perturbation.
    auto split = base;
    GO local_x = std::numeric_limits<GO>::max(), upper_x = 0;
    GO local_z = std::numeric_limits<GO>::max(), cross_z = 0;
    for (auto& face : split)
    {
        const auto lid = mesh->owned_face_map()->getLocalElement(face.face);
        const auto normal = mesh->face_normal(lid);
        const auto center = mesh->face_centroid(lid);
        if (std::abs(normal.z) > .5)
        {
            face.arithmetic_radius = M{};
            local_z = std::min(local_z, face.face);
        }
        if (std::abs(normal.x) > .5 && center.z > 1)
            local_x = std::min(local_x, face.face);
    }
    Teuchos::reduceAll(*mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_MIN, 1, &local_x, &upper_x);
    Teuchos::reduceAll(*mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_MIN, 1, &local_z, &cross_z);
    for (auto& face : split)
        if (face.face == upper_x)
            face.volume = face.volume + radius.scaled(.125);
    const auto closed = detail::close_general_vof_phi(mesh, cells, split, 4 * 1024 * 1024);
    for (std::size_t i = 0; i < split.size(); ++i)
        if (!split[i].arithmetic_radius.sign())
            EXPECT_EQ(closed.correction[i].sign(), 0);
    auto incompatible = split;
    for (auto& face : incompatible)
        if (face.face == cross_z)
            face.volume = face.volume + radius.scaled(.125);
    EXPECT_THROW((void) detail::close_general_vof_phi(mesh, cells, incompatible, 4 * 1024 * 1024), std::runtime_error);
    EXPECT_NO_THROW((void) detail::close_general_vof_phi(mesh, cells, split, 4 * 1024 * 1024));
}

TEST(GeneralVoFPhiClosureTest, ExactRepresentedRateRetainsUnderflowingAreaProduct)
{
    Meshes::AMR::AMRGrid grid;
    grid.root_blocks = {1, 1, 1};
    grid.block_cells = {2, 2, 2};
    grid.extent = {2, 2, 1};
    grid.periodic = {true, true, true};
    VoF::AdaptiveVoF geometry(std::make_shared<const Meshes::AMR::CartesianAMR>(grid));
    const auto mesh = geometry.mesh();
    detail::GeneralVoFProjectionArithmetic arithmetic(1024 * 1024);
    const double eta = std::numeric_limits<double>::denorm_min();
    std::vector<detail::GeneralVoFRoundedValue> values;
    {
        const auto lease = mesh->acquire_execution_view();
        for (std::size_t f = 0; f < mesh->num_owned_faces(); ++f)
            values.push_back(arithmetic.leaf(eta * mesh->face_normal(LO(f)).x));
    }
    const auto result = detail::certify_general_vof_projection_rate(mesh, values, 4 * 1024 * 1024);
    EXPECT_TRUE(result.arithmetic_admitted);
    const auto lease = mesh->acquire_execution_view();
    for (std::size_t f = 0; f < values.size(); ++f)
    {
        EXPECT_EQ(result.owned_face_radius[f].sign(), 0);
        const double normal = mesh->face_normal(LO(f)).x;
        if (std::abs(normal) > .5)
        {
            EXPECT_EQ(values[f].value * mesh->face_area(LO(f)), 0);
            EXPECT_NE(result.owned_rate[f].sign(), 0);
            EXPECT_EQ(result.owned_rate[f].scaled(2).compare(M::from_value(eta * normal)), 0);
        }
    }
}

TEST(GeneralVoFPhiClosureTest, TemporaryLeaseKeepsPersistentCapAndRecovers)
{
    constexpr std::size_t cap = 1024;
    auto half = M::from_value(.5);
    auto third = M::from_value(1).divided_by(M::from_value(3));
    for (unsigned i = 0; i < 12; ++i)
    {
        half = half.squared();
        third = third.squared();
    }
    ASSERT_LT(half.storage_bytes(), cap);
    ASSERT_LT(third.storage_bytes(), cap);
    EXPECT_THROW((void) detail::vof_exact::add(half, half, cap), std::runtime_error);
    const auto sum = detail::vof_exact::leased_add(half, half, cap);
    EXPECT_EQ(sum.compare(half.scaled(2)), 0);
    EXPECT_LE(sum.storage_bytes(), cap);
    EXPECT_THROW((void) detail::vof_exact::leased_add(half, third, cap), std::runtime_error);
    EXPECT_NO_THROW((void) detail::vof_exact::leased_add(half, half, cap));
}
TEST(GeneralVoFPhiClosureTest, FixedPrimaryReferenceSurvivesRoundedZeroInitialResidual)
{
    const double eta = std::numeric_limits<double>::denorm_min();
    const auto initial = M::from_value(eta).scaled(.5);
    EXPECT_EQ(static_cast<double>(initial.enclosure().value), 0);
    // Independently rounded face products cancel, but the original represented
    // area/velocity inputs retain a half-eta physical residual.
    EXPECT_EQ(eta * 1.25, eta * .75);
    const auto cancellation = M::from_value(eta).scaled(1.25) - M::from_value(eta).scaled(.75);
    EXPECT_EQ(cancellation.compare(initial), 0);
    const auto limit = detail::general_vof_projection_divergence_limit(initial, eta, 0, .25, 1, 4096);
    EXPECT_EQ(limit.compare(initial.scaled(.25)), 0);
    EXPECT_NE(limit.sign(), 0);
    // Pressure polishing changes neither the stored initial reference nor limit.
    const auto later = initial.scaled(.01);
    EXPECT_LT(later.compare(limit), 0);
    EXPECT_EQ(detail::general_vof_projection_divergence_limit(initial, eta, 0, .25, 1, 4096).compare(limit), 0);
}

TEST(GeneralVoFPhiClosureTest, PacketArenaSeparatesResidentInputsFromExchangePeakAndRecovers)
{
    Meshes::AMR::AMRGrid grid;
    grid.block_cells = {2, 2, 2};
    grid.extent = {2, 2, 2};
    grid.periodic = {true, true, true};
    VoF::AdaptiveVoF geometry(std::make_shared<const Meshes::AMR::CartesianAMR>(grid));
    const auto mesh = geometry.mesh();
    auto positive = M::from_value(1).divided_by(M::from_value(3));
    for (unsigned i = 0; i < 11; ++i)
        positive = positive.squared();
    const auto negative = positive.scaled(-1);
    detail::GeneralVoFProjectionArithmetic arithmetic(1024 * 1024);
    std::vector<detail::GeneralVoFRoundedValue> values;
    std::vector<M> rate, radius(mesh->num_owned_faces());
    {
        const auto lease = mesh->acquire_execution_view();
        for (std::size_t f = 0; f < mesh->num_owned_faces(); ++f)
        {
            const auto n = mesh->face_normal(LO(f));
            rate.push_back(n.x + n.y + n.z > 0 ? positive : negative);
            values.push_back(arithmetic.leaf(0)); // Diagnostic-only tiny circulation.
        }
    }
    constexpr std::size_t budget = 512 * 1024;
    const auto accepted = detail::certify_general_vof_projection_rate(mesh, values, budget, &rate, &radius);
    EXPECT_TRUE(accepted.arithmetic_admitted);
    EXPECT_GE(accepted.packet_exchange_cap, 4 * accepted.packet_input_cap);
    EXPECT_LE(accepted.packet_exchange_cap, 4 * accepted.packet_input_cap + 3);
    EXPECT_LE(accepted.packet_resident_input_bytes + accepted.packet_exchange_peak_bytes, accepted.packet_arena_cap);
    EXPECT_LE(
        accepted.packet_resident_input_bytes + accepted.packet_resident_output_bytes + accepted.packet_codec_peak_bytes,
        accepted.packet_arena_cap);
    if (mesh->owned_cell_map()->getComm()->getSize() == 1)
    {
        // This actual packed-buffer peak would be rejected if the engine were
        // incorrectly given only the outgoing payload cap again.
        EXPECT_GT(accepted.packet_exchange_peak_bytes, accepted.packet_input_cap);
        EXPECT_EQ(accepted.packet_sent_bytes, accepted.packet_received_bytes);
    }
    EXPECT_THROW(
        (void) detail::certify_general_vof_projection_rate(mesh, values, 8192, &rate, &radius), std::runtime_error);
    const auto recovered = detail::certify_general_vof_projection_rate(mesh, values, budget, &rate, &radius);
    EXPECT_TRUE(recovered.arithmetic_admitted);
    for (std::size_t f = 0; f < rate.size(); ++f)
        EXPECT_EQ(recovered.owned_rate[f].compare(rate[f]), 0);
}

TEST(GeneralVoFPhiClosureTest, LeasedMaximumSeparatesNegativeScalarGrowthAndCollectivePacketArena)
{
    Meshes::AMR::AMRGrid grid;
    grid.block_cells = {2, 2, 2};
    grid.extent = {2, 2, 2};
    VoF::AdaptiveVoF geometry(std::make_shared<const Meshes::AMR::CartesianAMR>(grid));
    const auto map = geometry.mesh()->owned_cell_map();
    auto magnitude = M::from_value(3);
    for (unsigned i = 0; i < 14; ++i)
        magnitude = magnitude.squared();
    const auto negative = magnitude.scaled(-1);
    constexpr std::size_t scalar_cap = 21640, communication_arena = 1024 * 1024;
    ASSERT_LT(magnitude.storage_bytes(), scalar_cap);
    ASSERT_GT(magnitude.storage_bytes() + M::from_value(-1).storage_bytes(), scalar_cap / 8);
    std::vector<M> values{negative};
    // Keep the original momentum reduction's single-budget admission unchanged.
    EXPECT_THROW((void) detail::vof_exact::maximum_absolute(values, *map, scalar_cap), std::runtime_error);
    const auto accepted = detail::vof_exact::leased_maximum_absolute(values, *map, scalar_cap, communication_arena);
    EXPECT_EQ(accepted.compare(magnitude), 0);
    const auto record_wire = magnitude.encode(scalar_cap).size() + 2 * sizeof(std::uint64_t);
    const auto small_arena = 8 * (record_wire + 128);
    if (map->getComm()->getSize() > 1)
    {
        // Each sender's one scalar fits. Root's aggregate receive buffers do
        // not; all owners, including empty ones, must reject and recover.
        EXPECT_THROW(
            (void) detail::vof_exact::leased_maximum_absolute(values, *map, scalar_cap, small_arena), std::exception);
    }
    else
        EXPECT_NO_THROW((void) detail::vof_exact::leased_maximum_absolute(values, *map, scalar_cap, small_arena));
    const auto recovered = detail::vof_exact::leased_maximum_absolute(values, *map, scalar_cap, communication_arena);
    EXPECT_EQ(recovered.compare(magnitude), 0);
    EXPECT_THROW(
        (void) detail::vof_exact::leased_maximum_absolute(values, *map, 1024, communication_arena), std::runtime_error);
    EXPECT_NO_THROW((void) detail::vof_exact::leased_maximum_absolute(values, *map, scalar_cap, communication_arena));
}
} // namespace
