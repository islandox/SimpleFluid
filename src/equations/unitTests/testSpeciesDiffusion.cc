#include "equations/SpeciesDiffusion.hh"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <limits>

namespace
{
namespace R = SimpleFluid::Reacting;
using P = R::SpeciesPhase;
constexpr R::SpeciesId counter = 9, one = 14, two = 29;
std::shared_ptr<const R::SpeciesCatalog> schema()
{
    return std::make_shared<const R::SpeciesCatalog>(
        std::vector<R::SpeciesDefinition>{{counter, "finite liquid counter", .010, {{"X", 1}}, {P::Liquid}},
            {one, "first solute", .020, {{"X", 2}}, {P::Dissolved}},
            {two, "second solute", .040, {{"X", 4}}, {P::Dissolved}}},
        7);
}
R::SpeciesDiffusionBinding binding(const R::SpeciesCatalog& catalog)
{
    R::SpeciesDiffusionBinding result;
    result.counterSpecies = counter;
    result.identity = "manufactured finite-counter diffusion";
    result.schemaFingerprint = catalog.fingerprint();
    return result;
}
R::CellSpeciesState state(const R::SpeciesCatalog& catalog, double solvent = 100., double a = 1., double b = 2.)
{
    R::CellSpeciesState result(catalog);
    result.set_amount(catalog, counter, P::Liquid, solvent);
    result.set_amount(catalog, one, P::Dissolved, a);
    result.set_amount(catalog, two, P::Dissolved, b);
    return result;
}
R::CanonicalDiffusionFace face()
{
    R::CanonicalDiffusionFace result;
    result.identity = 83;
    result.owner = 0;
    result.neighbor = 1;
    result.moles = {0., .25, -.1};
    result.ownerMolarEnthalpy = {0., 100., 200.};
    result.neighborMolarEnthalpy = {0., 300., 400.};
    result.ownerCounterMolarEnthalpy = 10.;
    result.neighborCounterMolarEnthalpy = 20.;
    result.ownerVelocity = {2., 0., 0.};
    result.neighborVelocity = {-1., 0., 0.};
    return result;
}
long double mass(const R::SpeciesCatalog& catalog, const R::CellSpeciesState& values)
{
    long double result = 0.;
    for (const auto& species : catalog.species())
        for (std::size_t p = 0; p < 4; ++p)
            result += static_cast<long double>(species.molarMass) *
                      values.phase_amount(catalog, species.id, static_cast<P>(p));
    return result;
}

std::shared_ptr<const R::SpeciesCatalog> zero_counter_schema()
{
    auto species = schema()->species();
    species[0].id = 0;
    std::rotate(species.begin(), species.begin() + 1, species.end()); // Explicit zero is not catalog slot zero.
    return std::make_shared<const R::SpeciesCatalog>(std::move(species), 7);
}
R::CellSpeciesState zero_counter_state(const R::SpeciesCatalog& catalog, double counter_moles = 100.)
{
    R::CellSpeciesState result(catalog);
    result.set_amount(catalog, 0, P::Liquid, counter_moles);
    result.set_amount(catalog, one, P::Dissolved, 1.);
    result.set_amount(catalog, two, P::Dissolved, 2.);
    return result;
}
R::CanonicalDiffusionFace zero_counter_face()
{
    auto packet = face();
    std::rotate(packet.moles.begin(), packet.moles.begin() + 1, packet.moles.end());
    std::rotate(
        packet.ownerMolarEnthalpy.begin(), packet.ownerMolarEnthalpy.begin() + 1, packet.ownerMolarEnthalpy.end());
    std::rotate(packet.neighborMolarEnthalpy.begin(), packet.neighborMolarEnthalpy.begin() + 1,
        packet.neighborMolarEnthalpy.end());
    return packet;
}

TEST(SpeciesDiffusionTest, TwoSolutesAndFiniteCounterUseOneOppositeSignedDonorEnthalpyPacket)
{
    const auto catalog = schema();
    const std::vector initial{state(*catalog), state(*catalog)};
    const auto packet = face();
    const auto result = R::preview_species_diffusion(*catalog, initial, binding(*catalog), std::span{&packet, 1});
    ASSERT_EQ(result.faces.size(), 1u);
    EXPECT_NEAR(result.faces[0].counterMoles, -.1, 1e-15);
    EXPECT_NEAR(result.faces[0].enthalpy, -17., 1e-13); // .25*100 - .1*400 - .1*20.
    EXPECT_NEAR(result.faces[0].momentum[0], .015, 1e-15);
    EXPECT_DOUBLE_EQ(result.cells[0].enthalpy, -result.cells[1].enthalpy);
    EXPECT_DOUBLE_EQ(result.cells[0].momentum[0], -result.cells[1].momentum[0]);
    EXPECT_NEAR(result.cells[0].enthalpy, 17., 1e-13);
    EXPECT_DOUBLE_EQ(result.state[0].amount(*catalog, one, P::Dissolved), .75);
    EXPECT_NEAR(result.state[0].amount(*catalog, two, P::Dissolved), 2.1, 1e-15);
    EXPECT_NEAR(result.state[1].amount(*catalog, counter, P::Liquid), 99.9, 1e-13);
    for (std::size_t c = 0; c < initial.size(); ++c)
        EXPECT_NEAR(static_cast<double>(mass(*catalog, result.state[c]) - mass(*catalog, initial[c])), 0., 3e-16);
    // Applying cell-net moles times each local h would spuriously create 29 J.
    EXPECT_DOUBLE_EQ(result.cells[0].enthalpy + result.cells[1].enthalpy, 0.);
    EXPECT_EQ(initial[0].amounts, state(*catalog).amounts);
}

TEST(SpeciesDiffusionTest, ExactlyZeroPrimaryMassFluxNeverDebitsAnEmptyCounterSpecies)
{
    const auto catalog = schema();
    const std::vector initial{state(*catalog, 0., 2., 0.), state(*catalog, 0., 0., 1.)};
    auto packet = face();
    packet.moles = {0., 2., -1.};
    const auto result = R::preview_species_diffusion(*catalog, initial, binding(*catalog), std::span{&packet, 1});
    EXPECT_DOUBLE_EQ(result.faces[0].counterMoles, 0.);
    EXPECT_DOUBLE_EQ(result.faces[0].massResidual, 0.);
    EXPECT_DOUBLE_EQ(result.state[0].amount(*catalog, counter, P::Liquid), 0.);
    EXPECT_DOUBLE_EQ(result.state[1].amount(*catalog, counter, P::Liquid), 0.);
    EXPECT_DOUBLE_EQ(result.state[0].amount(*catalog, two, P::Dissolved), 1.);
    EXPECT_DOUBLE_EQ(result.state[1].amount(*catalog, one, P::Dissolved), 2.);
}

TEST(SpeciesDiffusionTest, AllFaceDonorsAreReservedJointlyWithoutBorrowingSimultaneousIncomingFlux)
{
    const auto catalog = schema();
    const std::vector initial{state(*catalog, 100., 1.), state(*catalog), state(*catalog)};
    auto first = face();
    first.identity = 1;
    first.moles = {0., .6, 0.};
    auto second = first;
    second.identity = 2;
    second.neighbor = 2;
    auto incoming = first;
    incoming.identity = 3;
    incoming.moles = {0., -.2, 0.};
    const std::vector packets{first, second, incoming};
    EXPECT_THROW(
        (void) R::preview_species_diffusion(*catalog, initial, binding(*catalog), packets), std::runtime_error);
    EXPECT_DOUBLE_EQ(initial[0].amount(*catalog, one, P::Dissolved), 1.);
}

TEST(SpeciesDiffusionTest, FiniteCounterIsSharedAcrossAllFacesAndCannotBeAnInfiniteSolvent)
{
    const auto catalog = schema();
    const std::vector initial{state(*catalog, 1.), state(*catalog), state(*catalog)};
    auto first = face();
    first.identity = 1;
    first.moles = {0., -.4, 0.};
    auto second = first;
    second.identity = 2;
    second.neighbor = 2;
    const std::vector packets{first, second}; // Each face asks for .8 counter mol from owner zero.
    EXPECT_THROW(
        (void) R::preview_species_diffusion(*catalog, initial, binding(*catalog), packets), std::runtime_error);
    EXPECT_DOUBLE_EQ(initial[0].amount(*catalog, counter, P::Liquid), 1.);
}

TEST(SpeciesDiffusionTest, OwnedSidePreviewsMatchOneCombinedCanonicalFaceAcrossAPartition)
{
    const auto catalog = schema();
    const std::vector initial{state(*catalog), state(*catalog)};
    const auto packet = face();
    const auto full = R::preview_species_diffusion(*catalog, initial, binding(*catalog), std::span{&packet, 1});
    auto leftFace = packet;
    leftFace.neighbor.reset();
    auto rightFace = packet;
    rightFace.owner.reset();
    rightFace.neighbor = 0;
    const std::vector left{initial[0]}, right{initial[1]};
    const auto a = R::preview_species_diffusion(*catalog, left, binding(*catalog), std::span{&leftFace, 1});
    const auto b = R::preview_species_diffusion(*catalog, right, binding(*catalog), std::span{&rightFace, 1});
    EXPECT_EQ(a.state[0].amounts, full.state[0].amounts);
    EXPECT_EQ(b.state[0].amounts, full.state[1].amounts);
    EXPECT_EQ(a.cells[0].change.amounts, full.cells[0].change.amounts);
    EXPECT_EQ(b.cells[0].change.amounts, full.cells[1].change.amounts);
    EXPECT_DOUBLE_EQ(a.cells[0].enthalpy + b.cells[0].enthalpy, 0.);
    EXPECT_TRUE(a.faces[0].owner.has_value());
    EXPECT_FALSE(b.faces[0].owner.has_value());
}

TEST(SpeciesDiffusionTest, EmptyOwnerAndZeroFluxHaveNoInventedInventoryOrEnergy)
{
    const auto catalog = schema();
    const auto empty = R::preview_species_diffusion(*catalog, {}, binding(*catalog), {});
    EXPECT_TRUE(empty.state.empty());
    EXPECT_TRUE(empty.cells.empty());
    EXPECT_TRUE(empty.faces.empty());
    const std::vector initial{state(*catalog, 0., 0., 0.), state(*catalog, 0., 0., 0.)};
    auto packet = face();
    packet.moles = {0., 0., 0.};
    const auto zero = R::preview_species_diffusion(*catalog, initial, binding(*catalog), std::span{&packet, 1});
    EXPECT_EQ(zero.state[0].amounts, initial[0].amounts);
    EXPECT_DOUBLE_EQ(zero.cells[0].enthalpy, 0.);
    EXPECT_DOUBLE_EQ(zero.faces[0].counterMoles, 0.);
}

TEST(SpeciesDiffusionTest, ForeignSchemaMetadataNonfinitePropertiesAndDuplicateFacesRejectAtomically)
{
    const auto catalog = schema();
    const std::vector initial{state(*catalog), state(*catalog)};
    auto packet = face();
    auto selected = binding(*catalog);
    selected.schemaFingerprint ^= 1;
    EXPECT_THROW(
        (void) R::preview_species_diffusion(*catalog, initial, selected, std::span{&packet, 1}), std::invalid_argument);
    selected = binding(*catalog);
    selected.revision = 0;
    EXPECT_THROW(
        (void) R::preview_species_diffusion(*catalog, initial, selected, std::span{&packet, 1}), std::invalid_argument);
    selected = binding(*catalog);
    selected.transported.phase = static_cast<P>(999);
    EXPECT_THROW(
        (void) R::preview_species_diffusion(*catalog, initial, selected, std::span{&packet, 1}), std::invalid_argument);
    selected = binding(*catalog);
    packet.moles[1] = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(
        (void) R::preview_species_diffusion(*catalog, initial, selected, std::span{&packet, 1}), std::invalid_argument);
    packet = face();
    packet.neighborMolarEnthalpy[2] = std::numeric_limits<double>::infinity();
    EXPECT_THROW(
        (void) R::preview_species_diffusion(*catalog, initial, selected, std::span{&packet, 1}), std::invalid_argument);
    const std::vector duplicate{face(), face()};
    EXPECT_THROW((void) R::preview_species_diffusion(*catalog, initial, selected, duplicate), std::invalid_argument);
    auto species = catalog->species();
    std::swap(species[1], species[2]);
    const R::SpeciesCatalog foreign(species, catalog->revision());
    EXPECT_THROW((void) R::preview_species_diffusion(foreign, initial, binding(foreign), {}), std::invalid_argument);
}

TEST(SpeciesDiffusionTest, CounterCannotBePrescribedTwiceInOneCompartment)
{
    auto species = schema()->species();
    species[0].allowedPhases.push_back(P::Dissolved);
    const R::SpeciesCatalog catalog(species);
    const std::vector initial{state(catalog), state(catalog)};
    auto selected = binding(catalog);
    selected.counter.phase = P::Dissolved;
    auto packet = face();
    packet.moles[0] = .1;
    EXPECT_THROW(
        (void) R::preview_species_diffusion(catalog, initial, selected, std::span{&packet, 1}), std::invalid_argument);
}

TEST(SpeciesDiffusionTest, CanonicalOrderingAndCompensatedEventsRetainRepeatedSubUlpTransfers)
{
    const auto catalog = schema();
    const std::vector initial{state(*catalog, 0., 1e20, 0.), state(*catalog, 1e20, 0., 0.)};
    std::vector<R::CanonicalDiffusionFace> packets;
    for (std::size_t id = 0; id < 10000; ++id)
    {
        auto packet = face();
        packet.identity = id;
        packet.moles = {0., 1., 0.};
        packet.ownerMolarEnthalpy = packet.neighborMolarEnthalpy = {0., 0., 0.};
        packet.ownerCounterMolarEnthalpy = packet.neighborCounterMolarEnthalpy = 0.;
        packet.ownerVelocity = packet.neighborVelocity = {};
        packets.push_back(std::move(packet));
    }
    const auto first = R::preview_species_diffusion(*catalog, initial, binding(*catalog), packets);
    std::reverse(packets.begin(), packets.end());
    const auto replay = R::preview_species_diffusion(*catalog, initial, binding(*catalog), packets);
    EXPECT_EQ(first.state[0].amounts, replay.state[0].amounts);
    EXPECT_EQ(first.cells[0].change.amounts, replay.cells[0].change.amounts);
    EXPECT_DOUBLE_EQ(first.cells[0].change.amount(*catalog, one, P::Dissolved), -10000.);
    EXPECT_DOUBLE_EQ(first.state[0].amount(*catalog, one, P::Dissolved), static_cast<double>(1e20L - 10000.L));
    EXPECT_LT(first.state[0].amount(*catalog, one, P::Dissolved), 1e20);
    EXPECT_DOUBLE_EQ(first.state[1].amount(*catalog, one, P::Dissolved), 10000.);
    EXPECT_DOUBLE_EQ(first.cells[1].change.amount(*catalog, counter, P::Liquid), -20000.);
}

TEST(SpeciesDiffusionTest, NonrepresentableMassTransfersRejectInsteadOfUnderflowingAway)
{
    const auto catalog = schema();
    const std::vector initial{state(*catalog, 1., std::numeric_limits<double>::denorm_min(), 0.), state(*catalog)};
    auto packet = face();
    packet.moles = {0., std::numeric_limits<double>::denorm_min(), 0.};
    EXPECT_THROW((void) R::preview_species_diffusion(*catalog, initial, binding(*catalog), std::span{&packet, 1}),
        std::invalid_argument);
    EXPECT_DOUBLE_EQ(initial[0].amount(*catalog, one, P::Dissolved), std::numeric_limits<double>::denorm_min());
}

TEST(SpeciesDiffusionTest, CounterSelectionIsRequiredEvenWhenZeroIsAValidCatalogId)
{
    const auto catalog = zero_counter_schema();
    ASSERT_EQ(catalog->index(0), 2U);
    const std::vector initial{zero_counter_state(*catalog), zero_counter_state(*catalog)};
    R::SpeciesDiffusionBinding selected;
    selected.identity = "counter selection must be explicit";
    selected.schemaFingerprint = catalog->fingerprint();
    EXPECT_FALSE(selected.counterSpecies.has_value());
    const auto packet = zero_counter_face();
    EXPECT_THROW(
        (void) R::preview_species_diffusion(*catalog, initial, selected, std::span{&packet, 1}), std::invalid_argument);
    auto zero_flux = packet;
    std::ranges::fill(zero_flux.moles, 0.);
    EXPECT_THROW((void) R::preview_species_diffusion(*catalog, initial, selected, std::span{&zero_flux, 1}),
        std::invalid_argument);
    EXPECT_THROW((void) R::preview_species_diffusion(*catalog, {}, selected, {}), std::invalid_argument);
    EXPECT_DOUBLE_EQ(initial[0].amount(*catalog, 0, P::Liquid), 100.);
    EXPECT_DOUBLE_EQ(initial[0].amount(*catalog, one, P::Dissolved), 1.);
}

TEST(SpeciesDiffusionTest, ExplicitZeroCounterIdConservesFiniteMaterialAndPairedDonorEnthalpy)
{
    const auto catalog = zero_counter_schema();
    const std::vector initial{zero_counter_state(*catalog), zero_counter_state(*catalog)};
    R::SpeciesDiffusionBinding selected;
    selected.identity = "ordinary explicitly selected ID zero";
    selected.schemaFingerprint = catalog->fingerprint();
    selected.counterSpecies = R::SpeciesId{0};
    ASSERT_TRUE(selected.counterSpecies.has_value());
    EXPECT_EQ(*selected.counterSpecies, 0U);
    const auto packet = zero_counter_face();
    const auto result = R::preview_species_diffusion(*catalog, initial, selected, std::span{&packet, 1});
    ASSERT_EQ(result.faces.size(), 1U);
    EXPECT_NEAR(result.faces[0].counterMoles, -.1, 1e-15);
    EXPECT_NEAR(result.state[0].amount(*catalog, 0, P::Liquid), 100.1, 1e-13);
    EXPECT_NEAR(result.state[1].amount(*catalog, 0, P::Liquid), 99.9, 1e-13);
    EXPECT_NEAR(result.faces[0].enthalpy, -17., 1e-13); // .25*100 - .1*400 - .1*20 J.
    EXPECT_DOUBLE_EQ(result.cells[0].enthalpy, -result.cells[1].enthalpy);
    for (const auto& species : catalog->species())
        for (std::size_t phase = 0; phase < 4; ++phase)
            EXPECT_DOUBLE_EQ(result.cells[0].change.amount(*catalog, species.id, static_cast<P>(phase)) +
                                 result.cells[1].change.amount(*catalog, species.id, static_cast<P>(phase)),
                0.);
    EXPECT_DOUBLE_EQ(initial[1].amount(*catalog, 0, P::Liquid), 100.);
}

TEST(SpeciesDiffusionTest, ExplicitZeroCounterStillHasAFiniteJointDonorBudget)
{
    const auto catalog = zero_counter_schema();
    const std::vector initial{zero_counter_state(*catalog), zero_counter_state(*catalog, .05)};
    R::SpeciesDiffusionBinding selected;
    selected.identity = "finite explicit zero counter";
    selected.schemaFingerprint = catalog->fingerprint();
    selected.counterSpecies = R::SpeciesId{0};
    const auto packet = zero_counter_face(); // Its neighbor owes .1 mol of the explicitly selected counter.
    EXPECT_THROW(
        (void) R::preview_species_diffusion(*catalog, initial, selected, std::span{&packet, 1}), std::runtime_error);
    EXPECT_DOUBLE_EQ(initial[1].amount(*catalog, 0, P::Liquid), .05);
    EXPECT_DOUBLE_EQ(initial[0].amount(*catalog, one, P::Dissolved), 1.);
}
} // namespace

TEST(SpeciesDiffusionTest, OpposingLargeFaceHeatRetainsIndependentRawJouleInEachCell)
{
    const auto catalog = schema();
    const std::vector initial{state(*catalog, 10, 2, 0), state(*catalog, 10, 2, 0)};
    auto outgoing = face(), incoming = face();
    outgoing.identity = 1;
    incoming.identity = 2;
    const double huge = std::ldexp(1., 60);
    outgoing.moles = {0, 1, 0};
    incoming.moles = {0, -1, 0};
    outgoing.ownerMolarEnthalpy = {0, huge, 0};
    outgoing.neighborMolarEnthalpy = {0, 0, 0};
    outgoing.ownerCounterMolarEnthalpy = 0;
    outgoing.neighborCounterMolarEnthalpy = -.5; // Counter = -2 mol: +1 J in canonical face sum.
    incoming.ownerMolarEnthalpy = {0, 0, 0};
    incoming.neighborMolarEnthalpy = {0, huge, 0};
    incoming.ownerCounterMolarEnthalpy = incoming.neighborCounterMolarEnthalpy = 0;
    outgoing.ownerVelocity = outgoing.neighborVelocity = {};
    incoming.ownerVelocity = incoming.neighborVelocity = {};
    const auto result =
        R::preview_species_diffusion(*catalog, initial, binding(*catalog), std::vector{outgoing, incoming});
    EXPECT_DOUBLE_EQ(result.cells[0].enthalpy, -1);
    EXPECT_DOUBLE_EQ(result.cells[1].enthalpy, 1);
    EXPECT_EQ(result.state[0].amounts, initial[0].amounts);
    EXPECT_EQ(result.state[1].amounts, initial[1].amounts);
    // Each public face total is rounded; it must not seed the physical cell sum.
    EXPECT_DOUBLE_EQ(result.faces[0].enthalpy, huge);
    EXPECT_DOUBLE_EQ(result.faces[1].enthalpy, -huge);
}

TEST(SpeciesDiffusionTest, CompleteLargeDepletionThenFiniteCreditRetainsPositiveEndpoint)
{
    const auto catalog = schema();
    const std::vector initial{state(*catalog, 10, 1e20, 0), state(*catalog, 1e21, 10, 0)};
    auto outgoing = face(), incoming = face();
    outgoing.identity = 1;
    incoming.identity = 2;
    outgoing.moles = {0, 1e20, 0};
    incoming.moles = {0, -1, 0};
    outgoing.ownerMolarEnthalpy = outgoing.neighborMolarEnthalpy = {0, 0, 0};
    incoming.ownerMolarEnthalpy = incoming.neighborMolarEnthalpy = {0, 0, 0};
    outgoing.ownerCounterMolarEnthalpy = outgoing.neighborCounterMolarEnthalpy = 0;
    incoming.ownerCounterMolarEnthalpy = incoming.neighborCounterMolarEnthalpy = 0;
    outgoing.ownerVelocity = outgoing.neighborVelocity = {};
    incoming.ownerVelocity = incoming.neighborVelocity = {};
    const auto result =
        R::preview_species_diffusion(*catalog, initial, binding(*catalog), std::vector{outgoing, incoming});
    EXPECT_DOUBLE_EQ(result.state[0].amount(*catalog, one, P::Dissolved), 1);
    EXPECT_DOUBLE_EQ(initial[0].amount(*catalog, one, P::Dissolved), 1e20);
    // The stock is authoritative; its net double report cannot encode +1 beside -1e20.
    EXPECT_DOUBLE_EQ(result.cells[0].change.amount(*catalog, one, P::Dissolved), -1e20);
}
