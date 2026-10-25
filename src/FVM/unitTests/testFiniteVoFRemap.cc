/** @file testFiniteVoFRemap.cc
 * @brief Independent plane, conservation, history and owner-pin remap oracles.
 */
#include "FVM/GeneralVoF.hh"
#include "geometry/PLIC/ConvexCellGeometry.hh"
#include "geometry/mesh/AMR/CartesianAMR.hh"
#include "utils/testing_environment.hh"
#include <Teuchos_CommHelpers.hpp>
#include <Tpetra_Core.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <gtest/gtest.h>
#include <map>
#include <stdexcept>

namespace
{
using namespace SimpleFluid;
using namespace SimpleFluid::VoF;
using namespace SimpleFluid::Meshes;
using namespace SimpleFluid::Meshes::AMR;
using Measure = SimpleFluid::PLIC::GeometryMeasure<SimpleFluid::ArithmeticMode::Exact>;
using Value = SimpleFluid::PLIC::DynamicGeometryValue;
using OracleGeometry = PLIC::ConvexCellGeometry<ArithmeticMode::Exact>;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);

struct Input
{
    std::shared_ptr<const GeneralVoF> state;
    SimpleFluid::PLIC::GeometryHalfspace plane;
};
Input input(double scale = 1, SimpleFluid::PLIC::GeometryPoint origin = {}, bool reverse = false,
    bool pure_carrier = false, unsigned char domain = 1, double intercept = 6)
{
    const auto comm = Tpetra::getDefaultComm();
    AMRGrid grid;
    grid.block_cells = {2, 2, 2};
    grid.origin = origin;
    grid.extent = {2 * scale, 2 * scale, 2 * scale};
    CartesianAMR mesh(grid);
    std::vector<std::uint64_t> owned;
    for (std::uint64_t id = 0; id < 8; ++id)
        if (comm->getRank() == 0)
            owned.push_back(id);
    auto detached = mesh.detach_owned(owned, {'f', 'p', 'r', static_cast<char>(domain)});
    auto hierarchy = std::make_shared<const DistributedAdaptiveHierarchy>(
        std::move(detached.owned), std::move(detached.families), detached.kernel, comm);
    SimpleFluid::PLIC::GeometryHalfspace plane{{1, 2, 3}, origin, intercept * scale};
    if (reverse)
    {
        for (auto& n : plane.normal)
            n = -n;
        plane.offset = -plane.offset;
    }
    std::vector<AdaptiveCellKey> order;
    std::vector<double> gas, carrier;
    // Explicit, test-only exact oracle. Neither the production remap nor its
    // acceptance path receives exact phase measures from this calculation.
    for (const auto& [key, leaf] : hierarchy->owned())
    {
        const auto geometry = OracleGeometry::from_geometry(*hierarchy->geometry(leaf));
        order.push_back(key);
        if (pure_carrier)
        {
            gas.push_back(0);
            carrier.push_back(geometry->moments().volume.as_double());
        }
        else
        {
            const auto cut = geometry->clip(plane);
            gas.push_back(cut.inside ? cut.inside->moments().volume.as_double() : 0);
            carrier.push_back(cut.outside ? cut.outside->moments().volume.as_double() : 0);
        }
    }
    GeneralVoFOptions certified;
    certified.arithmetic_mode = ArithmeticMode::BoundedCertification;
    return {std::make_shared<const GeneralVoF>(hierarchy, order, gas, carrier, certified), plane};
}
std::shared_ptr<const DistributedAdaptiveHierarchy> bound_target(
    const std::shared_ptr<const DistributedAdaptiveHierarchy>& source, const DistributedAdaptiveHierarchy::Plan& plan)
{
    const auto raw = source->apply(plan);
    const auto native = assemble_adaptive_native(*raw, raw->configuration_descriptor());
    return source->apply(plan, raw->with_partition(native.partition, native.mesh->owned_cell_map()));
}
DistributedAdaptiveHierarchy::Plan refine(const GeneralVoF& state)
{
    std::vector<AdaptiveMark> marks;
    for (const auto& key : state.owned_order())
        marks.push_back({key, AdaptiveSplit::Isotropic});
    return state.hierarchy()->plan_adaptation(marks);
}
GeneralVoFFiniteRemapOptions options()
{
    GeneralVoFFiniteRemapOptions result;
    result.phase_density = {.125, 3.5};
    result.channels = {{"gas_tracer", GeneralVoFFiniteRemapCarrier::GasVolume, true},
        {"carrier_tracer", GeneralVoFFiniteRemapCarrier::CarrierVolume, true},
        {"mass_tracer", GeneralVoFFiniteRemapCarrier::MixtureMass, true},
        {"signed_bulk", GeneralVoFFiniteRemapCarrier::BulkVolume, false}};
    return result;
}
std::vector<GeneralVoFFiniteOwnedChannels> channels(const GeneralVoF& state, bool inherited_error = false)
{
    std::vector<GeneralVoFFiniteOwnedChannels> result;
    for (std::size_t i = 0; i < state.owned_order().size(); ++i)
    {
        const auto p = state.phase_volumes()[i];
        const long double mass = .125L * p.gas + 3.5L * p.carrier;
        result.push_back({state.owned_order()[i], {{2.L * p.gas, 0}, {3.L * p.carrier, 0}, {5.L * mass, 0},
                                                      {-7.L * p.bulk, inherited_error ? p.bulk * 1e-22L : 0}}});
    }
    return result;
}
std::array<double, 2> wide_parts(long double value)
{
    const auto high = static_cast<double>(value);
    const auto remainder = value - static_cast<long double>(high);
    const auto low = static_cast<double>(remainder);
    if (!std::isfinite(high) || !std::isfinite(low) || static_cast<long double>(low) != remainder)
        throw std::runtime_error("Test oracle value cannot be split into two binary64 components.");
    return {high, low};
}
Measure exact_global_sum(std::span<const long double> local)
{
    const auto comm = Tpetra::getDefaultComm();
    std::vector<double> packed;
    for (const auto value : local)
    {
        const auto pair = wide_parts(value);
        packed.insert(packed.end(), pair.begin(), pair.end());
    }
    Measure result;
    // Exact test-only accumulation of the represented quantities. Broadcasting
    // original components avoids treating an MPI rounded sum as its own oracle.
    for (int owner = 0; owner < comm->getSize(); ++owner)
    {
        int count = owner == comm->getRank() ? int(packed.size()) : 0;
        Teuchos::broadcast(*comm, owner, 1, &count);
        std::vector<double> values = owner == comm->getRank() ? packed : std::vector<double>(count);
        Teuchos::broadcast(*comm, owner, count, values.data());
        for (const auto value : values)
            result = result + Measure::from_value(value);
    }
    return result;
}
Measure absolute(Measure value)
{
    return value.sign() < 0 ? value.scaled(-1) : value;
}
Measure exact_wide(long double value)
{
    const auto parts = wide_parts(value);
    return Measure::from_value(parts[0]) + Measure::from_value(parts[1]);
}
Measure exact_product(const Measure& a, const Measure& b)
{
    return b.sign() == 0 ? Measure{} : a.divided_by(Measure::from_value(1).divided_by(b));
}
void allocation_oracle(const GeneralVoF& source, const GeneralVoF& target, const GeneralVoF::FiniteRemapTrial& trial)
{
    const auto planes = source.reconstruct();
    std::map<AdaptiveCellKey, std::size_t> source_index;
    for (std::size_t i = 0; i < source.owned_order().size(); ++i)
        source_index.emplace(source.owned_order()[i], i);
    for (const auto& route : trial.owned_routes())
    {
        // Refinement preserves this fixture's owner. MPI migration has its
        // separate bit-for-bit copy oracle and does not enter this helper.
        ASSERT_EQ(route.source_owner, route.target_owner);
        const auto i = source_index.at(route.source_key);
        const auto& plane = planes[i];
        if (plane.pure_gas || plane.pure_carrier)
        {
            EXPECT_EQ(route.fit_allocation.value, 0);
            EXPECT_EQ(route.fit_allocation.absolute_error, 0);
            continue;
        }
        const auto source_geometry = OracleGeometry::from_geometry(
            *source.hierarchy()->geometry(source.hierarchy()->owned().at(route.source_key)));
        const auto piece = OracleGeometry::from_geometry(
            *target.hierarchy()->geometry(target.hierarchy()->owned().at(route.target_key)));
        const auto source_cut = source_geometry->clip(plane.gas);
        const auto piece_cut = piece->clip(plane.gas);
        const auto volume = Measure::volume(*source_geometry);
        const auto piece_volume = Measure::volume(*piece);
        const auto raw_gas = Measure::volume(*source_cut.inside);
        const auto raw_carrier = Measure::volume(*source_cut.outside);
        const auto piece_gas = piece_cut.inside ? Measure::volume(*piece_cut.inside) : Measure{};
        const auto piece_carrier = piece_cut.outside ? Measure::volume(*piece_cut.outside) : Measure{};
        const auto gas = Measure::from_value(source.gas_volumes()[i]);
        const auto carrier = Measure::from_value(source.carrier_volumes()[i]);
        // Independent exact model evaluation: form physical piece labels first,
        // then remove piece bulk and the uniformly transported source defect.
        // This never calls the production centered-allocation arithmetic.
        const auto labelled = exact_product(gas, piece_gas.divided_by(raw_gas)) +
                              exact_product(carrier, piece_carrier.divided_by(raw_carrier));
        const auto uniform_defect = exact_product(gas + carrier - volume, piece_volume.divided_by(volume));
        const auto expected = labelled - piece_volume - uniform_defect;
        const auto observed = exact_wide(route.fit_allocation.value);
        const auto radius = exact_wide(route.fit_allocation.absolute_error);
        EXPECT_LE(absolute(observed - expected).compare(radius), 0);
    }
}
void phase_balance(const GeneralVoF& before, const GeneralVoF& after)
{
    for (unsigned phase = 0; phase < 2; ++phase)
    {
        std::vector<long double> old, next, errors;
        for (std::size_t i = 0; i < before.phase_volumes().size(); ++i)
        {
            old.push_back(phase ? before.carrier_volumes()[i] : before.gas_volumes()[i]);
            const auto e = before.accumulated_phase_errors()[i];
            errors.push_back(phase ? e.carrier : e.gas);
        }
        for (std::size_t i = 0; i < after.phase_volumes().size(); ++i)
        {
            next.push_back(phase ? after.carrier_volumes()[i] : after.gas_volumes()[i]);
            const auto e = after.accumulated_phase_errors()[i];
            errors.push_back(phase ? e.carrier : e.gas);
        }
        const auto residual = exact_global_sum(next) - exact_global_sum(old);
        const auto bound = exact_global_sum(errors);
        EXPECT_LE(absolute(residual).compare(bound), 0);
    }
}
void channel_balance(std::span<const GeneralVoFFiniteOwnedChannels> before,
    std::span<const GeneralVoFFiniteOwnedChannels> after, std::size_t count)
{
    for (std::size_t channel = 0; channel < count; ++channel)
    {
        std::vector<long double> old, next, errors;
        for (const auto& row : before)
        {
            old.push_back(row.amounts.at(channel).value);
            errors.push_back(row.amounts.at(channel).absolute_error);
        }
        for (const auto& row : after)
        {
            next.push_back(row.amounts.at(channel).value);
            errors.push_back(row.amounts.at(channel).absolute_error);
        }
        EXPECT_LE(absolute(exact_global_sum(next) - exact_global_sum(old)).compare(exact_global_sum(errors)), 0);
    }
}
void plane_oracle(const GeneralVoF& state, SimpleFluid::PLIC::GeometryHalfspace plane)
{
    bool local_nonbinary = false;
    for (std::size_t i = 0; i < state.owned_order().size(); ++i)
    {
        const auto& leaf = state.hierarchy()->owned().at(state.owned_order()[i]);
        const auto geometry = OracleGeometry::from_geometry(*state.hierarchy()->geometry(leaf));
        const auto cut = geometry->clip(plane); // Explicit independent exact reference.
        const double gas = cut.inside ? cut.inside->moments().volume.as_double() : 0;
        const double carrier = cut.outside ? cut.outside->moments().volume.as_double() : 0;
        const auto bulk = geometry->moments().volume.as_double();
        const auto& p = state.phase_volumes()[i];
        // A plane-model accuracy regression, separate from the unchanged
        // stricter phase-closure/target-defect gates exercised by the remap.
        const auto approximation = 8 * state.options().reconstruction.orientation_tolerance * bulk;
        EXPECT_LE(std::abs(p.gas - gas), approximation + state.phase_roundoff_bounds()[i].gas);
        EXPECT_LE(std::abs(p.carrier - carrier), approximation + state.phase_roundoff_bounds()[i].carrier);
        EXPECT_GE(p.gas, 0);
        EXPECT_GE(p.carrier, 0);
        // Independent evidence that public occupancy provenance encloses the
        // represented physical inventories against the original exact cell.
        // This oracle is outside production and uses no remap residual helper.
        const auto physical_defect =
            Measure::from_value(p.gas) + Measure::from_value(p.carrier) - Measure::volume(*geometry);
        EXPECT_LE(
            absolute(physical_defect).compare(Measure::from_value(state.initial_occupancy_errors()[i].admitted_bound)),
            0);
        if (gas > 0 && carrier > 0)
            local_nonbinary = local_nonbinary || std::ldexp(gas / bulk, 16) != std::floor(std::ldexp(gas / bulk, 16));
    }
    int local = local_nonbinary, global = 0;
    Teuchos::reduceAll(*Tpetra::getDefaultComm(), Teuchos::REDUCE_MAX, 1, &local, &global);
    EXPECT_EQ(global, 1);
}
std::vector<long double> history_values(const GeneralVoF& state, unsigned phase)
{
    std::vector<long double> result;
    for (const auto e : state.accumulated_phase_errors())
        result.push_back(phase ? e.carrier : e.gas);
    return result;
}
std::vector<long double> category_values(const GeneralVoF& state, unsigned category)
{
    std::vector<long double> result;
    for (const auto& error : state.initial_occupancy_errors())
        result.push_back(category == 0   ? error.phase_input_bound
                         : category == 1 ? error.transferred_bound
                                         : error.fit_allocation_bound);
    return result;
}
void expect_empty_owners(const GeneralVoF& state)
{
    int empty = state.owned_order().empty(), total = 0;
    const auto comm = Tpetra::getDefaultComm();
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &empty, &total);
    EXPECT_EQ(total, comm->getSize() - 1);
}
std::array<double, 8> occupancy(const PhaseError& e)
{
    return {e.difference, e.admitted_bound, e.native_conversion, e.phase_input_bound, e.fit_residual, e.fit_bound,
        e.transferred_bound, e.fit_allocation_bound};
}
std::size_t slot(const AdaptiveCellKey& key)
{
    const auto& cart = std::get<CartesianCellAncestry>(key.ancestry);
    const std::size_t side = std::size_t(2) << cart.level;
    return cart.index[0] + side * (cart.index[1] + side * cart.index[2]);
}
std::vector<double> publication(const GeneralVoF& state, std::span<const GeneralVoFFiniteOwnedChannels> rows)
{
    constexpr std::size_t width = 31;
    const auto cells = state.hierarchy()->global_cells();
    std::vector<double> local(cells * width, 0), global(local.size());
    std::map<AdaptiveCellKey, const GeneralVoFFiniteOwnedChannels*> by_key;
    for (const auto& row : rows)
        by_key.emplace(row.key, &row);
    for (std::size_t i = 0; i < state.owned_order().size(); ++i)
    {
        const auto& key = state.owned_order()[i];
        auto p = local.data() + width * slot(key);
        const auto v = state.phase_volumes()[i];
        const auto r = state.phase_roundoff_bounds()[i], h = state.accumulated_phase_errors()[i];
        *p++ = v.bulk;
        *p++ = v.gas;
        *p++ = v.carrier;
        *p++ = r.gas;
        *p++ = r.carrier;
        *p++ = h.gas;
        *p++ = h.carrier;
        for (const auto value : occupancy(state.initial_occupancy_errors()[i]))
            *p++ = value;
        for (const auto value : by_key.at(key)->amounts)
        {
            for (const auto part : wide_parts(value.value))
                *p++ = part;
            for (const auto part : wide_parts(value.absolute_error))
                *p++ = part;
        }
    }
    Teuchos::reduceAll(*Tpetra::getDefaultComm(), Teuchos::REDUCE_SUM, int(local.size()), local.data(), global.data());
    return global;
}

TEST(FiniteVoFRemap, ObliqueIndependentPhasesSurviveRefineCoarsenAndUnitChanges)
{
    for (const bool shifted : {false, true})
    {
        const double scale = shifted ? 8 : 1;
        const SimpleFluid::PLIC::GeometryPoint origin =
            shifted ? SimpleFluid::PLIC::GeometryPoint{64, -32, 16} : SimpleFluid::PLIC::GeometryPoint{};
        // Keep the ordinary success case away from exact grid tangencies.
        // The tangent fixture has its separate strict arithmetic rejection test.
        const auto initial = input(scale, origin, shifted, false, 1, 6.137);
        const auto source = initial.state;
        const auto control = source->accepted_control();
        const auto identity = source->accepted_identity();
        const auto original_gas = source->gas_volumes(), original_carrier = source->carrier_volumes();
        const auto schema = options();
        const auto original_channels = channels(*source);
        const auto plan = refine(*source);
        const auto trial = source->plan_remap(plan, bound_target(source->hierarchy(), plan), original_channels, schema);
        const auto fine = trial.candidate();
        EXPECT_FALSE(trial.ownership_only());
        EXPECT_EQ(fine->hierarchy()->global_cells(), 64u);
        allocation_oracle(*source, *fine, trial);
        plane_oracle(*fine, initial.plane);
        phase_balance(*source, *fine);
        channel_balance(original_channels, trial.owned_channels(), schema.channels.size());
        expect_empty_owners(*fine);
        for (unsigned phase = 0; phase < 2; ++phase)
            EXPECT_GT(exact_global_sum(history_values(*fine, phase)).sign(), 0);
        std::vector<AdaptiveBytes> families;
        for (const auto& [key, family] : fine->hierarchy()->families())
            families.push_back(key);
        const auto coarsen = fine->hierarchy()->plan_adaptation({}, families);
        const auto coarse_trial =
            fine->plan_remap(coarsen, bound_target(fine->hierarchy(), coarsen), trial.owned_channels(), schema);
        const auto coarse = coarse_trial.candidate();
        EXPECT_EQ(coarse->hierarchy()->global_cells(), 8u);
        plane_oracle(*coarse, initial.plane);
        phase_balance(*fine, *coarse);
        channel_balance(trial.owned_channels(), coarse_trial.owned_channels(), schema.channels.size());
        for (unsigned phase = 0; phase < 2; ++phase)
            EXPECT_GE(
                (exact_global_sum(history_values(*coarse, phase)) - exact_global_sum(history_values(*fine, phase)))
                    .sign(),
                0);
        for (unsigned category = 0; category < 3; ++category)
        {
            EXPECT_GE((exact_global_sum(category_values(*fine, category)) -
                          exact_global_sum(category_values(*source, category)))
                          .sign(),
                0);
            EXPECT_GE((exact_global_sum(category_values(*coarse, category)) -
                          exact_global_sum(category_values(*fine, category)))
                          .sign(),
                0);
        }
        EXPECT_EQ(source->accepted_control(), control);
        EXPECT_EQ(source->accepted_identity(), identity);
        EXPECT_EQ(source->gas_volumes(), original_gas);
        EXPECT_EQ(source->carrier_volumes(), original_carrier);
        trial.validate_observations();
        coarse_trial.validate_observations();
    }
}

TEST(FiniteVoFRemap, MigrationCopiesNonbinaryInventoriesAndInheritedBoundsExactly)
{
    const auto initial = input(1, {}, false, false, 1, 6.137);
    const auto schema = options();
    const auto plan = refine(*initial.state);
    const auto refined = initial.state->plan_remap(
        plan, bound_target(initial.state->hierarchy(), plan), channels(*initial.state, true), schema);
    const auto source = refined.candidate();
    const auto before = publication(*source, refined.owned_channels());
    const auto comm = Tpetra::getDefaultComm();
    std::vector<int> destinations(source->owned_order().size(), (comm->getRank() + 1) % comm->getSize());
    const auto migration = source->hierarchy()->plan_migration(destinations);
    const auto trial =
        source->plan_remap(migration, bound_target(source->hierarchy(), migration), refined.owned_channels(), schema);
    EXPECT_TRUE(trial.ownership_only());
    const auto after = trial.candidate();
    EXPECT_EQ(publication(*after, trial.owned_channels()), before);
    phase_balance(*source, *after);
    channel_balance(refined.owned_channels(), trial.owned_channels(), schema.channels.size());
    expect_empty_owners(*after);
    for (unsigned phase = 0; phase < 2; ++phase)
        EXPECT_GT(exact_global_sum(history_values(*after, phase)).sign(), 0);
}

TEST(FiniteVoFRemap, TangentFittingArtifactRejectsWithoutSpendingAnotherPhaseBudget)
{
    const auto initial = input();
    const auto& source = initial.state;
    const auto control = source->accepted_control();
    const auto identity = source->accepted_identity();
    const auto values = channels(*source, true);
    const auto before = publication(*source, values);
    const auto plan = refine(*source);
    bool rejected = false;
    int local_phase_gate = 0;
    try
    {
        (void) source->plan_remap(plan, bound_target(source->hierarchy(), plan), values, options());
    }
    catch (const std::invalid_argument& error)
    {
        rejected = true;
        // This fixture's geometric fit produces a positive carrier receipt of
        // about 4.63e-45 with radius 2.99e-49 at the tangent child. Its own
        // unchanged relative budget is about 3.71e-59. No bulk/other-phase
        // or historical allowance may qualify that arithmetic uncertainty.
        local_phase_gate = std::string(error.what()).find("independent phase budget") != std::string::npos;
    }
    EXPECT_TRUE(rejected);
    // Only the rejecting owner retains its local diagnostic. Empty ranks
    // correctly receive the collective's generic remote-rejection message.
    int phase_gate_reports = 0;
    Teuchos::reduceAll(*Tpetra::getDefaultComm(), Teuchos::REDUCE_SUM, 1, &local_phase_gate, &phase_gate_reports);
    EXPECT_EQ(phase_gate_reports, 1);
    EXPECT_EQ(source->accepted_control(), control);
    EXPECT_EQ(source->accepted_identity(), identity);
    EXPECT_EQ(publication(*source, values), before);
    expect_empty_owners(*source);
}

TEST(FiniteVoFRemap, ForeignStalePlansSchemasAndCapsRejectWithoutPublicationChanges)
{
    const auto initial = input(1, {}, false, true);
    const auto source = initial.state;
    const auto control = source->accepted_control();
    const auto identity = source->accepted_identity();
    const auto gas = source->gas_volumes(), carrier = source->carrier_volumes();
    const auto schema = options();
    const auto values = channels(*source);
    const auto plan = refine(*source);
    const auto target = bound_target(source->hierarchy(), plan);
    const auto foreign = input(1, {}, false, true, 2).state;
    EXPECT_THROW((void) foreign->plan_remap(plan, target, channels(*foreign), schema), std::invalid_argument);
    const auto other_plan = refine(*source);
    EXPECT_THROW((void) source->plan_remap(plan, bound_target(source->hierarchy(), other_plan), values, schema),
        std::invalid_argument);
    const auto valid = source->plan_remap(plan, target, values, schema);
    EXPECT_THROW(
        (void) valid.candidate()->plan_remap(plan, target, valid.owned_channels(), schema), std::invalid_argument);
    auto malformed = schema;
    malformed.byte_budget = 1;
    EXPECT_THROW((void) source->plan_remap(plan, target, values, malformed), std::invalid_argument);
    malformed = schema;
    if (Tpetra::getDefaultComm()->getRank() == 0)
        malformed.channels[0].name.clear();
    EXPECT_THROW((void) source->plan_remap(plan, target, values, malformed), std::invalid_argument);
    if (Tpetra::getDefaultComm()->getSize() > 1)
    {
        malformed = schema;
        if (Tpetra::getDefaultComm()->getRank() == 0)
            malformed.channels[3].name += "_different";
        EXPECT_THROW((void) source->plan_remap(plan, target, values, malformed), std::invalid_argument);
    }
    auto missing = values;
    if (Tpetra::getDefaultComm()->getRank() == 0 && !missing.empty())
        missing.pop_back();
    EXPECT_THROW((void) source->plan_remap(plan, target, missing, schema), std::invalid_argument);
    EXPECT_EQ(source->accepted_control(), control);
    EXPECT_EQ(source->accepted_identity(), identity);
    EXPECT_EQ(source->gas_volumes(), gas);
    EXPECT_EQ(source->carrier_volumes(), carrier);
}

TEST(FiniteVoFRemap, WholeCellMigrationCannotGiveAChannelAnAbsentPhysicalCarrier)
{
    const auto source = input(1, {}, false, true).state;
    const auto comm = Tpetra::getDefaultComm();
    std::vector<int> destinations(source->owned_order().size(), (comm->getRank() + 1) % comm->getSize());
    const auto plan = source->hierarchy()->plan_migration(destinations);
    const auto target = bound_target(source->hierarchy(), plan);
    GeneralVoFFiniteRemapOptions schema;
    schema.channels = {{"gas_only", GeneralVoFFiniteRemapCarrier::GasVolume, true}};
    std::vector<GeneralVoFFiniteOwnedChannels> absent;
    for (const auto& key : source->owned_order())
        absent.push_back({key, {{0, 0}}});
    const auto accepted = source->plan_remap(plan, target, absent, schema);
    const auto accepted_state = accepted.candidate();
    EXPECT_EQ(accepted_state->hierarchy()->global_cells(), source->hierarchy()->global_cells());
    EXPECT_TRUE(std::all_of(
        accepted_state->gas_volumes().begin(), accepted_state->gas_volumes().end(), [](double v) { return v == 0; }));
    auto invalid = absent;
    if (comm->getRank() == 0 && !invalid.empty())
        invalid.front().amounts[0] = {1, 0};
    EXPECT_THROW((void) source->plan_remap(plan, target, invalid, schema), std::invalid_argument);
    schema.channels[0].nonnegative = false;
    invalid = absent;
    if (comm->getRank() == 0 && !invalid.empty())
        invalid.front().amounts[0] = {0, 1e-24L};
    EXPECT_THROW((void) source->plan_remap(plan, target, invalid, schema), std::invalid_argument);
    EXPECT_TRUE(
        std::all_of(source->gas_volumes().begin(), source->gas_volumes().end(), [](double v) { return v == 0; }));

    const auto split = refine(*source);
    const auto refinement = source->plan_remap(split, bound_target(source->hierarchy(), split));
    const auto fine = refinement.candidate();
    const auto fine_identity = fine->accepted_identity();
    const auto fine_control = fine->accepted_control();
    std::vector<AdaptiveBytes> families;
    for (const auto& [key, family] : fine->hierarchy()->families())
        families.push_back(key);
    const auto coarsen = fine->hierarchy()->plan_adaptation({}, families);
    const auto coarse_target = bound_target(fine->hierarchy(), coarsen);
    std::vector<GeneralVoFFiniteOwnedChannels> invalid_fine;
    for (const auto& key : fine->owned_order())
        invalid_fine.push_back({key, {{0, 0}}});
    if (comm->getRank() == 0 && !invalid_fine.empty())
        invalid_fine.front().amounts[0] = {-1, 0}; // Signed still requires a real carrier.
    EXPECT_THROW((void) fine->plan_remap(coarsen, coarse_target, invalid_fine, schema), std::invalid_argument);
    EXPECT_EQ(fine->accepted_identity(), fine_identity);
    EXPECT_EQ(fine->accepted_control(), fine_control);
    EXPECT_TRUE(std::all_of(fine->gas_volumes().begin(), fine->gas_volumes().end(), [](double v) { return v == 0; }));
}
} // namespace