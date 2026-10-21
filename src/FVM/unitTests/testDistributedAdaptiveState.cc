/** @file testDistributedAdaptiveState.cc
 * @brief Atomic extensive histories on actual migrated sparse native fields.
 */
#include "FVM/DistributedAdaptiveState.hh"
#include "FVM/Operators.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveWire.hh"
#include "geometry/mesh/AMR/DistributedCartesianKernel.hh"
#include "geometry/unitTests/mixed_amr_helpers.hh"
#include "parallel/unitTests/frozen_native_helpers.hh"
#include "utils/testing_environment.hh"
#include <Tpetra_Core.hpp>
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <gtest/gtest.h>
#include <iostream>
#include <limits>
namespace
{
using namespace SimpleFluid;
using namespace SimpleFluid::Meshes;
using namespace SimpleFluid::VoF;
using LO = DefaultTpetraTypes::local_ordinal_type;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);
std::shared_ptr<const ::SimpleFluid::Meshes::AMR::DistributedAdaptiveHierarchy> detached(bool periodic = false)
{
    ::SimpleFluid::Meshes::AMR::AMRGrid grid;
    grid.block_cells = {2, 2, 2};
    grid.max_level = 2;
    grid.periodic[0] = periodic;
    auto source = std::make_shared<::SimpleFluid::Meshes::AMR::CartesianAMR>(grid);
    auto comm = Tpetra::getDefaultComm();
    std::vector<std::uint64_t> owned;
    if (comm->getRank() == 0)
        for (std::uint64_t i = 0; i < 8; ++i)
            owned.push_back(i);
    auto boot = source->detach_owned(owned, {'s', 't', 'a', 't', 'e'});
    return std::make_shared<const ::SimpleFluid::Meshes::AMR::DistributedAdaptiveHierarchy>(
        std::move(boot.owned), std::move(boot.families), boot.kernel, comm);
}
std::vector<::SimpleFluid::Meshes::AMR::AdaptiveCellKey> keys(
    const ::SimpleFluid::Meshes::AMR::DistributedAdaptiveHierarchy& h)
{
    std::vector<::SimpleFluid::Meshes::AMR::AdaptiveCellKey> result;
    for (const auto& [key, leaf] : h.owned())
        result.push_back(key);
    return result;
}
std::vector<AdaptiveHistoryFrame> frames(const ::SimpleFluid::Meshes::AMR::DistributedAdaptiveHierarchy& h,
    std::span<const ::SimpleFluid::Meshes::AMR::AdaptiveCellKey> order, bool subnormal = true)
{
    std::vector<AdaptiveHistoryFrame> result;
    for (unsigned history = 0; history < 3; ++history)
    {
        AdaptiveHistoryFrame f;
        f.name = history == 0 ? "accepted" : history == 1 ? "previous_accepted" : "rounded_trace";
        f.carrier_density = 1000;
        f.gas_density = 2;
        f.pairs = {{"number", "moles"}};
        f.inventories = {{"mass", InventoryCarrier::MixtureMass, {}, true},
            {"momentum", InventoryCarrier::BulkVolume, {}, false}, {"number", InventoryCarrier::GasVolume, {}, true},
            {"moles", InventoryCarrier::GasVolume, {}, true},
            {"uniform_tracer", InventoryCarrier::BulkVolume, {}, true},
            {"linear_tracer", InventoryCarrier::BulkVolume, {}, false}};
        for (const auto& key : order)
        {
            const auto& leaf = h.owned().at(key);
            auto geometry = h.geometry(leaf);
            GeometryHalfspace plane{{1, .25, .125}, {0, 0, 0}, history ? .45 : .6};
            auto cut = geometry->clip(plane);
            const double v = h.metrics(leaf).volume;
            const double g = history == 2 ? (subnormal ? std::numeric_limits<double>::denorm_min() : 1e-200)
                             : cut.inside ? cut.inside->moments().volume.as_double()
                                          : 0;
            const double c = history == 2 ? v : cut.outside ? cut.outside->moments().volume.as_double() : 0;
            f.phases.gas_volume.push_back(g);
            f.phases.carrier_volume.push_back(c);
            f.phases.gas_halfspace.push_back(plane);
            f.phases.geometry_declaration.push_back(
                history == 2 ? AMRPhaseGeometry::PureCarrier : AMRPhaseGeometry::Plic);
            f.inventories[0].amount.push_back(1000 * c + 2 * g);
            f.inventories[1].amount.push_back(history == 2 ? -0.0 : (geometry->centroid()[0] < .5 ? -2 : 3) * v);
            f.inventories[2].amount.push_back(g);
            f.inventories[3].amount.push_back(2 * g);
            f.inventories[4].amount.push_back(3 * v);
            const auto p = geometry->centroid();
            f.inventories[5].amount.push_back((1 + 2 * p[0] - .75 * p[1] + .4 * p[2]) * v);
        }
        result.push_back(std::move(f));
    }
    return result;
}
std::shared_ptr<const DistributedAdaptiveState> state(bool subnormal = true)
{
    auto h = detached();
    auto order = keys(*h);
    std::reverse(order.begin(), order.end());
    auto values = frames(*h, order, subnormal);
    return std::make_shared<const DistributedAdaptiveState>(h, std::move(order), std::move(values));
}
void expect_bits(const DistributedAdaptiveState& s)
{
    auto expected = frames(*s.hierarchy(), s.owned_order());
    ASSERT_EQ(expected.size(), s.histories().size());
    for (std::size_t f = 0; f < expected.size(); ++f)
        for (std::size_t i = 0; i < s.owned_order().size(); ++i)
        {
            EXPECT_EQ(std::bit_cast<std::uint64_t>(s.histories()[f].phases.gas_volume[i]),
                std::bit_cast<std::uint64_t>(expected[f].phases.gas_volume[i]));
            for (std::size_t q = 0; q < expected[f].inventories.size(); ++q)
            {
                const auto actual = s.histories()[f].inventories[q].amount[i];
                EXPECT_EQ(std::bit_cast<std::uint64_t>(actual),
                    std::bit_cast<std::uint64_t>(expected[f].inventories[q].amount[i]));
                EXPECT_EQ(std::bit_cast<std::uint64_t>(
                              s.field(expected[f].name, expected[f].inventories[q].name)->value(LO(i))),
                    std::bit_cast<std::uint64_t>(actual));
            }
        }
}
void operators(const DistributedAdaptiveState& s, bool carried_linear = false)
{
    const auto mesh = s.mesh();
    ScalarCellFieldStored<> field(mesh, "state_linear_probe");
    VectorCellFieldStored<> gradient(mesh, "state_linear_gradient");
    const auto linear = [](MeshUtils::Vec3 p) { return 1 + 2 * p.x - .75 * p.y + .4 * p.z; };
    for (std::size_t c = 0; c < mesh->num_owned_cells(); ++c)
        field.set_owned_value(LO(c), linear(mesh->cell_centroid(LO(c))));
    field.sync_ghosts();
    auto boundary = [&](int b, std::size_t i)
    { return linear(mesh->face_centroid(mesh->boundary_face_batch(b).face_lids[i])); };
    auto condition = [&](int b, std::size_t i)
    { return BoundaryCondition{BoundaryConditionType::Dirichlet, boundary(b, i)}; };
    FVM::CellGradientCache<DefaultTpetraTypes, MeshHandle<>> cache(mesh);
    FVM::cell_gradient(field, condition, boundary, gradient, cache);
    for (std::size_t c = 0; c < mesh->num_owned_cells(); ++c)
        EXPECT_NEAR((gradient.value(LO(c)) - MeshUtils::Vec3{2, -.75, .4}).norm(), 0, 3e-10);
    ScalarFaceFieldStored<> flux(mesh, "state_canonical_flux");
    std::array<double, 2> local{}, global{};
    for (std::size_t f = 0; f < mesh->num_owned_faces(); ++f)
    {
        const auto value = mesh->face_centroid(LO(f)).dot(mesh->face_area_vector(LO(f)));
        flux.set_owned_value(LO(f), value);
        if (mesh->is_exterior_face(LO(f)))
            local[1] += value;
    }
    flux.sync_ghosts();
    const auto divergence = FVM::cell_divergence_from_fluxes<DefaultTpetraTypes>(*mesh, flux);
    for (std::size_t c = 0; c < divergence.size(); ++c)
    {
        const auto v = mesh->cell_volume(LO(c));
        local[0] += v;
        EXPECT_NEAR(divergence[c] * v, 3 * v, 3e-10 * v);
    }
    Teuchos::reduceAll(*mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_SUM, 2, local.data(), global.data());
    EXPECT_NEAR(global[1], 3 * global[0], 3e-10 * global[0]);
    // This second operator input is derived from the actual transported named
    // inventory, including imported ghost values, rather than repopulated from
    // the target geometry. Unequal Mixed cell volumes expose wrong routing.
    auto amount = s.field("accepted", "uniform_tracer");
    ScalarCellFieldStored<> carried(mesh, "carried_intensive");
    for (std::size_t c = 0; c < mesh->num_owned_cells(); ++c)
    {
        const auto intensive = amount->value(LO(c)) / mesh->cell_volume(LO(c));
        EXPECT_NEAR(intensive, 3, 3e-13);
        carried.set_owned_value(LO(c), intensive);
    }
    carried.sync_ghosts();
    for (std::size_t c = 0; c < mesh->num_cells(); ++c)
        EXPECT_NEAR(amount->local_value(LO(c)) / mesh->cell_volume(LO(c)), 3, 3e-13);
    auto constant_boundary = [](int, std::size_t) { return 3.; };
    auto constant_condition = [](int, std::size_t) { return BoundaryCondition{BoundaryConditionType::Dirichlet, 3}; };
    FVM::cell_gradient(carried, constant_condition, constant_boundary, gradient, cache);
    for (std::size_t c = 0; c < mesh->num_owned_cells(); ++c)
        EXPECT_NEAR(gradient.value(LO(c)).norm(), 0, 3e-10);
    if (carried_linear)
    {
        auto linear_amount = s.field("accepted", "linear_tracer");
        for (std::size_t c = 0; c < mesh->num_owned_cells(); ++c)
            carried.set_owned_value(LO(c), linear_amount->value(LO(c)) / mesh->cell_volume(LO(c)));
        carried.sync_ghosts();
        for (std::size_t c = 0; c < mesh->num_cells(); ++c)
            EXPECT_NEAR(linear_amount->local_value(LO(c)) / mesh->cell_volume(LO(c)),
                linear(mesh->cell_centroid(LO(c))), 3e-13);
        FVM::cell_gradient(carried, condition, boundary, gradient, cache);
        for (std::size_t c = 0; c < mesh->num_owned_cells(); ++c)
            EXPECT_NEAR((gradient.value(LO(c)) - MeshUtils::Vec3{2, -.75, .4}).norm(), 0, 3e-10);
    }
    EXPECT_EQ(mesh->connectivity_storage_bytes(), 0u);
    EXPECT_FALSE(mesh->has_materialized_indexer());
}
void memory_record(
    const char* tag, const DistributedAdaptiveState& state, const DistributedAdaptiveState::Plan* plan = nullptr)
{
    const auto memory = state.storage();
    auto comm = state.mesh()->owned_cell_map()->getComm();
    for (int rank = 0; rank < comm->getSize(); ++rank)
    {
        comm->barrier();
        if (rank == comm->getRank())
        {
            std::cout << "state_memory tag=" << tag << " rank=" << rank << " owned=" << memory.owned_cells
                      << " full_ghost=" << memory.full_ghost_cells << " outer_metric=" << memory.outer_metric_cells
                      << " frame_capacity=" << memory.frame_capacity_bytes << " schema_capacity=" << memory.schema_bytes
                      << " field_scalar_payload=" << memory.field_scalar_payload_bytes
                      << " state_metadata_estimate=" << memory.state_metadata_estimate_bytes
                      << " native_registry_estimate=" << memory.native_registry_estimate_bytes
                      << " map_gid_payload=" << memory.map_gid_payload_bytes
                      << " importer_routing_estimate=" << memory.importer_routing_estimate_bytes
                      << " native_report=" << memory.native.measured_bytes()
                      << " planner_owned=" << memory.planner.owned_bytes
                      << " planner_ancestors=" << memory.planner.ancestor_bytes
                      << " planner_directory=" << memory.planner.directory_bytes
                      << " planner_procedure=" << memory.planner.procedural_bytes;
            if (state.frozen_model())
                std::cout << " frozen_owned=" << memory.frozen_planner.owned_cells
                          << " frozen_metadata_capacity_estimate=" << memory.frozen_planner.metadata_bytes
                          << " frozen_routing_capacity=" << memory.frozen_planner.routing_bytes
                          << " frozen_canonical_routes=" << memory.frozen_planner.canonical_routes
                          << " frozen_procedural_capacity=" << memory.frozen_planner.procedural_bytes
                          << " frozen_native_fragments=" << memory.frozen_planner.procedural_fragments
                          << " frozen_shared_native_parts=" << memory.frozen_planner.shared_native_parts
                          << " frozen_explicit_capacity=" << memory.frozen_planner.explicit_bytes
                          << " frozen_explicit_payload=" << memory.frozen_planner.explicit_payload_bytes;
            if (plan)
            {
                std::cout << " accepted_local_application_wire=" << plan->migration_wire_bytes()
                          << " native_application_exchanged=" << plan->native_application_bytes_exchanged()
                          << " state_exchange_phase_scratch=" << plan->state_exchange_peak_bytes();
                if (auto peak = plan->native_packet_phase_peak_bytes())
                    std::cout << " native_packet_phase=" << *peak;
                else
                    std::cout << " native_packet_phase=unavailable";
            }
            if (memory.resident_bytes)
                std::cout << " process_resident_bytes=" << *memory.resident_bytes;
            else
                std::cout << " process_resident_bytes=unavailable";
            if (memory.process_peak_resident_bytes)
                std::cout << " process_peak_resident_bytes=" << *memory.process_peak_resident_bytes;
            else
                std::cout << " process_peak_resident_bytes=unavailable";
            std::cout << std::endl;
        }
    }
    comm->barrier();
}
TEST(DistributedAdaptiveState, OwnershipCopiesNamedHistoriesAndSignedSubnormalBitsOnActualMaps)
{
    auto source = state();
    expect_bits(*source);
    std::vector<int> destinations;
    std::size_t i = 0;
    const auto comm = Tpetra::getDefaultComm();
    for (const auto& key : source->owned_order())
        destinations.push_back(int(i++ % std::min(2, comm->getSize())));
    auto plan = source->plan_migration(destinations);
    auto moved = source->apply(plan);
    expect_bits(*source);
    expect_bits(*moved);
    operators(*moved, true);
    memory_record("ownership_source", *source);
    memory_record("ownership_candidate", *moved, &plan);
    int local_empty = moved->owned_order().empty(), empty = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local_empty, &empty);
    EXPECT_EQ(empty, std::max(0, comm->getSize() - 2));
    const auto memory = moved->storage();
    EXPECT_GT(memory.state_metadata_estimate_bytes, 0u);
    if (!moved->owned_order().empty())
        EXPECT_GT(memory.frame_capacity_bytes, 0u);
    if (comm->getRank() == 0)
        std::cout << "state actual empty owners=" << empty << " application bytes=" << plan.migration_wire_bytes()
                  << '\n';
}
TEST(DistributedAdaptiveState, RemapNoOpMigrationAndQualifiedReconstructionKeepSeparateControlState)
{
    auto source = state(false);
    std::vector<::SimpleFluid::Meshes::AMR::AdaptiveMark> marks;
    for (const auto& key : source->owned_order())
        marks.push_back({key, ::SimpleFluid::Meshes::AMR::AdaptiveSplit::Isotropic});
    auto transfer = source->plan_adaptation(marks);
    auto refined = source->apply(transfer);
    ASSERT_EQ(transfer.conservation_reports().size(), 3u);
    EXPECT_FALSE(refined->histories().front().reconstruction_ready);
    const auto observation = refined->hierarchy()->observation();
    auto noop = refined->apply(refined->plan_adaptation({}));
    EXPECT_EQ(noop->hierarchy()->observation(), observation);
    EXPECT_EQ(noop->field("accepted", "mass"), refined->field("accepted", "mass"));
    const auto comm = Tpetra::getDefaultComm();
    std::vector<int> dest;
    std::uint64_t local_changed = 0, changed = 0;
    for (std::size_t i = 0; i < noop->owned_order().size(); ++i)
    {
        const int rank = int(i % std::min(2, comm->getSize()));
        dest.push_back(rank);
        local_changed += rank != comm->getRank();
    }
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local_changed, &changed);
    auto migration = noop->plan_migration(dest);
    auto moved = noop->apply(migration);
    if (comm->getSize() > 1)
    {
        EXPECT_GT(changed, 0u);
        EXPECT_GT(moved->hierarchy()->partition_generation(), noop->hierarchy()->partition_generation());
    }
    memory_record("unready_history_migration", *moved, &migration);
    EXPECT_FALSE(moved->histories().front().reconstruction_ready);
    std::vector<AdaptiveHistoryReconstruction> reconstruction;
    for (std::size_t f = 0; f < moved->histories().size(); ++f)
    {
        AdaptiveHistoryReconstruction r;
        r.name = moved->histories()[f].name;
        r.gas_halfspace.assign(moved->owned_order().size(), {{1, .25, .125}, {0, 0, 0}, f ? .45 : .6});
        r.geometry_declaration.assign(
            moved->owned_order().size(), f == 2 ? AMRPhaseGeometry::PureCarrier : AMRPhaseGeometry::Plic);
        reconstruction.push_back(std::move(r));
    }
    auto invalid = reconstruction;
    std::size_t mixed = moved->owned_order().size();
    for (std::size_t i = 0; i < moved->owned_order().size(); ++i)
        if (moved->histories().front().phases.gas_volume[i] > 0 &&
            moved->histories().front().phases.carrier_volume[i] > 0)
        {
            mixed = i;
            break;
        }
    int candidate_rank = mixed < moved->owned_order().size() ? comm->getRank() : std::numeric_limits<int>::max();
    int selected_rank = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MIN, 1, &candidate_rank, &selected_rank);
    ASSERT_NE(selected_rank, std::numeric_limits<int>::max());
    int local_selected = 0, selected = 0;
    if (comm->getRank() == selected_rank)
    {
        EXPECT_GT(moved->histories().front().phases.gas_volume[mixed], 0);
        EXPECT_GT(moved->histories().front().phases.carrier_volume[mixed], 0);
        invalid.front().gas_halfspace[mixed].offset = -100;
        local_selected = 1;
    }
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local_selected, &selected);
    ASSERT_EQ(selected, 1);
    EXPECT_THROW(moved->with_reconstruction(invalid), std::invalid_argument);
    EXPECT_FALSE(moved->histories().front().reconstruction_ready);
    auto qualified = moved->with_reconstruction(reconstruction);
    EXPECT_TRUE(qualified->histories().front().reconstruction_ready);
    EXPECT_EQ(qualified->field("accepted", "mass"), moved->field("accepted", "mass"));
    operators(*qualified);
    std::vector<::SimpleFluid::Meshes::AMR::AdaptiveBytes> families;
    for (const auto& [key, family] : qualified->hierarchy()->families())
        families.push_back(key);
    auto coarsened = qualified->apply(qualified->plan_adaptation({}, families));
    EXPECT_EQ(coarsened->hierarchy()->global_cells(), 8u);
    EXPECT_EQ(source->hierarchy()->global_cells(), 8u);
}
TEST(DistributedAdaptiveState, InvalidPhysicalPlanesPairsAndRankLocalPayloadLimitsRejectWithoutPublication)
{
    auto accepted = state();
    auto h = detached();
    auto order = keys(*h);
    for (unsigned bad = 0; bad < 4; ++bad)
    {
        auto values = frames(*h, order);
        if (!order.empty())
        {
            if (bad == 0)
                values[0].phases.gas_halfspace[0].offset = -100;
            if (bad == 1)
                values[0].phases.geometry_declaration[0] = AMRPhaseGeometry::PureGas;
            if (bad == 2)
                values[0].inventories[2].carrier = InventoryCarrier::BulkVolume;
            if (bad == 3)
                values[0].phases.geometry_declaration[0] = static_cast<AMRPhaseGeometry>(99);
        }
        EXPECT_THROW((DistributedAdaptiveState(h, order, std::move(values))), std::invalid_argument);
    }
    auto values = frames(*h, order);
    AdaptiveStateOptions options;
    if (Tpetra::getDefaultComm()->getRank() == 0)
        options.byte_budget = 128;
    EXPECT_THROW((DistributedAdaptiveState(h, order, values, options)), std::invalid_argument);
    EXPECT_NO_THROW((DistributedAdaptiveState(h, order, values)));
    expect_bits(*accepted);
    std::vector<int> unchanged(accepted->owned_order().size(), Tpetra::getDefaultComm()->getRank());
    auto retained = accepted->plan_migration(unchanged);
    std::vector<AdaptiveHistoryReconstruction> reconstruction;
    for (const auto& frame : accepted->histories())
        reconstruction.push_back({frame.name, frame.phases.gas_halfspace, frame.phases.geometry_declaration});
    auto new_control = accepted->with_reconstruction(reconstruction);
    EXPECT_THROW(new_control->apply(retained), std::invalid_argument);
    expect_bits(*accepted->apply(retained));
}
TEST(DistributedAdaptiveState, MeasuredClippingWorkUsesActualCombinedWireCostBeforePublication)
{
    auto source = state();
    const auto h = source->hierarchy();
    auto comm = h->communicator();
    ::SimpleFluid::Meshes::AMR::AdaptiveWorkRecorder recorder(::SimpleFluid::Meshes::AMR::AdaptiveWorkUnit::Operations);
    std::vector<::SimpleFluid::Meshes::AMR::AdaptiveBalanceCell> cells;
    ::SimpleFluid::Meshes::AMR::adaptive_detail::collective(*comm,
        [&]
        {
            std::size_t row = 0;
            for (const auto& [key, leaf] : h->owned())
            {
                const auto geometry = h->geometry(leaf);
                auto cut = geometry->clip({{1, .25, .125}, {0, 0, 0}, .6});
                // Count the clipping call and each actually executed face-area and
                // inventory inspection, rather than labeling topology sizes as work.
                std::size_t interface = 0, population = 0;
                if (cut.inside)
                    for (std::size_t face = 0; face < cut.inside->num_faces(); ++face)
                    {
                        EXPECT_GT(cut.inside->face_geometry(face)->area().as_double(), 0);
                        ++interface;
                    }
                for (const auto& frame : source->histories())
                    for (const auto& quantity : frame.inventories)
                    {
                        EXPECT_TRUE(std::isfinite(quantity.amount[row]));
                        ++population;
                    }
                ++row;
                recorder.observe(key, leaf.revision, 1, interface, population);
                recorder.record_payload(
                    key, leaf.revision, ::SimpleFluid::Meshes::AMR::encode_adaptive_leaf(leaf).size());
                cells.push_back({key, leaf.revision, recorder.snapshot(key, leaf.revision)});
            }
        });
    auto proposal = ::SimpleFluid::Meshes::AMR::balance_owned_cells(
        cells, {}, {h->identity(), h->topology_generation(), h->partition_generation()}, comm);
    auto transaction = source->plan_migration(proposal);
    ASSERT_TRUE(transaction.balance());
    auto moved = source->apply(transaction);
    expect_bits(*moved);
    if (comm->getSize() > 1)
    {
        EXPECT_TRUE(transaction.balance()->accepted);
        EXPECT_TRUE(transaction.balance()->proposed.wire_bytes_exact);
        std::uint64_t local = transaction.migration_wire_bytes(), global = 0;
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local, &global);
        EXPECT_EQ(transaction.balance()->proposed.moved_bytes, global);
        EXPECT_LT(transaction.balance()->final.maximum_compute, transaction.balance()->before.maximum_compute);
    }
    if (comm->getSize() > 1)
    {
        // Exercise the same final cost gate for the exact native hierarchy.
        ASSERT_TRUE(proposal.accepted);
        const auto estimate = proposal.proposed.moved_bytes;
        ASSERT_GT(proposal.proposed.moved_cells, 0u);
        auto declared = source->histories();
        const auto extra = estimate / proposal.proposed.moved_cells / (5 * sizeof(double)) + 2;
        ASSERT_LT(extra, 4096u);
        for (std::uint64_t f = 0; f < extra; ++f)
        {
            auto frame = source->histories().front();
            frame.name = "exact_cost_history_" + std::to_string(f);
            declared.push_back(std::move(frame));
        }
        DistributedAdaptiveState cost_source(h, source->owned_order(), declared);
        auto calibration = cost_source.plan_migration(proposal.localdestinations);
        std::uint64_t local = calibration.migration_wire_bytes(), global = 0;
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local, &global);
        ASSERT_GT(global, estimate);
        ::SimpleFluid::Meshes::AMR::AdaptiveBalanceOptions options;
        const auto benefit = proposal.before.maximum_compute - proposal.proposed.maximum_compute;
        options.migration_weight = benefit / (.5 * (double(estimate) + double(global)));
        auto initial = ::SimpleFluid::Meshes::AMR::balance_owned_cells(cells, {},
            {h->identity(), h->topology_generation(), h->partition_generation()}, comm, options,
            std::span<const int>(proposal.localdestinations));
        ASSERT_TRUE(initial.accepted);
        const auto oracle = ::SimpleFluid::Meshes::AMR::qualify_balance_wire_bytes(initial, local, comm, options);
        ASSERT_FALSE(oracle.accepted);
        auto rejected = cost_source.plan_migration(initial);
        ASSERT_TRUE(rejected.balance());
        EXPECT_FALSE(rejected.balance()->accepted);
        EXPECT_TRUE(rejected.balance()->proposed.wire_bytes_exact);
        EXPECT_EQ(rejected.balance()->proposed.moved_bytes, global);
        EXPECT_EQ(rejected.migration_wire_bytes(), 0u);
        EXPECT_EQ(rejected.native_application_bytes_exchanged(), 0u);
        EXPECT_EQ(rejected.state_exchange_peak_bytes(), 0u);
        auto unchanged = cost_source.apply(rejected);
        EXPECT_EQ(unchanged->hierarchy()->identity(), h->identity());
        EXPECT_EQ(unchanged->owned_order(), cost_source.owned_order());
        for (const auto& frame : declared)
            for (const auto& quantity : frame.inventories)
                EXPECT_EQ(unchanged->field(frame.name, quantity.name), cost_source.field(frame.name, quantity.name));
    }
    operators(*moved, true);
}
TEST(DistributedAdaptiveState, CallerDeclaredExtraFullGhostKeysImportActualNamedFields)
{
    auto comm = Tpetra::getDefaultComm();
    if (comm->getSize() < 2)
        GTEST_SKIP() << "Explicit remote reconstruction support requires MPI.";
    auto h = detached();
    auto order = keys(*h);
    AdaptiveStateOptions options;
    if (comm->getRank() == 1)
        options.additional_full_ghosts.push_back({0, ::SimpleFluid::Meshes::AMR::CartesianCellAncestry{0, {1, 1, 1}}});
    DistributedAdaptiveState snapshot(h, order, frames(*h, order), options);
    if (comm->getRank() == 1)
    {
        ASSERT_EQ(snapshot.mesh()->num_owned_cells(), 0u);
        ASSERT_EQ(snapshot.mesh()->num_cells(), 1u);
        EXPECT_GT(snapshot.field("accepted", "mass")->local_value(0), 0);
        EXPECT_TRUE(snapshot.hierarchy()->native_partition()->geometry()->certified_sparse_cell_geometry(
            snapshot.mesh()->cell_global_id(0)));
    }
    EXPECT_EQ(snapshot.mesh()->connectivity_storage_bytes(), 0u);
}
TEST(DistributedAdaptiveState, MixedUnequalNativeSeamMigratesRemapsAndRestoresPolyhedralFamily)
{
    auto comm = Tpetra::getDefaultComm();
    auto root = std::make_shared<::SimpleFluid::Meshes::AMR::MixedAMR>(test::mixed_joined_grid());
    auto source = root->adapt({test::first_product(*root)});
    std::vector<std::uint64_t> owned;
    if (comm->getRank() == 0)
        for (std::uint64_t cell = 0; cell < source->geometry()->num_cells(); ++cell)
            owned.push_back(cell);
    auto boot = source->detach_owned(owned, {'m', 'i', 'x', 'e', 'd', '_', 's', 't', 'a', 't', 'e'});
    std::weak_ptr<const MultiRegionMesh> old = source->geometry();
    source.reset();
    root.reset();
    EXPECT_TRUE(old.expired());
    auto hierarchy = std::make_shared<const ::SimpleFluid::Meshes::AMR::DistributedAdaptiveHierarchy>(
        std::move(boot.owned), std::move(boot.families), boot.kernel, comm);
    auto order = keys(*hierarchy);
    auto snapshot =
        std::make_shared<const DistributedAdaptiveState>(hierarchy, order, frames(*hierarchy, order, false));
    std::vector<int> destinations;
    std::size_t ordinal = 0;
    for (const auto& key : snapshot->owned_order())
        destinations.push_back(int(ordinal++ % std::min(2, comm->getSize())));
    auto migration = snapshot->plan_migration(destinations);
    auto moved = snapshot->apply(migration);
    operators(*moved, true);
    memory_record("mixed_source", *snapshot);
    memory_record("mixed_migrated", *moved, &migration);
    std::vector<::SimpleFluid::Meshes::AMR::AdaptiveMark> marks;
    for (const auto& [key, leaf] : moved->hierarchy()->owned())
        if (leaf.provider == ::SimpleFluid::Meshes::AMR::AdaptiveProviderKind::Polyhedral)
            marks.push_back({key, ::SimpleFluid::Meshes::AMR::AdaptiveSplit::Isotropic});
    auto refined = moved->apply(moved->plan_adaptation(marks));
    operators(*refined);
    std::uint64_t count = refined->owned_order().size(), largest = 0;
    std::uint64_t nonempty = count ? count : std::numeric_limits<std::uint64_t>::max(), smallest = 0;
    int local_empty = count == 0, empty = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local_empty, &empty);
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &count, &largest);
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MIN, 1, &nonempty, &smallest);
    EXPECT_EQ(empty, std::max(0, comm->getSize() - 2));
    if (comm->getSize() > 1)
        EXPECT_GT(largest, smallest);
    memory_record("mixed_refined_uneven", *refined);
    if (comm->getRank() == 0)
        std::cout << "mixed observed empty=" << empty << " min_nonempty=" << smallest << " max_owned=" << largest
                  << std::endl;
    std::vector<AdaptiveHistoryReconstruction> reconstruction;
    for (std::size_t f = 0; f < refined->histories().size(); ++f)
    {
        AdaptiveHistoryReconstruction r;
        r.name = refined->histories()[f].name;
        r.gas_halfspace.assign(refined->owned_order().size(), {{1, .25, .125}, {0, 0, 0}, f ? .45 : .6});
        r.geometry_declaration.assign(
            refined->owned_order().size(), f == 2 ? AMRPhaseGeometry::PureCarrier : AMRPhaseGeometry::Plic);
        reconstruction.push_back(std::move(r));
    }
    auto qualified = refined->with_reconstruction(reconstruction);
    std::vector<::SimpleFluid::Meshes::AMR::AdaptiveBytes> restore;
    for (const auto& [key, family] : qualified->hierarchy()->families())
        if (family.parent.provider == ::SimpleFluid::Meshes::AMR::AdaptiveProviderKind::Polyhedral)
            restore.push_back(key);
    auto restored = qualified->apply(qualified->plan_adaptation({}, restore));
    EXPECT_EQ(restored->hierarchy()->global_cells(), moved->hierarchy()->global_cells());
    operators(*restored);
}
TEST(DistributedAdaptiveState, UnsupportedWholeModelOwnershipPreservesEveryDeclaredHistoryFlagAndBit)
{
    auto comm = Tpetra::getDefaultComm();
    auto source = std::make_shared<test::AliasHierarchy>(test::mixed_source());
    std::vector<std::uint64_t> owned;
    if (comm->getRank() == 0)
        for (std::uint64_t cell = 0; cell < source->native_geometry()->num_cells(); ++cell)
            owned.push_back(cell);
    auto boot = ::SimpleFluid::Meshes::AMR::detach_frozen_native(*source, owned,
        ::SimpleFluid::Meshes::AMR::AdaptiveBytes{'f', 'r', 'o', 'z', 'e', 'n', '_', 's', 't', 'a', 't', 'e'}, comm);
    std::weak_ptr<test::AliasHierarchy::Data> old = source->data;
    source.reset();
    EXPECT_TRUE(old.expired());
    auto model = std::make_shared<const ::SimpleFluid::Meshes::AMR::FrozenNativeModel>(std::move(boot), comm);
    auto native = model->publication();
    std::vector<::SimpleFluid::Meshes::AMR::AdaptiveCellKey> order;
    std::vector<AdaptiveHistoryFrame> histories(2);
    for (std::size_t f = 0; f < histories.size(); ++f)
    {
        auto& frame = histories[f];
        frame.name = f ? "declared_previous" : "declared_exact_accepted";
        frame.reconstruction_ready = f == 0;
        frame.inventories = {{"momentum", InventoryCarrier::BulkVolume, {}, false},
            {"number", InventoryCarrier::GasVolume, {}, true}, {"moles", InventoryCarrier::GasVolume, {}, true}};
        frame.pairs = {{"number", "moles"}};
    }
    for (const auto& [key, bundle] : model->owned())
    {
        order.push_back(key);
        const auto volume = native.partition->geometry()->cell_volume(native.owned_gids.at(key));
        for (auto& frame : histories)
        {
            frame.phases.gas_volume.push_back(std::numeric_limits<double>::denorm_min());
            frame.phases.carrier_volume.push_back(volume);
            frame.phases.geometry_declaration.push_back(AMRPhaseGeometry::PureCarrier);
            frame.inventories[0].amount.push_back(-0.0);
            frame.inventories[1].amount.push_back(std::numeric_limits<double>::denorm_min());
            frame.inventories[2].amount.push_back(2 * std::numeric_limits<double>::denorm_min());
        }
    }
    DistributedAdaptiveState snapshot(model, order, histories);
    std::vector<int> destinations;
    std::size_t ordinal = 0;
    for (const auto& key : snapshot.owned_order())
        destinations.push_back(int(ordinal++ % std::min(2, comm->getSize())));
    auto calibration = snapshot.plan_migration(destinations);
    if (comm->getSize() > 1)
    {
        ::SimpleFluid::Meshes::AMR::AdaptiveWorkRecorder work(::SimpleFluid::Meshes::AMR::AdaptiveWorkUnit::Operations);
        std::vector<::SimpleFluid::Meshes::AMR::AdaptiveBalanceCell> cells;
        ::SimpleFluid::Meshes::AMR::adaptive_detail::collective(*comm,
            [&]
            {
                std::size_t row = 0;
                for (const auto& key : snapshot.owned_order())
                {
                    const auto gid = native.owned_gids.at(key);
                    EXPECT_GT(native.partition->geometry()->cell_volume(gid), 0);
                    std::size_t face_queries = 0, inspections = 0;
                    for (auto face : native.partition->geometry()->cell_faces(gid))
                    {
                        EXPECT_GT(native.partition->geometry()->face_area(face), 0);
                        ++face_queries;
                    }
                    for (const auto& history : snapshot.histories())
                        for (const auto& quantity : history.inventories)
                        {
                            EXPECT_TRUE(std::isfinite(quantity.amount[row]));
                            ++inspections;
                        }
                    const auto revision = model->owned().at(key).revision();
                    work.observe(key, revision, 1, face_queries, inspections);
                    work.record_payload(key, revision, model->owned().at(key).encoded_payload_bytes());
                    cells.push_back({key, revision, work.snapshot(key, revision)});
                    ++row;
                }
            });
        ::SimpleFluid::Meshes::AMR::AdaptiveBalanceOptions options;
        const ::SimpleFluid::Meshes::AMR::AdaptiveBalanceSource pin{
            model->identity(), model->topology_generation(), model->partition_generation()};
        auto proposal = ::SimpleFluid::Meshes::AMR::balance_owned_cells(
            cells, {}, pin, comm, options, std::span<const int>(destinations));
        const auto benefit = proposal.before.maximum_compute - proposal.proposed.maximum_compute;
        ASSERT_GT(benefit, 0);
        std::uint64_t local_wire = calibration.migration_wire_bytes(), wire = 0;
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local_wire, &wire);
        auto native_probe = model->plan_migration(destinations);
        const auto native_global = native_probe.application_bytes();
        ASSERT_GT(native_global, 0u);
        // Correct cost W accepts; incorrectly reducing already-global native N
        // again charges W+(ranks-1)N and rejects at this midpoint threshold.
        options.migration_weight = benefit / (double(wire) + .5 * (comm->getSize() - 1) * double(native_global));
        proposal = ::SimpleFluid::Meshes::AMR::balance_owned_cells(
            cells, {}, pin, comm, options, std::span<const int>(destinations));
        ASSERT_TRUE(proposal.accepted) << "Native standalone estimate must admit this exact-wire threshold probe.";
        auto forged = proposal;
        if (!forged.revisions.empty())
            ++forged.revisions.front();
        EXPECT_THROW(snapshot.plan_migration(forged), std::invalid_argument);
        auto qualified = snapshot.plan_migration(proposal);
        ASSERT_TRUE(qualified.balance());
        EXPECT_TRUE(qualified.balance()->accepted);
        EXPECT_TRUE(qualified.balance()->proposed.wire_bytes_exact);
        EXPECT_EQ(qualified.balance()->proposed.moved_bytes, wire);
        EXPECT_EQ(snapshot.apply(qualified)->frozen_model()->global_cells(), model->global_cells());
        // Keep the recorder's truthful standalone native payload samples. Add
        // actual declared named histories until their encoded values alone cost
        // more than that estimate, so rejection must occur at exact qualification.
        const auto estimate = proposal.proposed.moved_bytes;
        const auto moving = proposal.proposed.moved_cells;
        ASSERT_GT(moving, 0u);
        auto rejection_histories = histories;
        const auto extra = estimate / moving / (5 * sizeof(double)) + 2;
        ASSERT_LT(extra, 4096u);
        for (std::uint64_t f = 0; f < extra; ++f)
        {
            auto frame = histories.front();
            frame.name = "declared_cost_history_" + std::to_string(f);
            frame.reconstruction_ready = f % 2 == 0;
            rejection_histories.push_back(std::move(frame));
        }
        DistributedAdaptiveState rejection_source(model, order, rejection_histories);
        const auto exact_calibration = rejection_source.plan_migration(destinations);
        local_wire = exact_calibration.migration_wire_bytes();
        std::uint64_t rejection_wire = 0;
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local_wire, &rejection_wire);
        ASSERT_GT(rejection_wire, estimate);
        options.migration_weight = benefit / (.5 * (double(estimate) + double(rejection_wire)));
        auto expensive = ::SimpleFluid::Meshes::AMR::balance_owned_cells(
            cells, {}, pin, comm, options, std::span<const int>(destinations));
        ASSERT_TRUE(expensive.accepted);
        EXPECT_FALSE(expensive.proposed.wire_bytes_exact);
        EXPECT_EQ(expensive.proposed.moved_bytes, estimate);
        auto exact_oracle =
            ::SimpleFluid::Meshes::AMR::qualify_balance_wire_bytes(expensive, local_wire, comm, options);
        ASSERT_FALSE(exact_oracle.accepted);
        auto rejected = rejection_source.plan_migration(expensive);
        ASSERT_TRUE(rejected.balance());
        EXPECT_FALSE(rejected.balance()->accepted);
        EXPECT_TRUE(rejected.balance()->proposed.wire_bytes_exact);
        EXPECT_EQ(rejected.balance()->proposed.moved_bytes, rejection_wire);
        EXPECT_EQ(rejected.migration_wire_bytes(), 0u);
        EXPECT_EQ(rejected.native_application_bytes_exchanged(), 0u);
        EXPECT_FALSE(rejected.native_packet_phase_peak_bytes());
        EXPECT_EQ(rejected.state_exchange_peak_bytes(), 0u);
        auto unchanged = rejection_source.apply(rejected);
        EXPECT_EQ(unchanged->frozen_model()->identity(), model->identity());
        EXPECT_EQ(unchanged->frozen_model()->partition_generation(), model->partition_generation());
        EXPECT_EQ(unchanged->owned_order(), rejection_source.owned_order());
        ASSERT_EQ(unchanged->histories().size(), rejection_source.histories().size());
        for (std::size_t f = 0; f < rejection_histories.size(); ++f)
        {
            const auto& frame = rejection_histories[f];
            EXPECT_EQ(unchanged->histories()[f].reconstruction_ready, frame.reconstruction_ready);
            for (const auto& quantity : frame.inventories)
            {
                EXPECT_EQ(
                    unchanged->field(frame.name, quantity.name), rejection_source.field(frame.name, quantity.name));
                for (std::size_t row = 0; row < quantity.amount.size(); ++row)
                    EXPECT_EQ(
                        std::bit_cast<std::uint64_t>(
                            unchanged->histories()[f].inventories[&quantity - frame.inventories.data()].amount[row]),
                        std::bit_cast<std::uint64_t>(quantity.amount[row]));
            }
        }
    }
    auto moved = snapshot.apply(calibration);
    memory_record("frozen_source", snapshot);
    memory_record("frozen_migrated", *moved, &calibration);
    if (!moved->owned_order().empty())
    {
        const auto accounting = moved->storage();
        EXPECT_GT(accounting.frozen_planner.procedural_fragments, 0u);
        EXPECT_GT(accounting.frozen_planner.canonical_routes, 0u);
        EXPECT_GT(accounting.frozen_planner.routing_bytes, 0u);
    }
    ASSERT_TRUE(moved->frozen_model());
    EXPECT_FALSE(moved->hierarchy());
    ASSERT_EQ(moved->histories().size(), 2u);
    EXPECT_TRUE(moved->histories()[0].reconstruction_ready);
    EXPECT_FALSE(moved->histories()[1].reconstruction_ready);
    for (const auto& frame : moved->histories())
        for (std::size_t i = 0; i < moved->owned_order().size(); ++i)
        {
            EXPECT_EQ(std::bit_cast<std::uint64_t>(frame.inventories[0].amount[i]), std::bit_cast<std::uint64_t>(-0.0));
            EXPECT_EQ(frame.phases.gas_volume[i], std::numeric_limits<double>::denorm_min());
            EXPECT_EQ(frame.inventories[1].amount[i], std::numeric_limits<double>::denorm_min());
            EXPECT_EQ(frame.inventories[2].amount[i], 2 * std::numeric_limits<double>::denorm_min());
        }
    auto noop = moved->apply(moved->plan_adaptation({}));
    EXPECT_EQ(noop->field("declared_exact_accepted", "number"), moved->field("declared_exact_accepted", "number"));
    std::vector<::SimpleFluid::Meshes::AMR::AdaptiveMark> marks;
    for (const auto& [key, bundle] : moved->frozen_model()->owned())
        if (key.region == 42)
            marks.push_back({key, ::SimpleFluid::Meshes::AMR::AdaptiveSplit::Isotropic});
    EXPECT_THROW(moved->plan_adaptation(marks), std::invalid_argument);
    EXPECT_TRUE(moved->histories()[0].reconstruction_ready);
    EXPECT_FALSE(moved->histories()[1].reconstruction_ready);
}
TEST(DistributedAdaptiveState, PeriodicPhysicalImageHistoriesRemapMigrateAndCoarsenWithCarriedTracer)
{
    auto h = detached(true);
    auto order = keys(*h);
    auto source = std::make_shared<const DistributedAdaptiveState>(h, order, frames(*h, order, false));
    const auto check = [](const DistributedAdaptiveState& s)
    {
        const auto mesh = s.mesh();
        auto amount = s.field("accepted", "uniform_tracer");
        ScalarCellFieldStored<> constant(mesh, "periodic_carried_intensive");
        VectorCellFieldStored<> gradient(mesh, "periodic_carried_gradient");
        for (std::size_t c = 0; c < mesh->num_owned_cells(); ++c)
        {
            const auto value = amount->value(LO(c)) / mesh->cell_volume(LO(c));
            EXPECT_NEAR(value, 3, 3e-13);
            constant.set_owned_value(LO(c), value);
        }
        constant.sync_ghosts();
        for (std::size_t c = 0; c < mesh->num_cells(); ++c)
            EXPECT_NEAR(amount->local_value(LO(c)) / mesh->cell_volume(LO(c)), 3, 3e-13);
        auto boundary = [](int, std::size_t) { return 3.; };
        auto condition = [](int, std::size_t) { return BoundaryCondition{BoundaryConditionType::Dirichlet, 3}; };
        FVM::CellGradientCache<DefaultTpetraTypes, MeshHandle<>> cache(mesh);
        FVM::cell_gradient(constant, condition, boundary, gradient, cache);
        for (std::size_t c = 0; c < mesh->num_owned_cells(); ++c)
            EXPECT_NEAR(gradient.value(LO(c)).norm(), 0, 3e-10);
        ScalarFaceFieldStored<> flux(mesh, "periodic_constant_velocity_flux");
        for (std::size_t f = 0; f < mesh->num_owned_faces(); ++f)
            flux.set_owned_value(LO(f), MeshUtils::Vec3{2, -.75, .4}.dot(mesh->face_area_vector(LO(f))));
        flux.sync_ghosts();
        auto divergence = FVM::cell_divergence_from_fluxes<DefaultTpetraTypes>(*mesh, flux);
        for (std::size_t c = 0; c < divergence.size(); ++c)
            EXPECT_NEAR(divergence[c] * mesh->cell_volume(LO(c)), 0, 3e-13);
        EXPECT_EQ(mesh->connectivity_storage_bytes(), 0u);
    };
    check(*source);
    std::vector<::SimpleFluid::Meshes::AMR::AdaptiveMark> marks;
    for (const auto& key : source->owned_order())
        marks.push_back({key, ::SimpleFluid::Meshes::AMR::AdaptiveSplit::Isotropic});
    auto refined = source->apply(source->plan_adaptation(marks));
    check(*refined);
    std::vector<int> destinations;
    std::size_t ordinal = 0;
    for (const auto& key : refined->owned_order())
        destinations.push_back(int(ordinal++ % std::min(2, Tpetra::getDefaultComm()->getSize())));
    auto moved = refined->apply(refined->plan_migration(destinations));
    check(*moved);
    std::vector<AdaptiveHistoryReconstruction> reconstruction;
    for (std::size_t f = 0; f < moved->histories().size(); ++f)
    {
        AdaptiveHistoryReconstruction r;
        r.name = moved->histories()[f].name;
        r.gas_halfspace.assign(moved->owned_order().size(), {{1, .25, .125}, {0, 0, 0}, f ? .45 : .6});
        r.geometry_declaration.assign(
            moved->owned_order().size(), f == 2 ? AMRPhaseGeometry::PureCarrier : AMRPhaseGeometry::Plic);
        reconstruction.push_back(std::move(r));
    }
    auto ready = moved->with_reconstruction(reconstruction);
    std::vector<::SimpleFluid::Meshes::AMR::AdaptiveBytes> restore;
    for (const auto& [key, family] : ready->hierarchy()->families())
        restore.push_back(key);
    auto coarse = ready->apply(ready->plan_adaptation({}, restore));
    EXPECT_EQ(coarse->hierarchy()->global_cells(), 8u);
    check(*coarse);
}
} // namespace
