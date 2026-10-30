/** @file testGeneralVoFFinite.cc */
#include "FVM/details/GeneralVoF/ConstructionControls.tcc"
#include "FVM/details/GeneralVoF/PlaneOrientation.hh"
#include "FVM/details/GeometryWireProtocol.hh"
#include "general_vof_finite_test_helpers.hh"
#include "geometry/swept_volume/FiniteVoFTransport.hh"
namespace
{
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);
constexpr auto overlay_mode = ArithmeticMode::BoundedCertification;
using OverlayGeometry = PLIC::ConvexCellGeometry<overlay_mode>;
void reset_dispatch_statistics()
{
    GeneralVoF::reset_geometry_statistics();
    PLIC::reset_geometry_statistics();
}
void expect_static_geometry_work()
{
    const auto fvm = GeneralVoF::geometry_statistics().bounded;
    const auto executable = PLIC::geometry_statistics();
    EXPECT_EQ(fvm.runtime_dispatches, 0u);
    EXPECT_EQ(executable.runtime_dispatches, 0u);
    // Empty owners may legitimately do no work; the collective must exercise
    // the fixed-mode geometry provider rather than passing vacuously.
    const std::uint64_t local = fvm.constructions + fvm.volume_evaluations + fvm.partition_checks;
    std::uint64_t global = 0;
    Teuchos::reduceAll(*Tpetra::getDefaultComm(), Teuchos::REDUCE_SUM, 1, &local, &global);
    EXPECT_GT(global, 0u);
}

void check_budget_diagnostics(const GeneralVoF& state, const GeneralVoF::Trial& trial)
{
    const auto& faces = trial.finite_face_chain_diagnostics();
    const auto& cells = trial.finite_cell_closure_diagnostics();
    const auto comm = Tpetra::getDefaultComm();
    const auto collectively_valid = [&](bool valid)
    {
        int local = valid, global = 0;
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_MIN, 1, &local, &global);
        return global != 0;
    };
    bool valid = faces.size() == state.tiles().size() && cells.size() == state.owned_order().size();
    EXPECT_EQ(faces.size(), state.tiles().size());
    EXPECT_EQ(cells.size(), state.owned_order().size());
    std::map<std::uint64_t, const GeneralVoFFaceChainDiagnostic*> by_face;
    std::uint64_t local_max = 0, global_max = 0;
    for (const auto& face : faces)
    {
        const bool inserted = by_face.emplace(face.face, &face).second;
        EXPECT_TRUE(inserted);
        valid = valid && inserted && std::isfinite(face.control.value) &&
            std::isfinite(face.constructed.value) && std::isfinite(face.signed_difference.value) &&
            std::isfinite(face.allowed_difference);
        local_max = std::max(local_max, face.face);
        EXPECT_EQ(face.control.arithmetic_mode, ArithmeticMode::Ordinary);
        EXPECT_EQ(face.constructed.arithmetic_mode, ArithmeticMode::Ordinary);
        EXPECT_EQ(face.signed_difference.value, face.constructed.value - face.control.value);
        EXPECT_GT(face.allowed_difference, 0);
        EXPECT_LE(std::abs(face.signed_difference.value), face.allowed_difference);
    }
    for (const auto& tile : state.tiles()) valid = valid && by_face.contains(tile.face_gid);
    if (!collectively_valid(valid)) return;
    const auto geometry = state.owned_finite_geometry<ArithmeticMode::Ordinary>();
    valid = geometry.size() == cells.size();
    EXPECT_EQ(geometry.size(), cells.size());
    for (std::size_t c = 0; c < cells.size(); ++c)
        valid = valid && cells[c].key == state.owned_order()[c];
    if (!collectively_valid(valid)) return;
    for (std::size_t c = 0; c < cells.size(); ++c)
    {
        const auto& cell = cells[c];
        EXPECT_EQ(cell.key, state.owned_order()[c]);
        std::size_t count = 0;
        fp::Sum budgets(ArithmeticMode::Ordinary), actual(ArithmeticMode::Ordinary);
        for (const auto& tile : state.tiles())
            if (tile.first == cell.key || (tile.second && *tile.second == cell.key))
            {
                ++count;
                const auto& face = *by_face.at(tile.face_gid);
                budgets.add(fp::make_value(ArithmeticMode::Ordinary, face.allowed_difference));
                actual.add(fp::make_value(ArithmeticMode::Ordinary, std::abs(face.signed_difference.value)));
                EXPECT_LE(face.allowed_difference,
                    state.options().reconstruction.volume_tolerance * cell.quota_scale);
            }
        EXPECT_EQ(cell.canonical_face_count, count);
        EXPECT_GT(cell.quota_scale, 0);
        EXPECT_LE(cell.quota_scale * count,
            std::min(static_cast<long double>(state.phase_volumes()[c].bulk), geometry[c]->volume().value));
        EXPECT_LE(budgets.value().value, cell.allowed_closure);
        EXPECT_EQ(cell.absolute_geometry_difference, actual.value().value);
        EXPECT_EQ(cell.combined_residual.value,
            cell.canonical_residual.value + cell.signed_geometry_difference.value + cell.signed_phase_cut_closure);
        EXPECT_EQ(cell.allowed_closure,
            state.options().reconstruction.volume_tolerance * geometry[c]->volume().value);
        EXPECT_LE(std::abs(cell.combined_residual.value), cell.allowed_closure);
    }
    // Compare actual replica quantities, not just their configured allowances.
    // A high/low double pair retains every long-double bit in these fixtures.
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &local_max, &global_max);
    EXPECT_LT(global_max, 1000000u);
    if (global_max >= 1000000u) return; // The same reduced value governs every rank.
    const auto size = std::size_t(global_max + 1);
    std::vector<double> low(8 * size, std::numeric_limits<double>::infinity());
    std::vector<double> high(8 * size, -std::numeric_limits<double>::infinity());
    std::vector<double> minimum(low.size()), maximum(high.size());
    for (const auto& face : faces)
    {
        const std::array values{face.control.value, face.constructed.value,
            face.signed_difference.value, face.allowed_difference};
        for (std::size_t i = 0; i < values.size(); ++i)
        {
            const double head = static_cast<double>(values[i]);
            const double tail = static_cast<double>(values[i] - static_cast<long double>(head));
            EXPECT_EQ(static_cast<long double>(head) + tail, values[i]);
            const auto slot = 8 * face.face + 2 * i;
            low[slot] = high[slot] = head;
            low[slot + 1] = high[slot + 1] = tail;
        }
    }
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MIN, int(low.size()), low.data(), minimum.data());
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, int(high.size()), high.data(), maximum.data());
    for (std::size_t i = 0; i < minimum.size(); ++i)
        if (std::isfinite(minimum[i])) EXPECT_EQ(minimum[i], maximum[i]);
}

TEST(GeneralVoFFinite, ReconstructionResolutionFailureRetainsTypedClassification)
{
    namespace failure = SimpleFluid::detail::vof_failure;
    PLIC::PolyhedralResolutionAssessment assessment;
    assessment.reason = PLIC::PolyhedralResolutionAssessment::Reason::NoDirectionalSignal;
    failure::Record record;
    failure::classify(record, std::make_exception_ptr(PLIC::PolyhedralResolutionRequired(assessment)));
    EXPECT_EQ(record.category, failure::Category::Resolution);
    EXPECT_TRUE(record.has(failure::Field::Resolution));
    EXPECT_EQ(record.resolution_reason, int(assessment.reason));
}

TEST(GeneralVoFFinite, SmallPhasePolicyIsValidatedAndAgreedOnEmptyOwners)
{
    const auto input = fixture(); // Pure phases; validation must precede reconstruction.
    const auto comm = Tpetra::getDefaultComm();
    const bool changed_rank = comm->getRank() == comm->getSize() - 1;
    if (comm->getSize() > 2 && changed_rank)
        EXPECT_TRUE(input.order.empty());
    for (const auto mode : {ArithmeticMode::Ordinary, ArithmeticMode::BoundedCertification})
    {
        SCOPED_TRACE(static_cast<int>(mode));
        GeneralVoFOptions options;
        options.arithmetic_mode = mode;
        for (const double invalid : {-1., 1.01, std::numeric_limits<double>::infinity(),
                 std::numeric_limits<double>::quiet_NaN()})
        {
            options.reconstruction.minimum_optimized_phase_length_ratio = changed_rank ? invalid : .05;
            EXPECT_THROW((void) GeneralVoF(input.hierarchy, input.order, input.gas, input.carrier, options),
                std::invalid_argument);
        }
        for (const double valid : {0., 1.})
        {
            options.reconstruction.minimum_optimized_phase_length_ratio = valid;
            EXPECT_NO_THROW((void) GeneralVoF(input.hierarchy, input.order, input.gas, input.carrier, options));
        }
        if (comm->getSize() > 1)
        {
            options.reconstruction.minimum_optimized_phase_length_ratio = changed_rank ? .1 : .05;
            EXPECT_THROW((void) GeneralVoF(input.hierarchy, input.order, input.gas, input.carrier, options),
                std::invalid_argument);
        }
    }
}

TEST(GeneralVoFFinite, ConstructionAlgorithmBindsSmallPhasePolicy)
{
    namespace controls = SimpleFluid::VoF::construction_controls;
    GeneralVoFOptions options;
    const auto defaults = controls::algorithm(options);
    options.reconstruction.minimum_optimized_phase_length_ratio = 0;
    const auto disabled = controls::algorithm(options);
    EXPECT_NE(defaults, disabled);
    options.reconstruction.minimum_optimized_phase_length_ratio = 1;
    const auto gradient_only = controls::algorithm(options);
    EXPECT_NE(defaults, gradient_only);
    EXPECT_NE(disabled, gradient_only);
    options.byte_budget /= 2;
    options.sweep.byte_budget /= 2;
    EXPECT_EQ(gradient_only, controls::algorithm(options));
}

TEST(GeneralVoFFinite, FaceWorkLimitIsPositiveAndAgreedIncludingEmptyOwners)
{
    const auto input = fixture();
    const auto comm = Tpetra::getDefaultComm();
    const bool changed_rank = comm->getRank() == comm->getSize() - 1;
    if (comm->getSize() > 2 && changed_rank)
        EXPECT_TRUE(input.order.empty());
    GeneralVoFOptions options;
    EXPECT_EQ(options.maximum_face_geometry_operations, 64u * 1024 * 1024);
    options.maximum_face_geometry_operations = changed_rank ? 0 : 64u * 1024 * 1024;
    EXPECT_THROW((void) GeneralVoF(input.hierarchy, input.order, input.gas, input.carrier, options), std::invalid_argument);
    options.maximum_face_geometry_operations = 1;
    EXPECT_NO_THROW((void) GeneralVoF(input.hierarchy, input.order, input.gas, input.carrier, options));
    if (comm->getSize() > 1)
    {
        options.maximum_face_geometry_operations = changed_rank ? 2 : 1;
        EXPECT_THROW((void) GeneralVoF(input.hierarchy, input.order, input.gas, input.carrier, options),
            std::invalid_argument);
    }
}

TEST(GeneralVoFFinite, ConstructionAlgorithmBindsFaceWorkLimit)
{
    namespace controls = SimpleFluid::VoF::construction_controls;
    GeneralVoFOptions options;
    const auto original = controls::algorithm(options);
    --options.maximum_face_geometry_operations;
    const auto changed = controls::algorithm(options);
    EXPECT_NE(original, changed);
    options.byte_budget /= 2;
    options.sweep.byte_budget /= 2;
    EXPECT_EQ(changed, controls::algorithm(options));
}

TEST(GeneralVoFFinite, OrientationWirePreservesEveryMethodAndRejectsUnknownWords)
{
    namespace codec = SimpleFluid::VoF::plane_wire_detail;
    namespace wire = SimpleFluid::VoF::general_vof_detail::wire;
    for (const auto method : {PLIC::PolyhedralOrientationMethod::SuppliedNormal,
             PLIC::PolyhedralOrientationMethod::Gradient, PLIC::PolyhedralOrientationMethod::AngularVolumeFit})
    {
        wire::Writer writer;
        codec::orientation(writer, method);
        EXPECT_EQ(writer.data.size(), sizeof(std::uint64_t));
        wire::Reader reader{writer.data};
        EXPECT_EQ(codec::orientation(reader), method);
        EXPECT_NO_THROW(reader.done());
    }
    for (const auto invalid : {std::uint64_t(3), std::numeric_limits<std::uint64_t>::max()})
    {
        wire::Writer writer;
        writer.word(invalid);
        wire::Reader reader{writer.data};
        EXPECT_THROW((void) codec::orientation(reader), std::invalid_argument);
        wire::Writer outbound;
        EXPECT_THROW(codec::orientation(outbound, static_cast<PLIC::PolyhedralOrientationMethod>(invalid)),
            std::invalid_argument);
        EXPECT_TRUE(outbound.data.empty());
    }
}

TEST(GeneralVoFFinite, StencilResidualMeasurementWirePreservesBothValuesAndRejectsUnknownWords)
{
    namespace codec = SimpleFluid::VoF::plane_wire_detail;
    namespace wire = SimpleFluid::VoF::general_vof_detail::wire;
    EXPECT_FALSE(PLIC::PolyhedralInterface{}.stencil_volume_residual_measured);
    for (const bool measured : {false, true})
    {
        wire::Writer writer;
        codec::boolean(writer, measured);
        EXPECT_EQ(writer.data.size(), sizeof(std::uint64_t));
        wire::Reader reader{writer.data};
        PLIC::PolyhedralInterface decoded;
        decoded.stencil_volume_residual_measured = codec::boolean(reader);
        EXPECT_EQ(decoded.stencil_volume_residual_measured, measured);
        EXPECT_NO_THROW(reader.done());
    }
    for (const auto invalid : {std::uint64_t(2), std::numeric_limits<std::uint64_t>::max()})
    {
        wire::Writer writer;
        writer.word(invalid);
        wire::Reader reader{writer.data};
        EXPECT_THROW((void) codec::boolean(reader), std::invalid_argument);
    }
}

TEST(GeneralVoFFinite, SelectedGeometryModeStaysStaticThroughConstructionAndReconstruction)
{
    auto input = fixture();
    for (std::size_t i = 0; i < input.order.size(); ++i)
    {
        const auto geometry = input.hierarchy->geometry(input.hierarchy->owned().at(input.order[i]));
        const auto phases = geometry->phases({{1, 0, 0}, {}, .75});
        input.gas[i] = phases.gas.as_double();
        input.carrier[i] = phases.carrier.as_double();
    }
    for (const auto mode : {ArithmeticMode::Ordinary, ArithmeticMode::BoundedCertification})
    {
        SCOPED_TRACE(static_cast<int>(mode));
        GeneralVoFOptions options;
        options.arithmetic_mode = mode;
        reset_dispatch_statistics();
        GeneralVoF state(input.hierarchy, input.order, input.gas, input.carrier, options);
        expect_static_geometry_work();
        reset_dispatch_statistics();
        const auto planes = state.reconstruct();
        EXPECT_EQ(planes.size(), state.owned_order().size());
        expect_static_geometry_work();
        std::uint64_t local_mixed = 0, global_mixed = 0;
        for (const auto& plane : planes)
            local_mixed += !plane.pure_gas && !plane.pure_carrier;
        Teuchos::reduceAll(*Tpetra::getDefaultComm(), Teuchos::REDUCE_SUM, 1, &local_mixed, &global_mixed);
        EXPECT_GT(global_mixed, 0u);
    }
}

TEST(GeneralVoFFinite, OrdinaryDefaultAndBoundedOptInKeepSeparateAcceptancePolicies)
{
    const auto input = fixture();
    GeneralVoF ordinary(input.hierarchy, input.order, input.gas, input.carrier);
    GeneralVoFOptions options;
    options.arithmetic_mode = ArithmeticMode::BoundedCertification;
    GeneralVoF certified(input.hierarchy, input.order, input.gas, input.carrier, options);
    EXPECT_EQ(ordinary.arithmetic_mode(), ArithmeticMode::Ordinary);
    EXPECT_EQ(certified.arithmetic_mode(), ArithmeticMode::BoundedCertification);
    auto controls = circulation(ordinary);
    for (auto& face : controls)
        face.bounded_integrated_volume =
            SimpleFluid::PLIC::DynamicGeometryValue{face.integrated_volume, 1e-3L, ArithmeticMode::Ordinary};
    auto bounded_controls = controls;
    for (auto& face : bounded_controls)
        face.bounded_integrated_volume->arithmetic_mode = ArithmeticMode::BoundedCertification;
    // Each runtime adapter must carry the selected policy. Ordinary consumes
    // nominal values; bounded certification rejects the same unresolved widths.
    EXPECT_THROW((void) ordinary.plan_transport(stationary(ordinary), bounded_controls, 1), std::invalid_argument);
    const auto trial = ordinary.plan_transport(stationary(ordinary), controls, 1);
    const auto moved = ordinary.apply(trial);
    EXPECT_EQ(moved->arithmetic_mode(), ArithmeticMode::Ordinary);
    EXPECT_THROW((void) certified.plan_transport(stationary(certified), bounded_controls, 1), std::invalid_argument);
    EXPECT_THROW((void) certified.apply(trial), std::invalid_argument);
    EXPECT_EQ(certified.gas_volumes(), input.gas);
    EXPECT_EQ(certified.carrier_volumes(), input.carrier);
}

TEST(GeneralVoFFinite, OrdinaryModeRejectsInvalidPhysicalDataAndImplicitExactViews)
{
    const auto input = fixture();
    GeneralVoF state(input.hierarchy, input.order, input.gas, input.carrier);
    EXPECT_THROW((void) state.inventory_measures(), std::invalid_argument);
    EXPECT_THROW((void) state.geometric_phases(), std::invalid_argument);
    EXPECT_THROW((void) state.export_transfer_source(false), std::invalid_argument);
    auto gas = input.gas;
    if (Tpetra::getDefaultComm()->getRank() == 0 && !gas.empty()) gas.front() = -1;
    EXPECT_THROW((void) GeneralVoF(input.hierarchy, input.order, gas, input.carrier), std::invalid_argument);
    gas = input.gas;
    if (Tpetra::getDefaultComm()->getRank() == 0 && !gas.empty())
        gas.front() = std::numeric_limits<double>::infinity();
    EXPECT_THROW((void) GeneralVoF(input.hierarchy, input.order, gas, input.carrier), std::invalid_argument);
    GeneralVoFOptions mismatch;
    mismatch.arithmetic_mode = Tpetra::getDefaultComm()->getSize() == 1 ? static_cast<ArithmeticMode>(99) :
        Tpetra::getDefaultComm()->getRank() == 0 ? ArithmeticMode::Ordinary : ArithmeticMode::BoundedCertification;
    EXPECT_THROW((void) GeneralVoF(input.hierarchy, input.order, input.gas, input.carrier, mismatch),
        std::invalid_argument);
    EXPECT_EQ(state.gas_volumes(), input.gas);
    EXPECT_EQ(state.carrier_volumes(), input.carrier);
}

TEST(GeneralVoFFinite, SparseOverlayRetainsDisjointAndContactAtomsAndCoefficients)
{
    const std::vector<SimpleFluid::geometry::swept_volume::OverlayAtom<overlay_mode>> atoms{{OverlayGeometry::box({0, 0, 0}, {1, 1, 1}), 0},
        {OverlayGeometry::box({0, 0, 2}, {1, 1, 3}), 1}, {OverlayGeometry::box({0, 0, 4}, {1, 1, 5}), -1}};
    const std::array pieces{OverlayGeometry::box({3, 0, 0}, {4, 1, 5}), OverlayGeometry::box({1, 0, 0}, {2, 1, 5})};
    for (const auto& geometry : pieces)
        for (const int sign : {-1, 1})
            for (const int incidence : {-1, 1})
            {
                SimpleFluid::geometry::swept_volume::Work work;
                work.limits.maximum_pair_tests = atoms.size();
                SimpleFluid::PLIC::reset_geometry_statistics();
                const auto result = SimpleFluid::geometry::swept_volume::overlay(atoms, {geometry, sign}, incidence, work);
                ASSERT_EQ(result.size(), atoms.size());
                EXPECT_EQ(work.pair_tests, atoms.size());
                EXPECT_GT(work.bytes, 0u);
                for (std::size_t i = 0; i < atoms.size(); ++i)
                {
                    EXPECT_EQ(result[i].geometry.get(), atoms[i].geometry.get());
                    EXPECT_EQ(result[i].coefficient, atoms[i].coefficient);
                }
                const auto stats = SimpleFluid::PLIC::geometry_statistics();
                EXPECT_EQ(stats.support_distances, 0u);
                EXPECT_EQ(stats.volume_evaluations, 0u);
                EXPECT_EQ(stats.constructions, 0u);
                EXPECT_EQ(stats.intersections, 0u);
            }
}

TEST(GeneralVoFFinite, SparseOverlayRetainsZeroCoefficientAtomsForLaterActivation)
{
    const auto root = OverlayGeometry::box({0, 0, 0}, {1, 1, 1});
    const auto far = OverlayGeometry::box({3, 0, 0}, {4, 1, 1});
    const auto half = OverlayGeometry::box({.5, 0, 0}, {1, 1, 1});
    SimpleFluid::geometry::swept_volume::Work work;
    auto atoms = SimpleFluid::geometry::swept_volume::overlay<overlay_mode>({{root, 0}}, {far, -1}, -1, work);
    ASSERT_EQ(atoms.size(), 1u);
    EXPECT_EQ(atoms.front().geometry.get(), root.get());
    EXPECT_EQ(atoms.front().coefficient, 0);
    atoms = SimpleFluid::geometry::swept_volume::overlay(std::move(atoms), {half, 1}, 1, work);
    ASSERT_EQ(atoms.size(), 2u);
    unsigned zero = 0, one = 0;
    for (const auto& atom : atoms)
    {
        zero += atom.coefficient == 0;
        one += atom.coefficient == 1;
        const auto volume = atom.geometry->volume();
        EXPECT_LE(volume.lower(), .5L);
        EXPECT_GE(volume.upper(), .5L);
    }
    EXPECT_EQ(zero, 1u);
    EXPECT_EQ(one, 1u);
    EXPECT_EQ(root->volume().value, 1);
    EXPECT_EQ(root->volume().absolute_error, 0);
}

TEST(GeneralVoFFinite, SparseOverlayFastPathStillChargesPairAtomAndByteCaps)
{
    const auto root = OverlayGeometry::box({0, 0, 0}, {1, 1, 1});
    const SimpleFluid::geometry::swept_volume::SignedPiece<overlay_mode> far{OverlayGeometry::box({3, 0, 0}, {4, 1, 1}), 1};
    const std::vector<SimpleFluid::geometry::swept_volume::OverlayAtom<overlay_mode>> atoms{{root, 0}};
    const auto rejected = [&](SimpleFluid::geometry::swept_volume::Work& work, const char* reason)
    {
        try
        {
            (void) SimpleFluid::geometry::swept_volume::overlay(atoms, far, 1, work);
            FAIL() << "Sparse overlay must honor the original resource caps.";
        }
        catch (const std::invalid_argument& error)
        {
            EXPECT_NE(std::string(error.what()).find(reason), std::string::npos);
        }
    };
    SimpleFluid::geometry::swept_volume::Work pair;
    pair.limits.maximum_pair_tests = 0;
    rejected(pair, "pair-test budget");
    EXPECT_EQ(pair.pair_tests, 1u);
    SimpleFluid::geometry::swept_volume::Work atom;
    atom.limits.maximum_atoms = 0;
    rejected(atom, "atom buffer capacity");
    EXPECT_EQ(atom.pair_tests, 1u);
    SimpleFluid::geometry::swept_volume::Work bytes;
    bytes.limits.byte_budget = sizeof(SimpleFluid::geometry::swept_volume::OverlayAtom<overlay_mode>) - 1;
    rejected(bytes, "byte budget");
    EXPECT_EQ(bytes.pair_tests, 1u);
    SimpleFluid::geometry::swept_volume::Work exact;
    exact.limits.byte_budget = sizeof(SimpleFluid::geometry::swept_volume::OverlayAtom<overlay_mode>);
    const auto result = SimpleFluid::geometry::swept_volume::overlay(atoms, far, 1, exact);
    ASSERT_EQ(result.size(), 1u);
    EXPECT_EQ(result.front().geometry.get(), root.get());
    EXPECT_EQ(result.front().coefficient, 0);
    EXPECT_EQ(exact.bytes, sizeof(SimpleFluid::geometry::swept_volume::OverlayAtom<overlay_mode>));
    EXPECT_EQ(root->volume().value, 1);
    EXPECT_EQ(root->volume().absolute_error, 0);
}

TEST(GeneralVoFFinite, OrdinaryTransportKeepsIndependentPhasesCanonicalRoutesAndEmptyOwners)
{
    struct StatisticsScope
    {
        StatisticsScope()
        {
            GeneralVoF::enable_geometry_statistics(true);
            GeneralVoF::reset_geometry_statistics();
            PLIC::reset_geometry_statistics();
            SimpleFluid::PLIC::enable_backend_statistics<ArithmeticMode::Exact>(true);
            SimpleFluid::PLIC::reset_backend_statistics<ArithmeticMode::Exact>();
        }
        ~StatisticsScope()
        {
            GeneralVoF::enable_geometry_statistics(false);
            SimpleFluid::PLIC::enable_backend_statistics<ArithmeticMode::Exact>(false);
        }
    } statistics_scope;
    const auto input = fixture();
    GeneralVoF state(input.hierarchy, input.order, input.gas, input.carrier);
    EXPECT_EQ(state.arithmetic_mode(), ArithmeticMode::Ordinary);
    for (const auto& geometry : state.owned_finite_geometry<ArithmeticMode::Ordinary>())
        EXPECT_EQ(geometry->arithmetic_mode(), ArithmeticMode::Ordinary);
    const auto velocities = stationary(state);
    const auto controls = circulation(state);
    const auto prepared = GeneralVoF::geometry_statistics();
    EXPECT_EQ(prepared.bounded.runtime_dispatches, 0u);
    EXPECT_EQ(PLIC::geometry_statistics().runtime_dispatches, 0u);
    const auto trial = state.plan_transport(velocities, controls, 1);
    const auto candidate = state.apply(trial);
    expect_static_geometry_work();
    check_budget_diagnostics(state, trial);
    EXPECT_TRUE(trial.transfers().empty());
    EXPECT_TRUE(trial.donor_budgets().empty());
    const auto ledger = trial.prepare_transport();
    EXPECT_EQ(ledger->family, GeneralVoFTransportFamily::InterfaceLocalSwept);
    EXPECT_TRUE(trial.finite_donor_budgets().empty());
    EXPECT_TRUE(trial.finite_transfers().empty());
    ASSERT_EQ(ledger->signed_donor_budgets.size(), input.order.size());
    ASSERT_EQ(candidate->accumulated_phase_errors().size(), input.order.size());
    fp::Sum gas(state.arithmetic_mode()), carrier(state.arithmetic_mode());
    for (std::size_t i = 0; i < candidate->phase_volumes().size(); ++i)
    {
        const auto& p = candidate->phase_volumes()[i];
        EXPECT_GE(p.gas, 0);
        EXPECT_GE(p.carrier, 0);
        gas.add(fp::make_value(state.arithmetic_mode(), p.gas, trial.phase_roundoff_bounds()[i].gas));
        carrier.add(fp::make_value(state.arithmetic_mode(), p.carrier, trial.phase_roundoff_bounds()[i].carrier));
        EXPECT_GE(candidate->accumulated_phase_errors()[i].gas, trial.phase_roundoff_bounds()[i].gas);
        EXPECT_GE(candidate->accumulated_phase_errors()[i].carrier, trial.phase_roundoff_bounds()[i].carrier);
    }
    const auto comm = Tpetra::getDefaultComm();
    double local[4] = {static_cast<double>(gas.value().value), static_cast<double>(carrier.value().value),
        fp::upper_double(gas.value().absolute_error), fp::upper_double(carrier.value().absolute_error)}, global[4]{};
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 4, local, global);
    EXPECT_NEAR(global[0], 1., global[2] + 1e-14);
    EXPECT_NEAR(global[1], 7., global[3] + 1e-14);
    for (const auto& route : ledger->signed_receipts)
    {
        EXPECT_EQ(route.recipient_owner, comm->getRank());
        EXPECT_EQ(route.amount.bulk.arithmetic_mode, ArithmeticMode::Ordinary);
        EXPECT_EQ(route.amount.gas.arithmetic_mode, ArithmeticMode::Ordinary);
        EXPECT_EQ(route.amount.carrier.arithmetic_mode, ArithmeticMode::Ordinary);
        EXPECT_TRUE(std::isfinite(route.amount.gas.value));
        EXPECT_TRUE(std::isfinite(route.amount.carrier.value));
    }
    int local_empty = input.order.empty(), empty = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local_empty, &empty);
    EXPECT_EQ(empty, std::max(0, comm->getSize() - 2));
    const auto observed = GeneralVoF::geometry_statistics();
    const auto& statistics = observed.bounded;
    EXPECT_EQ(observed.exact.source_coordinate_exports, 0);
    EXPECT_EQ(observed.exact.source_support_exports, 0);
    EXPECT_EQ(observed.exact.volume_evaluations, 0);
    EXPECT_EQ(observed.exact.moment_evaluations, 0);
    EXPECT_EQ(observed.exact.clip_constructions, 0);
    EXPECT_EQ(observed.exact.intersection_constructions, 0);
    EXPECT_EQ(observed.exact.measure_values, 0);
    EXPECT_EQ(observed.exact.measure_arithmetic, 0);
    EXPECT_EQ(observed.exact.cover_certifications, 0);
    const auto native = SimpleFluid::PLIC::backend_statistics<ArithmeticMode::Exact>();
    EXPECT_EQ(native.source_coordinate_exports, 0);
    EXPECT_EQ(native.source_support_exports, 0);
    EXPECT_EQ(native.volume_evaluations, 0);
    EXPECT_EQ(native.moment_evaluations, 0);
    EXPECT_EQ(native.clip_constructions, 0);
    EXPECT_EQ(native.intersection_constructions, 0);
    EXPECT_EQ(native.measure_values, 0);
    EXPECT_EQ(native.measure_arithmetic, 0);
    EXPECT_EQ(native.cover_certifications, 0);
    EXPECT_EQ(statistics.source_imports, prepared.bounded.source_imports);
    EXPECT_EQ(statistics.interval_operations, 0);
    const std::uint64_t local_work = statistics.constructions - prepared.bounded.constructions +
                                     statistics.volume_evaluations - prepared.bounded.volume_evaluations;
    std::uint64_t global_work = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local_work, &global_work);
    // Opposite-pure faces need swept transport, but their computational
    // witnesses no longer require measured ConvexGeometry children.
    EXPECT_EQ(global_work, 0u);
    const std::uint64_t local_witnesses = ledger->diagnostics.aggregate_witnesses;
    std::uint64_t global_witnesses = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local_witnesses, &global_witnesses);
    EXPECT_GT(global_witnesses, 0u);
    EXPECT_EQ(state.gas_volumes(), input.gas);
    EXPECT_EQ(state.carrier_volumes(), input.carrier);
}
TEST(GeneralVoFFinite, RankLocalMalformedAndCrossPolicyControlsRejectWithoutChangingPublication)
{
    const auto input = fixture();
    GeneralVoF state(input.hierarchy, input.order, input.gas, input.carrier);
    const auto identity = state.accepted_identity();
    const auto control = state.accepted_control();
    auto faces = circulation(state);
    if (Tpetra::getDefaultComm()->getRank() == 0 && !faces.empty()) faces.front().face ^= 1;
    EXPECT_THROW(state.plan_transport(stationary(state), faces, 1), std::invalid_argument);
    EXPECT_EQ(state.accepted_identity(), identity);
    EXPECT_EQ(state.accepted_control(), control);
    faces = circulation(state, true);
    if (Tpetra::getDefaultComm()->getRank() == 0 && !faces.empty())
        faces.front().bounded_integrated_volume = SimpleFluid::PLIC::DynamicGeometryValue{0, 0, ArithmeticMode::Exact};
    EXPECT_THROW(state.plan_transport(stationary(state), faces, 1), std::invalid_argument);
    EXPECT_EQ(state.gas_volumes(), input.gas);
    EXPECT_EQ(state.carrier_volumes(), input.carrier);
}
TEST(GeneralVoFFinite, OwnerFaceBudgetsCoverCoarseFineTilesAndEmptyOwners)
{
    const auto input = coarse_fine_fixture();
    ASSERT_EQ(input.hierarchy->global_cells(), 72u);
    GeneralVoF state(input.hierarchy, input.order, input.gas, input.carrier);
    auto controls = circulation(state, true);
    const auto trial = state.plan_transport(stationary(state), controls, 1);
    check_budget_diagnostics(state, trial);
    std::uint64_t local_count = 0, maximum = 0;
    for (const auto& cell : trial.finite_cell_closure_diagnostics())
        local_count = std::max(local_count, std::uint64_t(cell.canonical_face_count));
    const auto comm = Tpetra::getDefaultComm();
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &local_count, &maximum);
    EXPECT_GT(maximum, 6u); // The coarse seam uses actual fine subfaces.
    int empty = input.order.empty(), total_empty = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &empty, &total_empty);
    EXPECT_EQ(total_empty, std::max(0, comm->getSize() - 2));
}

TEST(GeneralVoFFinite, OrdinaryFaceAllowanceCannotHideMaterialCellControlDefect)
{
    const auto input = fixture();
    GeneralVoF state(input.hierarchy, input.order, input.gas, input.carrier);
    auto controls = circulation(state, true);
    std::uint64_t local = std::numeric_limits<std::uint64_t>::max(), chosen = 0;
    for (const auto& face : state.tiles())
        if (!face.exterior) local = std::min(local, face.face_gid);
    const auto comm = Tpetra::getDefaultComm();
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MIN, 1, &local, &chosen);
    for (auto& face : controls)
        if (face.face == chosen) face.integrated_volume = 1e-3;
    const auto identity = state.accepted_identity();
    const auto control = state.accepted_control();
    EXPECT_THROW((void) state.plan_transport(stationary(state), controls, 1), std::invalid_argument);
    EXPECT_EQ(state.accepted_identity(), identity);
    EXPECT_EQ(state.accepted_control(), control);
    EXPECT_EQ(state.gas_volumes(), input.gas);
    EXPECT_EQ(state.carrier_volumes(), input.carrier);
}
TEST(GeneralVoFFinite, UnderresolvedNeighborhoodRequestsRefinementWithoutDeletingAcceptedPhase)
{
    const auto input = fixture(true);
    GeneralVoF state(input.hierarchy, input.order, input.gas, input.carrier);
    const auto requests = state.resolution_requests();
    unsigned count = unsigned(requests.size()), global = 0;
    Teuchos::reduceAll(*Tpetra::getDefaultComm(), Teuchos::REDUCE_SUM, 1, &count, &global);
    EXPECT_EQ(global, 8u);
    for (const auto& request : requests)
    {
        EXPECT_FALSE(request.assessment.admitted());
        EXPECT_GT(request.assessment.local_cell_span, request.assessment.requested_cell_span);
    }
    EXPECT_THROW(state.reconstruct(), std::invalid_argument);
    EXPECT_THROW(state.plan_transport(stationary(state), circulation(state), 1), std::invalid_argument);
    EXPECT_EQ(state.gas_volumes(), input.gas);
    EXPECT_EQ(state.carrier_volumes(), input.carrier);
    const auto unchanged = state.plan_transport(stationary(state), circulation(state, true), 1);
    EXPECT_EQ(state.apply(unchanged)->accepted_identity(), state.accepted_identity());
}

std::shared_ptr<const DistributedAdaptiveHierarchy> bound_candidate(
    const std::shared_ptr<const DistributedAdaptiveHierarchy>& source, const DistributedAdaptiveHierarchy::Plan& plan)
{
    const auto raw = source->apply(plan);
    const auto native = assemble_adaptive_native(*raw, raw->configuration_descriptor());
    return source->apply(plan, raw->with_partition(native.partition, native.mesh->owned_cell_map()));
}
double original_cell_marker(const AdaptiveCellKey& key)
{
    const auto& ancestry = std::get<CartesianCellAncestry>(key.ancestry);
    double marker = 1;
    for (unsigned a = 0; a < 3; ++a)
        marker += (1u << a) * std::floor(std::ldexp(double(ancestry.index[a]) + .5, -int(ancestry.level)));
    return marker;
}
GeneralVoFFiniteRemapOptions channel_options()
{
    GeneralVoFFiniteRemapOptions options;
    options.phase_density = {.1, 1};
    options.channels = {{"gas_marker", GeneralVoFFiniteRemapCarrier::GasVolume, true},
        {"cell_marker", GeneralVoFFiniteRemapCarrier::BulkVolume, true}};
    return options;
}
std::vector<GeneralVoFFiniteOwnedChannels> initial_channels(const GeneralVoF& state)
{
    std::vector<GeneralVoFFiniteOwnedChannels> result;
    for (std::size_t c = 0; c < state.owned_order().size(); ++c)
    {
        const auto& key = state.owned_order()[c];
        result.push_back({key, {{2 * state.gas_volumes()[c], 0},
            {state.phase_volumes()[c].bulk * original_cell_marker(key), 0}}});
    }
    return result;
}
void check_remap(const GeneralVoF::FiniteRemapTrial& trial, unsigned expected_cells)
{
    const auto state = trial.candidate();
    EXPECT_EQ(state->inventory_policy(), PhaseInventoryPolicy::Finite);
    EXPECT_EQ(state->hierarchy()->global_cells(), expected_cells);
    std::map<AdaptiveCellKey, const GeneralVoFFiniteOwnedChannels*> channels;
    for (const auto& item : trial.owned_channels())
        ASSERT_TRUE(channels.emplace(item.key, &item).second);
    double local[2]{}, global[2]{};
    for (std::size_t c = 0; c < state->owned_order().size(); ++c)
    {
        const auto& key = state->owned_order()[c];
        const auto& p = state->phase_volumes()[c];
        const auto& output = channels.at(key)->amounts;
        ASSERT_EQ(output.size(), 2u);
        EXPECT_GE(p.gas, 0);
        EXPECT_GE(p.carrier, 0);
        EXPECT_NEAR(output[0].value, 2 * p.gas, output[0].absolute_error + 1e-14);
        EXPECT_NEAR(output[1].value, p.bulk * original_cell_marker(key), output[1].absolute_error + 1e-14);
        local[0] += p.gas; local[1] += p.carrier;
    }
    Teuchos::reduceAll(*Tpetra::getDefaultComm(), Teuchos::REDUCE_SUM, 2, local, global);
    EXPECT_NEAR(global[0], 1, 2e-14);
    EXPECT_NEAR(global[1], 7, 2e-14);
    trial.validate_observations();
}
TEST(GeneralVoFFinite, FiniteRefineCoarsenPreservesPhasesWeightedChannelsAndSource)
{
    const auto input = fixture();
    GeneralVoFOptions options;
    options.phase_cleanup.relative_cutoff = 0;
    const auto source =
        std::make_shared<const GeneralVoF>(input.hierarchy, input.order, input.gas, input.carrier, options);
    const auto control = source->accepted_control();
    std::vector<AdaptiveMark> marks;
    for (std::size_t c = 0; c < source->owned_order().size(); ++c)
        if (source->gas_volumes()[c] == 1)
            marks.push_back({source->owned_order()[c], AdaptiveSplit::Isotropic});
    const auto plan = source->hierarchy()->plan_adaptation(marks);
    const auto target = bound_candidate(source->hierarchy(), plan);
    const auto channels = initial_channels(*source);
    reset_dispatch_statistics();
    const auto trial = source->plan_remap(plan, target, channels, channel_options());
    expect_static_geometry_work();
    EXPECT_FALSE(trial.ownership_only());
    // The Cartesian provider refines the complete native block containing
    // the mark: all eight original leaves acquire eight children.
    check_remap(trial, 64);
    EXPECT_EQ(source->accepted_control(), control);
    EXPECT_EQ(source->gas_volumes(), input.gas);
    EXPECT_EQ(source->carrier_volumes(), input.carrier);
    const auto refined = trial.candidate();
    std::vector<AdaptiveBytes> families;
    for (const auto& [key, family] : refined->hierarchy()->families())
        families.push_back(key);
    const auto coarsen = refined->hierarchy()->plan_adaptation({}, families);
    const auto coarse_target = bound_candidate(refined->hierarchy(), coarsen);
    reset_dispatch_statistics();
    const auto restored = refined->plan_remap(coarsen, coarse_target, trial.owned_channels(), channel_options());
    expect_static_geometry_work();
    check_remap(restored, 8);
}
TEST(GeneralVoFFinite, FiniteMigrationCopiesInventoriesAndRejectsForeignSchemaAtomically)
{
    const auto input = fixture();
    GeneralVoFOptions options;
    options.phase_cleanup.relative_cutoff = 0;
    GeneralVoF source(input.hierarchy, input.order, input.gas, input.carrier, options);
    const auto comm = Tpetra::getDefaultComm();
    std::vector<int> destinations(source.hierarchy()->owned().size(), (comm->getRank() + 1) % comm->getSize());
    const auto plan = source.hierarchy()->plan_migration(destinations);
    const auto target = bound_candidate(source.hierarchy(), plan);
    const auto channels = initial_channels(source);
    const auto trial = source.plan_remap(plan, target, channels, channel_options());
    EXPECT_TRUE(trial.ownership_only());
    check_remap(trial, 8);
    const auto migrated = trial.candidate();
    for (std::size_t c = 0; c < migrated->owned_order().size(); ++c)
    {
        const auto gas = original_cell_marker(migrated->owned_order()[c]) == 1 ? 1. : 0.;
        EXPECT_EQ(migrated->gas_volumes()[c], gas);
        EXPECT_EQ(migrated->carrier_volumes()[c], 1 - gas);
        EXPECT_EQ(migrated->phase_roundoff_bounds()[c].gas, 0);
        EXPECT_EQ(migrated->phase_roundoff_bounds()[c].carrier, 0);
    }
    auto invalid = channel_options();
    if (comm->getRank() == 0) invalid.channels[0].name.clear();
    EXPECT_THROW(source.plan_remap(plan, target, channels, invalid), std::invalid_argument);
    invalid = channel_options();
    invalid.byte_budget = 1;
    EXPECT_THROW(source.plan_remap(plan, target, channels, invalid), std::invalid_argument);
    EXPECT_EQ(source.gas_volumes(), input.gas);
    EXPECT_EQ(source.carrier_volumes(), input.carrier);
}

TEST(GeneralVoFFinite, MixedOwnershipMigrationPreservesOrientationAndMeasuredResidual)
{
    const auto comm = Tpetra::getDefaultComm();
    const auto ordinal = [](const AdaptiveCellKey& key)
    {
        const auto index = std::get<CartesianCellAncestry>(key.ancestry).index;
        return std::size_t(index[0] + 2 * index[1] + 4 * index[2]);
    };
    const auto diagnostics = [](const PLIC::PolyhedralInterface& plane)
    {
        return std::array{double(plane.orientation_method), double(plane.fit_converged), double(plane.fit_iterations),
            double(plane.stencil_volume_residual_measured), plane.stencil_volume_residual, plane.stencil_condition,
            plane.orientation_step};
    };
    for (const double fraction : {1e-6, .1, .5})
    {
        SCOPED_TRACE(fraction);
        auto input = fixture();
        for (std::size_t c = 0; c < input.order.size(); ++c)
        {
            const auto index = std::get<CartesianCellAncestry>(input.order[c].ancestry).index;
            input.gas[c] = index[0] == 0 ? fraction : 0;
            input.carrier[c] = 1 - input.gas[c];
        }
        GeneralVoFOptions options;
        options.phase_cleanup.relative_cutoff = 0; // This case carries coupled remap channels.
        GeneralVoF source(input.hierarchy, input.order, input.gas, input.carrier, options);
        const auto before = source.reconstruct();
        constexpr std::size_t width = 7;
        std::array<double, 8 * width> local{}, expected{};
        for (std::size_t c = 0; c < source.owned_order().size(); ++c)
        {
            const auto fields = diagnostics(before[c]);
            std::copy(fields.begin(), fields.end(), local.begin() + width * ordinal(source.owned_order()[c]));
            if (input.gas[c] != 0)
            {
                EXPECT_EQ(before[c].orientation_method, fraction < .5
                                                            ? PLIC::PolyhedralOrientationMethod::Gradient
                                                            : PLIC::PolyhedralOrientationMethod::AngularVolumeFit);
                EXPECT_EQ(before[c].stencil_volume_residual_measured, fraction >= .000125);
            }
        }
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, int(local.size()), local.data(), expected.data());
        const std::vector<int> destinations(input.order.size(), (comm->getRank() + 1) % comm->getSize());
        const auto plan = source.hierarchy()->plan_migration(destinations);
        const auto target = bound_candidate(source.hierarchy(), plan);
        const auto trial = source.plan_remap(plan, target, initial_channels(source), channel_options());
        ASSERT_TRUE(trial.ownership_only());
        const auto moved = trial.candidate();
        const auto after = moved->reconstruct();
        for (std::size_t c = 0; c < moved->owned_order().size(); ++c)
        {
            const auto fields = diagnostics(after[c]);
            for (std::size_t field = 0; field < width; ++field)
                EXPECT_EQ(fields[field], expected[width * ordinal(moved->owned_order()[c]) + field]);
        }
    }
}

TEST(GeneralVoFFinite, SourceOccupancyFollowsGeometricDonorSharesAndRecordsSignedFitRedistribution)
{
    auto input = fixture();
    constexpr double input_defect = 4 * std::numeric_limits<double>::epsilon();
    for (std::size_t i = 0; i < input.order.size(); ++i)
    {
        const auto center = input.hierarchy->geometry(input.hierarchy->owned().at(input.order[i]))->centroid();
        input.gas[i] = center[0] < 1 ? .9 : 0;
        input.carrier[i] = 1 - input.gas[i] + (center[0] < 1 ? input_defect : 0);
    }
    GeneralVoF state(input.hierarchy, input.order, input.gas, input.carrier);
    const auto trial = state.plan_transport(stationary(state), circulation(state), 1);
    const auto moved = state.apply(trial);
    int local_redistributed = 0;
    for (std::size_t i = 0; i < input.order.size(); ++i)
    {
        const auto& before = state.initial_occupancy_errors()[i];
        const auto& after = moved->initial_occupancy_errors()[i];
        EXPECT_EQ(before.fit_allocation_bound, 0);
        EXPECT_LE(std::abs(after.difference), after.admitted_bound);
        const auto current_arithmetic = fp::bound_add(trial.phase_roundoff_bounds()[i].gas,
            trial.phase_roundoff_bounds()[i].carrier);
        // The carrier-only sweep from the left source removes the uniform
        // occupancy share from this route. That signed geometric redistribution
        // is a model contribution, not an enlarged arithmetic radius.
        if (input.gas[i] == 0 && after.fit_allocation_bound > current_arithmetic + input_defect / 256)
            ++local_redistributed;
    }
    int redistributed = 0;
    Teuchos::reduceAll(*Tpetra::getDefaultComm(), Teuchos::REDUCE_SUM, 1, &local_redistributed, &redistributed);
    EXPECT_GT(redistributed, 0);
    EXPECT_EQ(state.gas_volumes(), input.gas);
    EXPECT_EQ(state.carrier_volumes(), input.carrier);
}
TEST(GeneralVoFFinite, SamePureGeometryKeepsIndependentTraceInventoryWithoutInventingAnInterface)
{
    auto input = fixture();
    std::fill(input.gas.begin(), input.gas.end(), 1.);
    std::fill(input.carrier.begin(), input.carrier.end(), 1e-18);
    const std::vector<GeneralVoFPhaseGeometry> declarations(input.order.size(), GeneralVoFPhaseGeometry::PureGas);
    GeneralVoFOptions options;
    options.phase_cleanup.relative_cutoff = 0;
    GeneralVoF state(input.hierarchy, input.order, input.gas, input.carrier, options, declarations);
    const auto moved = state.apply(state.plan_transport(stationary(state), circulation(state), 1));
    const auto pure = moved->reconstruct();
    for (std::size_t i = 0; i < input.order.size(); ++i)
    {
        EXPECT_GT(moved->carrier_volumes()[i], 0);
        EXPECT_NEAR(moved->carrier_volumes()[i], 1e-18, moved->phase_roundoff_bounds()[i].carrier);
        EXPECT_TRUE(pure[i].finite_geometry);
        EXPECT_TRUE(pure[i].pure_gas);
        EXPECT_FALSE(pure[i].pure_carrier);
    }
}
TEST(GeneralVoFFinite, IncomingTinyGeometryInvalidatesPureHintAndIdentityPreservesSourceEvidence)
{
    auto input = fixture();
    constexpr double cut = 1 - 3. / 64 + 1e-6;
    for (std::size_t i = 0; i < input.order.size(); ++i)
    {
        const auto center = input.hierarchy->geometry(input.hierarchy->owned().at(input.order[i]))->centroid();
        input.gas[i] = center[0] < 1 ? cut : 0;
        input.carrier[i] = 1 - input.gas[i];
    }
    GeneralVoFOptions options;
    options.phase_cleanup.relative_cutoff = 0;
    GeneralVoF state(input.hierarchy, input.order, input.gas, input.carrier, options);
    const auto identity = state.plan_transport(stationary(state), circulation(state, true), 1);
    EXPECT_EQ(state.apply(identity)->accepted_identity(), state.accepted_identity());
    const auto moved = state.apply(state.plan_transport(stationary(state), circulation(state), 1));
    // Inspect accepted hints without fitting, resolution heuristics or exports.
    const auto declarations = moved->phase_geometry_declarations();
    ASSERT_EQ(declarations.size(), moved->owned_order().size());
    int local_tiny_receipts = 0;
    for (std::size_t i = 0; i < input.order.size(); ++i)
        if (input.gas[i] == 0 && moved->gas_volumes()[i] > 0)
        {
            ++local_tiny_receipts;
            EXPECT_LT(moved->gas_volumes()[i], 8 * std::numeric_limits<double>::epsilon());
            EXPECT_EQ(declarations[i], GeneralVoFPhaseGeometry::Infer);
        }
    int tiny_receipts = 0;
    Teuchos::reduceAll(*Tpetra::getDefaultComm(), Teuchos::REDUCE_SUM, 1, &local_tiny_receipts, &tiny_receipts);
    EXPECT_GT(tiny_receipts, 0);
}
TEST(GeneralVoFFinite, TinyReceiptArithmeticRejectionPreservesAcceptedIndependentInventories)
{
    auto input = fixture();
    // A resolved planar interface lies just past the ordinary swept pyramid's
    // apex. The unit-scale donor is mostly gas, but its right recipient receives
    // only a tiny gas cap. Bulk face geometry keeps its ordinary accuracy gate.
    constexpr double cut = 1 - 3. / 64 + 1e-6;
    for (std::size_t i = 0; i < input.order.size(); ++i)
    {
        const auto center = input.hierarchy->geometry(input.hierarchy->owned().at(input.order[i]))->centroid();
        input.gas[i] = center[0] < 1 ? cut : 0;
        input.carrier[i] = 1 - input.gas[i];
    }
    GeneralVoFOptions bounded;
    bounded.arithmetic_mode = ArithmeticMode::BoundedCertification;
    GeneralVoF state(input.hierarchy, input.order, input.gas, input.carrier, bounded);
    const auto identity = state.accepted_identity();
    const auto control = state.accepted_control();
    const auto phases = state.phase_volumes();
    const auto history = state.accumulated_phase_errors();
    auto faces = circulation(state);
    // A bounded canonical control can satisfy its unit-scale occupancy budget
    // while leaving a much smaller phase receipt uncertified. The same radius
    // is shared by every owner incidence and is never hidden in phase history.
    for (auto& face : faces)
        if (face.integrated_volume != 0)
            face.bounded_integrated_volume = SimpleFluid::PLIC::DynamicGeometryValue{face.integrated_volume, 1e-19L};
    std::string diagnostic;
    try
    {
        (void) state.plan_transport(stationary(state), faces, 1);
        FAIL() << "An uncertified tiny receipt must not borrow its donor's phase budget.";
    }
    catch (const std::invalid_argument& error)
    {
        diagnostic = error.what();
    }
    // Geometry may reject the uncertain tiny cut before recipient publication.
    // Both paths are arithmetic failures; neither requests spatial refinement
    // solely because the amount is small. The own-phase admission formula is
    // covered separately by the scalar ReceiptAdmissionUsesItsOwnPhaseAndExcludesFinalNarrowing regression.
    const int local_arithmetic_gate = diagnostic.find("recipient gas phase current-step") != std::string::npos ||
        diagnostic.find("support/topology sign unresolved") != std::string::npos;
    int observed_arithmetic_gate = 0;
    Teuchos::reduceAll(*Tpetra::getDefaultComm(), Teuchos::REDUCE_MAX, 1,
        &local_arithmetic_gate, &observed_arithmetic_gate);
    EXPECT_EQ(observed_arithmetic_gate, 1) << diagnostic;
    EXPECT_EQ(state.accepted_identity(), identity);
    EXPECT_EQ(state.accepted_control(), control);
    ASSERT_EQ(state.phase_volumes().size(), phases.size());
    ASSERT_EQ(state.accumulated_phase_errors().size(), history.size());
    for (std::size_t i = 0; i < phases.size(); ++i)
    {
        EXPECT_EQ(state.phase_volumes()[i].gas, phases[i].gas);
        EXPECT_EQ(state.phase_volumes()[i].carrier, phases[i].carrier);
        EXPECT_EQ(state.accumulated_phase_errors()[i].gas, history[i].gas);
        EXPECT_EQ(state.accumulated_phase_errors()[i].carrier, history[i].carrier);
    }
}

TEST(GeneralVoFFinite, ConstructionControlsBindEveryPinAndServeEmptyAuthorities)
{
    namespace controls = SimpleFluid::VoF::construction_controls;
    namespace wire = SimpleFluid::VoF::general_vof_detail::wire;
    const auto input = fixture();
    const auto comm = Tpetra::getDefaultComm();
    const std::uint64_t id = std::min(2, comm->getSize() - 1);
    const PLIC::GeometryPacketContext context{{101, 103, 107, 109}};
    wire::Writer writer;
    controls::header(writer, 1, id, context, ArithmeticMode::Ordinary);
    writer.word(127);
    controls::Records authority;
    std::size_t retained = 0;
    if (comm->getRank() == int(id))
    {
        authority.emplace(std::pair<std::uint64_t, std::uint64_t>{1, id}, writer.data);
        retained = sizeof(controls::Records::value_type) + 4 * sizeof(void*) + writer.data.size();
        if (comm->getSize() > 2)
            EXPECT_TRUE(input.order.empty());
    }
    std::set<std::uint64_t> requested;
    if (comm->getRank() == 0)
        requested.insert(id);
    const auto received = controls::request_records(
        1, requested, authority, context, ArithmeticMode::Ordinary, comm, 64 * 1024, retained);
    const bool complete = comm->getRank() == 0
                              ? received.contains(id) && received.at(id) == writer.data && received.size() == 1
                              : received.empty();
    int local = complete, global = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MIN, 1, &local, &global);
    EXPECT_EQ(global, 1);
    if (!global)
        return;
    for (std::size_t pin = 0; pin < context.publication.size(); ++pin)
    {
        auto stale = context;
        ++stale.publication[pin];
        EXPECT_THROW((void) controls::request_records(
                         1, requested, authority, stale, ArithmeticMode::Ordinary, comm, 64 * 1024, retained),
            std::invalid_argument);
    }
    EXPECT_THROW((void) controls::request_records(
                     1, requested, authority, context, ArithmeticMode::BoundedCertification, comm, 64 * 1024, retained),
        std::invalid_argument);
    auto unknown = requested;
    if (comm->getRank() == 0)
    {
        unknown.clear();
        unknown.insert(id + comm->getSize());
    }
    EXPECT_THROW((void) controls::request_records(
                     1, unknown, authority, context, ArithmeticMode::Ordinary, comm, 64 * 1024, retained),
        std::invalid_argument);
    const std::size_t local_limit = comm->getRank() == 0 ? 1 : 64 * 1024;
    EXPECT_THROW((void) controls::request_records(
                     1, requested, authority, context, ArithmeticMode::Ordinary, comm, local_limit, retained),
        std::invalid_argument);
    EXPECT_EQ(authority.size(), comm->getRank() == int(id) ? 1u : 0u);
}

TEST(GeneralVoFFinite, ConstructionDependenciesBindDonorRecipientAndAdmittedHalo)
{
    namespace controls = SimpleFluid::VoF::construction_controls;
    using Root = PLIC::ConstructionRootKey;
    GeneralVoF::Data data;
    auto topology = std::make_shared<GeneralVoF::Data::Topology>();
    data.topology = topology;
    GeneralVoF::Key donor{}, recipient{}, foreign{};
    recipient.region = 1;
    foreign.region = 2;
    SimpleFluid::VoF::general_vof_detail::Cell cell;
    cell.gid = 11;
    cell.revision = 7;
    cell.key = donor;
    data.cells.emplace(donor, cell);
    topology->by_gid.emplace(11, donor);
    GeneralVoF::Data::FaceAuthority face;
    face.first = recipient;
    face.second = donor;
    face.node_gids = {1, 2, 3};
    face.face_gid = 17;
    topology->tiles.push_back(face);
    topology->construction_faces.emplace(17,
        GeneralVoF::Data::Topology::SurfaceTileReference{GeneralVoF::Data::Topology::SurfaceTileOwner::OwnedTiles, 0});
    Root native{Root::Kind::Native, 11, 7, 0, {}};
    Root sweep{Root::Kind::Sweep, 17, 0, 0, {}};
    EXPECT_NO_THROW(controls::admit_root(data, native, 11, 7, recipient));
    EXPECT_NO_THROW(controls::admit_root(data, sweep, 11, 7, recipient));
    EXPECT_THROW(controls::admit_root(data, native, 12, 7, recipient), std::invalid_argument);
    EXPECT_THROW(controls::admit_root(data, native, 11, 8, recipient), std::invalid_argument);
    EXPECT_THROW(controls::admit_root(data, sweep, 11, 7, foreign), std::invalid_argument);
    sweep.entity = 99;
    EXPECT_THROW(controls::admit_root(data, sweep, 11, 7, recipient), std::invalid_argument);
    sweep.entity = 17;
    sweep.slot = 9;
    EXPECT_THROW(controls::admit_root(data, sweep, 11, 7, recipient), std::invalid_argument);
    sweep.slot = 0;
    sweep.revision = 1;
    EXPECT_THROW(controls::admit_root(data, sweep, 11, 7, recipient), std::invalid_argument);
    sweep.revision = 0;
    sweep.identity = {'u', 'n', 't', 'r', 'u', 's', 't', 'e', 'd'};
    EXPECT_THROW(controls::admit_root(data, sweep, 11, 7, recipient), std::invalid_argument);
    sweep.identity.clear();
    sweep.kind = Root::Kind::InputPlane;
    EXPECT_THROW(controls::admit_root(data, sweep, 11, 7, recipient), std::invalid_argument);
    sweep.kind = Root::Kind::LocalSource;
    EXPECT_THROW(controls::admit_root(data, sweep, 11, 7, recipient), std::invalid_argument);

    // Merely mentioning the donor as a cutting-plane source does not prove
    // that the atom descends from that donor, even if the final coordinates fit.
    using Geometry = PLIC::ConvexCellGeometry<ArithmeticMode::Ordinary>;
    const auto donor_geometry = Geometry::box({0, 0, 0}, {1, 1, 1});
    const auto other_base = Geometry::box({0, 0, 0}, {2, 1, 1});
    PLIC::ConstructionRegistry<ArithmeticMode::Ordinary> registry({{101, 103, 107, 109}});
    registry.register_native(11, 7, donor_geometry);
    registry.register_sweep(17, 0, other_base);
    std::size_t right = donor_geometry->faces().size();
    for (std::size_t f = 0; f < donor_geometry->faces().size(); ++f)
        if (std::all_of(donor_geometry->faces()[f].begin(), donor_geometry->faces()[f].end(),
                [&](auto vertex) { return donor_geometry->points()[vertex].value[0] == 1; }))
            right = f;
    ASSERT_LT(right, donor_geometry->faces().size());
    const auto borrowed = other_base->clip_by_face(*donor_geometry, right, false).inside;
    ASSERT_TRUE(borrowed);
    const auto decoded = Geometry::decode(borrowed->encode(registry), registry);
    ASSERT_TRUE(decoded);
    EXPECT_THROW(controls::require_donor_ancestry(registry, *decoded, 11, 7), std::invalid_argument);
    EXPECT_NO_THROW(controls::require_donor_ancestry(registry, *donor_geometry, 11, 7));
}

TEST(GeneralVoFFinite, MovingConstructionReplayPreservesControlsWithRankLocalByteBudgets)
{
    const auto input = fixture();
    const auto comm = Tpetra::getDefaultComm();
    GeneralVoFOptions options;
    options.byte_budget = (32u + std::size_t(comm->getRank())) * 1024 * 1024;
    options.sweep.byte_budget = (16u + std::size_t(comm->getRank())) * 1024 * 1024;
    const GeneralVoF state(input.hierarchy, input.order, input.gas, input.carrier, options);
    const auto node_count = state.nodes().size(), face_count = state.tiles().size();
    const auto trial = state.plan_transport(stationary(state), circulation(state), 1);
    EXPECT_EQ(trial.finite_node_trajectories().size(), node_count);
    EXPECT_EQ(trial.finite_face_chain_diagnostics().size(), face_count);
    EXPECT_EQ(state.nodes().size(), node_count);
    EXPECT_EQ(state.tiles().size(), face_count);
    check_budget_diagnostics(state, trial);
    const auto next = state.apply(trial);
    double local[2]{}, global[2]{};
    for (const auto& phases : next->phase_volumes())
    {
        EXPECT_GE(phases.gas, 0);
        EXPECT_GE(phases.carrier, 0);
        local[0] += phases.gas;
        local[1] += phases.carrier;
    }
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 2, local, global);
    EXPECT_NEAR(global[0], 1, 1e-14);
    EXPECT_NEAR(global[1], 7, 1e-14);
}

} // namespace

TEST(GeneralVoFFinite, GeometryRecordEntriesRejectLegacyEmptyPayloadsAndWrongKinds)
{
    namespace protocol = SimpleFluid::VoF::geometry_wire;
    namespace wire = SimpleFluid::VoF::general_vof_detail::wire;
    for (const auto record :
        {protocol::Record::NativeShellRequest, protocol::Record::NativeShellReply, protocol::Record::ExactTileRequest,
            protocol::Record::ExactTileReply, protocol::Record::ExactFaceOutput, protocol::Record::ExactControls,
            protocol::Record::NodeUnion, protocol::Record::NodePublication, protocol::Record::ExactAtomRoute})
    {
        wire::Writer current;
        protocol::write_header(current, record);
        ASSERT_EQ(current.data.size(), sizeof(protocol::Header));
        EXPECT_EQ(static_cast<std::uint8_t>(current.data[offsetof(protocol::Header, family)]), 3u);
        EXPECT_EQ(static_cast<std::uint8_t>(current.data[offsetof(protocol::Header, version)]), 3u);
        EXPECT_EQ(static_cast<std::uint8_t>(current.data[offsetof(protocol::Header, record)]),
            static_cast<std::uint8_t>(record));
        wire::Writer repeated;
        protocol::write_header(repeated, record);
        EXPECT_EQ(repeated.data, current.data);
        wire::Reader accepted{current.data};
        EXPECT_NO_THROW(protocol::read_header(accepted, record));
        EXPECT_EQ(accepted.offset, protocol::header_bytes);
        wire::Writer legacy;
        legacy.word(0);
        legacy.word(0);
        legacy.word(0);
        wire::Reader old{legacy.data};
        EXPECT_THROW(protocol::read_header(old, record), std::invalid_argument);
        for (const auto previous_version : {std::uint8_t{2}, std::uint8_t{3}})
        {
            wire::Writer legacy_frame;
            legacy_frame.word(0x5346564f46574952ULL);
            legacy_frame.byte(previous_version);
            legacy_frame.byte(static_cast<std::uint8_t>(record));
            wire::Reader previous_frame{legacy_frame.data};
            EXPECT_THROW(protocol::read_header(previous_frame, record), std::invalid_argument);
        }
        for (std::size_t length = 0; length < protocol::header_bytes; ++length)
        {
            wire::Reader truncated{std::span<const char>(current.data).first(length)};
            EXPECT_THROW(protocol::read_header(truncated, record), std::invalid_argument);
        }
        for (std::size_t i = 0; i < protocol::header_bytes; ++i)
            if (i != offsetof(protocol::Header, family) && i != offsetof(protocol::Header, version) &&
                i != offsetof(protocol::Header, record))
            {
                EXPECT_EQ(current.data[i], 0);
                auto padding = current.data;
                padding[i] = 1;
                wire::Reader nonzero{padding};
                EXPECT_THROW(protocol::read_header(nonzero, record), std::invalid_argument);
            }
        auto changed = current.data;
        changed[offsetof(protocol::Header, family)] = char(255);
        wire::Reader unknown_family{changed};
        EXPECT_THROW(protocol::read_header(unknown_family, record), std::invalid_argument);
        changed = current.data;
        changed[offsetof(protocol::Header, record)] = char(255);
        wire::Reader unknown_record{changed};
        EXPECT_THROW(protocol::read_header(unknown_record, record), std::invalid_argument);
        changed = current.data;
        changed[offsetof(protocol::Header, version)] = 2; // Immediate preceding stamp, including empty records.
        wire::Reader previous{changed};
        EXPECT_THROW(protocol::read_header(previous, record), std::invalid_argument);
        changed[offsetof(protocol::Header, version)] = char(protocol::version + 1);
        wire::Reader future{changed};
        EXPECT_THROW(protocol::read_header(future, record), std::invalid_argument);
        wire::Reader foreign{current.data};
        const auto other =
            record == protocol::Record::NodeUnion ? protocol::Record::NodePublication : protocol::Record::NodeUnion;
        EXPECT_THROW(protocol::read_header(foreign, other), std::invalid_argument);
    }
}
