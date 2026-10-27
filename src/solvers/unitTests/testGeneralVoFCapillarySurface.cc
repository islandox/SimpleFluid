/** @file testGeneralVoFCapillarySurface.cc
 * @brief Independent geometry, component-volume, topology and collective capillary contracts.
 */
#include "FVM/details/VoFFailure.hh"
#include "general_vof_solver_test_helpers.hh"
#include "utils/CompensatedSum.hh"
#include <functional>
#include <optional>
#include <tuple>
namespace
{
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);
using Model = VoF::GeneralVoFCapillaryModel;
using Kind = VoF::GeneralVoFSurfaceKind;
using Reason = VoF::GeneralVoFSurfaceResolutionReason;
using Point = PLIC::GeometryPoint;
long double sum_wide(const Teuchos::Comm<int>& comm, long double value)
{
    // This Trilinos build serializes double, but not long double. Two binary64
    // pieces preserve the local wide accumulator; summing the bounded scalar
    // payload in rank order also preserves its low part across empty owners.
    const double head = static_cast<double>(value);
    const double local[2]{head, static_cast<double>(value - static_cast<long double>(head))};
    std::vector<double> gathered(2 * comm.getSize());
    Teuchos::gatherAll(comm, 2, local, static_cast<int>(gathered.size()), gathered.data());
    SimpleFluid::detail::CompensatedSum<> total;
    for (const auto piece : gathered)
        total += static_cast<long double>(piece);
    return total.value();
}
struct SourceSnapshot
{
    std::uint64_t control;
    std::shared_ptr<const void> identity;
    std::vector<double> gas, carrier;
    std::vector<VoF::GeneralVoFPhaseGeometry> declarations;
    std::vector<std::array<double, 2>> history, roundoff;
    explicit SourceSnapshot(const VoF::GeneralVoF& source)
        : control(source.accepted_control()), identity(source.accepted_identity()), gas(source.gas_volumes()),
          carrier(source.carrier_volumes()), declarations(source.phase_geometry_declarations())
    {
        for (const auto& p : source.accumulated_phase_errors())
            history.push_back({p.gas, p.carrier});
        for (const auto& p : source.phase_roundoff_bounds())
            roundoff.push_back({p.gas, p.carrier});
    }
    void check(const VoF::GeneralVoF& source) const
    {
        EXPECT_EQ(source.accepted_control(), control);
        EXPECT_EQ(source.accepted_identity(), identity);
        EXPECT_EQ(source.gas_volumes(), gas);
        EXPECT_EQ(source.carrier_volumes(), carrier);
        EXPECT_EQ(source.phase_geometry_declarations(), declarations);
        const SourceSnapshot after(source);
        EXPECT_EQ(after.history, history);
        EXPECT_EQ(after.roundoff, roundoff);
    }
};
std::shared_ptr<const VoF::GeneralVoF> binary_box(std::array<unsigned, 3> counts,
    const std::function<bool(Point)>& is_gas, bool traces = false,
    const std::function<std::optional<double>(Point)>& mixed_fraction = {}, unsigned support_rings = 2)
{
    const auto comm = Tpetra::getDefaultComm();
    std::vector<std::uint64_t> owned;
    const auto owners = std::min(2, comm->getSize());
    for (std::uint64_t gid = 0; gid < std::uint64_t(counts[0]) * counts[1] * counts[2]; ++gid)
        if (int(gid % owners) == comm->getRank())
            owned.push_back(gid);
    const AMR::AdaptiveBytes domain{'c', 'a', 'p', 'i', 'l', 'l', 'a', 'r', 'y', 'b', 'o', 'x'};
    auto bootstrap = [&]
    {
        if (std::all_of(counts.begin(), counts.end(), [](auto n) { return n % 2 == 0; }))
        {
            AMR::AMRGrid grid;
            // Small root blocks preserve the same tensor mesh without a large
            // single-block bootstrap/family payload.
            grid.root_blocks = {counts[0] / 2, counts[1] / 2, counts[2] / 2};
            grid.block_cells = {2, 2, 2};
            grid.extent = {double(counts[0]), double(counts[1]), double(counts[2])};
            return AMR::CartesianAMR(grid).detach_owned(owned, domain);
        }
        // Cartesian blocks require even counts; an odd, centered box uses the
        // same conforming tensor cells through the extruded native provider.
        AMR::ExtrudedAMRGrid grid;
        const auto nx = counts[0], ny = counts[1], nz = counts[2];
        for (unsigned y = 0; y <= ny; ++y)
            for (unsigned x = 0; x <= nx; ++x)
                grid.xy_nodes.push_back({double(x), double(y), 0});
        for (unsigned y = 0; y < ny; ++y)
            for (unsigned x = 0; x < nx; ++x)
            {
                const auto a = (nx + 1) * y + x;
                grid.base_cells.push_back({a, a + 1, a + nx + 2, a + nx + 1});
            }
        for (unsigned x = 0; x < nx; ++x)
        {
            grid.boundaries.push_back({x, x + 1, "wall"});
            grid.boundaries.push_back({ny * (nx + 1) + x, ny * (nx + 1) + x + 1, "wall"});
        }
        for (unsigned y = 0; y < ny; ++y)
        {
            grid.boundaries.push_back({(nx + 1) * y, (nx + 1) * (y + 1), "wall"});
            grid.boundaries.push_back({(nx + 1) * y + nx, (nx + 1) * (y + 1) + nx, "wall"});
        }
        for (unsigned z = 0; z <= nz; ++z)
            grid.z_edges.push_back(double(z));
        return AMR::ExtrudedAMR(grid).detach_owned(owned, domain);
    }();
    auto hierarchy = std::make_shared<const AMR::DistributedAdaptiveHierarchy>(
        std::move(bootstrap.owned), std::move(bootstrap.families), bootstrap.kernel, comm);
    std::vector<VoF::GeneralVoF::Key> order;
    std::vector<double> gas, carrier;
    std::vector<VoF::GeneralVoFPhaseGeometry> declarations;
    for (const auto& [key, leaf] : hierarchy->owned())
    {
        const auto shell = hierarchy->geometry(leaf);
        const auto volume = shell->volume().as_double();
        const auto center = shell->centroid();
        const bool g = is_gas(center);
        const auto fraction = mixed_fraction ? mixed_fraction(center) : std::optional<double>{};
        order.push_back(key);
        gas.push_back(fraction ? *fraction * volume : g ? volume : traces ? 1e-25 : 0);
        carrier.push_back(fraction ? (1 - *fraction) * volume : g ? traces ? 1e-25 : 0 : volume);
        declarations.push_back(fraction ? VoF::GeneralVoFPhaseGeometry::Infer
                               : g      ? VoF::GeneralVoFPhaseGeometry::PureGas
                                        : VoF::GeneralVoFPhaseGeometry::PureCarrier);
    }
    VoF::GeneralVoFOptions options;
    options.support_rings = support_rings;
    return std::make_shared<const VoF::GeneralVoF>(hierarchy, order, gas, carrier, options, declarations);
}
bool cube(const Point& p, double lower, double upper)
{
    return p[0] > lower && p[0] < upper && p[1] > lower && p[1] < upper && p[2] > lower && p[2] < upper;
}
VoF::GeneralVoFSurfaceOptions filtered(double width = 2)
{
    VoF::GeneralVoFSurfaceOptions options;
    options.model = Model::FilteredBinary;
    options.filter_width = width;
    return options;
}
void check_owners(const VoF::GeneralVoF& source)
{
    const auto comm = source.hierarchy()->communicator();
    int local[2]{!source.owned_order().empty(), source.owned_order().empty()}, global[2]{};
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 2, local, global);
    EXPECT_EQ(global[0], std::min(2, comm->getSize()));
    EXPECT_EQ(global[1], comm->getSize() - std::min(2, comm->getSize()));
}
void check_components(
    const VoF::GeneralVoF& source, const VoF::GeneralVoFSurfaceCatalog& catalog, std::vector<long double> expected)
{
    const auto comm = source.hierarchy()->communicator();
    EXPECT_EQ(catalog.source_control, source.accepted_control());
    EXPECT_EQ(catalog.model, Model::FilteredBinary);
    std::set<VoF::GeneralVoFSurfaceKey> keys;
    std::map<std::uint64_t, long double> local_integral;
    unsigned local_patches = 0;
    for (const auto& patch : catalog.patches)
    {
        EXPECT_TRUE(keys.insert(patch.key).second);
        EXPECT_EQ(patch.key.kind, Kind::FilteredCell);
        EXPECT_GT(patch.area, 0);
        EXPECT_TRUE(std::isfinite(patch.area));
        EXPECT_GE(patch.vertices.size(), 3u);
        EXPECT_GT(patch.cell_scale, 0);
        EXPECT_EQ(catalog.adjacency.count(patch.key), 1u);
        if (patch.owner == comm->getRank())
        {
            ++local_patches;
            // Divergence theorem on physical polygon centroids and normals is
            // independent of the tetrahedral volume reported by the extractor.
            for (unsigned axis = 0; axis < 3; ++axis)
                local_integral[patch.component] +=
                    static_cast<long double>(patch.area) * patch.centroid[axis] * patch.normal[axis] / 3;
        }
    }
    unsigned patches = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local_patches, &patches);
    EXPECT_GT(patches, 0u);
    std::sort(expected.begin(), expected.end());
    for (const auto& component : catalog.components)
    {
        EXPECT_TRUE(component.closed);
        EXPECT_GT(component.filter_width_min, 0);
        EXPECT_GE(component.filter_width_max, component.filter_width_min);
        EXPECT_GT(component.isovalue, 0);
        EXPECT_LT(component.isovalue, 1);
        const auto tolerance = source.options().reconstruction.volume_tolerance * component.original_volume;
        EXPECT_LE(std::abs(component.smoothed_volume - component.original_volume), tolerance);
        EXPECT_LE(std::abs(component.extracted_volume - component.original_volume), tolerance);
        EXPECT_TRUE(std::any_of(expected.begin(), expected.end(),
            [&](auto v) { return std::abs(v - component.original_volume) <= tolerance; }));
    }
    // On the two parity owners the retained halo covers the complete box. Empty
    // owners retain no components and still participate in these reductions.
    unsigned local_components = 0;
    long double local_volume = 0, local_geometric_volume = 0;
    for (const auto& component : catalog.components)
        if (component.owner == comm->getRank())
        {
            ++local_components;
            local_volume += component.original_volume;
        }
    for (const auto& [id, volume] : local_integral)
        local_geometric_volume += volume;
    unsigned components = 0;
    long double volume = 0, geometric_volume = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local_components, &components);
    volume = sum_wide(*comm, local_volume);
    geometric_volume = sum_wide(*comm, local_geometric_volume);
    EXPECT_EQ(components, expected.size());
    long double target = 0;
    for (const auto v : expected)
        target += v;
    const auto tolerance = source.options().reconstruction.volume_tolerance * target;
    EXPECT_LE(std::abs(volume - target), tolerance);
    EXPECT_LE(std::abs(geometric_volume - target), tolerance);
    std::uint64_t previous = 0;
    bool first = true;
    for (std::size_t ordinal = 0; ordinal < expected.size(); ++ordinal)
    {
        auto local_id = std::numeric_limits<std::uint64_t>::max();
        for (const auto& component : catalog.components)
            if ((first || component.identity > previous) && component.identity < local_id)
                local_id = component.identity;
        std::uint64_t id = 0;
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_MIN, 1, &local_id, &id);
        EXPECT_NE(id, std::numeric_limits<std::uint64_t>::max());
        long double local_original = 0, original = 0, integrated = 0;
        for (const auto& component : catalog.components)
            if (component.identity == id && component.owner == comm->getRank())
                local_original = component.original_volume;
        const long double local_integrated = local_integral[id];
        original = sum_wide(*comm, local_original);
        integrated = sum_wide(*comm, local_integrated);
        EXPECT_LE(std::abs(integrated - original), source.options().reconstruction.volume_tolerance * original);
        previous = id;
        first = false;
    }
}
void expect_resolution_rejection(const VoF::GeneralVoF& source, Reason reason, double width = 2)
{
    const auto comm = source.hierarchy()->communicator();
    const SourceSnapshot snapshot(source);
    int rejected = 0, structured = 0;
    try
    {
        (void) source.capillary_surface(filtered(width));
    }
    catch (const std::exception& error)
    {
        rejected = 1;
        const auto* failure = dynamic_cast<const SimpleFluid::detail::vof_failure::Failure*>(&error);
        if (failure && failure->local_cause())
            try
            {
                std::rethrow_exception(failure->local_cause());
            }
            catch (const VoF::GeneralVoFSurfaceResolutionError& cause)
            {
                EXPECT_EQ(cause.reason(), reason) << cause.what();
                structured = 1;
            }
            catch (const std::exception& cause)
            {
                ADD_FAILURE() << "Expected a surface resolution request, got: " << cause.what();
            }
        else if (const auto* cause = dynamic_cast<const VoF::GeneralVoFSurfaceResolutionError*>(&error))
        {
            EXPECT_EQ(cause->reason(), reason) << cause->what();
            structured = 1;
        }
    }
    int global_rejected = 0, global_structured = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &rejected, &global_rejected);
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &structured, &global_structured);
    EXPECT_EQ(global_rejected, comm->getSize());
    EXPECT_GT(global_structured, 0);
    snapshot.check(source);
}
TEST(GeneralVoFCapillarySurface, DefaultFilterPreservesCubeVolumeAndRejectsUnderresolvedCurvature)
{
    const auto source = binary_box({6, 6, 6}, [](Point p) { return cube(p, 2, 4); }, true);
    check_owners(*source);
    const SourceSnapshot snapshot(*source);
    const auto catalog = source->capillary_surface(filtered());
    check_components(*source, catalog, {8});
    for (const auto& component : catalog.components)
    {
        EXPECT_DOUBLE_EQ(component.filter_width_min, 2);
        EXPECT_DOUBLE_EQ(component.filter_width_max, 2);
    }
    snapshot.check(*source);
    auto material = inviscid();
    material.surface_tension = .07;
    GeneralVoFSolverOptions options;
    options.capillary = filtered();
    try
    {
        (void) GeneralVoFSolver(source, material, options);
        FAIL() << "The two-cell cube must not bypass the unchanged curvature residual gate.";
    }
    catch (const detail::vof_failure::Failure& failure)
    {
        EXPECT_EQ(failure.record().category, detail::vof_failure::Category::Resolution);
        EXPECT_TRUE(failure.record().has(detail::vof_failure::Field::Resolution));
        EXPECT_EQ(failure.record().resolution_reason, int(Reason::DeficientFit));
    }
    snapshot.check(*source);
}
TEST(GeneralVoFCapillarySurface, ResolvedFilteredBinaryDropHasInspectibleCurvature)
{
    const auto source = binary_box(
        {6, 6, 6},
        [](Point p)
        {
            const auto x = p[0] - 3, y = p[1] - 3, z = p[2] - 3;
            return x * x + y * y + z * z < 5.5;
        },
        true);
    const SourceSnapshot snapshot(*source);
    const auto catalog = source->capillary_surface(filtered());
    check_components(*source, catalog, {56});
    auto material = inviscid();
    material.surface_tension = .07;
    GeneralVoFSolverOptions options;
    options.capillary = filtered();
    GeneralVoFSolver solver(source, material, options);
    long double local_area = 0, local_weighted_curvature = 0;
    std::map<VoF::GeneralVoFSurfaceKey, double> area;
    for (const auto& patch : catalog.patches)
        area.emplace(patch.key, patch.area);
    for (const auto& fit : solver.capillary_diagnostics().fits)
    {
        EXPECT_TRUE(std::isfinite(fit.curvature));
        EXPECT_LE(fit.normalized_residual, .15);
        EXPECT_LE(fit.condition, 1e8);
        local_area += area.at(fit.key);
        local_weighted_curvature += area.at(fit.key) * fit.curvature;
    }
    const auto comm = source->hierarchy()->communicator();
    const auto total_area = sum_wide(*comm, local_area);
    const auto mean = sum_wide(*comm, local_weighted_curvature) / total_area;
    EXPECT_GT(total_area, 0);
    EXPECT_GT(mean, 0);
    RecordProperty("mean_curvature", double(mean));
    RecordProperty("volume_equivalent_sphere_curvature", 2 / std::cbrt(3 * 56 / (4 * std::acos(-1.))));
    snapshot.check(*source);
}

TEST(GeneralVoFCapillarySurface, FilteringPreservesEachOfTwoDistinctEnclosedComponentVolumes)
{
    const auto source = binary_box({14, 6, 6},
        [](Point p)
        {
            return (p[0] > 3 && p[0] < 5 && p[1] > 3 && p[1] < 4 && p[2] > 3 && p[2] < 4) ||
                   (p[0] > 9 && p[0] < 11 && p[1] > 2 && p[1] < 4 && p[2] > 2 && p[2] < 4);
        });
    const SourceSnapshot snapshot(*source);
    check_components(*source, source->capillary_surface(filtered(.25)), {2, 8});
    snapshot.check(*source);
}
TEST(GeneralVoFCapillarySurface, FilteredEnclosedComponentCoexistsWithUnrelatedMixedPlane)
{
    const auto source = binary_box(
        {12, 6, 6}, [](Point p) { return cube(p, 2, 4) || p[0] > 10; }, false,
        [](Point p) -> std::optional<double>
        {
            if (p[0] > 9 && p[0] < 10)
                return .5;
            return {};
        });
    const SourceSnapshot snapshot(*source);
    const auto catalog = source->capillary_surface(filtered(.25));
    unsigned local[3]{}, global[3]{};
    for (const auto& patch : catalog.patches)
        if (patch.owner == source->hierarchy()->communicator()->getRank())
        {
            ++local[patch.key.kind == Kind::MixedCell ? 0 : patch.key.kind == Kind::FilteredCell ? 1 : 2];
            if (patch.key.kind == Kind::MixedCell)
                EXPECT_NEAR(patch.centroid[0], 9.5, source->options().reconstruction.volume_tolerance);
        }
    Teuchos::reduceAll(*source->hierarchy()->communicator(), Teuchos::REDUCE_SUM, 3, local, global);
    EXPECT_EQ(global[0], 36u);
    EXPECT_GT(global[1], 0u);
    EXPECT_EQ(global[2], 0u);
    for (const auto& component : catalog.components)
        EXPECT_EQ(component.original_volume, 8);
    snapshot.check(*source);
}
TEST(GeneralVoFCapillarySurface, UnequalCellVolumesRetainVolumeWeightedFilteredConservation)
{
    constexpr unsigned n = 4;
    const auto coordinate = [](unsigned x) { return x == 2 ? 1.75 : double(x); };
    AMR::ExtrudedAMRGrid grid;
    for (unsigned y = 0; y <= n; ++y)
        for (unsigned x = 0; x <= n; ++x)
            grid.xy_nodes.push_back({coordinate(x), double(y), 0});
    for (unsigned y = 0; y < n; ++y)
        for (unsigned x = 0; x < n; ++x)
        {
            const auto a = (n + 1) * y + x;
            grid.base_cells.push_back({a, a + 1, a + n + 2, a + n + 1});
        }
    for (unsigned x = 0; x < n; ++x)
    {
        grid.boundaries.push_back({x, x + 1, "wall"});
        grid.boundaries.push_back({n * (n + 1) + x, n * (n + 1) + x + 1, "wall"});
    }
    for (unsigned y = 0; y < n; ++y)
    {
        grid.boundaries.push_back({(n + 1) * y, (n + 1) * (y + 1), "wall"});
        grid.boundaries.push_back({(n + 1) * y + n, (n + 1) * (y + 1) + n, "wall"});
    }
    for (unsigned z = 0; z <= n; ++z)
        grid.z_edges.push_back(double(z));
    const auto comm = Tpetra::getDefaultComm();
    const auto provider = std::make_shared<AMR::ExtrudedAMR>(grid);
    std::vector<std::uint64_t> owned;
    for (std::uint64_t gid = 0; gid < std::uint64_t(n) * n * n; ++gid)
        if (int(gid % std::min(2, comm->getSize())) == comm->getRank())
            owned.push_back(gid);
    auto bootstrap =
        provider->detach_owned(owned, {'u', 'n', 'e', 'q', 'u', 'a', 'l', 'c', 'a', 'p', 'i', 'l', 'l', 'a', 'r', 'y'});
    auto hierarchy = std::make_shared<const AMR::DistributedAdaptiveHierarchy>(
        std::move(bootstrap.owned), std::move(bootstrap.families), bootstrap.kernel, comm);
    std::vector<VoF::GeneralVoF::Key> order;
    std::vector<double> gas, carrier;
    std::vector<VoF::GeneralVoFPhaseGeometry> declarations;
    long double local_geometric_volume = 0, geometric_volume = 0;
    double local_minimum = std::numeric_limits<double>::infinity(), local_maximum = 0;
    for (const auto& [key, leaf] : hierarchy->owned())
    {
        const auto geometry = hierarchy->geometry(leaf);
        const double volume = geometry->volume().as_double();
        const bool is_gas = cube(geometry->centroid(), 1, 3);
        order.push_back(key);
        gas.push_back(is_gas ? volume : 1e-25);
        carrier.push_back(is_gas ? 1e-25 : volume);
        declarations.push_back(
            is_gas ? VoF::GeneralVoFPhaseGeometry::PureGas : VoF::GeneralVoFPhaseGeometry::PureCarrier);
        if (is_gas)
        {
            local_geometric_volume += volume;
            local_minimum = std::min(local_minimum, volume);
            local_maximum = std::max(local_maximum, volume);
        }
    }
    double minimum = 0, maximum = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MIN, 1, &local_minimum, &minimum);
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &local_maximum, &maximum);
    geometric_volume = sum_wide(*comm, local_geometric_volume);
    EXPECT_DOUBLE_EQ(minimum, .75);
    EXPECT_DOUBLE_EQ(maximum, 1.25);
    EXPECT_LT(minimum, maximum);
    // Independent box widths are (.75 + 1.25) by 2 by 2. The target uses
    // geometric gas declarations and excludes the independently retained traces.
    EXPECT_EQ(geometric_volume, 8);
    const auto source =
        std::make_shared<const VoF::GeneralVoF>(hierarchy, order, gas, carrier, VoF::GeneralVoFOptions{}, declarations);
    const SourceSnapshot snapshot(*source);
    check_components(*source, source->capillary_surface(filtered(.25)), {geometric_volume});
    snapshot.check(*source);
}
TEST(GeneralVoFCapillarySurface, SingleCellPlateauRequestsResolutionWithoutChangingInventories)
{
    const auto source = binary_box({4, 4, 4}, [](Point p) { return cube(p, 1, 2); }, true);
    // At width .25 all eight source-cell nodes are 15/128. A positive-volume
    // plateau makes a regular strict-superlevel surface at volume 1 unresolved.
    expect_resolution_rejection(*source, Reason::VolumeConstraint, .25);
}

TEST(GeneralVoFCapillarySurface, FilterRejectsEdgeAndVertexContactsAsAmbiguousTopology)
{
    for (const bool vertex : {false, true})
    {
        SCOPED_TRACE(vertex ? "vertex contact" : "edge contact");
        const auto source = binary_box({4, 4, 4},
            [vertex](Point p)
            {
                return cube(p, 1, 2) || (p[0] > 2 && p[0] < 3 && p[1] > 2 && p[1] < 3 &&
                                            (vertex ? p[2] > 2 && p[2] < 3 : p[2] > 1 && p[2] < 2));
            });
        expect_resolution_rejection(*source, Reason::AmbiguousTopology);
    }
}
TEST(GeneralVoFCapillarySurface, FilterRejectsMergingCavitiesAndIncompleteSupport)
{
    {
        const auto source = binary_box({8, 8, 8}, [](Point p)
            { return cube(p, 3, 4) || (p[0] > 5 && p[0] < 6 && p[1] > 3 && p[1] < 4 && p[2] > 3 && p[2] < 4); });
        expect_resolution_rejection(*source, Reason::ComponentMerging);
    }
    {
        const auto source = binary_box({6, 6, 6}, [](Point p) { return cube(p, 1, 4) && !cube(p, 2, 3); });
        expect_resolution_rejection(*source, Reason::UnsupportedCavity, .25);
    }
    {
        const auto source = binary_box({4, 4, 4}, [](Point p) { return cube(p, 0, 2); });
        expect_resolution_rejection(*source, Reason::IncompleteSupport);
    }
}
TEST(GeneralVoFCapillarySurface, MixedAndAlignedPlanarSectionsShareOneImmutableSourceCatalog)
{
    AMR::ExtrudedAMRGrid grid;
    for (unsigned y = 0; y <= 3; ++y)
        for (unsigned x = 0; x <= 3; ++x)
            grid.xy_nodes.push_back({x + (y >= 2 ? .25 : 0), double(y), 0});
    for (unsigned y = 0; y < 3; ++y)
        for (unsigned x = 0; x < 3; ++x)
        {
            const auto a = 4 * y + x;
            grid.base_cells.push_back({a, a + 1, a + 5, a + 4});
        }
    for (unsigned x = 0; x < 3; ++x)
    {
        grid.boundaries.push_back({x, x + 1, "wall"});
        grid.boundaries.push_back({12 + x, 13 + x, "wall"});
    }
    for (unsigned y = 0; y < 3; ++y)
    {
        grid.boundaries.push_back({4 * y, 4 * (y + 1), "wall"});
        grid.boundaries.push_back({4 * y + 3, 4 * (y + 1) + 3, "wall"});
    }
    grid.z_edges = {0, 1, 2, 3};
    const auto comm = Tpetra::getDefaultComm();
    const auto provider = std::make_shared<AMR::ExtrudedAMR>(grid);
    std::vector<std::uint64_t> owned;
    for (std::uint64_t gid = 0; gid < 27; ++gid)
        if (int(gid % std::min(2, comm->getSize())) == comm->getRank())
            owned.push_back(gid);
    auto bootstrap =
        provider->detach_owned(owned, {'p', 'l', 'a', 'n', 'a', 'r', 't', 'r', 'a', 'n', 's', 'i', 't', 'i', 'o', 'n'});
    auto hierarchy = std::make_shared<const AMR::DistributedAdaptiveHierarchy>(
        std::move(bootstrap.owned), std::move(bootstrap.families), bootstrap.kernel, comm);
    std::vector<VoF::GeneralVoF::Key> order;
    std::vector<double> gas, carrier;
    for (const auto& [key, leaf] : hierarchy->owned())
    {
        const auto center = hierarchy->geometry(leaf)->centroid();
        const unsigned strip = static_cast<unsigned>(center[1]);
        // Integrate the width between x=1 and the affine left wall directly.
        // Every cell has unit volume; these values do not use a PLIC clip.
        const double shift = strip == 0 ? 0 : strip == 1 ? .125 : .25;
        const bool first_column = center[0] - shift < 1;
        const double g = first_column ? 1 - shift : 0;
        order.push_back(key);
        gas.push_back(g);
        carrier.push_back(1 - g);
    }
    const auto source = std::make_shared<const VoF::GeneralVoF>(hierarchy, order, gas, carrier);
    const SourceSnapshot snapshot(*source);
    const auto planes = source->reconstruction_patches();
    const auto catalog = source->capillary_surface();
    EXPECT_EQ(catalog.source_control, source->accepted_control());
    std::set<VoF::GeneralVoFSurfaceKey> keys;
    unsigned local[3]{}, global[3]{};
    for (const auto& patch : catalog.patches)
    {
        EXPECT_TRUE(keys.insert(patch.key).second);
        if (patch.owner != comm->getRank())
            continue;
        ++local[patch.key.kind == Kind::MixedCell ? 0 : 1];
        for (const auto& neighbor : catalog.adjacency.at(patch.key))
            if (neighbor.kind != patch.key.kind)
                ++local[2];
    }
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 3, local, global);
    EXPECT_EQ(global[0], 6u);
    EXPECT_EQ(global[1], 3u);
    EXPECT_GT(global[2], 0u);
    const auto after = source->reconstruction_patches();
    EXPECT_EQ(after.size(), planes.size());
    for (std::size_t i = 0; i < std::min(after.size(), planes.size()); ++i)
    {
        EXPECT_EQ(after[i].cell, planes[i].cell);
        EXPECT_EQ(after[i].interface.gas.normal, planes[i].interface.gas.normal);
        EXPECT_EQ(after[i].interface.gas.anchor, planes[i].interface.gas.anchor);
        EXPECT_EQ(after[i].interface.gas.offset, planes[i].interface.gas.offset);
    }
    snapshot.check(*source);
    auto material = inviscid();
    material.surface_tension = .07;
    GeneralVoFSolver solver(source, material);
    for (const auto curvature : solver.curvature())
        EXPECT_EQ(curvature, 0);
    const auto initial = solver.diagnostics();
    const auto advanced = solver.step(.01);
    EXPECT_EQ(advanced.maximum_velocity, 0);
    EXPECT_EQ(advanced.gas_mass, initial.gas_mass);
    EXPECT_EQ(advanced.liquid_mass, initial.liquid_mass);
    snapshot.check(*source);
}
TEST(GeneralVoFCapillarySurface, RejectedInvalidInputsRetainAcceptedCapillaryDiagnostics)
{
    const auto source = phases(1);
    auto material = inviscid();
    material.surface_tension = .07;
    GeneralVoFSolver solver(source, material);
    const auto phase_state = solver.phases();
    const auto time = solver.time();
    const auto curvature = solver.curvature();
    const auto velocity = solver.velocity();
    const auto face_velocity = solver.face_velocity();
    const auto pressure = solver.pressure();
    const auto preparation = solver.prepared_transport();
    const auto capillary_image = [&]
    {
        using Fit = std::tuple<VoF::GeneralVoFSurfaceKey, double, double, double, std::size_t>;
        using Component =
            std::tuple<std::uint64_t, int, long double, long double, long double, double, double, double, bool>;
        std::vector<Fit> fits;
        std::vector<Component> components;
        const auto& diagnostics = solver.capillary_diagnostics();
        for (const auto& fit : diagnostics.fits)
            fits.emplace_back(fit.key, fit.curvature, fit.condition, fit.normalized_residual, fit.samples);
        for (const auto& component : diagnostics.components)
            components.emplace_back(component.identity, component.owner, component.original_volume,
                component.smoothed_volume, component.extracted_volume, component.filter_width_min,
                component.filter_width_max, component.isovalue, component.closed);
        return std::tuple{diagnostics.model, fits, components};
    };
    const auto image = capillary_image();
    const auto check = [&]
    {
        EXPECT_EQ(solver.phases(), phase_state);
        EXPECT_EQ(solver.time(), time);
        EXPECT_EQ(solver.curvature(), curvature);
        EXPECT_EQ(solver.velocity(), velocity);
        EXPECT_EQ(solver.face_velocity(), face_velocity);
        EXPECT_EQ(solver.pressure(), pressure);
        EXPECT_EQ(solver.prepared_transport(), preparation);
        EXPECT_EQ(capillary_image(), image);
    };
    // These are input failures; they establish diagnostic rollback without
    // claiming that a candidate geometry failed capillary resolution.
    EXPECT_THROW((void) solver.step(-.01), std::exception);
    check();
    std::vector<VoF::Vector> invalid(source->owned_order().size());
    if (!invalid.empty())
        invalid.front()[0] = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(solver.initialize(invalid), std::exception);
    check();
}
TEST(GeneralVoFCapillarySurface, SpatialPartitionIncompleteHaloRejectsWithResolutionRecordOnEveryRank)
{
    const auto comm = Tpetra::getDefaultComm();
    if (comm->getSize() < 2)
        GTEST_SKIP() << "Requires two spatial owners; four ranks also exercise two empty owners.";
    AMR::AMRGrid grid;
    grid.block_cells = {12, 4, 4};
    grid.extent = {12, 4, 4};
    const auto provider = std::make_shared<AMR::CartesianAMR>(grid);
    std::vector<std::uint64_t> owned;
    for (std::uint64_t gid = 0; gid < 12 * 4 * 4; ++gid)
        if (int((gid % 12) / 6) == comm->getRank())
            owned.push_back(gid);
    auto bootstrap =
        provider->detach_owned(owned, {'s', 'p', 'a', 't', 'i', 'a', 'l', 'c', 'a', 'p', 'i', 'l', 'l', 'a', 'r', 'y'});
    auto hierarchy = std::make_shared<const AMR::DistributedAdaptiveHierarchy>(
        std::move(bootstrap.owned), std::move(bootstrap.families), bootstrap.kernel, comm);
    std::vector<VoF::GeneralVoF::Key> order;
    std::vector<double> gas, carrier;
    for (const auto& [key, leaf] : hierarchy->owned())
    {
        const auto geometry = hierarchy->geometry(leaf);
        const auto center = geometry->centroid();
        const bool is_gas =
            center[0] > 5 && center[0] < 7 && center[1] > 1 && center[1] < 3 && center[2] > 1 && center[2] < 3;
        const auto volume = geometry->volume().as_double();
        order.push_back(key);
        gas.push_back(is_gas ? volume : 0);
        carrier.push_back(is_gas ? 0 : volume);
    }
    VoF::GeneralVoFOptions options;
    options.support_rings = 1;
    const auto source = std::make_shared<const VoF::GeneralVoF>(hierarchy, order, gas, carrier, options);
    check_owners(*source);
    const auto resident = source->support_cells();
    if (comm->getRank() < 2)
    {
        EXPECT_LT(resident.size(), 12u * 4u * 4u);
        // Width-two diffusion from the x=5/6 core needs these remote columns
        // before interpolation. Spatial shards deliberately omit one of them;
        // physical walls in y/z do not explain this missing interior geometry.
        const double missing_x = comm->getRank() == 0 ? 8.5 : 3.5;
        EXPECT_TRUE(std::none_of(
            resident.begin(), resident.end(), [&](const auto& cell) { return cell.centroid[0] == missing_x; }));
    }
    else
        EXPECT_TRUE(resident.empty());
    const SourceSnapshot snapshot(*source);
    int rejected = 0;
    try
    {
        (void) source->capillary_surface(filtered());
    }
    catch (const std::exception& error)
    {
        rejected = 1;
        const auto* failure = dynamic_cast<const SimpleFluid::detail::vof_failure::Failure*>(&error);
        EXPECT_NE(failure, nullptr) << error.what();
        if (failure)
        {
            const auto& record = failure->record();
            EXPECT_EQ(record.category, SimpleFluid::detail::vof_failure::Category::Resolution);
            EXPECT_TRUE(record.has(SimpleFluid::detail::vof_failure::Field::Resolution));
            EXPECT_EQ(record.resolution_reason, int(Reason::IncompleteSupport));
        }
    }
    int rejected_ranks = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &rejected, &rejected_ranks);
    EXPECT_EQ(rejected_ranks, comm->getSize());
    snapshot.check(*source);
}
TEST(GeneralVoFCapillarySurface, MpiCapillaryFailureDuringRepartitionRetainsCompleteAcceptedPublication)
{
    const auto comm = Tpetra::getDefaultComm();
    if (comm->getSize() < 2)
        GTEST_SKIP() << "Requires parity-to-spatial migration; four ranks also exercise two empty owners.";
    const auto source = binary_box(
        {10, 10, 10},
        [](Point p)
        {
            double radius_squared = 0;
            for (const auto coordinate : p)
                radius_squared += (coordinate - 5) * (coordinate - 5);
            return radius_squared < 5.5;
        },
        false, {}, 2);
    check_owners(*source);
    unsigned local_gas_cells = 0, gas_cells = 0;
    for (const auto volume : source->gas_volumes())
        local_gas_cells += volume > 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local_gas_cells, &gas_cells);
    EXPECT_EQ(gas_cells, 56u);
    const SourceSnapshot source_snapshot(*source);
    auto material = inviscid();
    material.surface_tension = .07;
    GeneralVoFSolverOptions options;
    options.capillary = filtered();
    // Initialization must satisfy the existing fit gates. A failed initial
    // fit is a test failure, rather than permission to relax those gates.
    GeneralVoFSolver solver(source, material, options);
    const auto phase_state = solver.phases();
    const auto mesh = solver.mesh();
    const auto time = solver.time();
    const auto gas_fraction = solver.gas_fraction();
    const auto curvature = solver.curvature();
    const auto velocity = solver.velocity();
    const auto face_velocity = solver.face_velocity();
    const auto pressure = solver.pressure();
    const auto preparation = solver.prepared_transport();
    const auto momentum = solver.momentum_transport_diagnostics();
    const auto capillary_image = [&]
    {
        using Fit = std::tuple<VoF::GeneralVoFSurfaceKey, double, double, double, std::size_t>;
        using Component =
            std::tuple<std::uint64_t, int, long double, long double, long double, double, double, double, bool>;
        std::vector<Fit> fits;
        std::vector<Component> components;
        const auto& diagnostics = solver.capillary_diagnostics();
        for (const auto& fit : diagnostics.fits)
            fits.emplace_back(fit.key, fit.curvature, fit.condition, fit.normalized_residual, fit.samples);
        for (const auto& component : diagnostics.components)
            components.emplace_back(component.identity, component.owner, component.original_volume,
                component.smoothed_volume, component.extracted_volume, component.filter_width_min,
                component.filter_width_max, component.isovalue, component.closed);
        return std::tuple{diagnostics.model, fits, components};
    };
    const auto flow_image = [&]
    {
        const auto& d = solver.diagnostics();
        return std::tuple{d.time, d.gas_volume, d.gas_mass, d.liquid_mass, d.momentum, d.kinetic_energy,
            d.maximum_velocity, d.maximum_divergence, d.pressure_residual, d.pressure_iterations};
    };
    const auto capillary = capillary_image();
    const auto flow = flow_image();
    unsigned local_fits = solver.capillary_diagnostics().fits.size(), fits = 0;
    unsigned local_components = solver.capillary_diagnostics().components.size(), components = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local_fits, &fits);
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local_components, &components);
    EXPECT_GT(fits, 0u);
    EXPECT_EQ(components, 1u);
    std::vector<int> destinations;
    for (const auto& key : source->owned_order())
        destinations.push_back(
            source->hierarchy()->geometry(source->hierarchy()->owned().at(key))->centroid()[0] < 5 ? 0 : 1);
    int rejected = 0;
    try
    {
        // The complete parity stencil becomes two spatial shards. The same
        // two-ring retention cannot supply the remote filtered geometry, so
        // candidate curvature must reject before any accepted publication.
        solver.repartition(destinations);
    }
    catch (const std::exception& error)
    {
        rejected = 1;
        const auto* failure = dynamic_cast<const SimpleFluid::detail::vof_failure::Failure*>(&error);
        EXPECT_NE(failure, nullptr) << error.what();
        if (failure)
        {
            const auto& record = failure->record();
            EXPECT_EQ(record.category, SimpleFluid::detail::vof_failure::Category::Resolution);
            EXPECT_TRUE(record.has(SimpleFluid::detail::vof_failure::Field::Resolution));
            EXPECT_EQ(record.resolution_reason, int(Reason::IncompleteSupport));
        }
    }
    int rejected_ranks = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &rejected, &rejected_ranks);
    EXPECT_EQ(rejected_ranks, comm->getSize());
    EXPECT_EQ(solver.phases(), phase_state);
    EXPECT_EQ(solver.mesh(), mesh);
    EXPECT_EQ(solver.time(), time);
    EXPECT_EQ(solver.gas_fraction(), gas_fraction);
    EXPECT_EQ(solver.curvature(), curvature);
    EXPECT_EQ(solver.velocity(), velocity);
    EXPECT_EQ(solver.face_velocity(), face_velocity);
    EXPECT_EQ(solver.pressure(), pressure);
    EXPECT_EQ(solver.prepared_transport(), preparation);
    EXPECT_EQ(capillary_image(), capillary);
    EXPECT_EQ(flow_image(), flow);
    for (unsigned axis = 0; axis < 3; ++axis)
    {
        const auto& after = solver.momentum_transport_diagnostics()[axis];
        EXPECT_EQ(after.physical_relative_residual, momentum[axis].physical_relative_residual);
        EXPECT_EQ(after.implicit_iterations, momentum[axis].implicit_iterations);
        EXPECT_EQ(after.defect_corrections, momentum[axis].defect_corrections);
    }
    source_snapshot.check(*source);
}
TEST(GeneralVoFCapillarySurface, DivergentRankModelControlsRejectBeforeAnyConditionalCollective)
{
    const auto comm = Tpetra::getDefaultComm();
    if (comm->getSize() < 2)
        GTEST_SKIP() << "Requires different rank controls; four ranks also exercise two empty owners.";
    const auto source = phases(1);
    check_owners(*source);
    const SourceSnapshot snapshot(*source);
    VoF::GeneralVoFSurfaceOptions options;
    if (comm->getRank() == comm->getSize() - 1)
        options.model = Model::FilteredBinary;
    EXPECT_THROW((void) source->capillary_surface(options), std::exception);
    options = {};
    if (comm->getRank() == comm->getSize() - 1)
        options.filter_width = 3;
    EXPECT_THROW((void) source->capillary_surface(options), std::exception);
    snapshot.check(*source);
    const auto foreign = phases(1);
    const auto& selected = comm->getRank() == comm->getSize() - 1 ? *foreign : *source;
    EXPECT_THROW((void) selected.face_interfaces(), std::exception);
    EXPECT_THROW((void) selected.capillary_surface(), std::exception);
    snapshot.check(*source);
    const auto recovered = source->capillary_surface();
    EXPECT_EQ(recovered.source_control, source->accepted_control());
}
} // namespace
