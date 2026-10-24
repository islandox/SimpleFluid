/** @file GeneralVoFInternal.hh
 * @brief Private accepted-state layout and shared transport support declarations.
 */
#pragma once
#include "FVM/GeneralVoF.hh"
#include "FVM/details/ExactLiveStorage.hh"
#include "FVM/details/BoundedPhaseArithmetic.hh"
#include "FVM/details/DistributedAMROverlap.hh"
#include "FVM/details/BoundedVoFTransport.hh"
#include "FVM/details/ExactPhaseAccumulator.hh"
#include "FVM/details/ExactPhaseLabelMap.hh"
#include "FVM/details/ExperimentalPhaseBackend.hh"
#include "FVM/details/PhaseArithmetic.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveWire.hh"
#include "utils/CompensatedSum.hh"
#include <Tpetra_Core.hpp>
#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <tuple>
namespace SimpleFluid::VoF
{
namespace Exact
{
struct SIMPLEFLUID_FVM_LOCAL GeneralVoF
{
    using State = VoF::GeneralVoF;
    using Data = State::Data;
    using Trial = State::Trial;
    using Key = State::Key;
    static Trial plan_transport(const State&, std::span<const GeneralVoFNodeVelocity>,
        std::span<const GeneralVoFFaceVolume>, double);
    static void validate_transfer_target(const State&, const ExactPhaseTransferResult&);
    static ExactPhaseTransferSource export_transfer_source(const State&, bool);
    static std::shared_ptr<const State> adopt_transfer(const State&, const ExactPhaseTransferResult&);
};
} // namespace Exact
namespace general_vof_detail
{
namespace AMR = Meshes::AMR;
namespace wire = AMR::adaptive_detail;
using Key = AMR::AdaptiveCellKey;
using Measure = Meshes::ExactGeometryMeasure;
namespace fp = finite_detail;
struct Cell
{
    Key key;
    std::uint64_t gid = 0, revision = 0;
    int owner = -1;
    Meshes::GeometryPoint centroid{};
    double native_volume = 0;
    std::size_t canonical_face_count = 0;
    long double ordinary_quota_scale = 0;
    PolyhedralPhaseSample phase;
    Meshes::BoundedConvexGeometryPtr finite_geometry;
    PhaseError occupancy;
    PhaseRoundoff roundoff, finite_history;
    std::shared_ptr<const ExperimentalPhaseInventory> experimental_exact;
    GeneralVoFPhaseGeometry declaration = GeneralVoFPhaseGeometry::Infer;
    std::optional<PolyhedralInterface> retained_interface;
    bool inherited = false;
};
} // namespace general_vof_detail
using namespace general_vof_detail;
struct GeneralVoF::Data
{
    struct Wall
    {
        AMR::AdaptiveBytes identity;
        Meshes::ExactPlanarFacePtr geometry;
        Meshes::BoundedPoint finite_normal;
        Meshes::BoundedFaceConstraint finite_constraint;
    };
    struct Topology
    {
        AMR::AdaptiveNativePublication native;
        AMR::AdaptiveNativeDraft draft;
        std::vector<AMR::AdaptiveTile> tiles;
        std::vector<AMR::AdaptiveNode> nodes;
        std::map<std::uint64_t, Meshes::BoundedPoint> finite_nodes;
        std::map<std::uint64_t, Key> by_gid;
        std::map<Key, std::set<Key>> adjacency;
        std::map<std::uint64_t, std::set<Key>> node_fans;
        std::map<std::uint64_t, std::vector<Wall>> node_walls;
        std::vector<Key> order;
        std::map<Key, std::vector<std::size_t>> cell_tiles;
        std::map<std::uint64_t, std::vector<std::size_t>> node_tiles;
    };
    std::shared_ptr<const Topology> topology;
    std::shared_ptr<const Hierarchy> hierarchy;
    AMR::HierarchyObservation observation;
    std::shared_ptr<const Hierarchy> supplied_hierarchy;
    AMR::HierarchyObservation supplied_observation;
    GeneralVoFOptions options;
    std::uint64_t control = 0;
    std::weak_ptr<const void> adopted_source;
    std::uint64_t adopted_source_control = 0, adopted_transaction = 0, adopted_target_control = 0;
    std::map<Key, Cell> cells;
    std::vector<double> gas, carrier;
    std::vector<PhaseVolumes> phases;
    std::vector<PhaseRoundoff> roundoff, phase_history;
    std::vector<PhaseError> occupancy;
    std::shared_ptr<const std::vector<Meshes::ExactPhaseMeasures>> experimental_catalog;
};
namespace general_vof_detail
{
SIMPLEFLUID_FVM_LOCAL PhaseInventoryPolicy inventory_policy_for(ArithmeticMode mode);
SIMPLEFLUID_FVM_LOCAL Meshes::BoundedGeometryLimits geometry_limits(ArithmeticMode mode, std::size_t bytes = 64 * 1024 * 1024);
SIMPLEFLUID_FVM_LOCAL long double phase_allowance(Meshes::BoundedGeometryValue source, double relative);
SIMPLEFLUID_FVM_LOCAL Meshes::ExactPhaseMeasures phase_measures(const Cell& cell);
SIMPLEFLUID_FVM_LOCAL Measure finite_bound(double value);
SIMPLEFLUID_FVM_LOCAL void occupancy(Cell& cell);
SIMPLEFLUID_FVM_LOCAL void validate_cell(const Cell& cell, const GeneralVoFOptions& options);
SIMPLEFLUID_FVM_LOCAL void real(wire::Writer& w, double value);
SIMPLEFLUID_FVM_LOCAL double real(wire::Reader& r);
SIMPLEFLUID_FVM_LOCAL void finite_real(wire::Writer& w, long double value);
SIMPLEFLUID_FVM_LOCAL long double finite_real(wire::Reader& r);
SIMPLEFLUID_FVM_LOCAL void finite_value(wire::Writer& w, Meshes::BoundedGeometryValue v);
SIMPLEFLUID_FVM_LOCAL Meshes::BoundedGeometryValue finite_value(wire::Reader& r, std::optional<ArithmeticMode> expected = {});
SIMPLEFLUID_FVM_LOCAL void plane(wire::Writer& w, const PolyhedralInterface& p);
SIMPLEFLUID_FVM_LOCAL PolyhedralInterface plane(wire::Reader& r, std::optional<ArithmeticMode> mode = {});
SIMPLEFLUID_FVM_LOCAL Meshes::ExactPhaseMeasures phases(const Meshes::ExactConvexFragment& fragment, const PolyhedralInterface& p);
SIMPLEFLUID_FVM_LOCAL GeneralVoFPhaseEnclosure diagnostic(const Meshes::ExactPhaseMeasures& measures);
SIMPLEFLUID_FVM_LOCAL double output(const Measure& measure);
SIMPLEFLUID_FVM_LOCAL Meshes::ExactPhaseMeasures geometric_labels(const Cell& cell);
SIMPLEFLUID_FVM_LOCAL PolyhedralPhaseSample geometric_sample(const Cell& cell);
SIMPLEFLUID_FVM_LOCAL void refresh_experimental_catalog(GeneralVoF::Data& data);
SIMPLEFLUID_FVM_LOCAL void current(const GeneralVoF::Data& data);
SIMPLEFLUID_FVM_LOCAL void retained_budget(const GeneralVoF::Data& data);
// Every incident owner registers its actual cell once. The node authority
// authenticates those registrations and returns the complete canonical fan;
// geometric coincidence and whichever ghost rows happen to be resident play no
// part in defining the interpolation stencil.
SIMPLEFLUID_FVM_LOCAL void complete_node_fans(GeneralVoF::Data& data, GeneralVoF::Data::Topology& topology);
SIMPLEFLUID_FVM_LOCAL std::uint64_t transaction(Teuchos::RCP<const Teuchos::Comm<int>> comm);
SIMPLEFLUID_FVM_LOCAL void agree_transaction(Teuchos::RCP<const Teuchos::Comm<int>> comm, std::uint64_t id);
SIMPLEFLUID_FVM_LOCAL void agree_options(const GeneralVoFOptions& options, Teuchos::RCP<const Teuchos::Comm<int>> comm);
SIMPLEFLUID_FVM_LOCAL void import_phases(GeneralVoF::Data& data);
SIMPLEFLUID_FVM_LOCAL FinitePolyhedralPhaseSample finite_sample(const Cell& cell);
SIMPLEFLUID_FVM_LOCAL std::vector<FinitePolyhedralPhaseSample> finite_stencil(const GeneralVoF::Data& data, const Key& key);
SIMPLEFLUID_FVM_LOCAL PolyhedralInterface reconstruct_finite_cell(const GeneralVoF::Data& data, const Key& key);
SIMPLEFLUID_FVM_LOCAL std::map<Key, PolyhedralInterface> reconstruct_finite_local(const GeneralVoF::Data& data);
SIMPLEFLUID_FVM_LOCAL std::map<Key, PolyhedralInterface> reconstruct_exact_local(const GeneralVoF::Data& data);
SIMPLEFLUID_FVM_LOCAL std::map<Key, PolyhedralInterface> reconstruct_local(const GeneralVoF::Data& data);
SIMPLEFLUID_FVM_LOCAL std::map<Key, PolyhedralInterface> import_planes(const GeneralVoF::Data& data);
} // namespace general_vof_detail
} // namespace SimpleFluid::VoF
