/** @file AdaptiveVoFContext.hh
 * @brief Private native transport state and context shared by adaptive solver implementations.
 */
#pragma once
#include "solvers/AdaptiveVoFSolver.hh"

#include "equations/ReactingGasProperties.hh"
#include "solvers/BelosLinearSolver.hh"
#include "solvers/details/ConservativeVoFTransport.hh"
#include "solvers/details/VoFProjectionWorkspace.hh"
#include "utils/CompensatedSum.hh"

#include <Teuchos_Array.hpp>
#include <Teuchos_CommHelpers.hpp>
#include <Tpetra_Core.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iterator>
#include <limits>
#include <map>
#include <numbers>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <utility>


namespace SimpleFluid::detail::adaptive_vof
{

using Pack = DefaultTpetraTypes;
using LO = Pack::local_ordinal_type;
using GO = Pack::global_ordinal_type;
using Vector = VoF::Vector;
using Inventory = VoF::ConservedQuantity;
using Carrier = VoF::InventoryCarrier;

using Hierarchy = AdaptiveVoFSolver::Hierarchy;
using Handle = AdaptiveVoFSolver::Handle;
SIMPLEFLUID_SOLVERS_LOCAL void require_all(bool valid, const char* message);
SIMPLEFLUID_SOLVERS_LOCAL double maximum(double local);
SIMPLEFLUID_SOLVERS_LOCAL double sum(double local);
SIMPLEFLUID_SOLVERS_LOCAL void identical(double value, const char* message);
SIMPLEFLUID_SOLVERS_LOCAL void identical_name(const std::string& name);
SIMPLEFLUID_SOLVERS_LOCAL double dot(const Vector& a, const Vector& b);
SIMPLEFLUID_SOLVERS_LOCAL Vector convert(const MeshUtils::Vec3& v);
SIMPLEFLUID_SOLVERS_LOCAL double mixture(double f, double liquid, double gas);
SIMPLEFLUID_SOLVERS_LOCAL double carrier_amount(Carrier carrier, double volume, double fraction, const VoFMaterial& m);
SIMPLEFLUID_SOLVERS_LOCAL double carrier_flux(Carrier carrier, double gas, double liquid, const VoFMaterial& m);
SIMPLEFLUID_SOLVERS_LOCAL void validate_options(const VoFMaterial& m, const VoFSolverOptions& o);
struct AdaptiveVoFState
{
    std::vector<double> f, p, k, face_u;
    std::vector<Vector> u;
    std::vector<Inventory> inventories;
    VoFStepDiagnostics diagnostics;
    std::size_t steps = 0;
    std::optional<VoF::LowMachState> low;
    VoF::LowMachDiagnostics low_diagnostics;
    VoF::LowMachTransport low_flux;
    Vector low_pressure_impulse_scale{};
    double low_momentum_operation_bound = 0;
};

struct SIMPLEFLUID_SOLVERS_LOCAL AdaptiveVoFContext
{
    using State = AdaptiveVoFState;
    struct Face
    {
        LO owner = -1, neighbor = -1;
        Vector normal{}, owner_tangent{}, neighbor_tangent{};
        double area = 0, owner_distance = 0, neighbor_distance = 0, distance = 0;
        bool boundary = false, active = false;
    };
    std::shared_ptr<VoF::AdaptiveVoF> geometry;
    SP<const Handle> mesh;
    VoFMaterial material;
    VoFSolverOptions options;
    std::vector<double> volume;
    std::vector<Vector> spacing;
    std::vector<Face> faces;
    std::size_t row_capacity = 1;
    GO gauge = 0;
    detail::VoFProjectionWorkspace projection;
    detail::ConservativeVoFTransport conservative_transport;
    std::shared_ptr<const VoF::LowMachOptions> low_options;

    AdaptiveVoFContext(std::shared_ptr<const Hierarchy> h, VoFMaterial m, VoFSolverOptions o);

    // Call only around local geometry reads. Imports, reductions, matrix
    // completion and linear solves remain outside this borrowed lease.
    template<class Work> void with_geometry(Work&& work) const
    {
        const auto execution = mesh->acquire_execution_view();
        std::forward<Work>(work)();
    }

    std::size_t owned() const { return mesh->num_owned_cells(); }
    bool is_owned(LO id) const { return id >= 0 && static_cast<std::size_t>(id) < owned(); }

    void validate_fraction(const std::vector<double>& f) const;

    std::vector<double> density(const State& s) const;

    std::vector<Vector> exchange_vectors(const std::vector<Vector>& values) const;

    /** Weighted least squares with physical periodic-image displacement. */
    std::vector<Vector> gradients(const std::vector<double>& overlap) const;

    std::vector<double> checked_faces(const std::vector<double>& values) const;

    std::vector<double> interpolate(const std::vector<Vector>& values) const;

    std::vector<Vector> average_faces(const std::vector<double>& values) const;

    std::vector<double> divergence(const std::vector<double>& velocity) const;

    double max_abs(const std::vector<double>& values) const;

    std::vector<double> extended_curvature(const State& s) const;

    enum class BoundaryProjection
    {
        Step,
        InitialCell,
        InitialFace
    };

    void project(State& s, const std::vector<double>& rho_owned, double dt,
        const std::vector<Vector>& predictor_change, bool capillary, BoundaryProjection boundary,
        const std::vector<double>* integrated_target = nullptr,
        const std::vector<double>* volume_susceptibility = nullptr,
        const std::vector<double>* pressure_reference = nullptr);

    std::vector<Vector> viscous_force(const State& s, Vector* traction_scale = nullptr) const;

    double inventory_total(const Inventory& q) const;

    double inventory_l1(const Inventory& q) const;

    Vector momentum_l1(const State& s) const;

    void validate_inventory(const Inventory& q, const State& s) const;

    std::vector<Inventory> transport_inventories(
        const State& old, const State& next, const VoF::AdaptiveAdvection& transport);

    void budget(State& s) const;

    double stable_timestep(const State& s) const;

    // Sparse conservative representation exchange.

    double gas_e(const VoF::IdealGasSpecies& gas, double temperature) const;

    double low_mass(const VoF::LowMachState& s, std::size_t c) const;

    Vector drift_momentum(const VoF::LowMachState& s, std::size_t c) const;

    Vector low_drift_momentum_l1(const VoF::LowMachState& state) const;

    double low_momentum_roundoff_factor(const State& state) const;

    void validate_low_options(const VoF::LowMachOptions& o) const;

    std::vector<double> low_values(const std::vector<double>& input, bool nonnegative, const char* message) const;

    std::vector<Vector> low_vectors(const std::vector<Vector>& input, const char* message) const;

    void validate_low_inventory(const VoF::LowMachState& s) const;

    struct LowTotals
    {
        double mass = 0, hydrogen = 0, inert = 0, energy = 0, energy_l1 = 0;
        Vector momentum{}, momentum_l1{};
    };

    LowTotals low_totals(const VoF::LowMachState& s) const;

    struct ThermoEvaluation
    {
        std::vector<double> temperature, rho_l, rho_r, p_l, p_r, partial, gas, liquid, dispersed;
        std::vector<std::vector<double>> bubble_volume, bubble_radius, bubble_pressure;
        double material_volume = 0;
    };

    std::optional<ThermoEvaluation> evaluate_low(const VoF::LowMachState& s, const std::vector<double>& mechanical,
        const std::vector<double>& f, const std::vector<double>& k, double p0) const;

    void assign_thermo(VoF::LowMachState& s, ThermoEvaluation evaluation, double p0) const;

    struct PhasePressureSusceptibility
    {
        std::vector<double> gas, carrier, total; // -dV_phase/dpi at fixed extensive inventories and internal energy.
    };

    PhasePressureSusceptibility low_pressure_susceptibility(const VoF::LowMachState& state) const;

    double domain_volume() const;

    void recover_closed_thermo(VoF::LowMachState& s, const std::vector<double>& mechanical, const std::vector<double>& f,
        const std::vector<double>& k) const;

    void normalize_low_pressure(State& s) const;

    std::vector<double> low_viscosity(const State& s) const;

    double low_conductivity(const State& state, std::size_t cell, const VoF::LiquidThermodynamicPoint& liquid) const;

    double low_thermal_iteration_error(const State& candidate, const State& guess) const;

    /** No fictitious radius is assigned to an empty cell. At a particle front
     * extend the populated-side evaluated closure; otherwise distance interpolate.
     * Constant-slip cell closures extend throughout the available liquid carrier.
     */
    Vector population_face_drift(const Face& face, const std::vector<Vector>& drift,
        const std::vector<double>& number) const;

    /** Support of the moments on which the supplied source closure was evaluated.
     * Source depletion matters just as much as birth: retaining a depleted cell
     * would interpolate its zero drift with a populated neighbor's evaluated law.
     */
    std::vector<double> population_closure_number(const VoF::DispersedPopulation& accepted,
        const VoF::PopulationSource& source) const;

    double stable_low_timestep(const State& s) const;

    void budget_low(State& s) const;

    VoF::LowMachSources normalize_low_sources(const State& old, const VoF::LowMachSources& input) const;

    using LowTransportScalar = detail::ConservativeVoFTransport::Result;

    LowTransportScalar transport_low_scalar(const std::vector<double>& old, const std::vector<double>& source,
        const std::vector<double>& capacity, const std::vector<double>& carrier_flux, bool positive);

    struct LowFluxWork
    {
        VoF::LowMachTransport published;
        std::vector<double> liquid_volume, gas_volume;
        std::vector<std::vector<double>> dispersed_volume;
        std::vector<std::vector<double>> population_number_trace, population_molar_trace, relative_population_moles;
        std::vector<double> energy_rhs;
        std::vector<Vector> momentum_rhs;
    };

    LowFluxWork transport_low(const State& old, State& candidate, const State& guess,
        const VoF::AdaptiveAdvection& geometric, const VoF::PhaseVolumeIncrement& increment,
        const VoF::LowMachSources& source, double dt, int corrector);

    void recover_low_volume_fluxes(const State& candidate, LowFluxWork& work) const;

    State initialize_low(const VoF::LowMachInitialState& initial);

    State advance_low(const State& old, double dt, const VoF::LowMachSources& input, const VoF::LowMachState* hint);

    State remap_low(const State& old, AdaptiveVoFContext& target) const;
    // Sparse conservative representation exchange.
    // Every sparse parcel is owned once. Only its source and destination packets
    // are imported; neither a global solution nor a global parcel list is formed.
    struct RepresentationPacket
    {
        enum Quantity : std::size_t
        {
            Volume,
            LiquidMass,
            Hydrogen,
            Inert,
            Number,
            Dissolved,
            Donor,
            Energy,
            MomentumX,
            MomentumY,
            MomentumZ,
            PressureVolume,
            Width
        };
        using Data = std::vector<std::vector<double>>; // phase-major columns, native owned/sparse rows
        static std::size_t column(std::size_t phase, std::size_t quantity) { return phase * Width + quantity; }
    };

    struct RepresentationExchange
    {
        Teuchos::RCP<const Pack::map_type> source, target;
        std::unique_ptr<Pack::import_type> importer;
        RepresentationExchange(Teuchos::RCP<const Pack::map_type> owned_map, const std::set<GO>& ids);
        RepresentationPacket::Data transfer(const RepresentationPacket::Data& input, bool reverse) const;
    };

    RepresentationPacket::Data representation_packets(const State& state) const;

    State change_representation(const State& old, const VoF::RepresentationChangeRequest& request,
        VoF::RepresentationDiagnostics& diagnostics) const;

    State advance(const State& old, double dt);
};
} // namespace SimpleFluid::detail::adaptive_vof
