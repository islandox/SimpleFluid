/** @file ReactingVoFCoupler.hh
 * @brief Transactional aqueous radiolysis and resolved/dispersed gas exchange.
 */
#pragma once

#include "equations/BubblePopulationKinetics.hh"
#include "solvers/AdaptiveVoFSolver.hh"
#include "solvers/BubbleRepresentation.hh"

#include <functional>
#include <memory>
#include <vector>

namespace SimpleFluid
{

/** A declared physical stagnant-film closure; no empirical thickness is built in. */
struct ResolvedHydrogenFilm
{
    bool enabled = false;
    std::string model_name;
    std::string validity_description;
    /** Inputs: interface centroid [m], liquid T [K], liquid p [Pa]; output delta [m]. */
    std::function<double(const VoF::Vector&, double, double)> thickness;
    /** Must reject an interface outside the supplied local-planarity/mixing/calibration domain. */
    std::function<void(const VoF::InterfacePatch&, double, double, double)> validate;
};

struct DispersedCaptureModel
{
    bool enabled = false;
    std::string model_name;
    std::string validity_description;
    /** Population index, bubble radius [m], approach speed [m/s], liquid T [K] -> [0,1]. */
    std::function<double(std::size_t, double, double, double)> efficiency;
    std::function<void(std::size_t, double, double, double)> validate;
};

/** An explicitly designated gas reservoir inside the fixed CFD domain. */
struct ResolvedHeadspaceVent
{
    bool enabled = false;
    std::vector<Meshes::AMRBox> regions;
    std::string model_name;
    std::string validity_description;
    /** Headspace mole-weighted p [Pa], T [K], H2 mole fraction -> total mol/s leaving. */
    std::function<double(double, double, double)> molar_rate;
    std::function<void(double, double, double)> validate;
};

struct ReactingVoFOptions
{
    /** Micro kinetics and particle geometry/slip; Disabled leaves other supplied processes selectable.
     * The hybrid micro integrator currently requires LegacySubcycled. A nonzero
     * ConstantSlip speed points opposite the flow solver's physical gravity.
     * Returned/transported micro moments must remain paired under the legacy
     * conversion law; incompatible number/molar ratios reject without resetting
     * the representative molar size or changing the legacy conversion formula.
     */
    ReactingGasOptions radiolysis;
    BubbleGasPhysics::MolarPivotGrid upper_pivots;
    BubbleGasPhysics::BubblePopulationKernels upper_kernels;
    BubbleGasPhysics::BubblePopulationStepOptions population_step;
    std::string micro_population = "micro";
    std::vector<std::string> upper_populations;
    ResolvedHydrogenFilm resolved_transfer;
    DispersedCaptureModel capture;
    ResolvedHeadspaceVent headspace;
    BubbleRepresentationOptions representation;
    bool dissolved_diffusion = true;
    int maximum_correctors = 20; ///< At least two evaluations are needed to check the source fixed point.
    double relative_tolerance = 1e-7;
    double inventory_relative_tolerance = 2e-10;
};

struct ReactingVoFInput
{
    /** Liquid-absorbed radiation energy [J/owned cell]; yield is applied once, with no extra void weight. */
    std::vector<double> absorbed_radiation_energy;
    /** Already net thermal deposition [J/owned cell]; chemical-energy allocation is the caller's responsibility. */
    std::vector<double> net_thermal_energy;
};

struct ReactingVoFStatistics
{
    double time = 0;
    int correctors = 0;
    int reaction_substeps = 0;
    double hydrogen_before = 0; ///< Dissolved + dispersed + resolved H2, mol; excludes unreacted donor.
    double hydrogen_after = 0;
    double hydrogen_produced = 0;
    double hydrogen_captured = 0;
    double hydrogen_escaped = 0; ///< Only actual vent withdrawal counts as escape.
    double donor_consumed = 0;
    double hydrogen_error = 0;
    double donor_error = 0;
    double dissolved_to_resolved = 0;
    double resolved_to_dissolved = 0;
    double captured_number = 0;
    double cumulative_produced = 0;
    double cumulative_captured = 0;
    double cumulative_escaped = 0;
    double escaped_enthalpy = 0;
    double thermal_input = 0;
    double mass_error = 0;
    double energy_error = 0;
};

/**
 * External shared owner of a low-Mach AdaptiveVoFSolver and reaction closures.
 *
 * All operations except const accessors are collective. Construction requires
 * an initialized low-Mach solver and an explicitly supplied aqueous model.
 * The solver owns transported conservative state; this controller owns kinetics,
 * interface/capture/vent laws and their accepted cumulative ledgers. It never
 * calls the legacy full ReactingGasModel::advance and cannot double-transport
 * the same inventory. Callers must not mutate the retained solver during a
 * prepared trial; its accepted-state identity rejects stale commits.
 * Population names are resolved collectively for each source preview, so valid
 * solver reinitialization may reorder populations or fail for missing names
 * before any accepted inventory changes.
 *
 * Constant resolved surface tension is the declared Stage 3 scope. Unresolved
 * drainage/rupture is represented only by supplied capture/coalescence kernels;
 * grid-scale resolved contact retains geometric VoF coalescence. Representation
 * changes are a separate opt-in fixed-clock transaction with explicit birth
 * identities. Headspace ownership and empirical calibrations remain supplied.
 * The first-order reaction/transport split freezes accepted interface patches,
 * carrier apertures and designated headspace ownership throughout the source
 * correctors. Material coefficients and true aqueous capacity use the trial
 * thermodynamic state; the flow solver transports each inventory only once.
 *
 * Upper bins use fixed molar pivots; their physical radius is recovered from
 * their moments and the trial EOS. Their surface energy is not a separate
 * conserved reservoir. The supplied net thermal input owns chemistry and
 * unresolved interfacial-energy allocation. Diagnostic absorbed radiation is
 * never added to thermal energy a second time.
 */
class SIMPLEFLUID_SOLVERS_EXPORT ReactingVoFCoupler
{
public:
    using Checkpoint = std::uint64_t;
    explicit ReactingVoFCoupler(SP<AdaptiveVoFSolver> solver, ReactingVoFOptions options);
    ~ReactingVoFCoupler();
    ReactingVoFCoupler(ReactingVoFCoupler&&) noexcept;
    ReactingVoFCoupler& operator=(ReactingVoFCoupler&&) noexcept;
    ReactingVoFCoupler(const ReactingVoFCoupler&) = delete;
    ReactingVoFCoupler& operator=(const ReactingVoFCoupler&) = delete;

    class SIMPLEFLUID_SOLVERS_EXPORT Trial
    {
    public:
        [[nodiscard]] const VoF::LowMachState& state() const;
        /** Immutable integrated face ledgers of the candidate transport. */
        [[nodiscard]] const VoF::LowMachTransport& transport() const;
        [[nodiscard]] const ReactingVoFStatistics& statistics() const;
        [[nodiscard]] const BubbleRepresentationInspection& representation() const;
        [[nodiscard]] const VoF::RepresentationDiagnostics& representation_diagnostics() const;

    private:
        friend class ReactingVoFCoupler;
        explicit Trial(std::shared_ptr<const void> candidate);
        std::shared_ptr<const void> candidate_;
    };

    [[nodiscard]] Trial preview(double dt, const ReactingVoFInput& input);
    /** Register deterministic persistent birth requests; no physical inventory changes. */
    void request_births(const std::vector<BubbleBirthSeed>& seeds);
    [[nodiscard]] const std::vector<BubbleBirthRecord>& birth_registry() const noexcept;
    [[nodiscard]] const std::vector<BubbleConversionEvent>& conversion_history() const noexcept;
    /** Collective read-only component/eligibility inspection, including refinement requirements. */
    [[nodiscard]] BubbleRepresentationInspection inspect_representation() const;
    /** Fixed-clock conservative conversion; existing commit/checkpoint owns flow and registry together. */
    [[nodiscard]] Trial preview_representation();
    void commit(const Trial& trial);
    [[nodiscard]] ReactingVoFStatistics step(double dt, const ReactingVoFInput& input);
    [[nodiscard]] Checkpoint checkpoint();
    void restore(Checkpoint token);
    void release_checkpoint(Checkpoint token);
    void regrid(const VoF::RefinementPlan& plan);
    [[nodiscard]] const SP<AdaptiveVoFSolver>& solver() const noexcept;
    [[nodiscard]] const ReactingVoFOptions& options() const noexcept;
    [[nodiscard]] const ReactingVoFStatistics& statistics() const noexcept;

private:
    struct SIMPLEFLUID_SOLVERS_LOCAL Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace SimpleFluid
