// Independent OpenFOAM finite-volume reference for uniform thermal expansion.
// The fluid and mesh share the affine normal velocity: relative flux is zero.
#include "fvCFD.H"

#include <cmath>
#include <fstream>
#include <iomanip>

int main(int argc, char* argv[])
{
    Foam::argList::addOption("mode", "steady|transient", "Heating history or source-off equilibrium");
    #include "setRootCase.H"
    #include "createTime.H"
    #include "createMesh.H"
    const word mode(args.getOrDefault<word>("mode", "transient"));
    if (Pstream::parRun() || (mode != "steady" && mode != "transient"))
    {
        FatalErrorInFunction << "Use serial execution and -mode steady|transient" << exit(FatalError);
    }
    const IOdictionary water
    (
        IOobject("referenceWater", runTime.constant(), mesh, IOobject::MUST_READ, IOobject::NO_WRITE)
    );
    const scalar rho0 = readScalar(water.lookup("density_kg_m3"));
    const scalar cp = readScalar(water.lookup("specific_heat_capacity_J_kg_K"));
    const scalar beta = readScalar(water.lookup("thermal_expansion_1_K"));
    const scalar T0 = readScalar(water.lookup("temperature_K"));
    const scalar conductivity = readScalar(water.lookup("thermal_conductivity_W_m_K"));
    const scalar viscosity = readScalar(water.lookup("dynamic_viscosity_Pa_s"));
    const scalar absolutePressure = readScalar(water.lookup("absolute_pressure_Pa"));
    const scalar dt = 1.0, heating = 4.0e5;
    const label heatedSteps = 20, quietSteps = 5;
    const label steps = heatedSteps + (mode == "steady" ? quietSteps : 0);
    runTime.setDeltaT(dt);
    const pointField referencePoints(mesh.points());
    const scalarField cellMass(rho0 * mesh.V().field());
    volScalarField T(IOobject("T", runTime.timeName(), mesh, IOobject::MUST_READ, IOobject::AUTO_WRITE), mesh);
    T = dimensionedScalar("T0", dimTemperature, T0);
    T.correctBoundaryConditions();
    const dimensionedScalar thermalConductivity
    (
        "thermalConductivity", dimEnergy/dimTime/dimLength/dimTemperature, conductivity
    );
    volScalarField rhoCp
    (
        IOobject("rhoCp", runTime.timeName(), mesh, IOobject::NO_READ, IOobject::AUTO_WRITE), mesh,
        dimensionedScalar("rhoCp", dimEnergy/dimVolume/dimTemperature, rho0*cp), "zeroGradient"
    );
    surfaceScalarField relativeHeatCapacityFlux
    (
        IOobject("relativeHeatCapacityFlux", runTime.timeName(), mesh, IOobject::NO_READ, IOobject::NO_WRITE),
        mesh, dimensionedScalar(dimEnergy/dimTemperature/dimTime, Zero)
    );
    T.oldTime();
    rhoCp.oldTime();
    std::ofstream csv((runTime.path()/"history.csv").c_str());
    csv.exceptions(std::ios::badbit | std::ios::failbit);
    csv << std::setprecision(17)
        << "time_s,sample,temperature_K,level_m,volume_m3,liquid_mass_kg,energy_J,cumulative_heat_J,"
           "mass_residual_kg,energy_balance_residual_J,gcl_residual_m3_per_s,"
           "analytic_temperature_error_K,analytic_level_error_m,density_kg_m3,cp_J_kg_K,mu_Pa_s,k_W_m_K,"
           "nu_m2_s,thermal_diffusivity_m2_s,thermal_expansion_1_K,absolute_pressure_Pa\n";
    scalar level = 1, exactT = T0, cumulativeHeat = 0, previousT = T0, previousLevel = 1;
    label quietCount = 0;
    auto check = [](scalar residual, scalar tolerance, const char* name)
    {
        if (!std::isfinite(residual) || mag(residual) > tolerance)
        {
            FatalErrorInFunction << name << " residual=" << residual << ", tolerance=" << tolerance
                << exit(FatalError);
        }
    };
    for (label step = 0; step <= steps; ++step)
    {
        const scalar q = step <= heatedSteps ? heating : 0.0;
        if (step)
        {
            ++runTime;
            T.oldTime();
            rhoCp.oldTime();
            bool converged = false;
            // Picard coupling of actual moved geometry, conservative FV energy,
            // thermal liquid EOS, and sum(Mcell/rhoLiquid). No analytic solution
            // or SimpleFluid-produced state enters this iteration.
            for (label corrector = 0; corrector < 30; ++corrector)
            {
                pointField trialPoints(referencePoints);
                forAll(trialPoints, pointi)
                    trialPoints[pointi].z() *= level;
                mesh.movePoints(trialPoints);
                forAll(cellMass, celli)
                    rhoCp[celli] = cellMass[celli]*cp/mesh.V()[celli];
                rhoCp.correctBoundaryConditions();
                fvScalarMatrix energyEquation
                (
                    fvm::ddt(rhoCp, T) + fvm::div(relativeHeatCapacityFlux, T)
                  - fvm::laplacian(thermalConductivity, T)
                 == dimensionedScalar("q", dimEnergy/dimVolume/dimTime, q)
                );
                energyEquation.solve();
                T.correctBoundaryConditions();
                scalar targetLevel = 0;
                forAll(cellMass, celli)
                {
                    const scalar liquidDensity = rho0*(1-beta*(T[celli]-T0));
                    if (liquidDensity <= 1 || !std::isfinite(liquidDensity))
                        FatalErrorInFunction << "Invalid thermal liquid density" << exit(FatalError);
                    targetLevel += cellMass[celli]/liquidDensity; // A = 1 m2
                }
                if (mag(targetLevel-level) <= 1e-13)
                {
                    converged = true;
                    break;
                }
                level = targetLevel;
            }
            if (!converged)
                FatalErrorInFunction << "ALE thermal/geometry coupling did not converge" << exit(FatalError);
            // Independent analytic BE oracle, used exclusively for acceptance.
            const scalar a = 1-beta*(exactT-T0), b = q*dt/(rho0*cp);
            exactT += 2*b/(a+std::sqrt(a*a-4*beta*b));
        }
        scalar volume = 0, mass = 0, energy = 0, gcl = 0;
        forAll(cellMass, celli)
        {
            volume += mesh.V()[celli];
            mass += rhoCp[celli]/cp*mesh.V()[celli];
            energy += rhoCp[celli]*mesh.V()[celli]*T[celli];
            check(T[celli]-exactT, 2e-7, "Cell temperature analytic error");
        }
        if (step)
        {
            cumulativeHeat += q*volume*dt;
            const volScalarField meshDivergence(fvc::div(mesh.phi()));
            forAll(cellMass, celli)
                gcl = max(gcl, mag((mesh.V()[celli]-mesh.V0()[celli])/dt
                    - meshDivergence[celli]*mesh.V()[celli]));
        }
        const scalar temperature = energy/(mass*cp);
        const scalar exactLevel = 1/(1-beta*(exactT-T0));
        const scalar density = rho0*(1-beta*(temperature-T0));
        const scalar massResidual = mass-rho0;
        const scalar energyResidual = energy-rho0*cp*T0-cumulativeHeat;
        check(massResidual, 2e-10, "Liquid mass conservation");
        check(energyResidual, 5e-5, "Liquid energy conservation");
        check(gcl, 2e-11, "Mesh GCL");
        check(level-exactLevel, 5e-10, "Level analytic error");
        check(volume-level, 2e-11, "Mesh volume and level closure");
        if (step > heatedSteps)
        {
            check(temperature-previousT, 2e-8, "Steady temperature change");
            check(level-previousLevel, 2e-11, "Steady level change");
            ++quietCount;
        }
        check(runTime.value()-step*dt, 1e-13, "Accepted physical time");
        csv << runTime.value() << ",global," << temperature << ',' << level << ',' << volume << ',' << mass << ','
            << energy << ',' << cumulativeHeat << ',' << massResidual << ',' << energyResidual << ',' << gcl
            << ',' << temperature-exactT << ',' << level-exactLevel << ',' << density << ',' << cp << ','
            << viscosity << ',' << conductivity << ',' << viscosity/density << ',' << conductivity/(density*cp)
            << ',' << beta << ',' << absolutePressure << '\n';
        previousT = temperature;
        previousLevel = level;
    }
    if (mode == "steady" && quietCount != quietSteps)
        FatalErrorInFunction << "Steady state did not pass five consecutive source-off steps" << exit(FatalError);
    T.write();
    rhoCp.write();
    mesh.write();
    Info<< "planarALE " << mode << ": " << steps << " accepted steps, " << quietCount
        << " source-off convergence checks; wrote history.csv" << nl;
    return 0;
}
