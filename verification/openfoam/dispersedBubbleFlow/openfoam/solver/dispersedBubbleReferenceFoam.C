// Independent OpenFOAM FV reference for the constant-slip microbubble limit.
// Production is applied after implicit Euler/upwind transport, matching the
// documented operator splitting in the SimpleFluid model.
#include "fvCFD.H"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>

int main(int argc, char *argv[])
{
    #include "setRootCase.H"
    #include "createTime.H"
    #include "createMesh.H"
    if (Pstream::parRun()) FatalErrorInFunction << "Serial reference only" << exit(FatalError);
    IOdictionary properties(IOobject("verificationProperties", runTime.constant(), mesh,
        IOobject::MUST_READ, IOobject::NO_WRITE));
    const auto parameter = [&](const word& name) { return readScalar(properties.lookup(name)); };
    const word mode(properties.lookup("mode"));
    const scalar height = parameter("height"), width = parameter("width");
    const scalar dt = parameter("dt"), speed = parameter("carrier_velocity") + parameter("slip_velocity");
    const scalar radius = parameter("nucleation_radius");
    // Isothermal IF97 reference liquid. Carrier flow and slip are prescribed,
    // so viscosity/conductivity are recorded but no momentum/energy is solved.
    const scalar temperature = parameter("temperature_K"), absolutePressure = parameter("absolute_pressure_Pa");
    const scalar rho = parameter("density_kg_m3"), cp = parameter("specific_heat_capacity_J_kg_K");
    const scalar mu = parameter("dynamic_viscosity_Pa_s"), conductivity = parameter("thermal_conductivity_W_m_K");
    const scalar sigma = parameter("surface_tension_N_m");
    const scalar bubbleVolume = 4.0 * constant::mathematical::pi / 3.0 * pow3(radius);
    const scalar molesPerBubble = bubbleVolume * (absolutePressure + 2 * sigma / radius)
        / (parameter("gas_constant") * temperature);
    const scalar source = mode == "steady" ? parameter("power_density") * parameter("yield_mol_per_j")
        * parameter("release_efficiency") : 0.0;
    const scalar initial = parameter(mode + "_initial_moles");
    const label cells = label(parameter("cells"));
    const scalar dz = height / cells, volume = height * sqr(width);
    const scalar end = parameter(mode + "_end_time");
    const label steps = label(std::llround(end / dt));
    const label writeSteps = label(std::llround(parameter(mode + "_write_interval") / dt));
    if (mesh.nCells() != cells || writeSteps < 1 || mag(runTime.deltaTValue() - dt) > 1e-14)
        FatalErrorInFunction << "Unmatched mesh/time parameters" << exit(FatalError);
    volScalarField microMoles(IOobject("microMoles", runTime.timeName(), mesh,
        IOobject::NO_READ, IOobject::AUTO_WRITE), mesh,
        dimensionedScalar("initial", dimensionSet(0,-3,0,0,1,0,0), initial), "zeroGradient");
    volScalarField microNumber(IOobject("microNumber", runTime.timeName(), mesh,
        IOobject::NO_READ, IOobject::AUTO_WRITE), mesh,
        dimensionedScalar("initial", dimless/dimVolume, initial/molesPerBubble), "zeroGradient");
    surfaceScalarField phi(IOobject("phi", runTime.timeName(), mesh, IOobject::NO_READ, IOobject::NO_WRITE),
        mesh.Sf() & dimensionedVector("bubbleVelocity", dimVelocity, vector(0,0,speed)));
    forAll(phi.boundaryField(), patch)
    {
        if (mesh.boundary()[patch].name() != "zmax") phi.boundaryFieldRef()[patch] = 0.0;
    }
    const label outlet = mesh.boundaryMesh().findPatchID("zmax");
    if (outlet < 0) FatalErrorInFunction << "Missing outlet" << exit(FatalError);
    std::ofstream profiles((runTime.path()/"profiles.csv").c_str());
    std::ofstream history((runTime.path()/"history.csv").c_str());
    profiles << std::setprecision(17)
        << "time_s,sample,z_m,micro_moles_mol_m3,micro_number_m3,alpha_g,temperature_K,absolute_pressure_Pa,density_kg_m3,"
           "specific_heat_capacity_J_kg_K,dynamic_viscosity_Pa_s,thermal_conductivity_W_m_K,"
           "kinematic_viscosity_m2_s,thermal_diffusivity_m2_s,surface_tension_N_m,"
           "hydrogen_balance_mol,number_balance_relative\n";
    history << std::setprecision(17) << "time_s,inventory_mol,produced_mol,escaped_mol,outlet_mol_s,hydrogen_balance_mol,maximum_change_mol_m3\n";
    scalar escaped = 0, escapedNumber = 0, produced = 0, lastEscape = 0, maximumChange = 0;
    label steadyConsecutiveSteps = 0;
    auto writeCsv = [&](scalar time)
    {
        const scalar inventory = sum(microMoles.primitiveField()*mesh.V());
        const scalar number = sum(microNumber.primitiveField()*mesh.V());
        const scalar balance = inventory + escaped - initial*volume - produced;
        const scalar numberBalance = (number + escapedNumber - (initial*volume + produced)/molesPerBubble)
            / ((initial*volume + produced)/molesPerBubble);
        if (!std::isfinite(balance) || mag(balance) > 2e-13 || mag(numberBalance) > 2e-8)
            FatalErrorInFunction << "Conservation gate failed" << exit(FatalError);
        forAll(microMoles, cell)
        {
            const scalar z = mesh.C()[cell].z();
            const label sample = label(std::llround(z/dz - 0.5));
            profiles << time << ',' << sample << ',' << z << ',' << microMoles[cell] << ','
                     << microNumber[cell] << ',' << microNumber[cell]*bubbleVolume << ','
                     << temperature << ',' << absolutePressure << ',' << rho << ',' << cp << ','
                     << mu << ',' << conductivity << ',' << mu/rho << ',' << conductivity/(rho*cp) << ','
                     << sigma << ','
                     << balance << ',' << numberBalance << '\n';
        }
        history << time << ',' << inventory << ',' << produced << ',' << escaped << ','
                << lastEscape/dt << ',' << balance << ',' << maximumChange << '\n';
    };
    writeCsv(0);
    for (label step = 1; step <= steps; ++step)
    {
        ++runTime;
        if (mag(runTime.value()-step*dt) > 1e-11)
            FatalErrorInFunction << "Physical time does not match fixed step schedule" << exit(FatalError);
        const scalarField previous(microMoles.primitiveField());
        solve(fvm::ddt(microMoles) + fvm::div(phi, microMoles));
        solve(fvm::ddt(microNumber) + fvm::div(phi, microNumber));
        microMoles.correctBoundaryConditions();
        microNumber.correctBoundaryConditions();
        lastEscape = dt * sum(phi.boundaryField()[outlet] * microMoles.boundaryField()[outlet]);
        escaped += lastEscape;
        escapedNumber += dt * sum(phi.boundaryField()[outlet] * microNumber.boundaryField()[outlet]);
        microMoles.primitiveFieldRef() += source*dt;
        microNumber.primitiveFieldRef() += source*dt/molesPerBubble;
        produced += source*dt*volume;
        maximumChange = max(mag(microMoles.primitiveField() - previous));
        if (maximumChange < 1e-12 && mag(lastEscape/dt-source*volume) < 2e-11)
            ++steadyConsecutiveSteps;
        else steadyConsecutiveSteps = 0;
        if (min(microMoles.primitiveField()) < 0 || min(microNumber.primitiveField()) < 0)
            FatalErrorInFunction << "Negative microbubble moment" << exit(FatalError);
        if (step % writeSteps == 0) writeCsv(step*dt);
    }
    if (mode == "steady")
    {
        if (steadyConsecutiveSteps < 5)
            FatalErrorInFunction << "Steady convergence/outlet balance failed" << exit(FatalError);
        forAll(microMoles, cell)
        {
            const scalar exact = source*mesh.C()[cell].z()/speed;
            const scalar truncation = source*(0.5*dz/speed+dt);
            if (mag(microMoles[cell]-exact) > truncation*1.01+1e-12)
                FatalErrorInFunction << "Steady continuum profile failed" << exit(FatalError);
        }
    }
    else
    {
        scalar l1 = 0;
        forAll(microMoles, cell)
        {
            const scalar upper = mesh.C()[cell].z()+0.5*dz;
            const scalar exact = initial*std::clamp((upper-speed*end)/dz, scalar(0), scalar(1));
            l1 += mag(microMoles[cell]-exact)*dz/(initial*height);
        }
        if (l1 > 0.12 || escaped < 0.4*initial*volume || escaped > 0.6*initial*volume)
            FatalErrorInFunction << "Transient translating-front/escape gate failed" << exit(FatalError);
    }
    profiles.flush(); history.flush();
    if (!profiles.good() || !history.good()) FatalErrorInFunction << "CSV write failed" << exit(FatalError);
    Info << mode << " dispersed microbubble reference passed" << endl;
    return 0;
}
