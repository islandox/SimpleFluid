// Independent OpenFOAM FV implementation of the matched Menter-1994 SST.
// Original vorticity limiter, k-only 20*betaStar*k*omega production cap,
// frozen old-state coefficients, and integration through the viscous sublayer.
#pragma once
#include "StructuredCaseMesh.H"
#include "fvCFD.H"
#include <limits>

class ReferenceSST
{
    Foam::fvMesh& mesh_;
    const Foam::scalar nu_, rho0_, prt_, kMin_, omegaMin_, wallCoefficient_;
    static bool wall(const Foam::word& name) { return name == "xmin" || name == "xmax" || name == "zmin"; }
    Foam::wordList patches(const Foam::word& wallType) const
    {
        auto types = StructuredCaseMesh::patch_types(mesh_, "zeroGradient");
        forAll(types, p) if (wall(mesh_.boundary()[p].name())) types[p] = wallType;
        return types;
    }
    Foam::volScalarField scalarField(
        const Foam::word& name, Foam::dimensionSet dims, Foam::scalar value, bool wallFixed = false) const
    {
        return Foam::volScalarField(
            Foam::IOobject(name, mesh_.time().timeName(), mesh_, Foam::IOobject::NO_READ, Foam::IOobject::NO_WRITE),
            mesh_, Foam::dimensionedScalar(dims, value), patches(wallFixed ? "fixedValue" : "zeroGradient"));
    }
    void wallValues()
    {
        forAll(mesh_.boundary(), p) if (wall(mesh_.boundary()[p].name()))
        {
            kinetic.boundaryFieldRef()[p] == Foam::scalar(0);
            nut.boundaryFieldRef()[p] == Foam::scalar(0);
            // Wall-face Dirichlet omega, not an owner-cell constraint.
            const auto& delta = mesh_.deltaCoeffs().boundaryField()[p];
            omega.boundaryFieldRef()[p] == wallCoefficient_* nu_ / 0.075 * Foam::sqr(delta);
        }
        kinetic.correctBoundaryConditions();
        omega.correctBoundaryConditions();
        nut.correctBoundaryConditions();
    }
    void refreshNut(const Foam::volVectorField& U)
    {
        const Foam::volTensorField gradU(Foam::fvc::grad(U));
        forAll(nut, c)
        {
            const auto& g = gradU[c];
            const Foam::scalar rotation = Foam::sqrt(2 * Foam::magSqr(Foam::skew(g)));
            const Foam::scalar d = distance[c], w = omega[c], k = kinetic[c];
            const Foam::scalar f2 =
                Foam::tanh(Foam::sqr(Foam::max(2 * Foam::sqrt(k) / (0.09 * w * d), 500 * nu_ / (d * d * w))));
            nut[c] = 0.31 * k / Foam::max(0.31 * w, rotation * f2);
            if (!std::isfinite(k) || k < kMin_ || !std::isfinite(w) || w < omegaMin_ || !std::isfinite(nut[c]) ||
                nut[c] < 0)
                FatalErrorInFunction << "Nonfinite or invalid SST state" << Foam::exit(Foam::FatalError);
        }
        wallValues();
    }

public:
    Foam::volScalarField kinetic, omega, nut, distance, yplus;
    ReferenceSST(Foam::fvMesh& mesh, const Foam::IOdictionary& p)
        : mesh_(mesh), nu_(Foam::readScalar(p.lookup("kinematic_viscosity_m2_s"))),
          rho0_(Foam::readScalar(p.lookup("density_kg_m3"))), prt_(Foam::readScalar(p.lookup("sst_prandtl"))),
          kMin_(Foam::readScalar(p.lookup("sst_k_min"))), omegaMin_(Foam::readScalar(p.lookup("sst_omega_min"))),
          wallCoefficient_(Foam::readScalar(p.lookup("sst_wall_omega_coefficient"))),
          kinetic(scalarField("kSST", Foam::sqr(Foam::dimVelocity), Foam::readScalar(p.lookup("sst_initial_k")), true)),
          omega(scalarField("omegaSST", Foam::dimless / Foam::dimTime, 1, true)),
          nut(scalarField("nutSST", Foam::dimViscosity, 0, true)),
          distance(scalarField("wallDistance", Foam::dimLength, 1)), yplus(scalarField("wallYPlus", Foam::dimless, 0))
    {
        if (!(nu_ > 0 && rho0_ > 0 && prt_ > 0 && kMin_ > 0 && omegaMin_ > 0 && wallCoefficient_ > 0))
            FatalErrorInFunction << "Invalid SST reference inputs" << Foam::exit(Foam::FatalError);
        const auto width = Foam::readScalar(p.lookup("width"));
        const auto initialOmega = Foam::readScalar(p.lookup("sst_initial_omega"));
        forAll(distance, c)
        {
            const auto centre = mesh_.C()[c];
            distance[c] = Foam::min(Foam::min(centre.x(), width - centre.x()), centre.z());
            omega[c] = Foam::max(initialOmega, 6 * nu_ / (0.075 * Foam::sqr(distance[c])));
            nut[c] = kinetic[c] / omega[c];
        }
        distance.correctBoundaryConditions();
        wallValues();
    }
    Foam::scalar wallYPlus(const Foam::volVectorField& U)
    {
        yplus = Foam::dimensionedScalar(Foam::dimless, 0);
        Foam::scalar maximum = 0;
        forAll(mesh_.boundary(), p) if (wall(mesh_.boundary()[p].name()))
        {
            const auto& owners = mesh_.boundary()[p].faceCells();
            const auto normals = mesh_.boundary()[p].nf();
            forAll(owners, f)
            {
                const auto c = owners[f];
                const auto n = normals()[f];
                const auto tangential = U[c] - n * (U[c] & n);
                const Foam::scalar d = 1 / mesh_.deltaCoeffs().boundaryField()[p][f];
                const Foam::scalar value = Foam::sqrt(Foam::mag(tangential) * d / nu_);
                yplus[c] = Foam::max(yplus[c], value);
                maximum = Foam::max(maximum, value);
            }
        }
        Foam::reduce(maximum, Foam::maxOp<Foam::scalar>());
        return maximum;
    }
    void advance(const Foam::volVectorField& U, const Foam::surfaceScalarField& phi, const Foam::volScalarField& rho)
    {
        using namespace Foam;
        wallValues();
        const volTensorField gradU(fvc::grad(U));
        const volVectorField gradK(fvc::grad(kinetic)), gradW(fvc::grad(omega)), gradRho(fvc::grad(rho));
        refreshNut(U);
        auto dk = scalarField("DkSST", dimViscosity, nu_, true), dw = scalarField("DwSST", dimViscosity, nu_, true);
        auto sk = scalarField("SkSST", sqr(dimVelocity) / dimTime, 0),
             sw = scalarField("SwSST", dimless / sqr(dimTime), 0);
        auto lk = scalarField("LkSST", dimless / dimTime, 0), lw = scalarField("LwSST", dimless / dimTime, 0);
        const scalar gamma1 = 0.075 / 0.09 - 0.5 * sqr(0.41) / Foam::sqrt(0.09);
        const scalar gamma2 = 0.0828 / 0.09 - 0.856 * sqr(0.41) / Foam::sqrt(0.09);
        const scalar gravity = readScalar(mesh_.lookupObject<IOdictionary>("verificationProperties").lookup("gravity"));
        forAll(kinetic, c)
        {
            const scalar k = kinetic[c], w = omega[c], d = distance[c], n = nut[c], dot = gradK[c] & gradW[c];
            const scalar cd = max(2 * 0.856 * dot / w, scalar(1e-20));
            const scalar arg =
                min(max(Foam::sqrt(k) / (0.09 * w * d), 500 * nu_ / (d * d * w)), 4 * 0.856 * k / (cd * d * d));
            const scalar f1 = Foam::tanh(sqr(sqr(arg))), gamma = f1 * gamma1 + (1 - f1) * gamma2;
            const scalar strain2 = 2 * magSqr(symm(gradU[c])), production = n * strain2;
            const scalar cross = 2 * (1 - f1) * 0.856 * dot / w;
            const scalar gb = n / prt_ * gravity * gradRho[c].z() / rho0_;
            const scalar gw = gamma / (n + std::numeric_limits<double>::epsilon()) * gb;
            dk[c] = nu_ + (f1 * 0.85 + 1 - f1) * n;
            dw[c] = nu_ + (f1 * 0.5 + (1 - f1) * 0.856) * n;
            sk[c] = min(production, 20 * 0.09 * k * w) + max(gb, scalar(0));
            lk[c] = 0.09 * w + max(-gb / k, scalar(0));
            sw[c] = gamma * strain2 + max(cross, scalar(0)) + max(gw, scalar(0));
            lw[c] = (f1 * 0.075 + (1 - f1) * 0.0828) * w + max(-cross / w, scalar(0)) + max(-gw / w, scalar(0));
        }
        dk.correctBoundaryConditions();
        dw.correctBoundaryConditions();
        // Both coefficient/source sets refer to the same old k/omega state.
        const auto kSolve = solve(
            fvm::ddt(kinetic) + fvm::div(phi, kinetic) - fvm::laplacian(dk, kinetic) + fvm::Sp(lk, kinetic) == sk);
        const auto omegaSolve =
            solve(fvm::ddt(omega) + fvm::div(phi, omega) - fvm::laplacian(dw, omega) + fvm::Sp(lw, omega) == sw);
        if (!kSolve.converged() || !omegaSolve.converged())
            FatalErrorInFunction << "SST scalar solve did not converge" << exit(FatalError);
        kinetic.max(dimensionedScalar(kinetic.dimensions(), kMin_));
        omega.max(dimensionedScalar(omega.dimensions(), omegaMin_));
        refreshNut(U);
    }
};
