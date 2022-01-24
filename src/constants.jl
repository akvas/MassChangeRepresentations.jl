module Constants

using MassChangeRepresentations

const lovenumbers_ce, lovenumbers_cm = MassChangeRepresentations.import_love_numbers()

function loadlovenumber(a::MassChangeRepresentations.LoveNumbers, degree)
    if degree + 2 <= length(a.kn)
        return a.kn[degree+1]
    else
        return a.kn[end]
    end
end

function deformationlovenumbers(a::MassChangeRepresentations.LoveNumbers, degree)
    if degree + 2 <= length(a.kn)
        return a.hn[degree+1], a.ln[degree+1]
    else
        return a.hn[end], a.ln[end]
    end
end

function loadlovenumber(system::AbstractString, degree)
    if system == "CM"
        return loadlovenumber(lovenumbers_cm, degree)
    elseif system == "CE"
        return loadlovenumber(lovenumbers_ce, degree)
    end
end

function deformationlovenumbers(system::AbstractString, degree)
    if system == "CM"
        return deformationlovenumbers(lovenumbers_cm, degree)
    elseif system == "CE"
        return deformationlovenumbers(lovenumbers_ce, degree)
    end
end

struct ReferenceSystem
    GM::Float64
    semimajor_axis::Float64
    J2::Float64
    omega::Float64
    flattening::Float64
    coefficients::Vector{Float64}
end

function createreferencesystem(GM, omega, a; f=nothing, J2=nothing)

    if !xor(isnothing(f), isnothing(J2))
        throw(ArgumentError("Either J2 or f must be set."))
    end

    if isnothing(J2)
        e2 = f * (2 - f)
        e = sqrt(e2)
        e_prime = e / sqrt(1 - e2)

        n = collect(1:20)
        q0 = -2 * sum((-1).^n .* n .* e_prime.^(2 .*n.+1) ./ ((2 .*n.+1).*(2 .*n.+3)))
        J2 = (e2 - 4/15 * (omega^2*a^3)/GM*e^3/(2*q0))/3
    elseif isnothing(f)
        e = 0.1
        e0 = Inf64

        n = collect(1:20)
        while abs(e-e0) > 1e-16
            e0 = e
            e_prime = e / sqrt(1 - e^2)
            q0 = -2 * sum((-1).^n .* n .* e_prime.^(2 .*n.+1) ./ ((2 .*n.+1).*(2 .*n.+3)))
            e = sqrt(3 * J2 + 4 / 15 * (omega^2 * a^3) / GM * e^3 / (2 * q0))
        end
        e2 = e^2
        f = 1 - sqrt(1-e2)
    end

    coefficients = [1.0]
    n = 1
    while abs(coefficients[end]) > 1e-24
        factor = (n % 2 == 0) ? 1 : -1
        c2n = factor * (3 * e2^n * (1 - n + 5 * n * J2 / e2) / ((2 * n + 1) * (2 * n + 3) * sqrt(4 * n + 1)))
        append!(coefficients, c2n)
        n += 1
    end

    ReferenceSystem(GM, a, J2, omega, f, coefficients)
end

const GRS80 = createreferencesystem(3986005e8, 7292115.0e-11, 6378137.0, J2=108263e-8)
const WGS84 = createreferencesystem(3986004.418e8, 7292115.0e-11, 6378137.0, f=1/298.257223563)

module IERS2010

const c = 299792458.0

const G = 6.67428e-11

const GM = 3.986004418e14

const R = 6378136.6

end

end