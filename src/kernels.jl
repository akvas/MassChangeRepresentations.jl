

abstract type Kernel end

abstract type IsotropicKernel <: Kernel end
abstract type AnisotropicKernel <: Kernel end

struct Potential <: IsotropicKernel end

struct SurfaceDensity <: IsotropicKernel end

function coefficients(kernel::Potential, minimum_degree, maximum_degree)
    kn = ones(maximum_degree + 1)
    kn[1:minimum_degree] .= 0
    return kn
end

function coefficients(kernel::SurfaceDensity, minimum_degree, maximum_degree; system="CE")
    kn = Vector{Float64}(undef, maximum_degree + 1)
    kn[1:minimum_degree] .= 0
    for n in minimum_degree:maximum_degree
        kn[n + 1] = (1 + Constants.loadlovenumber(system, n)) / (2 * n + 1) * (4 * pi * Constants.IERS2010.G)
    end
    return kn
end

function inverse_coefficients(kernel::IsotropicKernel, minimum_degree, maximum_degree)
    kn = coefficients(kernel, minimum_degree, maximum_degree)
    for n in minimum_degree:maximum_degree
        kn[n + 1] = kn[n + 1] == 0.0 ? 0.0 : 1 / kn[n + 1]
    end
    return kn
end

function evaluate(kernel::IsotropicKernel, p, q, minimum_degree, maximum_degree)
    kn = coefficients(kernel, minimum_degree, maximum_degree) .* sqrt.(2.0.*(0:maximum_degree).+1.0)
    factors = summationfactors(maximum_degree)
    legendre_summation(cosangle(p, q), factors, kn)
end

function evaluate(kernel::IsotropicKernel, psi::Number, minimum_degree, maximum_degree)
    kn = coefficients(kernel, minimum_degree, maximum_degree) .* sqrt.(2.0.*(0:maximum_degree).+1.0)
    factors = summationfactors(maximum_degree)
    legendre_summation(cos(psi), factors, kn)
end


