using Printf

struct PotentialCoefficients
    GM::Float64
    R::Float64
    anm::Array{Float64}
end

function fromarray(coefficient_array, GM, R)

    maximum_degree = Int64(sqrt(length(coefficient_array)) - 1)
    anm = Array{Float64}(undef, maximum_degree+1, maximum_degree+1, 1)

    for n in 0:maximum_degree
        anm[n+1, 1, 1], = coefficient_array[n+1, 1]
        for m in 1:n
            anm[n+1, m+1, 1] = coefficient_array[n+1, m+1]
            anm[(n-m)+1, maximum_degree+2-m, 1] = coefficient_array[m, n+1]
        end
    end
    PotentialCoefficients(GM, R, anm)
end

maximumdegree(coefficients::PotentialCoefficients) = size(coefficients.anm, 1)-1



function Base.:copy(a::PotentialCoefficients)
    PotentialCoefficients(a.GM, a.R, copy(a.anm))
end

function Base.:+(a::PotentialCoefficients, b::PotentialCoefficients)

    if maximumdegree(a) > maximumdegree(b)
        res = copy(a)
        @simd for n in 0:maximumdegree(b)
            factor =  (b.GM/a.GM) * (b.R/a.R)^n
            res.anm[n+1, 1, :] .+= factor * b.anm[n+1, 1, :]
            @simd for m in 1:n
                res.anm[n+1, m+1, :] .+= factor * b.anm[n+1, m+1, :]
                res.anm[n-m+1, maximumdegree(a)+2-m, :] .+= factor * b.anm[n-m+1, maximumdegree(b)+2-m, :]
            end
        end
    else
        res = copy(b)
        @simd for n in 0:maximumdegree(a)
            factor =  (a.GM/b.GM) * (a.R/b.R)^n
            res.anm[n+1, 1, :] .+= factor * a.anm[n+1, 1, :]
            @simd for m in 1:n
                res.anm[n+1, m+1, :] .+= factor * a.anm[n+1, m+1, :]
                res.anm[n-m+1, maximumdegree(b)+2-m, :] .+= factor * a.anm[n-m+1, maximumdegree(a)+2-m, :]
            end
        end
    end
    return res
end

function Base.:*(a::PotentialCoefficients, b::Number)

    res = copy(a)
    res.anm[:,:,:] *= b
    return res
end

Base.:-(a::PotentialCoefficients, b::PotentialCoefficients) =  a+(b*-1)
Base.:/(a::PotentialCoefficients, b::Number) =  a*(1/b)


function Base.:-(a::PotentialCoefficients, b::Constants.ReferenceSystem)
    max_degree = (length(b.coefficients) - 1) * 2
    if maximumdegree(a) >= max_degree
        res = copy(a)
        @simd for k in 1:length(b.coefficients)
            factor =  (b.GM/a.GM) * (b.semimajor_axis/a.R)^(2*k-2)
            res.anm[2*k-1, 1, :] .-= factor * b.coefficients[k]
        end
    else
        anm = zeros(max_degree + 1, max_degree + 1, size(a.anm, 3))
        anm[1:2:max_degree+1, 1, :] = -b.coefficients
        @simd for n in 0:maximumdegree(a)
            factor =  (a.GM/b.GM) * (a.R/b.semimajor_axis)^n
            anm[n+1, 1, :] .+= factor * a.anm[n+1, 1, :]
            @simd for m in 1:n
                anm[n+1, m+1, :] .+= factor * a.anm[n+1, m+1, :]
                anm[n-m+1, max_degree+2-m, :] .+= factor * a.anm[n-m+1, maximumdegree(a)+2-m, :]
            end
        end
        res = PotentialCoefficients(b.GM, b.semimajor_axis, anm)
    end
    return res
end


function truncate(coefficients::PotentialCoefficients, maximum_degree)

    if maximum_degree < maximumdegree(coefficients)

        anm_new = zeros(maximum_degree+1, maximum_degree+1, size(coefficients.anm, 3))
        for m in 0:maximum_degree
            anm_new[m+1:end, m+1, :] = coefficients.anm[m+1:maximum_degree+1, m+1, :]
        end
        for m in 1:maximum_degree
            anm_new[1:maximum_degree+1-m, maximum_degree+2-m, :] = coefficients.anm[1:maximum_degree+1-m, maximumdegree(coefficients)+2-m, :]
        end
        c = PotentialCoefficients(coefficients.GM, coefficients.R, anm_new)
    else
        c = copy(coefficients)
    end
    return c
end

function zerodegrees!(coefficients::PotentialCoefficients, degree_range)
    for n in degree_range
        coefficients.anm[n+1, 1, :] .= 0.0
        for m in 1:n
            coefficients.anm[n+1, m+1, :] .= 0.0
            coefficients.anm[(n-m)+1, maximumdegree(coefficients)+2-m, :] .= 0.0
        end
    end
end

function Base.:display(a::PotentialCoefficients)
    println("GM ", a.GM)
    println("R  ", a.R)
    @printf "%4s%4s%20s%20s\n" "L" "M" "C" "S"
    max_degree = maximumdegree(a)
    if max_degree < 8
        for n in 0:maximumdegree(a)
            @printf "%4i%4i%24.15e%24.15e\n" n 0 a.anm[n+1, 1] 0.0
            for m in 1:n
                @printf "%4i%4i%24.15e%24.15e\n" n m a.anm[n+1, m+1] a.anm[(n-m)+1, maximumdegree(a)+2-m]
            end
        end
    else
        for n in 0:3
            @printf "%4i%4i%24.15e%24.15e\n" n 0 a.anm[n+1, 1] 0.0
            for m in 1:n
                @printf "%4i%4i%24.15e%24.15e\n" n m a.anm[n+1, m+1] a.anm[(n-m)+1, maximumdegree(a)+2-m]
            end
        end
        @printf "%4s%4s%20s%20s\n" "\u22ee" "\u22ee" "\u22ee" "\u22ee"
        for n in max_degree:max_degree
            for m in max_degree-7:max_degree
                @printf "%4i%4i%24.15e%24.15e\n" n m a.anm[n+1, m+1] a.anm[(n-m)+1, maximumdegree(a)+2-m]
            end
        end
    end
end

function applyinversekernel!(c::PotentialCoefficients, k::IsotropicKernel)

    kn = inverse_coefficients(k, 0, maximumdegree(c), c.GM, c.R)
    for n in 0:maximumdegree(c)
        c.anm[n+1, 1, :] *= kn[n+1]
        for m in 1:n
            c.anm[n+1, m+1, :] *= kn[n+1]
            c.anm[(n-m)+1, maximumdegree(c)+2-m, :] *= kn[n+1]
        end
    end
    return c
end

function applyinversekernel(c::PotentialCoefficients, k::IsotropicKernel)
    c2 = copy(c)
    applyinversekernel!(c2, k)
end
