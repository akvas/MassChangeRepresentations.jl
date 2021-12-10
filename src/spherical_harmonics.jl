

function recursionfactors(maximum_degree)

    fnm = Vector{Float64}(undef, (maximum_degree + 1) * (maximum_degree + 1))
    # order 0
    fnm[1] = 0
    for n in 1:maximum_degree
        fnm[2*n] = sqrt((2*n+1)/(n*n)*(2*n-1))
        fnm[2*n + 1] = -sqrt((2*n+1)/((n*n))*(n-1)*(n-1)/(2*n-3))
    end

    # order 1
    fnm[2*maximum_degree+2] = sqrt(3)
    for n in 2:maximum_degree
        fnm[(2*maximum_degree-1)+2*n] = sqrt((2*n+1)/((n+1)*(n-1))*(2*n-1))
        fnm[(2*maximum_degree-1)+2*n+1] = -sqrt((2*n+1)/((n+1)*(n-1))*(n-2)*(n)/(2*n-3))
    end

    # other orders
    for m in 2:maximum_degree-1
        fnm[(2*maximum_degree+2-m)*m+1] = sqrt((2 * m + 1) / (2 * m))
        for n in m+1:maximum_degree
            fnm[(2*maximum_degree-m)*m+2*n] = sqrt((2*n+1)/((n+m)*(n-m))*(2*n-1))
            fnm[(2*maximum_degree-m)*m+2*n+1] = -sqrt((2*n+1)/((n+m)*(n-m))*(n-m-1)*(n+m-1)/(2*n-3))
        end
    end
    # last order
    fnm[end] =  sqrt((2 * maximum_degree + 1) / (2 * maximum_degree))

    return fnm
end

function legendrefunctions!(p, factors, reference_radius, Pnm)

    maximum_degree = Int64(sqrt(length(factors)) - 1)
    rr_squared = (reference_radius * reference_radius) / (p[1]*p[1] + p[2]*p[2] + p[3]*p[3])
    p_norm = p.*(rr_squared / reference_radius)

    # order 0
    Pnm[1] = 1
    Pnm[2] = factors[2] * p_norm[3]
    for n in 2:maximum_degree+1
        Pnm[n + 1] = factors[2 * n] * p_norm[3] * Pnm[n] + factors[2 * n + 1] * rr_squared * Pnm[n - 1]
    end
    # initialize diagonal for order 1
    Pnm[maximum_degree+2] = factors[(2*maximum_degree+1)+1] * p_norm[1] * Pnm[1] # P_m,m = r(P_m-1,m-1)
    for m in 1:maximum_degree-1
        i1 = Int64((2*maximum_degree+3-m)*m/2+2) # index of P_m+1,m
        i2 = (2*maximum_degree-m)*m+2*m+2 # index of g_m+1,m
        # subdiagonal
        Pnm[i1] = factors[i2] * p_norm[3] * Pnm[i1 - 1]   # C_m+1,m = r(C_m,m)

        # orders m+2:maximum_degree
        for k in 1:maximum_degree-m-1
             Pnm[i1+k] = factors[i2+2*k] * p_norm[3] * Pnm[i1+k-1] + factors[i2+1+2*k]*rr_squared*Pnm[i1+k-2]  # C_n,m = r(C_n-1,m,S_n-1,m)
        end
        # initialize diagonal for next order
        Pnm[i1 + maximum_degree - m] = factors[i2 + 2*(maximum_degree-m)] * p_norm[1] * Pnm[i1-1] # C_m+1,m+1 = r(C_m,m,S_m,m)
    end

    return Pnm
end

function legendrefunctions(p, factor, reference_radius)
    Pnm = Vector{Float64}(undef,Int64(length(factors)*(sqrt(length(factors)) + 1)/2))
    legendrefunctions!(p, factor, reference_radius, Pnm)
end

function legendrefunctions!(t::Number, factors, Pnm)
    p = [sqrt(1.0-t*t), 0.0, t]
    return legendrefunctions!(p, factors, 1.0, Pnm)
end


function legendrefunctions(t::Number, factors)
    Pnm = Vector{Float64}(undef,Int64(length(factors)*(sqrt(length(factors)) + 1)/2))
    legendrefunctions!(t, factors, Pnm)
end

function sphericalharmonics!(p, factors, reference_radius, Ynm)

    maximum_degree = Int64(sqrt(length(factors)) - 1)
    rr_squared = (reference_radius * reference_radius) / (p[1]*p[1] + p[2]*p[2] + p[3]*p[3])
    p_norm = p.*(rr_squared / reference_radius)

    # order 0, only Cnm
    @inbounds Ynm[1] = 1
    @inbounds Ynm[2] = factors[2] * p_norm[3]
    for n in 2:maximum_degree+1
        @inbounds Ynm[n + 1] = factors[2 * n] * p_norm[3] * Ynm[n] + factors[2 * n + 1] * rr_squared * Ynm[n - 1]
    end
    # initialize diagonal for order 1
    @inbounds Ynm[maximum_degree+2] = factors[(2*maximum_degree+1)+1] * p_norm[1] * Ynm[1] # C_m,m = r(C_m-1,m-1)
    @inbounds Ynm[maximum_degree+3] = factors[(2*maximum_degree+1)+1] * p_norm[2] * Ynm[1] # S_m,m = r(C_m-1,m-1)

    for m in 1:maximum_degree-1
        i1 = 2*m*(maximum_degree-1)-maximum_degree+2-(m-2)*(m-1)+2*m+2 # index C_m+1,m
        i2 = (2*maximum_degree-m)*m+2*m+2 # index of g_m+1,m
        # subdiagonal
        @inbounds Ynm[i1] = factors[i2] * p_norm[3] * Ynm[i1 - 2]   # C_m+1,m = r(C_m,m)
        @inbounds Ynm[i1 + 1] = factors[i2] * p_norm[3] * Ynm[i1 - 1]   # C_m+1,m = r(C_m,m)

        # orders m+2:maximum_degree
        for k in 1:maximum_degree-m-1
            @inbounds Ynm[i1+2*k] = factors[i2+2*k] * p_norm[3] * Ynm[i1-2+2*k] + factors[i2+1+2*k]*rr_squared*Ynm[i1-4+2*k]  # C_n,m = r(C_n-1,m,S_n-1,m)
            @inbounds Ynm[i1+2*k+1] = factors[i2+2*k] * p_norm[3] * Ynm[i1-1+2*k] + factors[i2+1+2*k]*rr_squared*Ynm[i1-3+2*k]  # C_n,m = r(C_n-1,m,S_n-1,m)
        end

        # initialize diagonal for next order
        @inbounds Ynm[i1 + 2*(maximum_degree-m)] = factors[i2 + 2*(maximum_degree-m)] * (p_norm[1]*Ynm[i1-2]-p_norm[2]*Ynm[i1-1]) # C_m+1,m+1 = r(C_m,m,S_m,m)
        @inbounds Ynm[i1 + 2*(maximum_degree-m)+1] = factors[i2 + 2*(maximum_degree-m)] * (p_norm[2]*Ynm[i1-2]+p_norm[1]*Ynm[i1-1]) # S_m+1,m+1 = r(C_m,m,S_m,m)
    end

    return Ynm
end

function sphericalharmonics(p, factors, reference_radius)
    Ynm = Vector{Float64}(undef, length(factors))
    sphericalharmonics!(p, factors, reference_radius, Ynm)
end

function summationfactors(maximum_degree)

    factors = Vector{Float64}(undef, 2*(maximum_degree+1))
    @simd for k in maximum_degree+1:-1:2
        factors[2*k] = sqrt((2 * k - 1) * (2 * k + 1)) / k
        factors[2*k-1] = -sqrt((2 * k + 3) / (2 * k - 1)) * k / (k + 1)
    end

    return factors
end

function legendre_summation(t, factors, coefficients)

    b2 = 0.0
    b1 = 0.0
    for k in length(coefficients):-1:2
        bk = coefficients[k] + factors[2*k] * t * b1 + factors[2*k-1] * b2
        b2 = b1
        b1 = bk
    end

    coefficients[1] + sqrt(3) * t * b1 - 0.5 * sqrt(5) * b2
end