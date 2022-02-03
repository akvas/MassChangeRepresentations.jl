
export recursionfactors, legendrefunctions!, legendrefunctions, sphericalharmonics!, sphericalharmonics

"""
    recursionfactors(maximum_degree)

Compute recursion factors for associated Legendre functions and spherical harmonics.

The factors are stored in a matrix of size ``n_\\text{max}+1 \\times n_\\text{max}+1``.

"""
function recursionfactors(maximum_degree)

    F = Matrix{Float64}(undef, maximum_degree+1, maximum_degree+1)

    # order 0
    F[1, 1] = 0
    for n in 1:maximum_degree
        F[n+1, 1] = sqrt((2*n+1)/(n*n)*(2*n-1))
        F[n, maximum_degree + 1] = -sqrt((2*n+1)/((n*n))*(n-1)*(n-1)/(2*n-3))
    end

    # order 1
    F[2, 2] = sqrt(3)
    for n in 2:maximum_degree
        F[n+1, 2] = sqrt((2*n+1)/((n+1)*(n-1))*(2*n-1))
        F[n-1, maximum_degree] = -sqrt((2*n+1)/((n+1)*(n-1))*(n-2)*(n)/(2*n-3))
    end

     # other orders
    for m in 2:maximum_degree
        F[m+1,m+1] = sqrt((2 * m + 1) / (2 * m))
        for n in m+1:maximum_degree
            F[n+1, m+1] = sqrt((2*n+1)/((n+m)*(n-m))*(2*n-1))
            F[n-m, maximum_degree+1-m] = -sqrt((2*n+1)/((n+m)*(n-m))*(n-m-1)*(n+m-1)/(2*n-3))
        end
    end
    return F
end

function legendrefunctions!(p, F, reference_radius, Pnm; regular=false)

    maximum_degree = size(F, 1) - 1
    if regular
        rr_squared = (p[1]*p[1] + p[2]*p[2] + p[3]*p[3]) / (reference_radius * reference_radius)
        p_norm = p./reference_radius
        @inbounds Pnm[1, 1] = 1e280
    else
        rr_squared = (reference_radius * reference_radius) / (p[1]*p[1] + p[2]*p[2] + p[3]*p[3])
        p_norm = p.*(rr_squared / reference_radius)
        @inbounds Pnm[1, 1] = sqrt(rr_squared) * 1e280
    end

    # order 0
    @inbounds Pnm[2, 1] = F[2, 1] * p_norm[3] * Pnm[1, 1]
    for n in 2:maximum_degree
        @inbounds Pnm[n + 1, 1] = F[n+1, 1] * p_norm[3] * Pnm[n, 1] + F[n, end] * rr_squared * Pnm[n - 1, 1]
    end
    # initialize diagonal for order 1
    Pnm[2, 2] = F[2, 2] * p_norm[1] * Pnm[1, 1] # P_m,m = r(P_m-1,m-1)
    for m in 1:maximum_degree-1
        # subdiagonal
        @inbounds Pnm[m+2, m+1] = F[m+2,m+1] * p_norm[3] * Pnm[m+1,m+1]   # C_m+1,m = r(C_m,m)

        # orders m+2:maximum_degree
        for n in m+2:maximum_degree
            @inbounds Pnm[n+1,m+1] = F[n+1, m+1] * p_norm[3] * Pnm[n, m+1] + F[(n-m), maximum_degree+1-m]*rr_squared*Pnm[n-1, m+1]  # C_n,m = r(C_n-1,m,S_n-1,m)
        end
        # initialize diagonal for next order
        @inbounds Pnm[m+2,m+2] = F[m+2,m+2] * p_norm[1] * Pnm[m+1,m+1] # C_m+1,m+1 = r(C_m,m,S_m,m)
    end
    Pnm .*= 1e-280
    return Pnm
end

function sphericalharmonics!(p, F, reference_radius, Ynm; regular=false)

    maximum_degree = size(F,1)-1
    if regular
        rr_squared = (p[1]*p[1] + p[2]*p[2] + p[3]*p[3]) / (reference_radius * reference_radius)
        p_norm = p./reference_radius
        Ynm[1, 1] = 1e280
    else
        rr_squared = (reference_radius * reference_radius) / (p[1]*p[1] + p[2]*p[2] + p[3]*p[3])
        p_norm = p.*(rr_squared / reference_radius)
        Ynm[1, 1] = sqrt(rr_squared) * 1e280
    end

    # order 0, only Cnm
    Ynm[2, 1] = F[2, 1] * p_norm[3] * Ynm[1, 1]
    for n in 2:maximum_degree
        Ynm[n + 1, 1] = F[n+1, 1] * p_norm[3] * Ynm[n, 1] + F[n, end] * rr_squared * Ynm[n - 1, 1]
    end
    # initialize diagonal for order 1
    Ynm[2, 2] = F[2, 2] * p_norm[1] * Ynm[1, 1] # C_m,m = r(C_m-1,m-1)
    Ynm[1, maximum_degree+1] = F[2, 2] * p_norm[2] * Ynm[1, 1] # S_m,m = r(C_m-1,m-1)

    for m in 1:maximum_degree-1
        # subdiagonal
        Ynm[m+2, m+1]              = F[m+2,m+1] * p_norm[3] * Ynm[m+1,m+1]   # C_m+1,m = r(C_m,m)
        Ynm[2, maximum_degree+2-m] = F[m+2,m+1] * p_norm[3] * Ynm[1, maximum_degree+2-m]   # S_m+1,m = r(S_m,m)

        # orders m+2:maximum_degree
        for n in m+2:maximum_degree
           Ynm[n+1, m+1]                  = F[n+1, m+1] * p_norm[3] * Ynm[n, m+1]               + F[n-m, maximum_degree+1-m]*rr_squared*Ynm[n-1,m+1]  # C_n,m = r(C_n-1,m,S_n-1,m)
           Ynm[n-m+1, maximum_degree+2-m] = F[n+1, m+1] * p_norm[3] * Ynm[n-m,maximum_degree+2-m] + F[n-m, maximum_degree+1-m]*rr_squared*Ynm[n-m-1,maximum_degree+2-m]  # C_n,m = r(C_n-1,m,S_n-1,m)
           #Ynm[n+1,m+1] = 10
        end

        # initialize diagonal for next order
        Ynm[m+2,m+2]               = F[m+2,m+2] * (p_norm[1]*Ynm[m+1,m+1]-p_norm[2]*Ynm[1,maximum_degree+2-m]) # C_m+1,m+1 = r(C_m,m,S_m,m)
        Ynm[1, maximum_degree+1-m] = F[m+2,m+2] * (p_norm[2]*Ynm[m+1,m+1]+p_norm[1]*Ynm[1,maximum_degree+2-m]) # S_m+1,m+1 = r(C_m,m,S_m,m)
    end
    Ynm .*= 1e-280
    return Ynm
end


function legendrefunctions(p, factors, reference_radius)
    Pnm = Vector{Float64}(undef,Int64(length(factors)*(sqrt(length(factors)) + 1)/2))
    legendrefunctions!(p, factors, reference_radius, Pnm)
end

function legendrefunctions!(t::Number, factors, Pnm)
    p = [sqrt(1.0-t*t), 0.0, t]
    return legendrefunctions!(p, factors, 1.0, Pnm)
end

function legendrefunctions(t::Number, factors)
    Pnm = Matrix{Float64}(undef, size(factors))
    legendrefunctions!(t, factors, Pnm)
end

function sphericalharmonics(p, factors, reference_radius)
    Ynm = Matrix{Float64}(undef, size(factors))
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

function synthesis(grid::Grid, coefficients)
    pts = points(grid)

    factors = recursionfactors(maximumdegree(coefficients))

    Ynm = Matrix{Float64}(undef, maximumdegree(coefficients)+1, maximumdegree(coefficients)+1)
    output = zeros(size(coefficients.anm, 3), length(pts))
    for k in 1:length(pts)
        sphericalharmonics!(pts[k], factors, coefficients.R, Ynm)
        output[:, k] = sum(coefficients.anm .* Ynm, dims=(1, 2))[:]
    end
    return transpose(output*coefficients.GM/coefficients.R)
end

function synthesis(grid::RegularGrid, coefficients)

    ll = LonLat.(0, grid.parallels)
    pts = geodetic2point.(ll, 0, grid.semimajoraxis, grid.flattening)

    maximum_degree = maximumdegree(coefficients)
    factors = recursionfactors(maximum_degree)

    cs = Matrix{Float64}(undef, length(grid.meridians), 2*maximum_degree+1)
    cs[:, 1] .= 1.0
    for m in 1:maximum_degree
        cs[:, 2*m] .= cos.(m*grid.meridians)
        cs[:, 2*m+1] .= sin.(m*grid.meridians)
    end

    Pnm = Matrix{Float64}(undef, maximum_degree+1, maximum_degree+1)
    t = Matrix{Float64}(undef, size(coefficients.anm, 3), 2*maximum_degree+1)
    output = Matrix{Float64}(undef, pointcount(grid), size(coefficients.anm, 3))
    for k in 1:length(pts)
        legendrefunctions!(pts[k], factors, coefficients.R, Pnm)

        t[:, 1] = transpose(Pnm[:, 1]) * coefficients.anm[:, 1, :]
        for m in 1:maximum_degree
            t[:, 2*m] .= 0.0
            t[:, 2*m+1] .= 0.0

            @views LinearAlgebra.BLAS.gemv!('T', 1.0, coefficients.anm[m+1:end,m+1,:], Pnm[m+1:end,m+1], 1.0, t[:, 2*m])
            @views LinearAlgebra.BLAS.gemv!('T', 1.0, coefficients.anm[1:maximum_degree+1-m,maximum_degree+2-m,:], Pnm[m+1:end,m+1], 1.0, t[:, 2*m+1])
        end
        @views output[(k-1)*length(grid.meridians)+1:k*length(grid.meridians), :] = cs * transpose(t)
    end
    return output * coefficients.GM / coefficients.R
end

function quadrature(grid::Grid, values, maximum_degree, GM, R)
    pts = points(grid)

    areas = areaweights(grid)./ (GM * 4*pi).*R
    anm = zeros(maximum_degree+1, maximum_degree+1, size(values, 2))
    factors = recursionfactors(maximum_degree)
    Ynm = Matrix{Float64}(undef, maximum_degree+1,maximum_degree+1)
    for k in 1:length(pts)
        r = sqrt(pts[k][1]*pts[k][1] + pts[k][2]*pts[k][2] + pts[k][3]*pts[k][3])
        sphericalharmonics!(pts[k].*(R/r), factors, r, Ynm, regular=false)
        for l in 1:size(values, 2)
            anm[:, :, l] += Ynm * areas[k] * values[k, l]
        end
    end
    PotentialCoefficients(GM, R, anm)
end

function quadrature(grid::RegularGrid, values, maximum_degree, GM, R)
    ll = LonLat.(0, grid.parallels)
    pts = geodetic2point.(ll, 0, grid.semimajoraxis, grid.flattening)

    cs = Matrix{Float64}(undef, length(grid.meridians), 2*maximum_degree+1)
    cs[:, 1] .= 1.0
    for m in 1:maximum_degree
        cs[:, 2*m] .= cos.(m*grid.meridians)
        cs[:, 2*m+1] .= sin.(m*grid.meridians)
    end

    areas = areaweights(grid)./(GM*4*pi).*R

    anm = zeros(maximum_degree+1, maximum_degree+1, size(values, 2))
    factors = recursionfactors(maximum_degree)
    Pnm = Matrix{Float64}(undef, maximum_degree+1,maximum_degree+1)
    for k in 1:length(pts)
        r = sqrt(pts[k][1]*pts[k][1] + pts[k][2]*pts[k][2] + pts[k][3]*pts[k][3])
        legendrefunctions!(pts[k].*(R/r), factors, r, Pnm, regular=false)

        t = transpose(values[(k-1)*length(grid.meridians)+1:k*length(grid.meridians), :].*areas[(k-1)*length(grid.meridians)+1:k*length(grid.meridians)]) * cs
        anm[1:end, 1, :] += Pnm[1:end, 1] .* t[:, 1]
        for m in 1:maximum_degree
            anm[m+1:end, m+1, :] += Pnm[m+1:end, m+1, :] .* t[:, 2*m]
            anm[1:maximum_degree+1-m, maximum_degree+2-m, :] += Pnm[m+1:end, m+1, :] .* t[:, 2*m+1]
        end

    end
    PotentialCoefficients(GM, R, anm)
end


function analysis(grid::Grid, values, maximum_degree, GM, R, max_iteration=10, threshold=1e-23)

    c = quadrature(grid, values, maximum_degree, GM, R)
    for _ in 1:max_iteration
        d = quadrature(grid, values - synthesis(grid, c), maximum_degree, GM, R)
        if maximum(abs.(d.anm)) < threshold
            break
        end
        c.anm .+= d.anm
    end
    PotentialCoefficients(GM, R, c.anm)
end
