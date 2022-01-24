import GZip

struct LoveNumbers
    kn::Vector{Float64}
    hn::Vector{Float64}
    ln::Vector{Float64}
end

datadir = joinpath(dirname(pathof(MassChangeRepresentations)), "..", "data")

function import_love_numbers()
    fh = GZip.open(joinpath(datadir, "ak135-LLNs-complete.dat.gz"))
    maximum_degree = 46341
    readline(fh)
    kn = Vector{Float64}(undef, maximum_degree+1)
    hn = Vector{Float64}(undef, maximum_degree+1)
    ln = Vector{Float64}(undef, maximum_degree+1)
    kn[1] = 0
    hn[1] = 0
    ln[1] = 0
    for n in 1:maximum_degree
        line = readline(fh)
        hn[n+1] = parse(Float64, split(line)[2])
        ln[n+1] = parse(Float64, split(line)[3])
        kn[n+1] = parse(Float64, split(line)[4])
    end
    close(fh)
    lln_ce = LoveNumbers(kn,hn,ln)
    lln_cm = LoveNumbers(copy(kn),copy(hn),copy(ln))
    lln_cm.kn[2] = -1
    lln_cm.hn[2] = -1
    lln_cm.ln[2] = -1

    lln_ce, lln_cm
end
