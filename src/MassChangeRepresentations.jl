module MassChangeRepresentations

include("data.jl")
include("spherical_harmonics.jl")
include("grids.jl")
const lovenumbers_ce, lovenumbers_cm = import_love_numbers()

end # module
