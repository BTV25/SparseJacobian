module SparseJacobian

import SNOW
import Snopt
import DelimitedFiles
import ForwardDiff
import FLOWFarm; const ff = FLOWFarm
import SparseArrays
import LinearAlgebra
import SparsityDetection, SparseArrays, SparseDiffTools
import FLOWMath: gradient
import LinearAlgebra
import Colors
import Traceur

## Derivative calculation
function sparseAEPJacobian!(x::Vector{Float64},params,deriv)
    deriv.Jacobian .= 0
    for i = 1:length(params.windresource.wind_directions)
        # calculate pattern
        calculateSparsityPattern!(x,deriv.pattern,params,i,deriv.sorted_index)

        if !isequal(deriv.pattern,deriv.patterns[:,:,i])
        # if !isequal(sum(x->x>0,deriv.pattern),sum(x->x>0,deriv.patterns[:,:,i]))
            deriv.patterns[:,:,i] .= deriv.pattern
            allocateNewCache!(x,deriv,i)
        end

        # calculate state Jacobian
        calculateSparseJacobian!(x,deriv,i)

        # sum jacobian columns and add to other states
        deriv.Jacobian .= deriv.Jacobian .+ transpose(sum(deriv.jac,dims = 1) .* params.windresource.wind_probabilities[i])
    end
    deriv.Jacobian .= deriv.Jacobian .* params.obj_scale .* 365.25 .* 24
end

function calculateSparseJacobian!(x::Vector{Float64},deriv,currentState::Int64)
    deriv.jac .= sparse(deriv.patterns[:,:,currentState])
    if iszero(deriv.jac)
        return nothing
    end

    forwarddiff_color_jacobian!(deriv.jac,
                            deriv.functions[currentState],
                            x,
                            deriv.caches[currentState])
end

function calculateSparsityPattern!(x::Vector{Float64},pattern::Matrix{Float64},params,currentState::Int64,sorted_index::Vector{Int64})
    n = Int(length(x)/2)
    rot_x, rot_y = ff.rotate_to_wind_direction(x[1:n],x[n+1:end],params.windresource.wind_directions[currentState])::Tuple{Vector{Float64}, Vector{Float64}}
    rot_x .= rot_x ./ params.rotor_diameter
    rot_y .= rot_y ./ params.rotor_diameter
    pattern .= 0
    sorted_index .= sortperm(rot_x)::Vector{Int64}
    d = 4.5
    
    # determine if current turbine is affected by the other turbines
    for i = 1:Int(length(x)/2)
        xCurrent = rot_x[i]
        yCurrent = rot_y[i]
        for j = 1:Int(length(x)/2)
            jt = sorted_index[j]
            xOther = rot_x[jt]
            xdif = xCurrent - xOther
            if i == jt
                pattern[i,jt+n] = 1.0
                pattern[i,jt] = 1.0
                continue
            elseif xdif < 0
                break
            end

            yOther = rot_y[jt]
            if yOther < yCurrent
                yOther = 2*yCurrent - yOther
            end

            ydif = yOther - yCurrent

            if ydif < d
                pattern[i,jt+n] = 1.0
                pattern[i,jt] = 1.0
                continue
            elseif ydif > (d + xdif*tand(15)) #15
                continue
            else
                pattern[i,jt+n] = 1.0
                pattern[i,jt] = 1.0
            end
        end
    end
end

mutable struct deriv_struct{MF, SMFI, VF, AF3, AA, VI}
    pattern::MF
    jac::SMFI
    Jacobian::VF
    patterns::AF3
    caches::AA
    sorted_index::VI
    functions::AA
end

function allocateNewCache!(x::Vector{Float64},deriv,currentState::Int64)
    # global newCaches = newCaches+1
    deriv.jac .= dropzeros(sparse(deriv.patterns[:,:,currentState]))
    if iszero(deriv.jac)
        colors = collect(Int64,1:length(colors))
    else
        colors = matrix_colors(deriv.jac)
    end
    deriv.caches[currentState] = ForwardColorJacCache(deriv.functions[currentState],x,
                              dx = similar(x[1:Int(length(x)/2)]),
                              colorvec=colors,
                              sparsity = deriv.jac)
end

function allocateContainers(x::Vector{Float64},params)
    n = Int(length(x)/2)
    numStates = length(params.windresource.wind_directions)
    pattern = zeros(n,n*2)
    jac = spzeros(n,n*2)
    Jacobian = zeros(n*2)
    patterns = zeros(n,n*2,numStates)
    caches = Array{Any,1}(undef,numStates)
    functions = Array{Any,1}(undef,numStates)
    colors = zeros(Int,n*2)
    index = zeros(Int,n)

    for i = 1:numStates
        calculateSparsityPattern!(x,pattern,params,i,index)
        patterns[:,:,i] .= pattern
        jac .= dropzeros(sparse(patterns[:,:,i]))
        if iszero(jac)
            colors = collect(Int64,1:length(colors))
        else
            colors = matrix_colors(jac)
        end
        f = (wt_power,x) -> p_wrapper!(wt_power,x,params,i)
        functions[i] = f
        caches[i] = ForwardColorJacCache(f,x,
                              dx = similar(x[1:n]),
                              colorvec=colors,
                              sparsity = jac)
    end
    deriv = deriv_struct(pattern,jac,Jacobian,patterns,caches,index,functions)
    return deriv
end

## Optimization calculations
function opt!(g,df,dg,x)
    # calculate spacing constraint value and jacobian
    spacing_con = spacing_wrapper(x)

    # calculate boundary constraint and jacobian
    boundary_con = boundary_wrapper(x)

    # combine constaint values and jacobians into overall constaint value and jacobian arrays
    c = [spacing_con; boundary_con]
    g[:] = c[:]

    # calculate the objective function and jacobian (negative sign in order to maximize AEP)
    AEP = -aep_wrapper(x)[1]

    df[:] = -ForwardDiff.jacobian(aep_wrapper,x)
    ds_dx = ForwardDiff.jacobian(spacing_wrapper, x)
    db_dx = ForwardDiff.jacobian(boundary_wrapper, x)
    dcdx = [ds_dx; db_dx]
    dg[:] = dcdx[:]

    return AEP
end

function opt_fast!(g,df,dg,x)
    # calculate spacing constraint value and jacobian
    spacing_con = spacing_wrapper(x)

    # calculate boundary constraint and jacobian
    boundary_con = boundary_wrapper(x)

    # combine constaint values and jacobians into overall constaint value and jacobian arrays
    c = [spacing_con; boundary_con]
    g[:] = c[:]

    # calculate the objective function and jacobian (negative sign in order to maximize AEP)
    AEP = -aep_wrapper(x)[1]

    sparseAEPJacobian!(x,params,deriv)

    df[:] = -deriv.Jacobian
    ds_dx = ForwardDiff.jacobian(spacing_wrapper, x)
    db_dx = ForwardDiff.jacobian(boundary_wrapper, x)
    dcdx = [ds_dx; db_dx]
    dg[:] = dcdx[:]

    return AEP
end

## Farm setup
struct params_struct{MS, AF, F, AI, ACTM, WR, APM, SPAF, SPAF2D}
    model_set::MS
    rotor_points_y::AF
    rotor_points_z::AF
    turbine_z::AF
    rotor_diameter::AF
    boundary_center::AF
    boundary_radius::F
    obj_scale::F
    hub_height::AF
    turbine_yaw::AF
    ct_models::ACTM
    generator_efficiency::AF
    cut_in_speed::AF
    cut_out_speed::AF
    rated_speed::AF
    rated_power::AF
    windresource::WR
    power_models::APM
    i::AI
    dx::SPAF
    jac::SPAF2D
    color::SPAF
end

# set up boundary constraint wrapper function
function boundary_wrapper(x, params)

    # get number of turbines
    nturbines = Int(length(x)/2)
    
    # extract x and y locations of turbines from design variables vector
    turbine_x = x[1:nturbines]
    turbine_y = x[nturbines+1:end]

    # get and return boundary distances
    return ff.circle_boundary(params.boundary_center, params.boundary_radius, turbine_x, turbine_y)
end

# set up spacing constraint wrapper function
function spacing_wrapper(x, params)
    
    # get number of turbines
    nturbines = Int(length(x)/2)

    # extract x and y locations of turbines from design variables vector
    turbine_x = x[1:nturbines]
    turbine_y = x[nturbines+1:end]

    # get and return spacing distances
    return 2.0*params.rotor_diameter[1] .- ff.turbine_spacing(turbine_x,turbine_y)
end

# set up objective wrapper function
function aep_wrapper(x, params)

    # get number of turbines
    nturbines = Int(length(x)/2)

    # extract x and y locations of turbines from design variables vector
    turbine_x = x[1:nturbines] 
    turbine_y = x[nturbines+1:end]

    # calculate AEP
    obj_scale = params.obj_scale
    AEP = obj_scale*ff.calculate_aep(turbine_x, turbine_y, params.turbine_z, params.rotor_diameter,
                params.hub_height, params.turbine_yaw, params.ct_models, params.generator_efficiency, params.cut_in_speed,
                params.cut_out_speed, params.rated_speed, params.rated_power, params.windresource, params.power_models, params.model_set,
                rotor_sample_points_y=params.rotor_points_y,rotor_sample_points_z=params.rotor_points_z)
    
    # return the objective as an array
    return [AEP]
end

function p_wrapper(x)
    n = Int(length(x)/2)
    turbine_x = x[1:n]
    turbine_y = x[n+1:end]

    rot_x, rot_y = ff.rotate_to_wind_direction(turbine_x, turbine_y, params.windresource.wind_directions[state])
    sorted_turbine_index = sortperm(rot_x)
    turbine_velocities = ff.turbine_velocities_one_direction(rot_x, rot_y, params.turbine_z, params.rotor_diameter, params.hub_height, params.turbine_yaw,
                            sorted_turbine_index, params.ct_models, params.rotor_points_y, params.rotor_points_z, params.windresource,
                            params.model_set, wind_farm_state_id=state, velocity_only=true)

    wt_power = ff.turbine_powers_one_direction(params.generator_efficiency, params.cut_in_speed, params.cut_out_speed, params.rated_speed,
                            params.rated_power, params.rotor_diameter, turbine_velocities, params.turbine_yaw, params.windresource.air_density, params.power_models)
end

function p_wrapper!(wt_power,x,params,currentState)
    n = Int(length(x)/2)
    turbine_x = x[1:n]
    turbine_y = x[n+1:end]

    rot_x, rot_y = ff.rotate_to_wind_direction(turbine_x, turbine_y, params.windresource.wind_directions[currentState])
    sorted_turbine_index = sortperm(rot_x)
    turbine_velocities = ff.turbine_velocities_one_direction(rot_x, rot_y, params.turbine_z, params.rotor_diameter, params.hub_height, params.turbine_yaw,
                            sorted_turbine_index, params.ct_models, params.rotor_points_y, params.rotor_points_z, params.windresource,
                            params.model_set, wind_farm_state_id=currentState, velocity_only=true)

    wt_power .= ff.turbine_powers_one_direction(params.generator_efficiency, params.cut_in_speed, params.cut_out_speed, params.rated_speed,
                            params.rated_power, params.rotor_diameter, turbine_velocities, params.turbine_yaw, params.windresource.air_density, params.power_models)
    
    nothing
end

## dirs is number of wind directions 12, 72, 1
## num is number of turbines 38 is standard
## angle is angle of the farm, 0 is default
## returns turbine_x,turbine_y,params
function loadRoundFarm(dirs,num,angle)
    cd("/Users/benjaminvarela/.julia/dev/FLOWFarm/test")

    # scale objective to be between 0.1 and 1
    if num == 38
        obj_scale = 1E-18
    else
        obj_scale = 1E-13
    end

    # set initial turbine x and y locations
    diam = 80.0
    data = readdlm("inputfiles/layout_38turb_round.txt",  ' ', skipstart=1)
    turbine_x = data[:, 1].*diam
    nturbines = length(turbine_x)
    nturbines = num
    turbine_y = data[:, 2].*diam
    turbine_z = zeros(nturbines)

    turbine_x = turbine_x .- turbine_x[1]
    turbine_y = turbine_y .- turbine_y[1]

    angle = 270 + angle

    turbine_x, turbine_y = ff.rotate_to_wind_direction(turbine_x, turbine_y, angle*pi/180)

    turbine_z = zeros(nturbines)
    
    # set turbine base heights
    turbine_z = zeros(nturbines) .+ 0.0
    
    # set turbine yaw values
    turbine_yaw = zeros(nturbines)
    
    # set turbine design parameters
    rotor_diameter = zeros(nturbines) .+ diam # m
    hub_height = zeros(nturbines) .+ 70.0   # m
    cut_in_speed = zeros(nturbines) .+ 0.  # m/s 4.0
    cut_out_speed = zeros(nturbines) .+ 25.  # m/s
    rated_speed = zeros(nturbines) .+ 16.  # m/s
    rated_power = zeros(nturbines) .+ 2.0E6  # W
    generator_efficiency = zeros(nturbines) .+ 0.944
    
    # rotor swept area sample points (normalized by rotor radius)
    rotor_points_y = [0.0]
    rotor_points_z = [0.0]
    
    # set flow parameters
    if dirs == 12
        data = readdlm("inputfiles/windrose_nantucket_12dir.txt",  ' ', skipstart=1)
        winddirections = data[:, 1].*pi/180.0
        windspeeds = data[:,2]
        windprobabilities = data[:, 3]
        nstates = length(windspeeds)
    elseif dirs == 72
        data = readdlm("inputfiles/windrose_amalia_72dirs.txt",  ' ', skipstart=1)
        winddirections = data[:, 1].*pi/180.0
        windspeeds = data[:,2]
        windprobabilities = data[:, 3]
        nstates = length(windspeeds)
    else
        winddirections = [270.0*pi/180]
        windspeeds = [8.0]
        windprobabilities = [1.0]
        nstates = length(windspeeds)
    end
    
    
    air_density = 1.1716  # kg/m^3
    ambient_ti = 0.077
    shearexponent = 0.15
    ambient_tis = zeros(nstates) .+ ambient_ti
    measurementheight = zeros(nstates) .+ hub_height[1]
    
    # load power curve
    powerdata = readdlm("inputfiles/niayifar_vestas_v80_power_curve_observed.txt",  ',', skipstart=1)
    velpoints = powerdata[:,1]
    powerpoints = powerdata[:,2]*1E6
    
    # initialize power model
    power_model = ff.PowerModelCpPoints(velpoints, powerpoints)
    power_models = Vector{typeof(power_model)}(undef, nturbines)
    for i = 1:nturbines
        power_models[i] = power_model
    end
    
    # load thrust curve
    ctdata = readdlm("inputfiles/predicted_ct_vestas_v80_niayifar2016.txt",  ',', skipstart=1)
    velpoints = ctdata[:,1]
    ctpoints = ctdata[:,2]
    
    # initialize thurst model
    ct_model = ff.ThrustModelCtPoints(velpoints, ctpoints)
    ct_models = Vector{typeof(ct_model)}(undef, nturbines)
    for i = 1:nturbines
        ct_models[i] = ct_model
    end
    
    # initialize wind shear model
    wind_shear_model = ff.PowerLawWindShear(shearexponent)
    
    # get sorted indecies 
    # sorted_turbine_index = sortperm(turbine_x)
    
    # initialize the wind resource definition
    windresource = ff.DiscretizedWindResource(winddirections, windspeeds, windprobabilities, measurementheight, air_density, ambient_tis, wind_shear_model)
    
    # set up wake and related models
    wakedeficitmodel = ff.GaussYawVariableSpread()
    wakedeficitmodel.wec_factor[1] = 1.0
    
    wakedeflectionmodel = ff.GaussYawVariableSpreadDeflection()
    wakecombinationmodel = ff.LinearLocalVelocitySuperposition()
    localtimodel = ff.LocalTIModelNoLocalTI()
    
    # initialize model set
    model_set = ff.WindFarmModelSet(wakedeficitmodel, wakedeflectionmodel, wakecombinationmodel, localtimodel)
    
    # initialize sparse partial derivatives array for turbine velocity
    dx = spzeros(nturbines)
    
    # set wind farm boundary parameters
    boundary_center = [0.0,0.0]
    boundary_radius = 1225.8227848101264
    
    params = params_struct(model_set, rotor_points_y, rotor_points_z, turbine_z, 
    rotor_diameter, boundary_center, boundary_radius, obj_scale, hub_height, turbine_yaw, 
    ct_models, generator_efficiency, cut_in_speed, cut_out_speed, rated_speed, rated_power, 
    windresource, power_models, [0], dx, [spzeros(nturbines, 2*nturbines)], spzeros(2*nturbines));

    return turbine_x,turbine_y,params
end


spacing_wrapper(x) = spacing_wrapper(x, params)
aep_wrapper(x) = aep_wrapper(x, params)
boundary_wrapper(x) = boundary_wrapper(x, params)
end
