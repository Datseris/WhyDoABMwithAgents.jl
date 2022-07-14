cd(@__DIR__)
using Pkg; Pkg.activate(@__DIR__)
using Agents, LinearAlgebra
using Random

@agent Bird ContinuousAgent{2} begin
    speed::Float64
end

# The function `initialize_model` generates birds and returns a model object using default values.
function initialize_model(;
        n_birds = 100,
        speed = 1.0,
        cohere_factor = 0.25,
        separation = 4.0,
        separate_factor = 0.25,
        match_factor = 0.01,
        visual_distance = 5.0,
        extent = (100, 100),
    )
    properties = (; cohere_factor, separation, separate_factor, match_factor,
    visual_distance)
    properties = Dict(k=>v for (k,v) in pairs(properties))
    space2d = ContinuousSpace(extent; spacing = visual_distance/1.5)
    model = ABM(Bird, space2d; scheduler = Schedulers.Randomly(), properties)
    for _ in 1:n_birds
        vel = Tuple(rand(model.rng, 2) .+ 1)
        add_agent!(
            model,
            vel,
            speed,
        )
    end
    return model
end

function agent_step!(bird, model)
    ## Obtain the ids of neighbors within the bird's visual distance
    neighbor_ids = nearby_ids(bird, model, model.visual_distance)
    N = 0
    match = separate = cohere = (0.0, 0.0)
    ## Calculate behaviour properties based on neighbors
    for id in neighbor_ids
        N += 1
        neighbor = model[id].pos
        heading = neighbor .- bird.pos

        ## `cohere` computes the average position of neighboring birds
        cohere = cohere .+ heading
        if euclidean_distance(bird.pos, neighbor, model) < model.separation
            ## `separate` repels the bird away from neighboring birds
            separate = separate .- heading
        end
        ## `match` computes the average trajectory of neighboring birds
        match = match .+ model[id].vel
    end
    N = max(N, 1)
    ## Normalise results based on model input and neighbor count
    cohere = cohere ./ N .* model.cohere_factor
    separate = separate ./ N .* model.separate_factor
    match = match ./ N .* model.match_factor
    ## Compute velocity based on rules defined above
    bird.vel = (bird.vel .+ cohere .+ separate .+ match) ./ 2
    bird.vel = bird.vel ./ norm(bird.vel)
    ## Move bird according to new velocity and speed
    move_agent!(bird, model, bird.speed)
end

# ## Plotting the flock
using InteractiveDynamics
using GLMakie; GLMakie.activate!()

# The great thing about [`abmplot`](@ref) is its flexibility. We can incorporate the
# direction of the birds when plotting them, by making the "marker" function `am`
# create a `Polygon`: a triangle with same orientation as the bird's velocity.
# It is as simple as defining the following function:

const bird_polygon = Polygon(Point2f[(-0.5, -0.5), (1, 0), (-0.5, 0.5)])
function bird_marker(b::Bird)
    φ = atan(b.vel[2], b.vel[1]) #+ π/2 + π
    scale(rotate2D(bird_polygon, φ), 2)
end

# Where we have used the utility functions `scale` and `rotate2D` to act on a
# predefined polygon. We now give `bird_marker` to `abmplot`, and notice how
# the `as` keyword is meaningless when using polygons as markers.

model = initialize_model()
params = Dict(
    :cohere_factor => 0:0.05:0.5,
    :separation => 1:0.1:10,
    :separate_factor => 0:0.1:1.0,
    :match_factor => 0:0.001:0.1,
    :visual_distance => 1:0.25:10.0,
)

fig, = abmplot(model;
    agent_step!, am = bird_marker, params,
    figure = (resolution = (700, 700),),
    axis = (title = "Flocking", titlealign = :left),
)
fig
