using Random # reproducibility
Random.seed!(1234)

# The first thing is that we need a data structure to keep track of the agents.
# We will create a mutable struct, because it is the most convenient
# way to modify named properties of an object in the case
# where each property may have different data type
mutable struct ExampleAgent
    group::Int
    happy::Bool
end
# next, we define some constants that will be used in the simulation
const min_to_be_happy = 3 # how many nearby agents you need of same group
const grid_occupation = 0.8 # percentage of space occupied by agents
const grid_size = (30, 30)

# The **hardest problem to solve** in ABM is how to represent agents in space.
# How to store them based on their location, how to retrieve them
# and most importantly, how to find other agents nearby a given location.
# Of course, the actually hard part is to make this as performant as possible.

# The simplest (but not most performant) way to do this, is by using
# a dictionary, mapping filled positions (i, j) to existing agents.

function initialize()
    # Initialize the dict which stores the agents.
    model = Dict{Tuple{Int, Int}, ExampleAgent}()
    # Fill the grid with agents
    N = grid_size[1]*grid_size[2]*grid_occupation
    for n in 1:N
        # Half the agents belong to group 1, the other half to 2
        group = if n < N/2
            1
        else
            2
        end
        # Find random and empty position to put agent in
        pos = (rand(1:grid_size[1]), rand(1:grid_size[2]))
        while haskey(model, pos)
            pos = (rand(1:grid_size[1]), rand(1:grid_size[2]))
        end
        # create agent
        agent = ExampleAgent(group, false)
        # add agent into model
        model[pos] = agent
    end
    return model
end

model = initialize()

# create a function for visualizing the model
using GLMakie
function visualize(model)
    pos, agents = keys(model), values(model)
    xcoords = [p[1] for p in pos]
    ycoords = [p[2] for p in pos]
    colors = [a.group == 1 ? "blue" : "orange" for a in agents]
    scatter(xcoords, ycoords; color = colors, markersize = 15)
end

fig, ax = visualize(model)


# Create rules of simulation!
# These are the indices that, when added to "current" position,
# give the position of the neighbors
const neighborhood_offsets = [
    (1,0),  (1,1),  (1,-1),
    (0,1),          (0,-1),
    (-1,0), (-1,1), (-1,-1),
]
# and this is the simulation step!
function simulation_step!(model)
    # Crucial to pre-collect positions, otherwise termination never stops!
    filled_positions = collect(keys(model))
    for pos in filled_positions # loop over all locations on grid
        agent = model[pos]
        # First step: count neighbors of same group
        same = 0
        for offset in neighborhood_offsets
            near_pos = pos .+ offset
            # skip count if invalid position or no neighbor
            if !haskey(model, near_pos)
                continue
            end
            neighbor = model[near_pos]
            if neighbor.group == agent.group
                same += 1
            end
        end
        # Second part: relocate if not enough same neighbors
        if same ≥ min_to_be_happy
            agent.happy = true
        else # generate new unoccupied position
            newpos = (rand(1:grid_size[1]), rand(1:grid_size[2]))
            while haskey(model, newpos) # careful here, use `newpos`, not `pos`!
                newpos = (rand(1:grid_size[1]), rand(1:grid_size[2]))
            end
            model[newpos] = agent
            # Crucial to remove from previous position, otherwise termination never stops!
            # These are the things that take plenty of time to figure out...
            delete!(model, pos)
        end
    end
end

simulation_step!(model)
fig, ax = visualize(model)

# Collect data. Which data to collect?
# one can collect each position and each happiness status at each step
# but probably overkill. Instead, collect happiness ratios of each group.
# And to make further data analysis seamless,
# we immediatelly collect data as `DataFrame`.
using DataFrames
function collect_data(model, S)
    data = DataFrame((step = [0], happy_1 = [0], happy_2 = [0]))
    for s in 1:S
        simulation_step!(model)
        happy_1 = count(a.happy for a in values(model) if a.group == 1)
        happy_2 = count(a.happy for a in values(model) if a.group == 2)
        push!(data, (step = s, happy_1, happy_2) )
    end
    return data
end

model = initialize()
data = collect_data(model, 10)
# and plot it
fig, ax = lines(data[!, "step"], data[!, "happy_1"], color = "blue")
lines!(ax, data[!, "step"], data[!, "happy_2"], color = "orange")

# %% Benchmarking
using BenchmarkTools
model = initialize()
@btime simulation_step!($model)
