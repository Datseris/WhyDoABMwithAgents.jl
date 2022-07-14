cd(@__DIR__)
using Pkg; Pkg.activate(@__DIR__)
using Agents
using Random # for reproducibility

# Agents live in spaces in Agents.jl. In our example they live in
space = GridSpaceSingle((10,10); periodic = false)
# which is a specialized version of
space_general = GridSpace((10,10); periodic = false)
# whose documentation string says how to make the appropriate agent type.

# In Agents.jl agents are typically created with the `@agent` decorator,
# which creates a Julia struct that obtains some additional fields from another
# already existing agent type (these fields are ID and position)
@agent SchellingAgent GridAgent{2} begin
    group::Int
    happy::Bool
end

function initialize(; # create a model with agents and return it
        grid_size = (30, 30), # Dimension of the grid
        min_to_be_happy = 3,  # how many nearby agents you need of same group
        grid_occupation = 0.8 # percentage of space occupied by agents
    )
    # Grid space structure takes care of all the hard stuff in modelling
    # regarding agent position and moving and nearby agents
    space = GridSpaceSingle(grid_size; periodic = false)
    # All model-level properties should be put into a container
    properties = Dict(:min_to_be_happy => min_to_be_happy)
    rng = Random.Xoshiro(1234) # reproducibility with more control
    # `ABM` stands for `AgentBasedModel`
    model = ABM(SchellingAgent, space; properties, rng)

    N = grid_size[1]*grid_size[2]*grid_occupation
    for n in 1:N
        group = n < N/2 ? 1 : 2
        agent = SchellingAgent(n, (1, 1), group, false)
        # add agent at random unoccupied position. update position automatically
        add_agent_single!(agent, model)
    end
    return model
end

model = initialize()

# Define the agent stepping function (the rules of the game)
# Here we don't loop over agents. We define rules-per-agent.
# Looping is handled internally by Agents.jl
function agent_step!(agent, model)
    nearby_same = 0
    # with default arguments `nearby_agents` returns the maximum 8 nearby agents
    # as in the previous simpler case. It only returns agents that actually exist
    for neighbor in nearby_agents(agent, model)
        if agent.group == neighbor.group
            nearby_same += 1
        end
    end
    if nearby_same ≥ model.min_to_be_happy
        agent.happy = true
    else
        # move agent to a random unoccupied position. update position automatically
        move_agent_single!(agent, model)
    end
    return
end

step!(model, agent_step!) # same as `simulation_step!(model)` from before

# Visualization and interactive evolution
using InteractiveDynamics
using GLMakie; GLMakie.activate!() # for plotting
model = initialize()
groupcolor(a) = a.group == 1 ? :blue : :orange
groupmarker(a) = a.group == 1 ? :circle : :rect

fig, = abmplot(model;
    agent_step!, # simply providing the `agent_step!` function makes interactive GUI
    ac = groupcolor, am = groupmarker, as = 20,
    axis = (title = "Schelling's segregation model",),
    figure = (resolution = (800, 900),)
)
display(fig)

# Collect data, now that's something especially simple!
group1(agent) = agent.group == 1
group2(agent) = agent.group == 2
adata = [
    (:happy, count, group1),
    (:happy, count, group2),
]
# to explain `adata`: A 3-tuple means:
# (agent property to collect, aggregating function, filtering function)
model = initialize()
adf, mdf = run!(model, agent_step!, dummystep, 10; adata)

# `dummystep` is a function from Agents.jl that does nothing.
# it is given instead of the model stepping function, as our simulation
# has no model-level dynamics

# Benchmark
# %%
using BenchmarkTools
model = initialize()
@btime step!($model, agent_step!)
