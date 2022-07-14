# This simple script simulates agents going into a presentation about
# Agent Based Modelling using Julia. Unfortunately for them, one of the
# participants is carrying a deadline zombie virus that will soon take over...

# We start by loading the necessary packages
cd(@__DIR__)
using Pkg; Pkg.activate(@__DIR__)
using Agents
using InteractiveDynamics, GLMakie
using Random

# Download map of Hamburg given longitude, latitude
if !isfile("hamburg_map.json") # don't download again if file already exists.
    OSM.download_osm_network(:bbox;
        minlon = 9.9159, maxlon = 9.9711,
        minlat = 53.5428, maxlat = 53.5706,
        save_to_file_location = "hamburg_map.json",
        network_type = :all,
    )
end

"University of Applied Sciences Europe (lon, lat) coordinates."
const UoASE_COORDS = (9.93464, 53.55124)

# Create agent type with necessary properties
@agent HumanNotForLong OSMAgent begin
    zombie::Bool          # whether the agent has become a rabid zombie!
    speed::Float64        # in km / model step
    vision::Float64       # vision radius, in km
    stay_still::Int       # at some processes agents need to stay put for some steps
    km_travelled::Float64 # km already travelled; need to rest after some amount...
    km_capacity::Float64  # max capacity that can be traversed before resting
    victim::Int           # id of hunted agent, if zombie
end

# Initialize model using function and keywords (best practice)
function initialize_hamburg_abm(;
        max_speed = 0.005, max_vision = 0.02,
        max_km_capacity = 2.0, resting_time = 400,
        seed = 1234, total_participants = 50, countdown = 1500
    )
    # Create space instance from OSM
    space = OpenStreetMapSpace("hamburg_map.json")
    # Instantiate model
    rng = Random.Xoshiro(seed)
    properties = Dict(:step => 0, :countdown => countdown, :resting_time => resting_time)
    model = AgentBasedModel(HumanNotForLong, space; rng, properties)
    # Add random agents, who all want to go to the presentation place
    location_of_presentation = OSM.nearest_node(UoASE_COORDS, model)
    for i in 1:total_participants
        speed = max_speed*(0.5rand(model.rng) + 0.5)
        vision = max_vision*(0.5rand(model.rng) + 0.5)
        km_capacity = max_km_capacity*(0.75rand(model.rng) + 0.25)
        # Create an agent, and add it at a random position, automatically:
        # (everything after `model` is properties added to the agent)
        agent = add_agent!(model, false, speed, vision, 0, 0.0, km_capacity, 0)
        # Then, set a route for this agent towards the venue:
        success = plan_route!(agent, location_of_presentation, model)
        # Depending on position we initialize, there may not be a connection
        # to the presentation room; we randomly relocate until there is.
        while !success
            move_agent!(agent, model)
            success = plan_route!(agent, location_of_presentation, model)
        end
    end
    return model
end

model = initialize_hamburg_abm()

# Create the agent stepping function
# and the model stepping function (which just performs the countdown)
function model_step!(model)
    model.step += 1
    if model.step == model.countdown
        turn_to_zombie!(model[1]) # the infected one turns into zombie
    end
end

function turn_to_zombie!(agent)
    agent.zombie = true
    agent.vision /= 3
    agent.speed *= 2
end

function agent_step!(agent, model)
    if model.step < model.countdown
        # Crazy zombie apocalipse hasn't begun yet, casually go to lecture hall...
        move_along_route!(agent, model, agent.speed)
        return
    end
    # Now shit has hit the fan and you either run for your life, or you are
    # already a zombie and you NEED BRAINS!
    # But first, check if you need to rest more
    if agent.stay_still > 0
        agent.stay_still -= 1
        return
    end
    if agent.zombie # hunter mode
        hunter_mode!(agent, model)
    else
        hunted_mode!(agent, model)
    end
end

function hunter_mode!(zombie, model)
    if zombie.victim == 0 # Identify closest human
        d = Inf
        for potential_victim in nearby_agents(zombie, model, zombie.vision)
            potential_victim.zombie && continue # can't chase another zombie
            d2 = OSM.distance(zombie.pos, potential_victim.pos, model)
            if d2 < d
                d = d2
                zombie.victim = potential_victim.id
            end
        end
    end
    if zombie.victim != 0 && !model[zombie.victim].zombie
        # You have targeted a victim, chase it!
        victim = model[zombie.victim]
        plan_route!(zombie, victim.pos, model)
        move_along_route!(zombie, model, zombie.speed)
        if OSM.distance(zombie.pos, victim.pos, model) ≤ 0
            # you've reached the victim, time to eat brains
            turn_to_zombie!(victim)
            zombie.victim = 0
            # zombies only rest when they eat brains!
            initiate_resting!(zombie, model, 0.5)
            # The new zombie takes twice as long to become "operational"
            initiate_resting!(victim, model, 1)
        end
    else
        # You didn't find a victim, so start wondering around
        if is_stationary(zombie, model)
            OSM.plan_random_route!(zombie, model)
        end
        move_along_route!(zombie, model, zombie.speed)
    end
end

function hunted_mode!(agent, model)
    if is_stationary(agent, model)
        # You are scared and in shock, so you start running towards a random place
        OSM.plan_random_route!(agent, model)
    end
    # Move towards your random destination
    remaining = move_along_route!(agent, model, agent.speed)
    # Accumulate kilometers travelled while moving!
    agent.km_travelled += agent.speed - remaining
    # If you run too much, you get tired
    if agent.km_travelled > agent.km_capacity
        initiate_resting!(agent, model)
        agent.km_travelled = 0
    end
    return
end

function initiate_resting!(agent, model, factor = 1)
    agent.stay_still = model.resting_time*factor
end


# And finally, launch an interactive application for evolving the model
using GLMakie; GLMakie.activate!()
# Overwrite OSMMakie color defaults:
using GLMakie.Colors
default_colors = InteractiveDynamics.OSMMakie.WAYTYPECOLORS
default_colors["primary"] = colorant"#a1777f"
default_colors["secondary"] = colorant"#a18f78"
default_colors["tertiary"] = colorant"#b3b381"

zombie_color(a) = a.zombie ? to_color("green") : JULIADYNAMICS_COLORS[1]
zombie_size(a) = a.zombie ? 25 : 20
zombies(model) = count(a -> a.zombie, allagents(model))

fig, ax = abmplot(model;
    agent_step!, model_step!, as = zombie_size, ac = zombie_color,
    scatterkwargs = (strokewidth = 1.0, strokecolor = :black),
    enable_inspection = false,
)
display(fig)

# %% Make video
# For high quality output use CairoMakie backend
using CairoMakie; CairoMakie.activate!()
# For fast video output use GLMakie backend
# using GLMakie; GLMakie.activate!()

model = initialize_hamburg_abm()
abmvideo("hamburg_simulation.mp4", model, agent_step!, model_step!;
    as = zombie_size, ac = zombie_color,
    scatterkwargs = (strokewidth = 1.0, strokecolor = :black),
    title = "Zombie outbreak during a presentation in Hamburg",
    frames = 1200, framerate = 60, spf = 5,
    recordkwargs = (compression = 1, profile = "high"),
    figure = (resolution = (1100, 550),)
)