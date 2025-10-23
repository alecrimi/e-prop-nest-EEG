import nest
import nest.voltage_trace
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------
# 1. Reset NEST Kernel
# ---------------------------
nest.ResetKernel()

# ---------------------------
# 2. Parameters
# ---------------------------
N_exc = 800    # number of excitatory neurons
N_inh = 200    # number of inhibitory neurons
p_connect = 0.1  # connection probability
syn_weight_exc = 20.0
syn_weight_inh = -20.0
delay = 1.5

# Alzheimer-like effect: reduce connectivity
alzheimer_factor = 0.5  # 50% of normal connectivity
p_connect_alz = p_connect * alzheimer_factor

# ---------------------------
# 3. Create neurons
# ---------------------------
exc_neurons = nest.Create('iaf_psc_alpha', N_exc)
inh_neurons = nest.Create('iaf_psc_alpha', N_inh)

# ---------------------------
# 4. Create spike detector
# ---------------------------
spike_detector = nest.Create('spike_detector')

# ---------------------------
# 5. Connect neurons
# ---------------------------
# Excitatory -> Excitatory
nest.Connect(exc_neurons, exc_neurons,
             {'rule': 'pairwise_bernoulli', 'p': p_connect_alz},
             {'weight': syn_weight_exc, 'delay': delay})

# Excitatory -> Inhibitory
nest.Connect(exc_neurons, inh_neurons,
             {'rule': 'pairwise_bernoulli', 'p': p_connect_alz},
             {'weight': syn_weight_exc, 'delay': delay})

# Inhibitory -> Excitatory
nest.Connect(inh_neurons, exc_neurons,
             {'rule': 'pairwise_bernoulli', 'p': p_connect_alz},
             {'weight': syn_weight_inh, 'delay': delay})

# Inhibitory -> Inhibitory
nest.Connect(inh_neurons, inh_neurons,
             {'rule': 'pairwise_bernoulli', 'p': p_connect_alz},
             {'weight': syn_weight_inh, 'delay': delay})

# Connect to spike detector
nest.Connect(exc_neurons + inh_neurons, spike_detector)

# ---------------------------
# 6. Stimulate the network
# ---------------------------
poisson = nest.Create('poisson_generator', params={'rate': 800.0})
nest.Connect(poisson, exc_neurons, syn_spec={'weight': 20.0, 'delay': 1.0})

# ---------------------------
# 7. Simulate
# ---------------------------
sim_time = 1000.0  # ms
nest.Simulate(sim_time)

# ---------------------------
# 8. Plot spikes
# ---------------------------
events = nest.GetStatus(spike_detector, 'events')[0]
plt.figure(figsize=(12, 4))
plt.scatter(events['times'], events['senders'], s=2)
plt.xlabel("Time (ms)")
plt.ylabel("Neuron ID")
plt.title("Alzheimer-like network spike raster")
plt.show()
