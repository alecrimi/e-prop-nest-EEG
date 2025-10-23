import torch
import matplotlib.pyplot as plt
import numpy as np
from spikingjelly.clock_driven import neuron, functional, layer

# ---------------------------
# 1. Parameters
# ---------------------------
device = 'cpu'

N_exc = 800
N_inh = 200
N_total = N_exc + N_inh

p_connect = 0.1
alzheimer_factor = 0.5
p_connect_alz = p_connect * alzheimer_factor

syn_weight_exc = 0.05  # SpikeJelly uses smaller scale weights
syn_weight_inh = -0.05
delay = 1  # Not explicitly used in this simple example

sim_time = 1000  # ms
dt = 1.0  # ms
time_steps = int(sim_time / dt)

poisson_rate = 800.0  # Hz

# ---------------------------
# 2. Create recurrent connectivity
# ---------------------------
def create_connection_matrix(N_pre, N_post, p, weight):
    conn = (torch.rand(N_post, N_pre) < p).float() * weight
    return conn

# Excitatory connections
W_ee = create_connection_matrix(N_exc, N_exc, p_connect_alz, syn_weight_exc)
W_ei = create_connection_matrix(N_exc, N_inh, p_connect_alz, syn_weight_exc)

# Inhibitory connections
W_ie = create_connection_matrix(N_inh, N_exc, p_connect_alz, syn_weight_inh)
W_ii = create_connection_matrix(N_inh, N_inh, p_connect_alz, syn_weight_inh)

# Combine into full weight matrix
W = torch.zeros(N_total, N_total)
W[:N_exc, :N_exc] = W_ee
W[N_exc:, :N_exc] = W_ei
W[:N_exc, N_exc:] = W_ie
W[N_exc:, N_exc:] = W_ii

# ---------------------------
# 3. Create neurons
# ---------------------------
v_th = 1.0
tau = 20.0

neurons = neuron.LIFNode(v_threshold=v_th, tau=tau, detach_reset=True).to(device)

# ---------------------------
# 4. Simulation loop
# ---------------------------
spike_record = []

# Poisson input
poisson_input = torch.rand(time_steps, N_exc) < (poisson_rate * dt / 1000.0)
poisson_input = poisson_input.float().to(device)

# Initialize membrane potentials
v = torch.zeros(N_total, device=device)

for t in range(time_steps):
    x = torch.zeros(N_total, device=device)
    x[:N_exc] += poisson_input[t]  # input only to excitatory neurons

    # Recurrent input
    I = torch.matmul(W, spike_record[-1]) if spike_record else torch.zeros(N_total, device=device)
    x += I

    # Forward through LIF neuron
    spk = neurons(x)
    spike_record.append(spk)

# ---------------------------
# 5. Convert spikes to numpy for plotting
# ---------------------------
spike_record = torch.stack(spike_record)  # shape: [time_steps, N_total]
spike_times, neuron_ids = spike_record.nonzero(as_tuple=True)
spike_times = spike_times.cpu().numpy() * dt
neuron_ids = neuron_ids.cpu().numpy()

# ---------------------------
# 6. Plot raster
# ---------------------------
plt.figure(figsize=(12, 4))
plt.scatter(spike_times, neuron_ids, s=2)
plt.xlabel("Time (ms)")
plt.ylabel("Neuron ID")
plt.title("SpikeJelly Alzheimer's-like network raster")
plt.show()
