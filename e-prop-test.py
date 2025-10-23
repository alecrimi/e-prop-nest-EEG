# =======================================================
# Full NEST + e-prop EEG classification example
# =======================================================

import nest
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# -------------------------
# 1. Load or generate EEG data
# -------------------------
# Example dummy EEG dataset
n_samples = 100
n_channels = 8   # EEG electrodes
n_timepoints = 100  # number of ms/timepoints per trial
n_classes = 2

np.random.seed(42)
eeg_data = np.random.randn(n_samples, n_channels, n_timepoints)
eeg_labels = np.random.randint(0, n_classes, size=(n_samples,))

# Standardize channels
for i in range(n_samples):
    eeg_data[i] = StandardScaler().fit_transform(eeg_data[i].T).T

# -------------------------
# 2. EEG -> spike trains (rate coding)
# -------------------------
def eeg_to_spikes(signal, dt=1.0, max_rate=100):
    """
    Convert EEG channels to spike trains.
    signal: array (n_channels, n_timepoints)
    returns: list of spike times per neuron
    """
    spike_times = []
    n_channels, n_timepoints = signal.shape
    for ch in range(n_channels):
        neuron_spikes = []
        for t in range(n_timepoints):
            rate = (signal[ch, t] - signal[ch].min()) / (signal[ch].ptp() + 1e-12)
            rate *= max_rate
            if np.random.rand() < rate * dt / 1000.0:  # dt in ms, rate in Hz
                neuron_spikes.append(t * dt)
        spike_times.append(neuron_spikes)
    return spike_times

# -------------------------
# 3. Build NEST network
# -------------------------
nest.ResetKernel()

n_input = n_channels
n_hidden = 50
n_output = n_classes

# Input spike generators
input_neurons = nest.Create('spike_generator', n_input)

# Recurrent hidden layer
hidden_neurons = nest.Create('iaf_psc_alpha', n_hidden)

# Output neurons
output_neurons = nest.Create('iaf_psc_alpha', n_output)

# Connect input -> hidden
nest.Connect(input_neurons, hidden_neurons, syn_spec={'weight': 20.0, 'delay': 1.0})

# Recurrent hidden -> hidden
nest.Connect(hidden_neurons, hidden_neurons, syn_spec={'weight': 20.0, 'delay': 1.5})

# Hidden -> output
nest.Connect(hidden_neurons, output_neurons, syn_spec={'weight': 20.0, 'delay': 1.5})

# -------------------------
# 4. Setup e-prop supervised learning
# -------------------------
# Note: Replace with actual NEST e-prop synapse model
# Example placeholders
synapses = nest.GetConnections(source=hidden_neurons, target=output_neurons)
nest.SetStatus(synapses, {"eprop_learning": True})  # supervised e-prop

# -------------------------
# 5. Training loop
# -------------------------
def train_epoch(X, y, dt=1.0):
    for trial, label in zip(X, y):
        # Convert EEG trial to spike trains
        spike_times = eeg_to_spikes(trial, dt=dt)
        
        # Set spike times for input neurons
        for idx, neuron in enumerate(input_neurons):
            nest.SetStatus([neuron], {'spike_times': spike_times[idx]})
        
        # Define supervised target spikes for output neurons
        target_spikes = []
        for out_idx in range(n_output):
            if out_idx == label:
                # Target spike at end of trial
                target_spikes.append([n_timepoints * dt - 1])
            else:
                target_spikes.append([])
        
        nest.SetStatus(output_neurons, [{"spike_times_supervised": t} for t in target_spikes])
        
        # Simulate network for trial duration
        nest.Simulate(n_timepoints)

# -------------------------
# 6. Train/Test split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(eeg_data, eeg_labels, test_size=0.2, random_state=42)

n_epochs = 5
for epoch in range(n_epochs):
    train_epoch(X_train, y_train)
    print(f"Epoch {epoch+1}/{n_epochs} completed.")

# -------------------------
# 7. Evaluation
# -------------------------
def evaluate(X, y, dt=1.0):
    correct = 0
    for trial, label in zip(X, y):
        spike_times = eeg_to_spikes(trial, dt=dt)
        for idx, neuron in enumerate(input_neurons):
            nest.SetStatus([neuron], {'spike_times': spike_times[idx]})
        
        # Simulate trial
        nest.Simulate(n_timepoints)
        
        # Count spikes in output neurons
        spike_counts = [len(nest.GetStatus([neuron], 'events')[0]['times']) for neuron in output_neurons]
        pred_label = np.argmax(spike_counts)
        if pred_label == label:
            correct += 1
    accuracy = correct / len(y)
    return accuracy

test_acc = evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc*100:.2f}%")
