import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output
from qiskit import QuantumCircuit
from qiskit.circuit.library import RealAmplitudes
from qiskit_machine_learning.circuit.library import RawFeatureVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit_aer import AerSimulator
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from qiskit_machine_learning.neural_networks import EstimatorQNN
from PIL import Image
import os
import glob
from tqdm import tqdm
import time

# Parameters
n_qubits = 12
feature_dimension = 2**n_qubits  # 65,536 for 256x256 images
num_images = 10  # Change to 1000 for the second case
num_iterations = 200  # COBYLA maxiter
reps = 2  # RealAmplitudes layers
image_size = (64, 64)
path = r"C:\PycharmProjects\Quantum Denoiser\Denoiser\testimgs"

images = []

for file_name in os.listdir(path):
    try:
        file_path = os.path.join(path, file_name)
        if file_name.lower().endswith(('.jpg')):
            img = Image.open(file_path)
            images.append(img)
    except (PermissionError, FileNotFoundError) as e:
        print(f"Error: {e} - {file_path}")
    except Exception as e:
        print(f"An unexpected error occured: {e} - {file_path}")

print(f"Number of images loaded: {len(images)}")


# Process images
def data(image_list):
    processed_list = []
    for img in image_list:
        img = img.convert('L')  # Convert to grayscale
        img = img.resize(image_size)  # Resize to 256x256
        img_array = np.array(img).flatten() / 255.0  # Flatten and normalize to [0, 1]
        img_array = img_array / np.linalg.norm(img_array)  # Normalize to unit vector
        processed_list.append(img_array)
    return np.array(processed_list)
clean_data = data(images)
print(f"clean_data shape: {clean_data.shape}")

# Circuit setup
feature_map = RawFeatureVector(feature_dimension=feature_dimension)
mps_block = RealAmplitudes(num_qubits=n_qubits, reps=1)
qc = QuantumCircuit(n_qubits)
qc.compose(feature_map, inplace=True)
qc.compose(mps_block, inplace=True)

# Simulator and estimator
simulator = AerSimulator(method='statevector', max_parallel_threads=13)  # Try 'matrix_product_state' if sparse
estimator = Estimator(options={"backend": simulator})

# Observable
observable = SparsePauliOp.from_list([("Z" * n_qubits, 1)])


# QNN setup
qnn = EstimatorQNN(
    circuit=qc,
    input_params=feature_map.parameters,
    weight_params=mps_block.parameters,
    estimator=estimator,
    observables=observable
)

# Classifier
classifier = NeuralNetworkClassifier(
    qnn,
    optimizer=COBYLA(maxiter=num_iterations),
    callback=lambda weights, obj_func_eval: callback_graph(weights, obj_func_eval)
)

# Callback for plotting
objective_func_vals = []
def callback_graph(weights, obj_func_eval):
    clear_output(wait=True)
    objective_func_vals.append(obj_func_eval)
    plt.title("Objective function value against iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Objective function value")
    plt.plot(range(len(objective_func_vals)), objective_func_vals)
    plt.rcParams["figure.figsize"] = (12, 6)
    plt.show()
# Progress tracking with time estimation
class ProgressTracker:
    def __init__(self, total_iterations, evaluations_per_iter_estimate=260):
        self.total_iterations = total_iterations
        self.evaluations_per_iter = evaluations_per_iter_estimate
        self.total_evals = total_iterations * evaluations_per_iter_estimate
        self.eval_count = 0
        self.iter_count = 0
        self.pbar = tqdm(total=self.total_evals, desc="Evaluations", unit="eval")
        self.start_time = time.time()

    def eval_callback(self):
        self.eval_count += 1
        self.pbar.update(1)
        if self.eval_count % self.evaluations_per_iter == 0:
            self.iter_count += 1
            elapsed = time.time() - self.start_time
            eval_time = elapsed / self.eval_count
            remaining_evals = self.total_evals - self.eval_count
            time_left = remaining_evals * eval_time
            print(f"Iteration {self.iter_count}/{self.total_iterations}, Eval {self.eval_count}/{self.total_evals}, "
                  f"Time per eval: {eval_time*1000:.2f} ms, Estimated time left: {time_left/3600:.2f} hours")

    def close(self):
        self.pbar.close()

# Train classifier

labels = [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,1,1,1,1,1,1,1,1,1,1,1,1,1]
y = np.asarray(labels)
objective_func_vals = []
plt.rcParams["figure.figsize"] = (12, 6)
tracker = ProgressTracker(num_iterations, evaluations_per_iter_estimate=100)
print("Starting training...")
start_time = time.time()
classifier.fit(clean_data, y)
tracker.close()

# Final results
total_time = time.time() - start_time
hours = int(total_time // 3600)
minutes = int((total_time % 3600) // 60)
seconds = int(total_time % 60)
print(f"\nOptimization complete. Total time: {hours:02d}:{minutes:02d}:{seconds:02d}")
print(f"Accuracy score: {classifier.score(clean_data, y):.4f}")

classifier.predict(clean_data)
