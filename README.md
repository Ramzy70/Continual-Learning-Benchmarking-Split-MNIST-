# Continual Learning Benchmarking (Split-MNIST)

An implementation and comparative analysis of strategies to mitigate **catastrophic forgetting** in class-incremental learning scenarios.

### 🚀 Project Overview
This project evaluates the performance of different Continual Learning (CL) agents on the **Split-MNIST** benchmark, where a model must learn a sequence of five disjoint tasks (digit pairs) without losing performance on previous ones.

* **Architecture**: Multi-Layer Perception (MLP) with two hidden layers of 256 units each.
* **Benchmark**: Split-MNIST where the standard dataset is divided into five disjoint tasks, each consisting of two classes.
* **Key Challenge**: Addressing the primary obstacle of catastrophic forgetting, where optimization for a new task degrades performance on previously learned tasks.

### 🛠 Methods Evaluated
1.  **Naive Sequential Fine-Tuning**: Represents the "lower bound" of performance, training sequentially on each task without mechanisms to preserve past knowledge.
2.  **Experience Replay (Rehearsal)**: Mitigates forgetting by maintaining a small memory buffer of samples from previous tasks to "rehearse" during new training.
3.  **Elastic Weight Consolidation (EWC)**: A regularization-based method that constrains model parameters important for previous tasks using a Fisher Information Matrix.
4.  **Joint Training**: Represents the theoretical "upper bound" or ideal performance, where the agent is retrained on the accumulated dataset of all tasks seen so far.

### 📊 Performance Comparison
| Method | Avg Accuracy | Avg Forgetting | Key Observation |
| :--- | :--- | :--- | :--- |
| **Naive** | 19.62% | 99.28% | Severe Catastrophic Forgetting |
| **EWC ($\lambda=580k$)** | 43.74% | 53.04% | Balanced Stability-Plasticity trade-off |
| **Replay (Buffer=200)** | 88.45% | 11.29% | Efficiently bridges the gap to Joint Training |
| **Joint (Upper Bound)** | 96.97% | 0.0% | Ideal reference performance |

### 🔍 Robustness & Generalization
* **Task Order**: The Replay agent demonstrated order-robustness with an average accuracy of 89.89% on a shuffled task sequence.
* **Fashion-MNIST**: The strategy generalized to the more challenging Fashion-MNIST dataset with an average accuracy of 79.26%.
* **Backward Transfer**: Analysis revealed that the replay buffer can help the model escape poor local minima, effectively repairing knowledge for older tasks as new tasks are added.

### 🏗 Project Structure
* **Backbone Model**: Implementation of an MLP with 2 hidden layers and ReLU activations.
* **ReplayAgent**: Logic for fixed-size buffer storage and interleaved training steps.
* **EWCAgent**: Implementation of the EWC loss function and Fisher Information estimation with numerical stability fixes.
