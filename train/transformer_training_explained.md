## Neural Transformer Training on HDF5 Trajectory Datasets

This document explains how the neural transformer model is trained on the HDF5 trajectory datasets when `train/train.py` is executed. It first gives a high‑level view of the pipeline, then describes the key files in detail.

---

### High‑Level Pipeline

- **Entry point & configuration**
  - You run `train/train.py` with a YAML config (e.g. `cfg/Ant/transformer.yaml`).
  - `train.py` parses CLI args, loads the YAML into `cfg`, applies overrides, and chooses:
    - **Environment**: `NeuralEnvironment` with `neural_integrator_cfg['name'] == 'TransformerNeuralIntegrator'`.
    - **Algorithm**: `SequenceModelTrainer` when `cfg['algorithm']['name'] == 'SequenceModelTrainer'`.
  - For transformer configs it enforces:
    - `cfg['env']['neural_integrator_cfg']['name'] == 'TransformerNeuralIntegrator'`
    - `cfg['env']['neural_integrator_cfg']['num_states_history'] == cfg['algorithm']['sample_sequence_length']`

- **Environment and neural integrator**
  - `NeuralEnvironment` wraps a Warp simulation (`create_abstract_contact_env`) and creates `TransformerNeuralIntegrator`.
  - The neural integrator exposes helper methods used during training:
    - `get_contact_masks(contact_depths, contact_thicknesses)`
    - `process_neural_model_inputs(data_dict)` to assemble model inputs from raw tensors.
    - `convert_next_states_to_prediction(states, next_states, dt)` and the inverse for evaluation.

- **Model construction (transformer variant)**
  - `VanillaTrainer.__init__` (base class of `SequenceModelTrainer`) obtains an **input sample** from the integrator via:
    - `neural_integrator.get_neural_model_inputs()`.
  - It then builds `ModelMixedInput` with:
    - `input_sample`: example input dictionary (keys like `states`, `states_embedding`, `joint_acts`, `gravity_dir`, contact features).
    - `input_cfg`: which keys are used and how they are grouped (e.g. `inputs.low_dim`).
    - `network_cfg`: includes an `encoder` section, `transformer` section, and `model` (output head) section.
  - Inside `ModelMixedInput`:
    - Encoders (`MLPBase`) are built for each configured input group (e.g. `low_dim`).
    - If `network_cfg` has a `transformer` section:
      - A `GPTConfig` is instantiated with:
        - `n_layer`, `n_head`, `n_embd`, `block_size`, `bias`, `dropout`.
        - `vocab_size` is set to the **concatenated feature dimension** produced by encoders.
      - A `GPT` transformer is constructed and moved to the target device.
      - `self.is_transformer = True` and `self.feature_dim` is set to `n_embd`.
    - A final `MLPDeterministic` head maps transformer features to the integrator’s `prediction_dim` (e.g. delta states).

- **HDF5 trajectory dataset loading**
  - Because the algorithm is `SequenceModelTrainer`, the overridden `get_datasets` uses `TrajectoryDataset` (not `BatchTransitionDataset`):
    - `TrajectoryDataset(sample_sequence_length, hdf5_dataset_path, max_capacity)`.
  - `TrajectoryDataset` expects the HDF5 file layout:
    - Group `data` with attribute `mode == 'trajectory'`.
    - Datasets like `states`, `next_states`, contact fields, actions, etc. with shape `(T, B, feature_dims...)`.
    - Optional `traj_lengths` giving the valid length of each trajectory.
  - Internally, `TrajectoryDataset`:
    - Truncates to `num_trajectories` based on `max_capacity`.
    - Transposes each dataset from shape `(T, B, ...)` to `(B, T, ...)`.
    - Flattens feature dimensions to `(B, T, dim)`.
    - Precomputes a mapping from a global sample index to `(traj_idx, start_step)` for sliding windows of length `sample_sequence_length`.

- **Batched sequence sampling**
  - `VanillaTrainer.train()` creates a `torch.utils.data.DataLoader` on `self.train_dataset` with:
    - `batch_size = algo_cfg['batch_size']`
    - `collate_fn = None` (default) for `TrajectoryDataset`.
  - Each batch from the `DataLoader` is a dictionary:
    - For each key (e.g. `states`, `next_states`, contact tensors, actions):
      - Shape `(B, sample_sequence_length, dim)`.
  - For transformer training the sequence length is:
    - `T = cfg['algorithm']['sample_sequence_length']`
    - Must be ≤ `network_cfg['transformer']['block_size']` to satisfy GPT’s `block_size` constraint.

- **Preprocessing before the model**
  - Each batch goes through `VanillaTrainer.preprocess_data_batch(data)`:
    - All tensors are moved to `self.device` (CUDA).
    - Contact masks are computed:
      - `data['contact_masks'] = neural_integrator.get_contact_masks(data['contact_depths'], data['contact_thicknesses'])`.
    - The integrator augments `data` in-place via:
      - `neural_integrator.process_neural_model_inputs(data)` to build fields used by `ModelMixedInput` encoders.
    - The **prediction target** is computed using the neural integrator:
      - `data['target'] = neural_integrator.convert_next_states_to_prediction(states=data['states'], next_states=data['next_states'], dt=neural_env.frame_dt)`.
  - Result: `data` contains all model inputs plus a `target` tensor of shape `(B, T, prediction_dim)`.

- **Forward pass through encoders + transformer + head**
  - `VanillaTrainer.compute_loss(data, train)` calls:
    - `prediction = self.neural_model(data)`, where `self.neural_model` is `ModelMixedInput`.
  - Inside `ModelMixedInput.forward(input_dict)` for transformer:
    1. **Input normalization (optional)**  
       - If `network_cfg.normalize_input` is `True`, each key in `self.input_rms` is normalized using `RunningMeanStd`.
    2. **Encoder stage**
       - `extract_input_features` concatenates all configured inputs, each passed through its encoder (e.g. `MLPBase`).
       - This yields `features` of shape `(B, T, feature_dim)` where `feature_dim` equals `vocab_size` in `GPTConfig`.
    3. **Optional RNN**
       - For transformer configs `is_rnn == False`, so this step is skipped.
    4. **Transformer stage**
       - `features = self.transformer_model(features)`:
         - `GPT.forward(idx)` treats `features` as a sequence of continuous tokens:
           - `tok_emb = Linear(features)` maps `(B, T, feature_dim)` to `(B, T, n_embd)`.
           - Adds positional embeddings (`wpe`) and applies dropout.
           - Passes through `n_layer` `Block`s (LayerNorm + causal self‑attention + MLP).
           - Applies final LayerNorm and linear projection `lm_head` back to `(B, T, n_embd)`.
    5. **Output head**
       - `features` (now `(B, T, n_embd)`) is reshaped to `(B*T, n_embd)`.
       - `MLPDeterministic` maps to `(B*T, prediction_dim)` and is reshaped back to `(B, T, prediction_dim)`.
       - Optional `tanh` is applied if `output_tanh` is enabled.
       - If `normalize_output` is `True`, it uses `self.output_rms` (set from dataset statistics) to un‑normalize outputs.

- **Loss and optimization**
  - `VanillaTrainer.compute_loss`:
    - Reads `prediction_target = data['target']`.
    - Computes per‑dimension weights:
      - If `normalize_output`: `loss_weights = 1 / sqrt(output_rms.var + 1e-5)`.
      - Else: `loss_weights = 1`.
    - Uses MSE over weighted predictions:
      - `loss = MSE(prediction * loss_weights, prediction_target * loss_weights)`.
  - It also constructs **state‑space error statistics**:
    - Converts predictions back to next states:
      - `predicted_next_states = neural_integrator.convert_prediction_to_next_states(states=data['states'], prediction=prediction)`.
      - `neural_integrator.wrap2PI(predicted_next_states)` to keep angles bounded.
    - Logs per‑state and grouped errors (joint position / velocity norms and MSE).
  - `VanillaTrainer.one_epoch`:
    - For each batch, if `train == True`:
      - Zeroes gradients, backpropagates `loss.backward()`.
      - Optionally clips gradients with `clip_grad_norm_`.
      - Steps the optimizer (`Adam`) with scheduled learning rate.

- **Training loop and evaluation**
  - `VanillaTrainer.train()`:
    - Creates train and validation `DataLoader`s.
    - For each epoch:
      - Schedules learning rate (`constant`, `linear`, or `cosine`).
      - Runs `one_epoch(train=True, ...)` on the train set.
      - Runs `one_epoch(train=False, ...)` on each validation dataset.
      - Periodically calls `self.eval(epoch)`:
        - Uses `NeuralSimEvaluator` to run rollouts (using `NeuralEnvironment` in neural mode).
      - Logs metrics to TensorBoard and saves model checkpoints (best val, best eval, periodic, and final).

---

### Pipeline Diagrams

**Data and training flow**

```text
HDF5 file (mode='trajectory')
  └─ group 'data' (states, next_states, contacts, actions, ...)
          │
          ▼
TrajectoryDataset (sliding windows of length sample_sequence_length)
          │
          ▼
DataLoader (batches of sequences: dict[key] ∈ ℝ^{B×T×dim})
          │
          ▼
VanillaTrainer.preprocess_data_batch
  ├─ move to device
  ├─ compute contact_masks
  ├─ process_neural_model_inputs (integrator)
  └─ build 'target' (prediction target)
          │
          ▼
ModelMixedInput
  ├─ encoders (MLPs) → features ∈ ℝ^{B×T×feature_dim}
  ├─ GPT transformer (causal self-attention over time)
  └─ MLPDeterministic head → prediction ∈ ℝ^{B×T×prediction_dim}
          │
          ▼
MSE loss vs 'target' (+ state-based metrics)
          │
          ▼
Adam optimizer + LR schedule
```

**Call‑graph oriented view**

```text
train/train.py
  ├─ parse args, load YAML cfg
  ├─ NeuralEnvironment(**cfg['env'])
  │     └─ TransformerNeuralIntegrator(model, **neural_integrator_cfg)
  ├─ SequenceModelTrainer(neural_env, cfg, device)
  │     ├─ VanillaTrainer.__init__
  │     │     ├─ neural_integrator.get_neural_model_inputs()
  │     │     ├─ ModelMixedInput(input_sample, output_dim, input_cfg, network_cfg)
  │     │     └─ get_datasets(...)  (overridden)
  │     └─ SequenceModelTrainer.get_datasets
  │           └─ TrajectoryDataset(sample_sequence_length, hdf5_path, max_capacity)
  └─ if args.train:
        algo.train()
      else:
        algo.test()
```

---

## File‑by‑File Explanation

### `train/train.py`

- **What it does**
  - Acts as the main entry point for training and testing NeRD models, including the transformer‑based integrator.
  - Handles CLI parsing, YAML config loading, seeding, logging directory setup, and algorithm selection.
- **How it works for transformer training**
  - After loading `cfg`, it creates `NeuralEnvironment(**cfg['env'], device=args.device)`.
  - It reads `cfg['algorithm']['name']` to choose between:
    - `VanillaTrainer` (for standard, per‑step models).
    - `SequenceModelTrainer` (for sequence models like transformer or RNN).
  - For `SequenceModelTrainer`:
    - If `'transformer'` is in `cfg['network']`:
      - Asserts `cfg['env']['neural_integrator_cfg']['name'] == 'TransformerNeuralIntegrator'`.
      - Asserts `neural_integrator_cfg['num_states_history'] == cfg['algorithm']['sample_sequence_length']`.
    - Instantiates `SequenceModelTrainer(neural_env, cfg, model_checkpoint_path=args.checkpoint, device=args.device)`.
  - Finally:
    - Calls `algo.train()` when `args.train` is `True`.
    - Calls `algo.test()` otherwise.

### `algorithms/sequence_model_trainer.py`

- **What it does**
  - Specializes `VanillaTrainer` for **sequence models** (transformers, RNNs) that operate on windows of trajectories instead of single transitions.
- **How it works**
  - Stores `sample_sequence_length = cfg['algorithm'].get('sample_sequence_length', 1)` before calling `VanillaTrainer.__init__`.
  - Overrides `get_datasets`:
    - Uses `TrajectoryDataset` for both training and validation:
      - `self.train_dataset = TrajectoryDataset(sample_sequence_length, train_dataset_path, max_capacity)`.
      - For each validation set, constructs another `TrajectoryDataset` with the same `sample_sequence_length`.
    - Keeps `self.collate_fn = None`, so the default PyTorch collation stacks trajectories into shape `(B, T, dim)`.
  - All remaining training logic (statistics, loss, optimization, logging, rollouts) is inherited from `VanillaTrainer`.

### `algorithms/vanilla_trainer.py`

- **What it does**
  - Implements the **generic training pipeline** used by both per‑step and sequence models.
  - Handles model creation, dataset loading, statistics, training loop, validation, and evaluation.
- **How it works (relevant parts for transformer training)**
  - **Model creation**
    - Retrieves the neural integrator from `NeuralEnvironment`:
      - `self.neural_integrator = neural_env.integrator_neural`.
    - If `model_checkpoint_path` is `None`:
      - Calls `input_sample = self.neural_integrator.get_neural_model_inputs()`.
      - Creates `ModelMixedInput(input_sample, output_dim=neural_integrator.prediction_dim, input_cfg=cfg['inputs'], network_cfg=cfg['network'])`.
    - Assigns the model to the integrator via `self.neural_integrator.set_neural_model(self.neural_model)`.
  - **Dataset handling**
    - Reads paths and settings from `algo_cfg['dataset']` and defers to `self.get_datasets(...)`:
      - For transformers, this is the overridden method in `SequenceModelTrainer` that uses `TrajectoryDataset`.
  - **Dataset statistics**
    - Optionally computes per‑key mean/std over the train dataset via `compute_dataset_statistics`:
      - Wraps the dataset in a `DataLoader`.
      - For each batch: runs `preprocess_data_batch`, then updates `RunningMeanStd` for each key (including `target`).
    - Sets normalization objects on the model:
      - `self.neural_model.set_input_rms(self.dataset_rms)`.
      - `self.neural_model.set_output_rms(self.dataset_rms['target'])`.
  - **Preprocessing and loss**
    - `preprocess_data_batch`:
      - Moves every tensor to the chosen device.
      - Computes `contact_masks` (via integrator).
      - Calls `self.neural_integrator.process_neural_model_inputs(data)`.
      - Computes `data['target']` from `(states, next_states, dt)`.
    - `compute_loss`:
      - Optionally initializes RNN state (not used for transformer).
      - Calls `self.neural_model(data)` → `prediction`.
      - Builds `loss_weights` (based on output normalization) and computes weighted MSE vs `data['target']`.
      - Builds detailed state error metrics using integrator’s conversion utilities.
  - **Training loop (`train`)**
    - Wraps the training dataset in a `DataLoader` (sequence dataset for transformer).
    - For each epoch:
      - Computes learning rate via `get_scheduled_learning_rate`.
      - Runs `one_epoch` on train set (for `epoch > 0`) and on each validation set.
      - Runs `eval` periodically (NeuralSim rollouts).
      - Logs scalars and saves checkpoints (current, best validation, best eval, final).

### `utils/datasets.py`

- **What it does**
  - Provides dataset wrappers for HDF5 files:
    - `BatchTransitionDataset` for per‑transition training.
    - `TrajectoryDataset` for sequence‑based training (used by transformer and RNN models).
- **How `TrajectoryDataset` works for transformer training**
  - `load_dataset(hdf5_dataset_path)`:
    - Opens the file using `h5py.File(..., swmr=True, libver='latest')`.
    - Reads `mode = dataset['data'].attrs['mode']` and asserts `mode == 'trajectory'`.
    - Determines `num_transitions_per_trajectory` from `dataset['data']['states'].shape[0]` (T dimension).
    - Computes `num_trajectories` based on `max_capacity` and the dataset size.
    - For each key under `data`:
      - If key is `traj_lengths`: loads per‑trajectory valid lengths.
      - Else:
        - Slices to `[ :, :num_trajectories, ... ]`, type‑casts to `float32`.
        - Swaps axes from `(T, B, ...)` to `(B, T, ...)`.
        - Flattens feature dims to `(B, T, dim)` and stores as a NumPy array.
    - If `traj_lengths` is missing, assumes all trajectories are full length.
  - `update_sample_sequence_length(sample_sequence_length)`:
    - Stores `sample_sequence_length` and calls `build_index()`.
  - `build_index()`:
    - Computes total number of valid sequence windows:
      - For each trajectory `i`, adds:
        - `max(0, traj_lengths[i] - sample_sequence_length + 1)` windows.
    - Fills `self.mapping_index2traj` with `(traj_idx, start_step)` for each global sample index.
  - `__len__` / `__getitem__`:
    - `__len__` returns the number of windows.
    - `__getitem__(index)`:
      - Maps `index` to `(traj_idx, traj_step_idx)`.
      - For each key `k`, slices:
        - `self.dataset[k][traj_idx, traj_step_idx : traj_step_idx + sample_sequence_length]`.
      - Wraps slices as `torch.Tensor` and returns a dictionary.
  - When wrapped in a `DataLoader`:
    - Default collation stacks the per‑window tensors into `B×T×dim` tensors, exactly matching what `ModelMixedInput` and the transformer expect.

### `envs/neural_environment.py`

- **What it does**
  - Wraps the Warp simulation environment and exposes a clean PyTorch‑friendly API (`NeuralEnvironment`) for:
    - Stepping the environment using either ground‑truth or neural integrators.
    - Maintaining generalized state tensors.
    - Handling rendering and rollouts.
- **How it participates in transformer training**
  - At construction time, given `neural_integrator_cfg['name']`:
    - Creates a `TransformerNeuralIntegrator` instance for `neural_integrator_type == 'TransformerNeuralIntegrator'`.
  - Exposes:
    - `integrator_neural`: used by `VanillaTrainer` to access:
      - `get_neural_model_inputs`.
      - `get_contact_masks`.
      - `process_neural_model_inputs`.
      - `convert_next_states_to_prediction` and `convert_prediction_to_next_states`.
    - `frame_dt`: simulation frame time, passed into integrator conversions to compute derivative‑type targets.
  - During evaluation (`NeuralSimEvaluator`), `NeuralEnvironment`:
    - Switches between ground‑truth and neural integrators.
    - Performs rollouts to measure how well the learned transformer dynamics match the simulator.

### `integrators/integrator_neural_transformer.py`

- **What it does**
  - Specializes `StatefulNeuralIntegrator` for transformer‑based training and inference.
  - Maintains a **history of states and related inputs** so that the transformer can see a window of past frames.
- **How it works**
  - Inherits from `StatefulNeuralIntegrator` and thus shares all of its core functionality (pre‑ and post‑processing, conversion between state and prediction spaces).
  - `reset_states_history`:
    - Initializes `self.states_history` as a `deque` with `maxlen=self.num_states_history`.
  - `get_neural_model_inputs`:
    - If `states_history` is empty (e.g. when building `input_sample` before real data):
      - Creates a dictionary of zero tensors (with a fake time dimension of length 1) for:
        - `root_body_q`, `states`, `states_embedding`, `joint_acts`, `gravity_dir`, and contact‑related entries.
      - This defines the **shape and keys** for `ModelMixedInput`’s encoders.
    - Else:
      - Collates the entries in `states_history` using `torch.utils.data.default_collate`.
      - Permutes from `(history_length, B, dim)` to `(B, history_length, dim)` for each key.
      - Passes the result through `process_neural_model_inputs` to obtain the same processed structure as in training.
  - Alignment with training:
    - The `num_states_history` is forced to match `sample_sequence_length`, ensuring that:
      - The transformer always sees sequences of the same length during training and inference.

### `models/models.py` (transformer integration)

- **What it does**
  - Defines generic model wrappers for mixed inputs:
    - Encoders for different input groups (e.g. `low_dim`).
    - Optional RNN layers.
    - Optional transformer (GPT) sequence model.
    - Final MLP head for regression to the integrator’s prediction space.
- **How it works for transformer training**
  - `construct_input_encoders`:
    - Builds a `ModuleDict` of encoders based on `input_cfg` and `encoder_cfg`:
      - Aggregates all `low_dim` inputs (e.g. `states_embedding`, `joint_acts`, contact features, gravity direction) into a single encoder.
    - Computes `feature_dim` as the sum of encoder output sizes.
  - Transformer configuration:
    - When `"transformer"` is present in `network_cfg`:
      - Sets `vocab_size = feature_dim`.
      - Builds `GPTConfig` using parameters from `network_cfg['transformer']`.
      - Instantiates `self.transformer_model = GPT(gptconf)` and sets `self.is_transformer = True`.
      - Updates `feature_dim` to be `gptconf.n_embd` for the downstream MLP head.
  - Forward logic:
    - Both `evaluate` (single‑step) and `forward` (multi‑step) paths:
      - Build feature sequences via encoders.
      - Optionally pass through an RNN (not used for transformer setups).
      - Pass through the transformer model if enabled.
      - Apply the final `MLPDeterministic` head and optional output (un)normalization.
    - For training, `forward` is used:
      - Input dict: keys from the dataset and integrator processing.
      - Output: `prediction` with the same `(B, T, prediction_dim)` layout as `target`.

### `models/model_transformer.py`

- **What it does**
  - Implements a GPT‑style transformer, adapted from nanoGPT, but used here as a **continuous sequence feature extractor** rather than a language model.
- **How it works in this project**
  - `GPTConfig`:
    - Standard transformer hyperparameters (`n_layer`, `n_head`, `n_embd`, `block_size`, `dropout`, `bias`, `vocab_size`).
  - `GPT`:
    - Defines:
      - `wte`: a `Linear(vocab_size → n_embd)` that maps each per‑time‑step feature vector into an embedding (no discrete tokenization).
      - `wpe`: positional embeddings of shape `(block_size, n_embd)`.
      - A stack of `Block` modules (LayerNorm + `CausalSelfAttention` + MLP).
      - `lm_head`: a `Linear(n_embd → n_embd)` projection.
    - `forward(idx)`:
      - Expects `idx` of shape `(B, T, feature_dim)` where `feature_dim` equals `vocab_size`.
      - Applies `wte(idx)` and `wpe` (position‑wise), sums them, then applies dropout.
      - Runs the sequence through the stack of transformer blocks with **causal attention** (no future leakage).
      - Applies a final LayerNorm and `lm_head`, returning `(B, T, n_embd)`.
  - In training:
    - This transformer output is not directly compared to targets.
    - Instead, it feeds into the MLP head in `ModelMixedInput`, which produces the final prediction that is used in the MSE loss.

---

This documentation covers how, starting from `train/train.py`, the HDF5 trajectory dataset is turned into windowed sequences, preprocessed, passed through encoders and the GPT‑style transformer, and finally optimized via MSE loss against the integrator’s prediction targets.

