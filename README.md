# NMT Bug-Fixing Replication

A PyTorch replication of the paper *"An Empirical Study on Learning Bug-Fixing Patches in the Wild via Neural Machine Translation"* (Tufano et al., TOSEM 2019).

This project investigates the feasibility of using Neural Machine Translation (NMT) to automatically generate bug-fixing patches for Java methods. Given a buggy Java method, the model translates it into a fixed version by learning from real-world bug-fixing commits mined from GitHub repositories.

## Results

Two datasets are evaluated: **small** (up to 50 tokens) and **medium** (50 to 100 tokens). For each buggy snippet, the model generates $k \in [1, 50]$ patch candidates using vectorized beam search.

### 1. Accuracy & Syntactic Correctness Benchmarking

- **Accuracy (Perfect Predictions):** Percentage of test bugs where at least one of the $k$ generated candidates exactly matches the human-written fix.
- **Syntactic Correctness:** Percentage of total generated candidates ($\text{bugs} \times k$) that are syntactically valid Java methods (verified via `javalang`).

#### Small dataset (test: 5835 bugs)
<img width="4200" height="1500" alt="charts_small" src="https://github.com/user-attachments/assets/2682e291-60de-44fe-bbe1-70d44268a6ad" />


#### Medium dataset (test: 6546 bugs)
<img width="4200" height="1500" alt="charts_medium" src="https://github.com/user-attachments/assets/5affce54-ccaa-4b63-a710-b88f94b797e0" />


- **Exact-Match Accuracy:** Closely tracks the paper's baseline on the small dataset (**49.2%** vs. 50.2% at $k=50$) and outperforms the baseline on the medium dataset (**31.7%** vs. 28.6% at $k=50$).
- **Syntax Stability:** Maintains **>96.7%** syntactic validity across all beam widths on both datasets, avoiding the baseline's syntax degradation at wider beams (82%–85%).

---

### 2. Training & Inference Time Profiling

- **Training Time:** Using length-based bucketing (`BucketBatchSampler`) on single GPU T4 (Google Colab) and dual GPU P100 (Kaggle) reduces training time to **~2.5 hours** ($M_{small}$) and **~6 hours** ($M_{medium}$), compared to 6 hours and 15 hours on three GPUs in the original paper.
- **Inference Profiling Setup:** Measured across all beam widths using `time.perf_counter()` paired with explicit **CUDA barrier synchronization** (`torch.cuda.synchronize()`) on single GPU T4.

#### Small dataset (test: 5835 bugs)
<img width="2400" height="1500" alt="time_chart_small" src="https://github.com/user-attachments/assets/336e8410-289d-4a2a-8761-86af192c4649" />


#### Medium dataset (test: 6546 bugs)
<img width="2400" height="1500" alt="time_chart_medium" src="https://github.com/user-attachments/assets/51c034ed-c547-4ceb-b650-0b1c10e3ff25" />


- **Avg. Time per Bug (Green Line):** By batching $k$ beam branches in parallel on the GPU, scaling $k$ by $50\times$ only slightly increases per-bug latency from **0.038s to 0.060s** ($M_{small}$) and from **0.093s to 0.128s** ($M_{medium}$).
- **Avg. Time per Patch (Orange Line):** Amortized generation latency drops sharply as $k$ grows, converging to **~1.2ms/patch** ($M_{small}$) and **~2.6ms/patch** ($M_{medium}$) at $k=50$.



## Getting Started

Clone the repository and enter the project directory:

```bash
git clone https://github.com/khoa-td/nmt-bugfix-replication.git
cd nmt-bugfix-replication
```

---

## Option 1: Running Locally (Standard Setup)

Use this workflow to run the project directly on your machine. 

**1. Create and activate a virtual environment**
*(Required on modern Linux/macOS to avoid system package conflicts)*
```bash
python3 -m venv .venv           # or python -m venv .venv On Windows CMD
source .venv/bin/activate       # On Linux/macOS/WSL
# .venv\Scripts\activate        # On Windows CMD/PowerShell
```

**2. Install dependencies**
*(Note: We use `--no-cache-dir` to prevent disk space exhaustion, as PyTorch is a very large package).*
```bash
pip install --no-cache-dir -r requirements.txt
```

**3. Download datasets and pretrained checkpoints**
Downloads the pre-abstracted corpora and model weights from Hugging Face (`KhoaDuongDang/nmt-bugfix-data`):
```bash
# Download pre-abstracted dataset (extracts to myabs/out/)
python scripts/download_data.py abstract.zip

# (Optional) Download pretrained checkpoints & precomputed predictions to skip training/inference
python scripts/download_data.py checkpoints.zip
```
> **Note:** `abstract.zip` already contains the abstracted code ready for training. If you want to process the raw Java files from scratch using the Java AST tool (see the **Java Code Abstraction** section at the bottom), **do not** download `abstract.zip` here—only download `datasets.zip` in that section so you do not download both.


**4. Preprocess data and build vocabularies**
Splits the abstracted corpora into `train`/`eval`/`test` sets and builds token vocabularies in `processed/`:
```bash
python src/ProcessingData.py
```

**5. Train the NMT model**
*(Optional: Skip this step if you downloaded the pretrained checkpoints in Step 3)*
```bash
python src/train.py small
python src/train.py medium
```

**6. Run Beam Search evaluation and compute scores**
Generates candidate patches across beam widths $K \in \{1, 5, \dots, 50\}$ and evaluates both exact match accuracy and Java syntactic validity:

> **Note:** If you downloaded `checkpoints.zip` in the preparation step, you already have the precomputed predictions. You can skip the `evaluate.py` commands below and run `scores.py` directly to see the results instantly.

```bash
# For Small dataset
python src/evaluate.py small
python src/scores.py small

# For Medium dataset
python src/evaluate.py medium
python src/scores.py medium
```

**7. Generate comparison plots**
```bash
python src/visualize.py small
python src/visualize.py medium
```

---

## Option 2: Running with Docker

Use this workflow to run the project inside an isolated container with CUDA, Python, and Java pre-installed. No virtual environment is needed.

**1. Build the Docker image**
```bash
docker build -t nmt-bugfix .
```

**2. Start the container**
```bash
docker run --gpus all -it --rm nmt-bugfix
```

*(Note: Omit `--gpus all` if running on a CPU-only machine).*


**3. Run the pipeline inside the container**
```bash
# 1. Download pre-abstracted dataset
python scripts/download_data.py abstract.zip

# 2. (Optional) Download pretrained checkpoints & precomputed predictions to skip training/inference
python scripts/download_data.py checkpoints.zip

# 3. Preprocess data and build vocabularies
python src/ProcessingData.py

# 4. (Optional) Train from scratch — skip this if you downloaded the checkpoints above
python src/train.py small
python src/train.py medium

# 5. Evaluate and compute scores 
# (Note: You can skip the 'evaluate.py' commands if you downloaded checkpoints.zip in step 2)
python src/evaluate.py small
python src/scores.py small
python src/evaluate.py medium
python src/scores.py medium


# 6. Generate plots
python src/visualize.py small
python src/visualize.py medium
```

---

## Java Code Abstraction (Optional)

The datasets downloaded via `python scripts/download_data.py abstract.zip` are already abstracted. If you wish to run the Java AST abstraction tool (`myabs/`) on the raw Java methods from scratch instead (skipping `abstract.zip` in Step 3 above):

**1. Download the raw Java dataset and required JAR libraries**
```bash
python scripts/download_data.py datasets.zip
python scripts/setup_java_libs.py
```
*(This extracts the raw buggy/fixed Java methods into `datasets/` at the project root and downloads the required JARs into `myabs/lib/`).*

**2. Compile and run the Java abstractor**
Navigate into `myabs/` so the tool reads `idioms/idioms.csv` and outputs the abstracted files directly to `myabs/out/small/` and `myabs/out/medium/`:
```bash
cd myabs
javac -cp "lib/*" src/*.java -d bin

# Run abstraction for Small and Medium datasets (use "bin;lib/*" on Windows CMD/PowerShell)
java -cp "bin:lib/*" Main ../datasets/small/buggy ../datasets/small/fixed small
java -cp "bin:lib/*" Main ../datasets/medium/buggy ../datasets/medium/fixed medium

cd ..
```
Once finished and back at the project root (`cd ..`), continue directly from **Step 4** (`python src/ProcessingData.py`) in Option 1.

---
