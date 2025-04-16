<<<<<<< HEAD
# Microstructure Image GAN

This project trains a Generative Adversarial Network (GAN) to synthesize realistic microstructure images based on input RGB image data. It supports multi-class training and saves both intermediate and final results for further analysis.

---

## Installation Guide

Follow the steps below to set up the environment and run the project.

### ✅ Step 1: Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html)

#### 🔹 Windows

1. Download the Miniconda installer: [Miniconda for Windows](https://docs.conda.io/en/latest/miniconda.html#windows-installers)
2. Run the `.exe` installer and follow the prompts.
3. Open the **Anaconda Prompt** after installation.

#### 🔹 macOS

```bash
# Download and install Miniconda
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
bash Miniconda3-latest-MacOSX-x86_64.sh
```

Then restart your terminal.

#### 🔹 Linux

```bash
# Download and install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

Then restart your shell or run: `source ~/.bashrc`

---

### ✅ Step 2: Create the Conda Environment

Clone the repository (or download the files) and make sure you're in the root directory.

```bash
git clone https://github.com/cmaloney111/deep-microstructure-reconstruction.git
conda env create -f environment.yml
conda activate MicroRec
```

---

### ✅ Step 3: Navigate to the Training Script

```bash
cd src/new-final-gan
```

---

### ✅ Step 4: Prepare Your Data

Your RGB image folder should be structured like this:

```
/path/to/RGB-image-folder/
├── class_0/
│   ├── image_0.jpg
│   ├── image_1.jpg
│   └── ...
├── class_1/
│   ├── image_0.jpg
│   ├── image_1.jpg
│   └── ...
└── ...
```

Each subfolder represents a class label (e.g., microstructure type), and contains images in `.jpg`/`.jpeg`, `.png`, or `.tif` format.

---

### ✅ Step 5: Run the Training Script

```bash
python train.py --path /path/to/RGB-image-folder
```

**Arguments:**

- `--path`: Path to the root folder containing your image classes.

Example:

```bash
python train.py --path ./data/microstructures
```

---

## Outputs

After training, you will find:

- Generated images (images folder)
- Model checkpoints (models folder)
=======
# deep-microstructure-reconstruction
>>>>>>> 9934159ec532f3c5181e0e15ca62269aedc54c0a
