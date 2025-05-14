# 🧠 PyTorch Neural Network Collection

A modular collection of different neural network architectures implemented in **PyTorch**. Each network type lives in its own folder with separate code, data, and training logic — making it easy to understand, modify, and extend.

---

## 📁 Project Structure

Each folder represents a standalone neural network project. You can navigate into any of them to see details, run training, and inspect the model implementation.

## 🧠 Included Models

- **MLP (Multi-Layer Perceptron)** – for basic classification tasks using fully connected layers.
- **CNN (Convolutional neural network)** – feedforward neural network that learns features via filter (or kernel) optimization.
- comming soon

More model types (RNN, etc.) will be added progressively.

## 📦 Dependencies

#### Recommended: Install PyTorch with CUDA >= 12.1 Support (for NVIDIA GPUs)

If you have a compatible NVIDIA GPU (e.g., RTX 3060), you should install PyTorch with CUDA to take advantage of accelerated training and inference:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### Not Recommended: Install PyTorch without CUDA supprot
if you want to train and test it on CPU only, make sure to install the required dependencies before running any scripts. Most projects use:

```bash
pip install torch torchvision torchaudio
```
