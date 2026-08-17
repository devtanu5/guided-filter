# Guided Image Filtering Studio 🎨✨

A modern, high-performance Python implementation of the classic computer vision algorithm **"Guided Image Filtering"** by Kaiming He, Jian Sun, and Xiaoou Tang (TPAMI '12).

This project features advanced guided filtering applications—such as **single-image haze removal (dehazing)** and **edge-preserving detail enhancement**—packaged into a clean Command Line Interface (CLI) and a beautiful, interactive **Streamlit Web Application**.

---

## 🚀 Key Features

*   **Fast Box Filtering $O(N)$**: Core algorithm written in NumPy with integral/summed area-table box filter executing in linear time independent of kernel/filter size.
*   **Single-Image Dehazing**: Implementation of He's **Dark Channel Prior (DCP)** algorithm. Coarse transmission maps are refined using the Guided Filter to recover clean scene radiance without halos.
*   **Detail Enhancement**: Advanced edge-preserving detail sharpening. Color shifted errors are prevented using YCrCb luminance-only detail extraction.
*   **Joint Guided Filtering**: Filter a noisy or target image using a separate high-quality guide image (e.g. Flash/No-flash denoising).
*   **Interactive UI Studio**: A modern Streamlit dashboard to experiment with filter parameters in real-time, inspect intermediate outputs (transmission maps, dark channels), and export results.

---

## 🛠️ Installation & Setup

1.  **Clone the Repository**:
    ```bash
    git clone <your-github-repo-url>
    cd guided-filter
    ```

2.  **Create and Activate a Virtual Environment**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use: venv\Scripts\activate
    ```

3.  **Install Dependencies**:
    You can install the package in editable mode:
    ```bash
    pip install -e .
    ```
    Or install dependencies directly:
    ```bash
    pip install -r requirements.txt
    ```

---

## 🖥️ Interactive Web Studio

To launch the interactive web application, run:
```bash
streamlit run app.py
```
This opens a local server (default: `http://localhost:8501`) where you can:
*   Upload your own images or run presets.
*   Select the application: **Smoothing**, **Detail Enhancement**, **Dehazing**, or **Joint Filtering**.
*   Interactively adjust filter radius ($r$) and edge-preservation regularization ($\epsilon$) using sliders.
*   Visualize intermediate maps like the **Dark Channel** and **Guided Transmission Map** side-by-side.
*   Download processed results as high-resolution PNGs.

---

## ⌨️ Command Line Interface (CLI)

The command line utility `guided-filter-cli` (or running `python main.py` directly) supports all application tasks.

### 1. Edge-Preserving Smoothing
Smooth an image while preserving sharp edges:
```bash
python main.py --input data/cat.png --output smoothed_cat.png --task smooth --radius 4 --eps 0.04
```

### 2. Detail Enhancement (Sharpening)
Amplify details and sharpen the image (without creating halos):
```bash
python main.py --input data/Lenna.png --output enhanced_lenna.png --task enhance --radius 8 --eps 0.04 --factor 3.5
```

### 3. Single-Image Dehazing
Remove haze and restore true colors:
```bash
python main.py --input path/to/hazy_image.png --output clear_scene.png --task dehaze --omega 0.95 --guided-radius 40 --guided-eps 0.001
```

### 4. Joint Guided Filtering
Smooth a target image using a separate guide image:
```bash
python main.py --input target_noisy.png --guide guide_clean.png --output filtered.png --task joint --radius 4 --eps 0.01
```

---

## 🧠 Mathematical Background

### Guided Filter Formulation
The guided filter assumes a local linear relationship between the guide image $I$ and the filter output $q$ in a window $\omega_k$ centered at pixel $k$:

$$q_i = a_k I_i + b_k, \quad \forall i \in \omega_k$$

To determine the coefficients $a_k$ and $b_k$, we minimize the squared difference between the output $q$ and input $p$ while regularizing the coefficient magnitude:

$$E(a_k, b_k) = \sum_{i \in \omega_k} \left( (a_k I_i + b_k - p_i)^2 + \epsilon a_k^2 \right)$$

Solving via linear regression yields:

$$a_k = \frac{\frac{1}{|\omega|} \sum_{i\in\omega_k} I_i p_i - \mu_k \bar{p}_k}{\sigma_k^2 + \epsilon}, \quad b_k = \bar{p}_k - a_k \mu_k$$

where $\mu_k$ and $\sigma_k^2$ are the mean and variance of $I$ in $\omega_k$, and $\bar{p}_k$ is the mean of $p$ in $\omega_k$. Because a pixel $i$ is involved in multiple overlapping windows $\omega_k$, the final output is averaged:

$$q_i = \bar{a}_i I_i + \bar{b}_i$$

### Single-Image Dehazing via Dark Channel Prior (DCP)
The haze model is defined as:

$$I(x) = J(x)t(x) + A(1 - t(x))$$

where $I(x)$ is the observed intensity, $J(x)$ is the scene radiance (dehazed image), $t(x)$ is the transmission map, and $A$ is the atmospheric light. The coarse transmission map is estimated by:

$$t_{\text{rough}}(x) = 1 - \omega \cdot \min_c \left( \min_{y\in\Omega(x)} \frac{I_c(y)}{A_c} \right)$$

This coarse map contains blocking artifacts around object boundaries. We run the **Guided Filter** with the hazy image as the guide to refine $t_{\text{rough}}(x)$ into a smooth, edge-aligned transmission map $t(x)$. The clean scene is then recovered via:

$$J(x) = \frac{I(x) - A}{\max(t(x), t_0)} + A$$

---

## 📁 Repository Structure

```
guided-filter/
├── core/
│   ├── __init__.py
│   ├── filter.py          # GuidedFilter and MultiDimGuidedFilter classes
│   └── applications.py    # Detail enhancement, joint filtering, and dehazing
├── cv/
│   ├── __init__.py
│   ├── image.py           # Image format conversion (float32, uint8)
│   ├── pad.py             # Boundary padding functions (reflect, edge, zero)
│   └── smooth.py          # Box filter implementation
├── data/                  # Sample datasets and visualizations
├── test/
│   └── test_box_filter.py # Unit tests matching OpenCV blur outputs
├── tools/
│   └── visualize.py       # Helper functions for local matplotlib rendering
├── app.py                 # Streamlit Web App
├── main.py                # Command Line Interface (CLI) Entrypoint
├── pyproject.toml         # Package setup file
└── requirements.txt       # Project dependencies
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
