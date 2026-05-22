# OFDM Image Transmission System Simulation

This project implements an OFDM baseband communication system simulation driven by an image source in Python. The program converts a grayscale image to a bit stream, applies QPSK modulation, OFDM modulation, a multipath Rayleigh fading channel, AWGN noise, MMSE frequency-domain equalization, and QPSK demodulation to recover the image, and evaluates the impact of different cyclic prefix lengths using BER and PSNR.

The project is suitable as lab or teaching material for wireless communications, digital communications, OFDM principles, cyclic prefix, multipath channels, and frequency-domain equalization.

## Project Features

- Convert an input image to a binary bit stream and reconstruct a grayscale image at the receiver.
- Implement QPSK modulation and hard-decision demodulation.
- Implement 64-subcarrier OFDM modulation and demodulation.
- Compare different cyclic prefix lengths, including `CP = 0, 2, 4, 8, 16, 32`.
- Build a multipath Rayleigh fading channel with an exponential power delay profile.
- Add complex Gaussian white noise (AWGN) at the channel output.
- Use single-tap MMSE frequency-domain equalization.
- Output BER-SNR curves for different CP lengths.
- Output PSNR-SNR curves for different CP lengths.
- Summarize constellation diagrams before and after equalization at a given SNR for different CP lengths.
- Summarize the original and recovered images for different CP lengths.

## System Flow

```text
Input image
  -> Grayscale conversion and size normalization
  -> Image bit stream
  -> QPSK modulation
  -> Serial-to-parallel and subcarrier mapping
  -> IFFT to generate time-domain OFDM symbols
  -> Add cyclic prefix (CP)
  -> Multipath Rayleigh fading channel
  -> AWGN noise
  -> Remove cyclic prefix
  -> FFT to recover frequency-domain subcarriers
  -> MMSE frequency-domain equalization
  -> QPSK demodulation
  -> Bit stream recovery
  -> Image reconstruction
  -> BER / PSNR / constellation analysis
```

## Core Principles

### QPSK Modulation

The program maps every 2 bits to one QPSK symbol:

```text
00 -> (-1 - j) / sqrt(2)
01 -> (-1 + j) / sqrt(2)
10 -> ( 1 - j) / sqrt(2)
11 -> ( 1 + j) / sqrt(2)
```

All QPSK symbols are normalized so the average symbol power is approximately 1.

### OFDM Modulation

The program uses `N = 64` subcarriers. The transmitter groups QPSK symbols into blocks of 64 to form a frequency-domain OFDM symbol, then applies the IFFT to each block:

```text
x[n] = IFFT{X[k]}
```

Here `X[k]` is the QPSK data on frequency-domain subcarriers, and `x[n]` is the time-domain OFDM symbol.

### Cyclic Prefix (CP)

To combat inter-symbol interference caused by multipath, the program copies the last samples of each OFDM symbol to the front as a cyclic prefix.

The CP lengths compared in this project are:

```python
CP_LIST = [0, 2, 4, 8, 16, 32]
```

Channel length is:

```python
CHANNEL_LEN = 8
```

Therefore the maximum discrete multipath delay is about `CHANNEL_LEN - 1 = 7`. In theory, when `CP >= 7`, the cyclic prefix can effectively suppress ISI caused by multipath.

### Multipath Rayleigh Fading Channel

The program generates complex Rayleigh channel taps of length `CHANNEL_LEN` and applies an exponential decay power profile so earlier paths are stronger and later paths are weaker:

```python
power_profile = np.exp(-np.arange(channel_len) / 2.0)
```

The transmitted signal is linearly convolved with the channel impulse response and then AWGN noise is added.

### MMSE Frequency-Domain Equalization

After the FFT, the receiver performs single-tap MMSE equalization on each subcarrier:

```text
X_hat[k] = H*[k]Y[k] / (|H[k]|^2 + 1 / SNR)
```

Where:

- `Y[k]` is the received frequency-domain signal after FFT;
- `H[k]` is the channel frequency response;
- `H*[k]` is the complex conjugate of `H[k]`;
- `SNR` is the linear signal-to-noise ratio.

Compared with zero-forcing equalization, MMSE does not over-amplify noise at deep fades, so it usually provides better noise robustness.

## Project Structure

```text
OFDM_CP_Study/
│
├── OUTPUT/                # Results
├── figures/               # Figures
├── src/                   # Core code
│   └── main.py
├── report/                # PDF
├── README.md              # Project documentation
├── requirements.txt       # Python dependencies
├── LICENSE                # MIT license
└── .gitignore             # Git ignore file
```

## Dependencies

Python 3.9 or newer is recommended.

Required packages:

```text
numpy
matplotlib
pillow
```

Install dependencies:

```bash
pip install numpy matplotlib pillow
```

You can also use a virtual environment:

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install numpy matplotlib pillow
```

## Usage

1. Name the image to transmit `original_image.png` and place it in the project root.

2. Modify the path settings in `main.py` to match your local directory:

```python
IMAGE_PATH = r"figures\original_image.png"
RESULTS_DIR = r"OUTPUT"
```

3. Run the simulation:

```bash
python main.py
```

4. View the output results:

```text
OUTPUT/
```

During execution, the program prints the current multipath channel taps, plus the BER and PSNR for different CP lengths at the target SNR.

## Output Explanation

### 1. BER-SNR Performance Curves

Output path:

```text
OUTPUT/1_BER/ber_vs_snr_different_cp.png
```

This plot shows how BER changes with SNR for different CP lengths. When plotting BER curves, if a curve reaches `BER = 0`, the program truncates the curve from that point to avoid an uninformative vertical drop on a semilog scale.

### 2. Constellation Before/After Equalization

Output path:

```text
OUTPUT/2_balanced/constellation_summary_snr_18.png
```

This plot compares constellation diagrams at `target_snr = 18 dB` for different CP lengths. The layout is 3 columns by 4 rows:

```text
CP=0  Before EQ   CP=2  Before EQ   CP=4  Before EQ
CP=0  After EQ    CP=2  After EQ    CP=4  After EQ
CP=8  Before EQ   CP=16 Before EQ   CP=32 Before EQ
CP=8  After EQ    CP=16 After EQ    CP=32 After EQ
```

It shows how constellation points converge before and after MMSE equalization and how CP length affects multipath suppression.

### 3. Image Recovery Results

Output path:

```text
OUTPUT/3_figures/recovered_images_summary_snr_18.png
```

The left side shows the original image and the right side shows recovered images for different CP lengths. The layout is:

```text
Original    CP=0     CP=2     CP=4
Original    CP=8     CP=16    CP=32
```

This provides a direct view of how CP length affects recovery quality.

### 4. PSNR-SNR Performance Curves

Output path:

```text
OUTPUT/4_PSNR/psnr_vs_snr_different_cp.png
```

This plot shows how PSNR changes with SNR for different CP lengths. Higher PSNR indicates the recovered image is closer to the original.

## Key Parameters

| Parameter | Default | Meaning |
| --- | --- | --- |
| `N` | `64` | Number of OFDM subcarriers |
| `BITS_PER_SYMBOL` | `2` | Number of bits per QPSK symbol |
| `SNR_DB_RANGE` | `0:2:20 dB` | Simulated SNR range |
| `CP_LIST` | `[0, 2, 4, 8, 16, 32]` | Cyclic prefix length list |
| `IMAGE_SIZE` | `(128, 128)` | Normalized input image size |
| `CHANNEL_LEN` | `8` | Multipath Rayleigh channel tap length |
| `target_snr` | `18 dB` | Target SNR used for constellation and image summaries |
| `SEED` | `1` | Random seed for reproducibility |

## Metric Definitions

### BER

BER (bit error rate) measures the fraction of erroneous bits between the received and transmitted bit streams:

```text
BER = number of error bits / total number of bits
```

### PSNR

PSNR (peak signal-to-noise ratio) measures image reconstruction quality:

```text
PSNR = 10 * log10(255^2 / MSE)
```

Where `MSE` is the mean squared error between the original and recovered images.

## Typical Observations

You can usually observe:

- Higher SNR leads to lower BER and higher PSNR.
- When CP is too short, it cannot fully combat multipath delay spread and the recovered image may have obvious errors.
- When CP length is greater than or equal to the maximum discrete multipath delay, system performance typically improves significantly.
- MMSE equalization improves constellation point clustering and reduces noise amplification on deeply faded subcarriers.

## Assumptions and Limitations

This project focuses on cyclic prefix, multipath fading, frequency-domain equalization, and performance evaluation in an OFDM baseband link. To highlight the core principles, the current code makes the following simplifications:

- No channel coding, interleaving, or decoding.
- No pilot insertion or practical channel estimation; the receiver uses an ideal channel response.
- No timing synchronization, carrier frequency offset estimation, or phase synchronization.
- No guard subcarriers or DC subcarrier.
- No PAPR analysis.
- No RF upconversion, downconversion, or hardware chain.

Therefore, this project is better suited for OFDM principle verification and teaching simulation rather than a complete engineering-grade wireless system.

## Possible Extensions

You can extend the project with:

- LS or MMSE channel estimation.
- Pilot subcarrier design.
- Convolutional, LDPC, or Turbo coding.
- Interleaving and deinterleaving.
- Comparison of ZF, MMSE, and other equalizers.
- Higher-order modulation such as 16QAM or 64QAM.
- Carrier frequency offset (CFO) and symbol timing offset (STO).
- PAPR statistics and peak reduction.
- Packaging the script as a CLI with parameter inputs.

## Reproducibility

This project uses a fixed random seed:

```python
SEED = 1
RNG = np.random.default_rng(SEED)
```

Therefore, under the same input image, parameters, and dependency versions, the simulation results are reproducible.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Summary

This project is an OFDM system simulation for a wireless communications fundamentals course, emphasizing the relationship between cyclic prefix, multipath channels, MMSE equalization, and BER/PSNR performance analysis.
