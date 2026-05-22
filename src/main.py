import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# =========================
# PARAMETER SETTINGS
# =========================
SEED = 1
RNG = np.random.default_rng(SEED)

N = 64
BITS_PER_SYMBOL = 2
SNR_DB_RANGE = np.arange(0, 21, 2)
CP_LIST = [0, 2, 4, 8, 16, 32]
IMAGE_SIZE = (128, 128)
CHANNEL_LEN = 8

IMAGE_PATH = r"figures\original_image.png"
RESULTS_DIR = r"OUTPUT"


# =========================
# Image to bitstream conversion
# =========================
def load_image_to_bits(image_path, image_size):
    if image_path and os.path.exists(image_path):
        img = Image.open(image_path).convert("L")
    else:
        height, width = image_size
        y = np.linspace(0, 255, height, dtype=np.uint16)
        x = np.linspace(0, 255, width, dtype=np.uint16)
        gradient = (np.add.outer(y, x) // 2).astype(np.uint8)
        img = Image.fromarray(gradient, mode="L")

    img = img.resize((image_size[1], image_size[0]), Image.BILINEAR)
    img_array = np.array(img, dtype=np.uint8)

    bits = np.unpackbits(img_array.reshape(-1))
    return bits, img_array


def bits_to_image(bits, image_shape):
    total_bits = image_shape[0] * image_shape[1] * 8
    bits = bits[:total_bits].astype(np.uint8)

    bytes_arr = np.packbits(bits)
    return bytes_arr.reshape(image_shape)


# =========================
# QPSK modulation and demodulation
# =========================
def qpsk_mod(bits):
    """
    Note: must use a signed integer type.
    Using uint8 would underflow for 0 - 1 and corrupt constellation points.
    """
    bits = np.array(bits, dtype=np.int8).reshape(-1)

    if len(bits) % 2 == 1:
        bits = np.concatenate([bits, np.zeros(1, dtype=np.int8)])

    bits = bits.reshape((-1, 2))

    symbols = (2 * bits[:, 0] - 1) + 1j * (2 * bits[:, 1] - 1)
    symbols = symbols / np.sqrt(2)

    return symbols


def qpsk_demod(symbols):
    bits_hat = np.zeros((len(symbols), 2), dtype=np.uint8)

    bits_hat[:, 0] = (np.real(symbols) > 0).astype(np.uint8)
    bits_hat[:, 1] = (np.imag(symbols) > 0).astype(np.uint8)

    return bits_hat.reshape(-1)


# =========================
# OFDM modulation and demodulation
# =========================
def ofdm_modulate(symbols, n_subcarriers, cp_len):
    symbols = np.array(symbols, dtype=complex).reshape(-1)

    pad_len = (-len(symbols)) % n_subcarriers
    if pad_len > 0:
        symbols = np.concatenate([symbols, np.zeros(pad_len, dtype=complex)])

    num_ofdm_symbols = len(symbols) // n_subcarriers
    data_matrix = symbols.reshape((num_ofdm_symbols, n_subcarriers))

    # IFFT: frequency-domain subcarriers -> time-domain OFDM symbols
    time_signal = np.fft.ifft(data_matrix, axis=1)

    if cp_len > 0:
        cp_part = time_signal[:, -cp_len:]
        time_signal = np.concatenate([cp_part, time_signal], axis=1)

    tx_signal = time_signal.reshape(-1)

    return tx_signal, pad_len, num_ofdm_symbols


def ofdm_demodulate(rx_signal, n_subcarriers, cp_len, num_ofdm_symbols):
    symbol_len = n_subcarriers + cp_len
    needed_len = num_ofdm_symbols * symbol_len

    if len(rx_signal) < needed_len:
        rx_use = np.zeros(needed_len, dtype=complex)
        rx_use[:len(rx_signal)] = rx_signal
    else:
        rx_use = rx_signal[:needed_len]

    rx_matrix = rx_use.reshape((num_ofdm_symbols, symbol_len))

    if cp_len > 0:
        rx_matrix = rx_matrix[:, cp_len:]

    # FFT: time-domain OFDM symbols -> frequency-domain subcarriers
    freq_data = np.fft.fft(rx_matrix, axis=1)

    return freq_data.reshape(-1)


# =========================
# Channel model
# =========================
def generate_multipath_rayleigh_channel(channel_len):
    """
    Generate a multipath Rayleigh fading channel.
    Uses an exponentially decaying power profile so earlier taps are stronger.
    """
    power_profile = np.exp(-np.arange(channel_len) / 2.0)

    taps = (
        RNG.standard_normal(channel_len)
        + 1j * RNG.standard_normal(channel_len)
    ) / np.sqrt(2)

    taps *= np.sqrt(power_profile)

    norm = np.linalg.norm(taps)
    if norm > 0:
        taps = taps / norm

    return taps


def multipath_channel(tx_signal, h, snr_db):
    """
    Multipath channel + AWGN.
    No extra delay alignment; keep the natural start of the causal convolution.
    """
    rx_full = np.convolve(tx_signal, h, mode="full")

    # Keep the same length as the transmitted signal
    rx_signal = rx_full[:len(tx_signal)]

    signal_power = np.mean(np.abs(rx_signal) ** 2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear

    noise = np.sqrt(noise_power / 2) * (
        RNG.standard_normal(len(rx_signal))
        + 1j * RNG.standard_normal(len(rx_signal))
    )

    return rx_signal + noise


# =========================
# Frequency-domain equalization
# =========================
def frequency_equalization(rx_freq, h, n_subcarriers, num_ofdm_symbols, snr_db):
    H = np.fft.fft(h, n_subcarriers)
    H_all = np.tile(H, num_ofdm_symbols)
    snr_linear = 10 ** (snr_db / 10)

    # MMSE single-tap equalization. QPSK symbols are unit-power normalized,
    # so the noise term uses 1/SNR.
    rx_equalized = np.conj(H_all) * rx_freq / (np.abs(H_all) ** 2 + 1 / snr_linear)

    return rx_equalized


# =========================
# Metrics
# =========================
def calculate_ber(bits_tx, bits_rx):
    length = min(len(bits_tx), len(bits_rx))
    if length == 0:
        return 0.0

    return np.mean(bits_tx[:length] != bits_rx[:length])


def calculate_psnr(original_img, recovered_img):
    mse = np.mean(
        (original_img.astype(np.float64) - recovered_img.astype(np.float64)) ** 2
    )

    if mse == 0:
        return np.inf

    return 10 * np.log10(255 ** 2 / mse)


# =========================
# Main simulation
# =========================
def main():
    ber_dir = os.path.join(RESULTS_DIR, "1_BER")
    constellation_dir = os.path.join(RESULTS_DIR, "2_balanced")
    image_dir = os.path.join(RESULTS_DIR, "3_figures")
    psnr_dir = os.path.join(RESULTS_DIR, "4_PSNR")

    os.makedirs(ber_dir, exist_ok=True)
    os.makedirs(constellation_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(psnr_dir, exist_ok=True)

    bits_tx, original_img = load_image_to_bits(IMAGE_PATH, IMAGE_SIZE)
    Image.fromarray(original_img).save(os.path.join(image_dir, "original_image.png"))

    bits_tx_len = len(bits_tx)
    qpsk_symbols = qpsk_mod(bits_tx)

    h = generate_multipath_rayleigh_channel(CHANNEL_LEN)

    print("Multipath Rayleigh channel taps h =")
    print(h)
    print(
        f"Channel length = {CHANNEL_LEN}, max discrete delay is about {CHANNEL_LEN - 1}"
    )
    print(
        "In theory, CP length should be >= the max discrete delay; "
        "CP >= CHANNEL_LEN - 1 performs better."
    )

    ber_results = {}
    psnr_results = {}
    constellation_results = {}
    recovered_images = {}

    target_snr = 18

    for cp_len in CP_LIST:
        tx_signal, pad_len, num_ofdm_symbols = ofdm_modulate(qpsk_symbols, N, cp_len)

        ber_list = []
        psnr_list = []

        for snr_db in SNR_DB_RANGE:
            rx_signal = multipath_channel(tx_signal, h, snr_db)

            # No delay shift; demodulate directly
            rx_freq = ofdm_demodulate(rx_signal, N, cp_len, num_ofdm_symbols)

            # Frequency-domain MMSE single-tap equalization
            rx_eq = frequency_equalization(rx_freq, h, N, num_ofdm_symbols, snr_db)

            bits_rx = qpsk_demod(rx_eq)

            # Remove OFDM/QPSK padding to keep original image length
            bits_rx = bits_rx[:bits_tx_len]

            recovered_img = bits_to_image(bits_rx, IMAGE_SIZE)

            ber = calculate_ber(bits_tx, bits_rx)
            psnr = calculate_psnr(original_img, recovered_img)

            ber_list.append(ber)
            psnr_list.append(psnr)

            if snr_db == target_snr:
                recovered_images[cp_len] = recovered_img.copy()
                constellation_results[cp_len] = (rx_freq.copy(), rx_eq.copy())

        ber_results[cp_len] = np.array(ber_list)
        psnr_results[cp_len] = np.array(psnr_list)

    # =========================
    # Plot constellations before/after equalization
    # =========================
    cp_groups = [CP_LIST[:3], CP_LIST[3:]]
    fig, axes = plt.subplots(4, 3, figsize=(12, 12), squeeze=False)

    for group_row, cp_group in enumerate(cp_groups):
        before_row = 2 * group_row
        after_row = before_row + 1

        for col, cp_len in enumerate(cp_group):
            rx_freq, rx_eq = constellation_results[cp_len]
            num_points = min(2000, len(rx_freq))
            before_ax = axes[before_row, col]
            after_ax = axes[after_row, col]

            before_ax.scatter(
                np.real(rx_freq[:num_points]),
                np.imag(rx_freq[:num_points]),
                s=5,
                alpha=0.6,
            )
            before_ax.set_title(f"CP={cp_len} Before EQ")
            before_ax.set_xlabel("In-phase")
            before_ax.grid(True)

            after_ax.scatter(
                np.real(rx_eq[:num_points]),
                np.imag(rx_eq[:num_points]),
                s=5,
                alpha=0.6,
            )
            after_ax.set_title(f"CP={cp_len} After EQ")
            after_ax.set_xlabel("In-phase")
            after_ax.grid(True)

            if col == 0:
                before_ax.set_ylabel("Quadrature")
                after_ax.set_ylabel("Quadrature")

    fig.suptitle(
        f"Constellations Before/After Equalization at SNR={target_snr} dB"
    )
    fig.tight_layout()
    fig.savefig(
        os.path.join(constellation_dir, f"constellation_summary_snr_{target_snr}.png"),
        dpi=300,
    )
    plt.close(fig)

    # =========================
    # Plot original and recovered images
    # =========================
    fig = plt.figure(figsize=(12, 6), constrained_layout=True)
    gs = fig.add_gridspec(2, 4, width_ratios=[1.5, 1, 1, 1])

    ax_original = fig.add_subplot(gs[:, 0])
    ax_original.imshow(original_img, cmap="gray", vmin=0, vmax=255)
    ax_original.set_title("Original Image")
    ax_original.axis("off")

    for row, cp_group in enumerate(cp_groups):
        for col, cp_len in enumerate(cp_group):
            ax = fig.add_subplot(gs[row, col + 1])
            ax.imshow(recovered_images[cp_len], cmap="gray", vmin=0, vmax=255)
            ax.set_title(f"CP={cp_len}")
            ax.axis("off")

    fig.suptitle(
        f"Original and Recovered Images at SNR={target_snr} dB"
    )
    fig.savefig(
        os.path.join(image_dir, f"recovered_images_summary_snr_{target_snr}.png"),
        dpi=300,
    )
    plt.close(fig)

    # =========================
    # Plot BER curves
    # =========================
    fig = plt.figure(figsize=(8, 6))

    for cp_len in CP_LIST:
        ber_values = ber_results[cp_len]
        zero_indices = np.where(ber_values <= 0)[0]
        plot_end = zero_indices[0] if len(zero_indices) > 0 else len(ber_values)

        if plot_end == 0:
            continue

        plt.semilogy(
            SNR_DB_RANGE[:plot_end],
            ber_values[:plot_end],
            marker="o",
            label=f"CP={cp_len}",
        )

    plt.xlabel("SNR / dB")
    plt.ylabel("BER")
    plt.title("BER Performance with Different CP Lengths")
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(ber_dir, "ber_vs_snr_different_cp.png"), dpi=300)
    plt.close(fig)

    # =========================
    # Plot PSNR curves
    # =========================
    fig = plt.figure(figsize=(8, 6))

    for cp_len in CP_LIST:
        plt.plot(
            SNR_DB_RANGE,
            psnr_results[cp_len],
            marker="o",
            label=f"CP={cp_len}",
        )

    plt.xlabel("SNR / dB")
    plt.ylabel("PSNR / dB")
    plt.title("PSNR Performance with Different CP Lengths")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(psnr_dir, "psnr_vs_snr_different_cp.png"), dpi=300)
    plt.close(fig)

    # =========================
    # Print summary table
    # =========================
    target_idx = int(np.where(SNR_DB_RANGE == target_snr)[0][0])

    print(f"\nCP Length | BER@{target_snr} dB | PSNR@{target_snr} dB")
    print("------------------------------------------")
    for cp_len in CP_LIST:
        ber_val = ber_results[cp_len][target_idx]
        psnr_val = psnr_results[cp_len][target_idx]
        print(f"{cp_len:>5} | {ber_val:>8.3e} | {psnr_val:>10.2f}")

    print("\nSimulation complete. Results saved to:")
    print(RESULTS_DIR)


if __name__ == "__main__":
    main()
