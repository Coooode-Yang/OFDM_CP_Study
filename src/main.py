import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# =========================
# 中文字体设置
# =========================
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

# =========================
# 基础参数
# =========================
SEED = 1
RNG = np.random.default_rng(SEED)

N = 64
BITS_PER_SYMBOL = 2
SNR_DB_RANGE = np.arange(0, 21, 2)
CP_LIST = [0, 2, 4, 8, 16, 32]
IMAGE_SIZE = (128, 128)
CHANNEL_LEN = 8

IMAGE_PATH = r"E:\Study\无线通信基础\报告\image.png"
RESULTS_DIR = r"E:\Study\无线通信基础\报告\OUTPUT"


# =========================
# 图像与bit流转换
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
# QPSK调制与解调
# =========================
def qpsk_mod(bits):
    """
    注意：这里必须使用有符号整数类型。
    如果使用uint8，0 - 1会发生溢出，导致星座点错误。
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
# OFDM调制与解调
# =========================
def ofdm_modulate(symbols, n_subcarriers, cp_len):
    symbols = np.array(symbols, dtype=complex).reshape(-1)

    pad_len = (-len(symbols)) % n_subcarriers
    if pad_len > 0:
        symbols = np.concatenate([symbols, np.zeros(pad_len, dtype=complex)])

    num_ofdm_symbols = len(symbols) // n_subcarriers
    data_matrix = symbols.reshape((num_ofdm_symbols, n_subcarriers))

    # IFFT：频域子载波数据 -> 时域OFDM符号
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

    # FFT：时域OFDM符号 -> 频域子载波数据
    freq_data = np.fft.fft(rx_matrix, axis=1)

    return freq_data.reshape(-1)


# =========================
# 信道模型
# =========================
def generate_multipath_rayleigh_channel(channel_len):
    """
    生成多径瑞利衰落信道。
    使用指数衰减功率分布，使前面的路径功率较强，后面的路径功率较弱。
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
    多径信道 + AWGN。
    这里不再做额外delay对齐，保持因果卷积的自然起点。
    """
    rx_full = np.convolve(tx_signal, h, mode="full")

    # 截取与发送信号等长部分
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
# 频域均衡
# =========================
def frequency_equalization(rx_freq, h, n_subcarriers, num_ofdm_symbols, snr_db):
    H = np.fft.fft(h, n_subcarriers)
    H_all = np.tile(H, num_ofdm_symbols)
    snr_linear = 10 ** (snr_db / 10)

    # MMSE单抽头频域均衡。QPSK符号已归一化为单位平均功率，因此噪声项取1/SNR。
    rx_equalized = np.conj(H_all) * rx_freq / (np.abs(H_all) ** 2 + 1 / snr_linear)

    return rx_equalized


# =========================
# 指标计算
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
# 主仿真
# =========================
def main():
    ber_dir = os.path.join(RESULTS_DIR, "1_BER性能图")
    constellation_dir = os.path.join(RESULTS_DIR, "2_均衡前后星座图")
    image_dir = os.path.join(RESULTS_DIR, "3_图像恢复结果")

    os.makedirs(ber_dir, exist_ok=True)
    os.makedirs(constellation_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)

    bits_tx, original_img = load_image_to_bits(IMAGE_PATH, IMAGE_SIZE)
    Image.fromarray(original_img).save(os.path.join(image_dir, "original_image.png"))

    bits_tx_len = len(bits_tx)
    qpsk_symbols = qpsk_mod(bits_tx)

    h = generate_multipath_rayleigh_channel(CHANNEL_LEN)

    print("多径瑞利信道抽头 h = ")
    print(h)
    print(f"信道长度 = {CHANNEL_LEN}, 最大离散时延约为 {CHANNEL_LEN - 1}")
    print("理论上 CP 长度应大于等于最大离散时延，CP >= CHANNEL_LEN - 1 时效果较好。")

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

            # 不再做delay平移，直接OFDM解调
            rx_freq = ofdm_demodulate(rx_signal, N, cp_len, num_ofdm_symbols)

            # 频域MMSE单抽头均衡
            rx_eq = frequency_equalization(rx_freq, h, N, num_ofdm_symbols, snr_db)

            bits_rx = qpsk_demod(rx_eq)

            # 去除OFDM补零和QPSK补零，只保留原始图像bit长度
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
    # 汇总绘制均衡前后星座图
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
            before_ax.set_title(f"CP={cp_len} 均衡前")
            before_ax.set_xlabel("In-phase")
            before_ax.grid(True)

            after_ax.scatter(
                np.real(rx_eq[:num_points]),
                np.imag(rx_eq[:num_points]),
                s=5,
                alpha=0.6,
            )
            after_ax.set_title(f"CP={cp_len} 均衡后")
            after_ax.set_xlabel("In-phase")
            after_ax.grid(True)

            if col == 0:
                before_ax.set_ylabel("Quadrature")
                after_ax.set_ylabel("Quadrature")

    fig.suptitle(f"不同CP长度在SNR={target_snr} dB下的均衡前后星座图")
    fig.tight_layout()
    fig.savefig(
        os.path.join(constellation_dir, f"constellation_summary_snr_{target_snr}.png"),
        dpi=300,
    )
    plt.close(fig)

    # =========================
    # 汇总绘制原始图像与不同CP长度下的恢复图像
    # =========================
    fig = plt.figure(figsize=(12, 6), constrained_layout=True)
    gs = fig.add_gridspec(2, 4, width_ratios=[1.5, 1, 1, 1])

    ax_original = fig.add_subplot(gs[:, 0])
    ax_original.imshow(original_img, cmap="gray", vmin=0, vmax=255)
    ax_original.set_title("原始图像")
    ax_original.axis("off")

    for row, cp_group in enumerate(cp_groups):
        for col, cp_len in enumerate(cp_group):
            ax = fig.add_subplot(gs[row, col + 1])
            ax.imshow(recovered_images[cp_len], cmap="gray", vmin=0, vmax=255)
            ax.set_title(f"CP={cp_len}")
            ax.axis("off")

    fig.suptitle(f"原始图像与SNR={target_snr} dB下不同CP长度的恢复图像")
    fig.savefig(
        os.path.join(image_dir, f"recovered_images_summary_snr_{target_snr}.png"),
        dpi=300,
    )
    plt.close(fig)

    # =========================
    # 绘制 BER 曲线
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
    plt.title("不同CP长度下的BER性能")
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(ber_dir, "ber_vs_snr_different_cp.png"), dpi=300)
    plt.close(fig)

    # =========================
    # 绘制 PSNR 曲线
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
    plt.title("不同CP长度下的PSNR性能")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "psnr_vs_snr_different_cp.png"), dpi=300)
    plt.close(fig)

    # =========================
    # 输出结果表
    # =========================
    target_idx = int(np.where(SNR_DB_RANGE == target_snr)[0][0])

    print(f"\nCP长度 | BER@{target_snr}dB | PSNR@{target_snr}dB (dB)")
    print("--------------------------------")
    for cp_len in CP_LIST:
        ber_val = ber_results[cp_len][target_idx]
        psnr_val = psnr_results[cp_len][target_idx]
        print(f"{cp_len:>5} | {ber_val:>8.3e} | {psnr_val:>10.2f}")

    print("\n仿真完成，结果已保存到：")
    print(RESULTS_DIR)


if __name__ == "__main__":
    main()
