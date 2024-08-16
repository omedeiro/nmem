import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

datafile = r"SPG806_20240816_nMem_ICE_ber_D6_A4_C3_2024-08-16 08-44-17.mat"

measurement_dict = sio.loadmat(datafile)


write_1_read_0_error = np.where(
    measurement_dict["read_one_top"] > measurement_dict["threshold_bert"], 1, 0
).flatten()
write_0_read_1_error = np.where(
    measurement_dict["read_zero_top"] < measurement_dict["threshold_bert"], 1, 0
).flatten()

# plt.plot(write_1_read_0_error, ".")
# plt.show()


fft_write_1_read_0_error = np.fft.fft(write_1_read_0_error)
fft_write_0_read_1_error = np.fft.fft(write_0_read_1_error)

plt.plot(np.abs(fft_write_0_read_1_error), label="fft of write_0_read_1 errors")
plt.plot(np.abs(fft_write_1_read_0_error), label="fft of write_1_read_0 errors")
plt.ylim([0, 1000])
plt.legend()
plt.ylabel("Amplitude")
plt.xlabel("Frequency")
plt.show()


fftfreq_write_1_read_0_error = np.fft.fftfreq(
    len(write_1_read_0_error), d=measurement_dict["sample_time"].flatten()
)
fftfreq_write_0_read_1_error = np.fft.fftfreq(
    len(write_0_read_1_error), d=measurement_dict["sample_time"].flatten()
)

plt.plot(
    fftfreq_write_0_read_1_error,
    np.abs(fft_write_0_read_1_error),
    label="fft of write_0_read_1 errors",
)
plt.plot(
    fftfreq_write_1_read_0_error,
    np.abs(fft_write_1_read_0_error),
    label="fft of write_1_read_0 errors",
)
plt.ylim([0, 1000])
plt.xlim([-1e4, 1e4])
plt.ylabel("Amplitude")
plt.xlabel("Frequency [Hz]")
plt.legend()
plt.show()
