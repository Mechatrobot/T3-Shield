from PyQt5 import Qt, QtCore
from gnuradio import qtgui
from gnuradio import blocks
from gnuradio import fft
from gnuradio.fft import window
from gnuradio import gr
from gnuradio import uhd
import sip
import threading
import numpy as np
import csv
import os
import sys
import signal
from datetime import datetime
from scipy.signal import find_peaks, peak_widths
import matplotlib
#matplotlib.use('Qt5Agg')   # with TDMA plot
matplotlib.use('Agg')      # without TDMA plot
import matplotlib.pyplot as plt
import time
import numpy as np
from gnuradio import gr

class RBW_Filter(gr.sync_block):

    def __init__(self, fft_size, rbw, samp_rate):
        gr.sync_block.__init__(self,
            name="RBW_Filter",
            in_sig=[(np.float32, fft_size)],
            out_sig=[(np.float32, fft_size)]
        )
        self.fft_size = fft_size
        self.samp_rate = samp_rate
        self.rbw = rbw
        self.update_kernel()

    def update_kernel(self):
        freq_res = self.samp_rate / self.fft_size
        smooth_bins = max(1, int(self.rbw / freq_res))
        if smooth_bins > 1:
            sigma = smooth_bins / 3.0
            x = np.arange(smooth_bins) - smooth_bins // 2
            self.kernel = np.exp(-(x**2) / (2 * sigma**2))
            self.kernel /= np.sum(self.kernel)
        else:
            self.kernel = np.array([1.0])
        print(f"RBW Filter updated: {self.rbw/1e3:.1f} kHz, {smooth_bins} bins")

    def set_rbw(self, rbw):
        self.rbw = rbw
        self.update_kernel()

    def set_rbw_fraction(self, fraction):
        self.rbw = fraction * self.samp_rate
        self.update_kernel()

    def work(self, input_items, output_items):
        data = input_items[0]
        bin_bw = self.samp_rate / self.fft_size
        rbw_hz = len(self.kernel) * bin_bw
        noise_offset_db = 10 * np.log10(rbw_hz / bin_bw)

        for i, vec in enumerate(data):
            if len(self.kernel) > 1:
                smoothed = np.convolve(vec, self.kernel, mode="same")

                if len(smoothed) > self.fft_size:
                    smoothed = smoothed[:self.fft_size]
                elif len(smoothed) < self.fft_size:
                    smoothed = np.pad(smoothed, (0, self.fft_size - len(smoothed)), mode="constant")

                edge_cut = len(self.kernel) // 2
                smoothed[:edge_cut] = vec[:edge_cut]
                smoothed[-edge_cut:] = vec[-edge_cut:]

                output_items[0][i][:] = smoothed + noise_offset_db
            else:
                output_items[0][i][:] = vec
        return len(output_items[0])

class VBW_Filter(gr.sync_block):
    """Video Bandwidth Filter - Controls time domain averaging (FFT traces)"""
    def __init__(self, fft_size, vbw, samp_rate):
        gr.sync_block.__init__(self,
            name="VBW_Filter",
            in_sig=[(np.float32, fft_size)],
            out_sig=[(np.float32, fft_size)]
        )
        self.fft_size = fft_size
        self.samp_rate = samp_rate
        self.vbw = vbw
        self.previous_data = None
        self.update_alpha()

    def update_alpha(self):
        # Make VBW less aggressive for narrow spans
        span_factor = min(self.samp_rate / 1e6, 1.0)  # Scale with span
        max_vbw = 1e6 * span_factor
        min_vbw = 1e3 * span_factor
        normalized_vbw = (self.vbw - min_vbw) / (max_vbw - min_vbw)
        normalized_vbw = np.clip(normalized_vbw, 0.0, 1.0)
        # Increase minimum alpha for narrow spans to reduce smoothing
        min_alpha = 0.3 if self.samp_rate <= 1e6 else 0.01
        self.alpha = min_alpha + (0.99 - min_alpha) * normalized_vbw

    def set_vbw(self, vbw):
        self.vbw = vbw
        self.update_alpha()

    def work(self, input_items, output_items):
        data = input_items[0]
        for i, vec in enumerate(data):
            if self.previous_data is not None:
                filtered = self.alpha * vec + (1 - self.alpha) * self.previous_data
                output_items[0][i] = filtered
                self.previous_data = filtered.copy()
            else:
                output_items[0][i] = vec
                self.previous_data = vec.copy()
        return len(output_items[0])


# Cache the last VBW output frame for peak scanning (non-invasive)
_original_vbw_work = VBW_Filter.work
def _vbw_work_store(self, input_items, output_items):
    ret = _original_vbw_work(self, input_items, output_items)
    try:
        if len(output_items[0]) > 0:
            self._last_frame_db = np.array(output_items[0][-1], dtype=np.float32)
    except Exception:
        pass
    return ret
VBW_Filter.work = _vbw_work_store


class BandPower(gr.sync_block):
    """Compute average dB power over a frequency sub-band (from FFT trace)."""
    def __init__(self, fft_size, samp_rate, center_freq, target_freq, bandwidth):
        gr.sync_block.__init__(self,
            name="Band Power",
            in_sig=[(np.float32, fft_size)],
            out_sig=[np.float32])
        self.fft_size = fft_size
        self.samp_rate = samp_rate
        self.center_freq = center_freq
        self.target_freq = target_freq
        self.bandwidth = bandwidth
        self.update_freq_bins()

    def update_freq_bins(self):
        freq_res = self.samp_rate / self.fft_size
        self.freqs = self.center_freq - (self.samp_rate/2) + np.arange(self.fft_size) * freq_res
        self.band_mask = (self.freqs >= self.target_freq - self.bandwidth/2) & \
                         (self.freqs <= self.target_freq + self.bandwidth/2)

    def set_center_freq(self, center_freq):
        self.center_freq = center_freq
        self.update_freq_bins()

    def set_target(self, target_freq, bandwidth):
        self.target_freq = target_freq
        self.bandwidth = bandwidth
        self.update_freq_bins()

    def work(self, input_items, output_items):
        data_in = input_items[0]
        data_out = output_items[0]
        for i, vec in enumerate(data_in):
            band_power = np.mean(vec[self.band_mask])  # dB avg
            data_out[i] = band_power
        return len(data_out)


class PeakDetector(gr.sync_block):

    def __init__(self, fft_size, samp_rate, center_freq, p_threshold=-40, output_path=None):
        gr.sync_block.__init__(self,
            name="Peak Detector",
            in_sig=[(np.float32, fft_size)],
            out_sig=None)
        self.fft_size = fft_size
        self.samp_rate = samp_rate
        self.center_freq = center_freq
        self.p_threshold = p_threshold
        
        # Set output path to Desktop if not specified
        if output_path is None:
            desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
            self.output_path = os.path.join(desktop_path, "P_Val.csv")
        else:
            self.output_path = output_path
        
        self.update_freq_bins()
        if not os.path.isfile(self.output_path):
            with open(self.output_path, "w", newline="") as f:
                csv.writer(f).writerow(["Timestamp", "PeakFreq(MHz)", "PeakPower(dB)", "Bandwidth(Hz)", "Protocol", "Operator"])

    def update_freq_bins(self):
        freq_res = self.samp_rate / self.fft_size
        self.freqs = self.center_freq - (self.samp_rate / 2) + np.arange(self.fft_size) * freq_res

    def set_center_freq(self, center_freq):
        self.center_freq = center_freq
        self.update_freq_bins()

    def get_protocol(self, freq_hz,bandwidth_hz=None):
        if bandwidth_hz > 2e6:
            return "3G"
        # GSM-900 downlink (935–960 MHz) & uplink (890–915 MHz)
        elif (935e6 <= freq_hz <= 960e6) or (880e6 <= freq_hz <= 915e6):
            return "GSM"
        return "Unknown"

    def work(self, input_items, output_items):
        # This block no longer writes to CSV
        return len(input_items[0])

class TDMA_Slicer(gr.sync_block):
    """Smooth power (dB), threshold into 0/1 to visualize TDMA slots (~577 µs)."""
    def __init__(self, avg_len_samples=1, threshold_db=-40.0):
        gr.sync_block.__init__(self,
            name="TDMA_Slicer",
            in_sig=[np.float32],
            out_sig=[np.float32])
        self.avg_len = max(1, int(avg_len_samples))
        self.threshold = threshold_db
        self.buf = np.zeros(self.avg_len, dtype=np.float32)
        self.idx = 0
        self.filled = 0

    def set_avg_len(self, n):
        self.avg_len = max(1, int(n))
        self.buf = np.zeros(self.avg_len, dtype=np.float32)
        self.idx = 0
        self.filled = 0

    def set_threshold(self, th_db):
        self.threshold = float(th_db)

    def work(self, input_items, output_items):
        x = input_items[0]
        y = output_items[0]
        for i, v in enumerate(x):
            # ring buffer average
            self.buf[self.idx] = v
            self.idx = (self.idx + 1) % self.avg_len
            self.filled = min(self.filled + 1, self.avg_len)
            avg = np.sum(self.buf) / self.filled
            y[i] = 1.0 if avg >= self.threshold else 0.0
        return len(y)


# ---------- Add this probe sink class ----------
class ProbeSink(gr.sync_block):
    """Non-blocking circular buffer sink for capturing recent float samples."""
    def __init__(self, maxlen=200000):
        gr.sync_block.__init__(self, name="ProbeSink", in_sig=[np.float32], out_sig=None)
        self.maxlen = int(maxlen)
        self.buf = np.zeros(self.maxlen, dtype=np.float32)
        self.ptr = 0
        self.filled = 0
        self.lock = threading.Lock()

    def work(self, input_items, output_items):
        data = input_items[0]
        n = len(data)
        with self.lock:
            if n >= self.maxlen:
                # keep most recent chunk
                self.buf[:] = data[-self.maxlen:]
                self.ptr = 0
                self.filled = self.maxlen
            else:
                end = self.ptr + n
                if end <= self.maxlen:
                    self.buf[self.ptr:end] = data
                else:
                    part = self.maxlen - self.ptr
                    self.buf[self.ptr:] = data[:part]
                    self.buf[:end % self.maxlen] = data[part:]
                self.ptr = end % self.maxlen
                self.filled = min(self.maxlen, self.filled + n)
        return n

    def get_last(self, num):
        """Return a copy of the last `num` samples (most recent last)."""
        with self.lock:
            if self.filled == 0:
                return np.array([], dtype=np.float32)
            num = min(int(num), self.filled)
            start = (self.ptr - num) % self.maxlen
            if start + num <= self.maxlen:
                return self.buf[start:start+num].copy()
            else:
                first_part = self.maxlen - start
                return np.concatenate((self.buf[start:].copy(), self.buf[:num-first_part].copy()))

    def clear_buffer(self):
        with self.lock:
            self.buf.fill(0)
            self.ptr = 0
            self.filled = 0

# ----------------------------
# Operator assignment helper
# ----------------------------

def operator_for(center_hz, span_hz, freq_hz):

    start = center_hz - span_hz / 2.0
    if freq_hz < start or freq_hz > start + span_hz:
        return "Unknown"
    rel = (freq_hz - start) / span_hz
    if rel < (15.0 / 35.0):
        return "inwi"
    elif rel < (30.0 / 35.0):
        return "orange"
    else:
        return "IAM"


# ----------------------------
# Top block
# ----------------------------

class SpecAnalyser(gr.top_block, Qt.QWidget):

    def __init__(self):
        gr.top_block.__init__(self, "GSM_PVT_Spectrum_Improved", catch_exceptions=True)
        Qt.QWidget.__init__(self)
        self.setWindowTitle("GSM PVT & Spectrum - Improved Sweep Time")
        qtgui.util.check_set_qss()
        try:
            self.setWindowIcon(Qt.QIcon.fromTheme('gnuradio-grc'))
        except Exception:
            pass

        # Main layout
        self.top_scroll_layout = Qt.QVBoxLayout()
        self.setLayout(self.top_scroll_layout)
        self.top_scroll = Qt.QScrollArea()
        self.top_scroll.setFrameStyle(Qt.QFrame.NoFrame)
        self.top_scroll_layout.addWidget(self.top_scroll)
        self.top_scroll.setWidgetResizable(True)
        self.top_widget = Qt.QWidget()
        self.top_scroll.setWidget(self.top_widget)
        self.top_layout = Qt.QVBoxLayout(self.top_widget)
        self.top_grid_layout = Qt.QGridLayout()
        self.top_layout.addLayout(self.top_grid_layout)

        self.settings = Qt.QSettings("gnuradio/flowgraphs", "SpecAnalyser")
        try:
            geometry = self.settings.value("geometry")
            if geometry:
                self.restoreGeometry(geometry)
        except Exception as exc:
            print(f"Qt GUI: Could not restore geometry: {str(exc)}", file=sys.stderr)
        self.flowgraph_started = threading.Event()

        # ----------------------------
        # Parameters
        # ----------------------------

        self.samp_rate = 45e6    
        self.fft_size = 1024

        # Spectrum analyzer params
        self.rbw = 525e3         # Initial RBW
        self.vbw = 1e6        # Initial VBW
        self.k_total = 2.5      # Sweep time constant (adjustable)

        # Calculate initial sweep time using improved method
        self.sweep_time = self.calculate_sweep_time()
        self.update_rate = self.sweep_time

        # Default GSM downlink center
        self.center_freq = 900e6      # GSM-900 DL example
        self.rf_bw = 36e6               # USRP frontend BW

        # Band-power ROI for GSM channel
        self.band_target_freq = 947.5e6
        self.band_bandwidth = 200e3     # GSM channel bandwidth

        # PVT decimation for TDMA visibility (do not touch this chain)
        self.pvt_decim = 2           # 5MHz/100 = 50kSa/s
        self.tdma_threshold_db = -50.0
        self.tdma_avg_len = 1           # Minimal averaging to preserve bursts

        # Calculate expected slot duration at decimated rate
        decimated_fs = self.samp_rate / self.pvt_decim
        expected_samples_per_slot = int(577e-6 * decimated_fs)
        print(f"Initial: Expected samples per GSM slot at {decimated_fs/1e3:.1f} kSa/s: {expected_samples_per_slot}")
        print(f"Initial Sweep Time: {self.sweep_time*1000:.1f} ms")

        # ----------------------------
        # UI widgets (controls)
        # ----------------------------
        self.create_control_widgets()

        # ----------------------------
        # Blocks
        # ----------------------------
        # USRP
        self.uhd_usrp_source_0 = uhd.usrp_source(
            ",".join(('type=b200', 'serial=000000085')),
            uhd.stream_args(cpu_format="fc32", args='', channels=list(range(0, 1))),
        )
        self.uhd_usrp_source_0.set_samp_rate(self.samp_rate)
        self.uhd_usrp_source_0.set_time_unknown_pps(uhd.time_spec(0))
        self.uhd_usrp_source_0.set_center_freq(self.center_freq, 0)
        self.uhd_usrp_source_0.set_antenna("TX/RX", 0)
        self.uhd_usrp_source_0.set_bandwidth(self.rf_bw, 0)
        self.uhd_usrp_source_0.set_gain(30, 0)

        # FFT chain (spectrum)
        self.blocks_stream_to_vector_0 = blocks.stream_to_vector(gr.sizeof_gr_complex*1, self.fft_size)
        self.fft_vxx_0 = fft.fft_vcc(self.fft_size, True, window.hann(self.fft_size), True, 1)
        
        # Proper normalization for FFT (dB scale)
        self.blocks_complex_to_mag_squared_0 = blocks.complex_to_mag_squared(self.fft_size)
        # FFT scaling for proper power normalization
        fft_scale = 1.0 / (self.fft_size * (self.samp_rate / self.fft_size))
        self.blocks_multiply_const_fft = blocks.multiply_const_ff(fft_scale, self.fft_size)
        self.blocks_nlog10_ff_0 = blocks.nlog10_ff(10, self.fft_size, 0)

        self.rbw_filter = RBW_Filter(self.fft_size, self.rbw, self.samp_rate)
        self.vbw_filter = VBW_Filter(self.fft_size, self.vbw, self.samp_rate)
        self.peak_detector = PeakDetector(self.fft_size, self.samp_rate, self.center_freq, p_threshold=-65)
        self.band_power = BandPower(self.fft_size, self.samp_rate, self.center_freq, self.band_target_freq, self.band_bandwidth)

        # FFT displays
        self.qtgui_vector_sink_f_0 = qtgui.vector_sink_f(
            self.fft_size,
            self.center_freq - self.samp_rate/2,
            self.samp_rate / self.fft_size,
            'Frequency (Hz)',
            'Power (dB)',
            "RBW/VBW Spectrum",
            1, None
        )

        self.qtgui_freq_sink_x_0 = qtgui.freq_sink_c(
            self.fft_size,
            window.WIN_HANN,
            self.center_freq,
            self.samp_rate,
            "Live FFT",
            1, None
        )

        self.qtgui_freq_sink_x_0.set_update_time(self.update_rate)
        self.qtgui_vector_sink_f_0.set_update_time(self.update_rate)

        self.qtgui_vector_sink_f_0.set_y_axis(-120, 40)
        self.qtgui_vector_sink_f_0.enable_autoscale(False)
        self.qtgui_vector_sink_f_0.enable_grid(True)
        
        self.qtgui_freq_sink_x_0.set_update_time(self.update_rate)
        self.qtgui_freq_sink_x_0.set_y_axis(-120, 40)
        self.qtgui_freq_sink_x_0.enable_autoscale(False)
        self.qtgui_freq_sink_x_0.enable_grid(True)

        # Time-domain power (PVT) - do not change this chain
        self.blocks_complex_to_mag_squared_td = blocks.complex_to_mag_squared(1)
        pvt_scale = 1.0 / self.samp_rate
        self.pvt_normalize = blocks.multiply_const_ff(pvt_scale)
        self.pvt_mavg_len = 5
        self.pvt_mavg = blocks.moving_average_ff(self.pvt_mavg_len, 1.0/self.pvt_mavg_len, 4000)
        self.pvt_log10 = blocks.nlog10_ff(10, 1, 0)
        self.keep_one_in_n = blocks.keep_one_in_n(gr.sizeof_float, max(1, int(self.pvt_decim)))

        # Time sinks
        decimated_fs = self.samp_rate / max(1, int(self.pvt_decim))
        self.pvt_sink = qtgui.time_sink_f(
            8192,
            decimated_fs,
            "Power vs Time (dB) - GSM Bursts",
            1
        )
        self.pvt_sink.set_update_time(self.update_rate)
        self.pvt_sink.set_y_axis(-150, 0)
        self.pvt_sink.enable_grid(True)

        # TDMA slicer (binary)
        self.tdma_slicer = TDMA_Slicer(self.tdma_avg_len, self.tdma_threshold_db)
        self.tdma_sink = qtgui.time_sink_f(
            8192,
            decimated_fs,
            f"TDMA Slots (~{expected_samples_per_slot} samples/577μs slot)",
            1
        )
        self.tdma_sink.set_update_time(self.update_rate)
        self.tdma_sink.set_y_axis(-0.2, 1.2)
        self.tdma_sink.enable_grid(True)

        # Band power vs time (from FFT band)
        self.band_time_sink = qtgui.time_sink_f(
            1024,
            10,  # just for x-axis scaling
            "Band Power (FFT ROI) vs Time",
            1
        )
        self.band_time_sink.set_update_time(self.update_rate)
        self.band_time_sink.set_y_axis(-100, 40)
        self.band_time_sink.enable_grid(True)

        # ----------------------------
        # Automatic peak recenter/zoom + operator label
        # ----------------------------
        self._peak_threshold_db = -65.0
        self._zoom_span = 1e6       
        self._dwell_ms = 5000            
        self._cooldown_ms = 1000          # 2 seconds
        self._in_dwell = False
        self._cooldown = False
        self._saved_center = self.center_freq
        self._saved_span = self.samp_rate

        # Operator label
        self._operator_label = Qt.QLabel("Operator: —")
        self._operator_label.setAlignment(QtCore.Qt.AlignCenter)
        self.top_layout.addWidget(self._operator_label)

        # Scan timer
        self._peak_scan_timer = Qt.QTimer()
        self._peak_scan_timer.timeout.connect(self._scan_and_recentre)
        self._peak_scan_timer.start(250)

        # Dwell/cooldown timers
        self._dwell_timer = Qt.QTimer()
        self._dwell_timer.setSingleShot(True)
        self._dwell_timer.timeout.connect(self._restore_defaults)
        self._cooldown_timer = Qt.QTimer()
        self._cooldown_timer.setSingleShot(True)
        self._cooldown_timer.timeout.connect(self._end_cooldown)

        self.pvt_threshold = blocks.threshold_ff(0.5, 0.5, 0.0)   # threshold at 0.5
        self.tdma_threshold = blocks.threshold_ff(0.5, 0.5, 0.0)
        self.pvt_probe = ProbeSink()

        # Wrap and place widgets
        self.top_layout.addWidget(sip.wrapinstance(self.qtgui_vector_sink_f_0.qwidget(), Qt.QWidget))
        self.top_layout.addWidget(sip.wrapinstance(self.qtgui_freq_sink_x_0.qwidget(), Qt.QWidget))
        self.top_layout.addWidget(sip.wrapinstance(self.pvt_sink.qwidget(), Qt.QWidget))
        self.top_layout.addWidget(sip.wrapinstance(self.tdma_sink.qwidget(), Qt.QWidget))
        self.top_layout.addWidget(sip.wrapinstance(self.band_time_sink.qwidget(), Qt.QWidget))

        self.tdma_probe = blocks.probe_signal_f()

        # Add after existing parameters
        self.operator_users = {'inwi': 0, 'orange': 0, 'IAM': 0, 'Unknown': 0}
        self.peak_analysis_complete = False
        self.current_operator_peaks = {'inwi': [], 'orange': [], 'IAM': [], 'Unknown': []}

        # ----------------------------
        # Connections
        # ----------------------------

        self.connect((self.uhd_usrp_source_0, 0), (self.blocks_stream_to_vector_0, 0))
        self.connect((self.blocks_stream_to_vector_0, 0), (self.fft_vxx_0, 0))
        self.connect((self.fft_vxx_0, 0), (self.blocks_complex_to_mag_squared_0, 0))
        self.connect((self.blocks_complex_to_mag_squared_0, 0), (self.blocks_multiply_const_fft, 0))
        self.connect((self.blocks_multiply_const_fft, 0), (self.blocks_nlog10_ff_0, 0))
        self.connect((self.blocks_nlog10_ff_0, 0), (self.rbw_filter, 0))
        self.connect((self.rbw_filter, 0), (self.vbw_filter, 0))

        self.connect((self.vbw_filter, 0), (self.qtgui_vector_sink_f_0, 0))
        self.connect((self.vbw_filter, 0), (self.peak_detector, 0))
        self.connect((self.vbw_filter, 0), (self.band_power, 0))
        self.connect((self.band_power, 0), (self.band_time_sink, 0))

        # Live FFT view (QtGUI, complex spectrum)
        self.connect((self.uhd_usrp_source_0, 0), (self.qtgui_freq_sink_x_0, 0))

        # ================================================================
        # PVT chain (time-domain power vs time, with slicing)
        # ================================================================

        self.connect((self.uhd_usrp_source_0, 0), (self.blocks_complex_to_mag_squared_td, 0))
        self.connect((self.blocks_complex_to_mag_squared_td, 0), (self.pvt_normalize, 0))
        self.connect((self.pvt_normalize, 0), (self.pvt_mavg, 0))
        self.connect((self.pvt_mavg, 0), (self.pvt_log10, 0))
        self.connect((self.pvt_log10, 0), (self.keep_one_in_n, 0))

        # PVT plot
        self.connect((self.keep_one_in_n, 0), (self.pvt_sink, 0))

        # PVT threshold + slicer
        self.connect((self.keep_one_in_n, 0), (self.pvt_threshold, 0))
        self.connect((self.pvt_threshold, 0), (self.pvt_probe, 0))

        # ================================================================
        # TDMA Slicer (binary detection, same stream as PVT)
        # ================================================================
        self.connect((self.keep_one_in_n, 0), (self.tdma_slicer, 0))
        self.connect((self.tdma_slicer, 0), (self.tdma_sink, 0))
        self.connect((self.tdma_slicer, 0), (self.tdma_probe, 0))



        # buffer probe for TDMA binary only (keep a couple seconds)
        max_buffer_seconds = 4.0
        decimated_fs = int(self.samp_rate / max(1, int(self.pvt_decim)))
        maxlen = max(1024, int(decimated_fs * max_buffer_seconds))
        self.tdma_probe = ProbeSink(maxlen=maxlen)

        # Connect TDMA probe to the TDMA slicer output
        self.connect((self.tdma_slicer, 0), (self.tdma_probe, 0))
        self._orange_zero_user_detected = False
        self._orange_rescan_done = False
    # ----------------------------
    # Automatic re-centering + operator detection
    # ----------------------------

    def _scan_and_recentre(self):
            if self._in_dwell or self._cooldown:
                return
            try:
                buf = getattr(self.vbw_filter, "_last_frame_db", None)
                if buf is None:
                    return
                frame = np.array(buf, dtype=np.float32)
                
                peaks, props = find_peaks(frame, height=self._peak_threshold_db)
                if len(peaks) == 0:
                    return

                # Pre-calculate constants once
                freq_res = self.samp_rate / self.fft_size
                f_start = self.center_freq - self.samp_rate / 2.0
                
                # Vectorized calculations
                widths = peak_widths(frame, peaks, rel_height=0.1)[0]
                bandwidths_hz = widths * freq_res
                peak_freqs = f_start + peaks * freq_res
                peak_heights = props["peak_heights"]
                
                # Separate 3G and GSM
                is_3g = bandwidths_hz > 2e6
                is_gsm = (bandwidths_hz <= 300e3) & (peak_heights >= self._peak_threshold_db)
                
                # Handle 3G peaks - DETECT AND SAVE IMMEDIATELY (NO DWELL)
                if np.any(is_3g):
                    threeg_peaks = list(zip(peak_freqs[is_3g], bandwidths_hz[is_3g], peak_heights[is_3g]))
                    
                    # Filter 3G peaks by -65 dB threshold
                    threeg_peaks_filtered = [(freq, bw, power) for freq, bw, power in threeg_peaks if power > -65]
                    
                    threeg_info = f"3G_activity: {len(threeg_peaks_filtered)} peaks - "
                    for freq, bw, power in threeg_peaks_filtered[:2]:
                        threeg_info += f"{freq/1e6:.3f}MHz({bw/1e6:.1f}MHz, {power:.1f}dB) "
                    self._operator_label.setText(threeg_info)
                    self._operator_label.update()
                    
                    # WRITE 3G PEAKS TO CSV IMMEDIATELY (NO DWELL/ZOOM) - only if > -65 dB
                    if len(threeg_peaks_filtered) > 0:
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
                        desktop_path = os.path.join(os.path.expanduser("~"), "Desktop", "P_Val.csv")
                        file_exists = os.path.isfile(desktop_path)
                        
                        try:
                            with open(desktop_path, "a", newline="") as f:
                                writer = csv.writer(f)
                                if not file_exists:
                                    writer.writerow(["Timestamp", "PeakFreq(MHz)", "PeakPower(dB)", "Bandwidth(Hz)", "Protocol", "Operator"])
                                
                                for freq, bw, power in threeg_peaks_filtered:
                                    writer.writerow([
                                        timestamp,
                                        f"{freq/1e6:.6f}",
                                        f"{power:.2f}",
                                        f"{bw:.2f}",
                                        "3G",
                                        "3G"
                                    ])
                            print(f"✓ {len(threeg_peaks_filtered)} 3G peaks saved to CSV (>{-65}dB threshold, no zoom)")
                        except Exception as e:
                            print(f"Error writing 3G peaks to CSV: {e}")
                
                # ONLY use GSM peaks for GSM processing
                gsm_peaks = peaks[is_gsm]
                gsm_peak_freqs = peak_freqs[is_gsm]
                gsm_peak_heights = peak_heights[is_gsm]
                gsm_bandwidths = bandwidths_hz[is_gsm]
                
                if len(gsm_peaks) == 0 and not np.any(is_3g):
                    return

                # Continue with GSM-only peaks
                if not hasattr(self, '_peak_queue'):
                    self._peak_queue = []
                    self._current_peak_idx = 0
                    self._scanning_mode = True

                if (self._scanning_mode and 
                    not hasattr(self, '_processing_peaks') and 
                    self.samp_rate > 10e6):
                    
                    # Build peak list from GSM peaks ONLY
                    for i, freq in enumerate(gsm_peak_freqs):
                        is_duplicate = False
                        for existing_freq, _ in self._peak_queue:
                            if abs(freq - existing_freq) < 2 * self.rbw:
                                is_duplicate = True
                                break
                        if not is_duplicate:
                            # Store tuple: (freq, power) for GSM only
                            self._peak_queue.append((freq, gsm_peak_heights[i]))
                    
                    self._peak_queue.sort(key=lambda x: x[1], reverse=True)
                    self._peak_queue = self._peak_queue[:8]

                if (self._scanning_mode and 
                    len(self._peak_queue) >= 1 and
                    not hasattr(self, '_processing_peaks')):
                    
                    Qt.QTimer.singleShot(4000, self._start_processing_peaks)

                if (hasattr(self, '_processing_peaks') and 
                    self._processing_peaks and 
                    not self._in_dwell and 
                    hasattr(self, '_peak_freq_list') and
                    self._current_peak_idx < len(self._peak_freq_list)):
                    
                    peak_info = self._peak_freq_list[self._current_peak_idx]
                    peak_freq = peak_info[0]
                    
                    op = self.operator_for(self._saved_center, self._saved_span, peak_freq)
                    self._operator_label.setText(f"Operator: {op} | GSM Peak {self._current_peak_idx+1}/{len(self._peak_freq_list)}: {peak_freq/1e6:.6f} MHz")
                    self._enter_dwell(peak_freq)
                    self._current_peak_idx += 1

            except Exception as e:
                print(f"Error in _scan_and_recentre: {e}")

    def operator_for(self, center_hz, span_hz, freq_hz, bandwidth_hz=None):
            # Existing GSM operator logic
            start = center_hz - span_hz / 2.0
            if freq_hz < start or freq_hz > start + span_hz:
                return "Unknown"
            rel = (freq_hz - start) / span_hz
            if rel < (15.0 / 35.0):
                return "inwi"
            elif rel < (30.0 / 35.0):
                return "orange"
            else:
                return "IAM"

    def operator_for(self, center_hz, span_hz, freq_hz, bandwidth_hz=None):
        # Existing GSM operator logic
        start = center_hz - span_hz / 2.0
        if freq_hz < start or freq_hz > start + span_hz:
            return "Unknown"
        rel = (freq_hz - start) / span_hz
        if rel < (15.0 / 35.0):
            return "inwi"
        elif rel < (30.0 / 35.0):
            return "orange"
        else:
            return "IAM"

    def _clear_peak_queue(self):
        if hasattr(self, '_peak_queue'):
            self._peak_queue.clear()
            self._current_peak_idx = 0
            self._scanning_mode = True
            if hasattr(self, '_processing_peaks'):
                self._processing_peaks = False

    def _enter_dwell(self, center_hz):
        try:
            self._in_dwell = True
            self._current_peak_freq = center_hz
            
            # WRITE GSM PEAK TO CSV ONCE WHEN SELECTED
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
            operator = operator_for(self._saved_center, self._saved_span, center_hz)
            
            # Get current spectrum data to find peak power
            try:
                buf = getattr(self.vbw_filter, "_last_frame_db", None)
                if buf is not None:
                    frame = np.array(buf, dtype=np.float32)
                    freq_res = self.samp_rate / self.fft_size
                    f_start = self.center_freq - self.samp_rate / 2.0
                    
                    # Find the bin closest to center_hz
                    peak_bin = int((center_hz - f_start) / freq_res)
                    peak_bin = np.clip(peak_bin, 0, len(frame) - 1)
                    peak_power = frame[peak_bin]
                    
                    # Estimate bandwidth (use zoom span as approximation)
                    bandwidth_hz = self._zoom_span
                    
                    # Determine protocol (GSM only in dwell)
                    if (935e6 <= center_hz <= 960e6) or (880e6 <= center_hz <= 915e6):
                        protocol = "GSM"
                    else:
                        protocol = "Unknown"
                    
                    # Write to CSV
                    desktop_path = os.path.join(os.path.expanduser("~"), "Desktop", "P_Val.csv")
                    file_exists = os.path.isfile(desktop_path)
                    
                    with open(desktop_path, "a", newline="") as f:
                        writer = csv.writer(f)
                        if not file_exists:
                            writer.writerow(["Timestamp", "PeakFreq(MHz)", "PeakPower(dB)", "Bandwidth(Hz)", "Protocol", "Operator"])
                        writer.writerow([
                            timestamp,
                            f"{center_hz/1e6:.6f}",
                            f"{peak_power:.2f}",
                            f"{bandwidth_hz:.2f}",
                            protocol,
                            operator
                        ])
                    
                    print(f"✓ GSM Peak written to CSV: {center_hz/1e6:.6f} MHz, {operator}, {peak_power:.2f} dB")
            
            except Exception as e:
                print(f"Error writing peak to CSV: {e}")
            
            # Continue with normal dwell behavior (zoom and analyze)
            self.set_center_freq(center_hz + 200e3)
            self.set_samp_rate(self._zoom_span)
            
            self.tdma_probe.clear_buffer() 
            Qt.QTimer.singleShot(4000, lambda: self.plot_tdma_window())
            
            self._dwell_timer.start(self._dwell_ms)
        except Exception as e:
            print(f"Error in _enter_dwell: {e}")
            self._in_dwell = False

    def _restore_defaults(self):
        try:
            # Check for Orange zero-user anomaly
            if hasattr(self, '_current_peak_freq'):
                operator = operator_for(self._saved_center, self._saved_span, self._current_peak_freq)
                
                if operator == "orange" and not self._orange_rescan_done:
                    orange_peaks = self.current_operator_peaks.get('orange', [])
                    for peak_info in orange_peaks:
                        if abs(peak_info['freq'] - self._current_peak_freq) < 1000 and peak_info['users'] == 0:
                            self._orange_zero_user_detected = True
                            print(f"⚠️  Orange peak with 0 users detected at {self._current_peak_freq/1e6:.3f} MHz")
                            break
            
            # Restore defaults
            self.set_center_freq(self._saved_center)
            self.set_samp_rate(self._saved_span)
            self._in_dwell = False
            self._cooldown = True
            self._cooldown_timer.start(self._cooldown_ms)
            
            # Check if all peaks done
            all_done = (hasattr(self, '_processing_peaks') and 
                       self._processing_peaks and 
                       hasattr(self, '_peak_freq_list') and 
                       self._current_peak_idx >= len(self._peak_freq_list))
            
            # If done and Orange needs re-scan, do it now BEFORE final results
            if all_done and self._orange_zero_user_detected and not self._orange_rescan_done:
                print(f"\n🔄 Re-scanning Orange bandwidth before final results...")
                Qt.QTimer.singleShot(self._cooldown_ms + 500, self._rescan_orange)
                return
            
            # Otherwise finish normally
            if all_done:
                if self._orange_rescan_done:
                    print(f"✅ Orange re-scan complete\n")
                
                self.calculate_operator_totals()
                
                # Cleanup
                self._processing_peaks = False
                self._scanning_mode = True
                self._peak_queue.clear()
                self._current_peak_idx = 0
                self.current_operator_peaks = {'inwi': [], 'orange': [], 'IAM': [], 'Unknown': []}
                if hasattr(self, '_peak_freq_list'):
                    del self._peak_freq_list
                
                print("\n✅ Analysis complete")
                Qt.QTimer.singleShot(2000, self.close)
        
        except Exception as e:
            print(f"Error in _restore_defaults: {e}")
            self._in_dwell = False

    def _start_processing_peaks(self):
        # FIXED: Only start if not already processing
        if (hasattr(self, '_peak_queue') and 
            len(self._peak_queue) > 0 and
            not hasattr(self, '_processing_peaks')):
            
            self._processing_peaks = True
            self._scanning_mode = False  # Stop scanning
            self._current_peak_idx = 0
            # Store only GSM peaks (freq, power) - 3G already saved
            self._peak_freq_list = [(freq, pwr) for freq, pwr in self._peak_queue]
            print(f"Starting GSM peak processing: {len(self._peak_freq_list)} peaks queued")

    def _rescan_orange(self):
        """Re-scan Orange bandwidth only"""
        try:
            print(f"\n{'='*50}")
            print(f"🔄 ORANGE RE-SCAN (Zero User Detected)")
            print(f"{'='*50}")
            
            # Clear Orange results
            old_count = len(self.current_operator_peaks.get('orange', []))
            self.current_operator_peaks['orange'] = []
            print(f"Cleared {old_count} previous Orange peaks")
            
            # Fixed Orange band: 905 - 915 MHz
            orange_start = 905e6
            orange_end = 915e6
            orange_center = (orange_start + orange_end) / 2
            orange_bw = orange_end - orange_start
            
            print(f"Orange band: {orange_start/1e6:.2f} - {orange_end/1e6:.2f} MHz ({orange_bw/1e6:.2f} MHz)")
            print(f"Orange center: {orange_center/1e6:.2f} MHz")
            print(f"Using threshold: {self._peak_threshold_db} dB")
            
            # Store Orange band boundaries
            self._orange_band_start = orange_start
            self._orange_band_end = orange_end
            
            # Zoom to Orange bandwidth
            self.set_center_freq(orange_center)
            self.set_samp_rate(orange_bw * 1.2)  # 20% margin for edge coverage
            
            self._orange_rescan_done = True
            self._peak_queue.clear()
            self._scanning_mode = True
            self._current_peak_idx = 0
            
            # Collect peaks for 3 seconds
            self._orange_scan_timer = Qt.QTimer()
            self._orange_scan_timer.timeout.connect(self._collect_orange_peaks)
            self._orange_scan_timer.start(250)  # Check every 250ms
            
            print(f"Scanning for 3 seconds...\n")
            Qt.QTimer.singleShot(3000, self._finish_orange_scan)
            
        except Exception as e:
            print(f"Error in _rescan_orange: {e}")
            self._orange_rescan_done = False

    def _collect_orange_peaks(self):
        """Collect peaks during Orange re-scan"""
        try:
            buf = getattr(self.vbw_filter, "_last_frame_db", None)
            if buf is None:
                return
            
            frame = np.array(buf, dtype=np.float32)
            
            # Use same threshold as overall analysis
            peaks, props = find_peaks(frame, height=self._peak_threshold_db)
            
            if len(peaks) == 0:
                return
            
            freq_res = self.samp_rate / self.fft_size
            f_start = self.center_freq - self.samp_rate / 2.0
            
            for peak_idx in peaks:
                freq = f_start + peak_idx * freq_res
                power = props["peak_heights"][np.where(peaks == peak_idx)[0][0]]
                
                # Verify peak is within Orange band boundaries
                if not (self._orange_band_start <= freq <= self._orange_band_end):
                    continue
                
                # Check for duplicate with tolerance (avoid multiple zooms on same peak)
                # Use larger tolerance: 3x RBW or minimum 300 kHz
                duplicate_tolerance = max(3 * self.rbw, 500e3)
                is_duplicate = any(abs(freq - pf) < duplicate_tolerance for pf, _, _ in self._peak_queue)
                
                if not is_duplicate:
                    self._peak_queue.append((freq, power, None))
                    print(f"  Found Orange peak: {freq/1e6:.3f} MHz ({power:.1f} dB)")
                else:
                    print(f"  Skipped duplicate: {freq/1e6:.3f} MHz (within {duplicate_tolerance/1e3:.0f} kHz of existing peak)")
        
        except Exception as e:
            print(f"Error collecting Orange peaks: {e}")


    def _finish_orange_scan(self):
        """Finish Orange scan and start analysis"""
        try:
            self._orange_scan_timer.stop()
            
            if len(self._peak_queue) == 0:
                print(f"⚠️  No Orange peaks found during re-scan")
                print(f"Restoring to wide-span view\n")
                self.set_center_freq(self._saved_center)
                self.set_samp_rate(self._saved_span)
                self._orange_rescan_done = False
                return
            
            print(f"\n✅ Orange scan complete: {len(self._peak_queue)} peaks found")
            
            # Sort by power
            self._peak_queue.sort(key=lambda x: x[1], reverse=True)
            
            # Start analyzing Orange peaks
            self._processing_peaks = True
            self._scanning_mode = False
            self._current_peak_idx = 0
            self._peak_freq_list = [(freq, pwr) for freq, pwr, _ in self._peak_queue]
            
            print(f"Analyzing {len(self._peak_freq_list)} Orange peaks...\n")
        
        except Exception as e:
            print(f"Error finishing Orange scan: {e}")
            self._orange_rescan_done = False

    def _end_cooldown(self):
        self._cooldown = False

    # ----------------------------
    # Improved sweep time calculation methods
    # ----------------------------
    def calculate_sweep_time(self):
        """
        Calculate sweep time based on spectrum analyzer principles with GSM optimization
        """
        # Define the actual frequency span being displayed
        span = self.samp_rate  # Full bandwidth being analyzed
        
        # Traditional spectrum analyzer formula
        base_sweep_time = self.k_total * span / (self.rbw ** 2)
        
        # Add practical limits for GUI responsiveness
        min_sweep_time = 0.010  # 10ms minimum for GUI
        max_sweep_time = 1.000  # 1s maximum to avoid sluggish response
        
        sweep_time = max(min_sweep_time, min(base_sweep_time, max_sweep_time))
        
        # GSM-specific optimization
        # TDMA frame = 4.615ms (8 slots), each slot = 577μs
        # For burst detection, update rate should be at least 3x faster than burst period
        gsm_burst_period = 577e-6  # 577 microseconds
        max_update_for_gsm = 3
        
        # Choose the more restrictive requirement
        if sweep_time > max_update_for_gsm:
            print(f"Info: Sweep time {sweep_time*1000:.1f}ms limited to {max_update_for_gsm*1000:.1f}ms for GSM burst detection")
            sweep_time = max_update_for_gsm
        
        return sweep_time

    def get_adaptive_k_total(self, rbw):
        """
        Adjust k_total based on RBW for different measurement scenarios
        """
        if rbw <= 1e3:        # Very narrow RBW - need more averaging
            return 5.0
        elif rbw <= 10e3:     # Medium RBW - balanced
            return 2.5
        else:                 # Wide RBW - can be faster
            return 1.0

    def gsm_optimized_sweep_time(self):
        """
        GSM-optimized sweep time that balances spectrum quality with burst detection
        """
        base_sweep_time = self.k_total * self.samp_rate / (self.rbw ** 2)
        
        # GSM considerations:
        # - TDMA frame = 4.615ms (8 slots)
        # - Each slot = 577μs
        # - Need to capture burst structure
        
        gsm_frame_time = 4.615e-3
        desired_frames_per_sweep = 1  # Capture at least 1 full frame
        gsm_optimized_time = gsm_frame_time / desired_frames_per_sweep
        
        # Use the more restrictive (faster) requirement
        return min(base_sweep_time, gsm_optimized_time)

    # ----------------------------
    # UI controls
    # ----------------------------

    def create_control_widgets(self):
        panel = Qt.QFrame()
        panel.setFrameStyle(Qt.QFrame.StyledPanel)
        layout = Qt.QGridLayout(panel)

        # Center frequency control
        self._cf_range = qtgui.Range(880e6, 1.0e9, 100e3, self.center_freq, 200)
        self._cf_win = qtgui.RangeWidget(self._cf_range, self.set_center_freq, 'Center Freq [Hz]', "counter_slider", float)

        # Sample rate control
        self._fs_range = qtgui.Range(200e3, 50e6, 100e3, self.samp_rate, 200)
        self._fs_win = qtgui.RangeWidget(self._fs_range, self.set_samp_rate, 'Sample Rate [Hz]', "counter_slider", float)

        # RF frontend bandwidth
        self._rfbw_range = qtgui.Range(1e6, 40e6, 1e6, self.rf_bw, 200)
        self._rfbw_win = qtgui.RangeWidget(self._rfbw_range, self.set_rf_bw, 'RF Frontend BW [Hz]', "counter_slider", float)

        # RBW / VBW
        self.rbw_fraction = 0.002  # 0.2% of span by default
        self._rbw_range = qtgui.Range(0.001, 0.02, 0.001, self.rbw_fraction, 200)
        self._rbw_win = qtgui.RangeWidget(self._rbw_range, self.rbw_fraction_changed, 'RBW fraction of span', "counter_slider", float)

        self._rbw_label = Qt.QLabel(f"{self.rbw/1e3:.1f} kHz")
        self._rbw_label.setAlignment(QtCore.Qt.AlignCenter)

        self._rbw_hz_range = qtgui.Range(1e3, 500e3, 1e3, self.rbw, 200)
        self._rbw_hz_win = qtgui.RangeWidget(self._rbw_hz_range, self.rbw_changed, 'RBW [Hz]', "counter_slider", float)

        self._vbw_range = qtgui.Range(1e3, 1e6, 1e3, self.vbw, 200)
        self._vbw_win = qtgui.RangeWidget(self._vbw_range, self.vbw_changed, 'VBW [Hz]', "counter_slider", float)

        # Sweep time constant control (NEW)
        self._k_range = qtgui.Range(1, 12000, 1, self.k_total, 200)
        self._k_win = qtgui.RangeWidget(self._k_range, self.set_k_total, 'Sweep Time Factor (k)', "counter_slider", float)

        # PVT decimation
        self._pvtdec_range = qtgui.Range(0.01, 1000, 0.1, self.pvt_decim, 1)
        self._pvtdec_win = qtgui.RangeWidget(self._pvtdec_range, self.set_pvt_decim, 'PVT Decimation', "counter_slider", float)

        # TDMA slicer controls
        self._th_range = qtgui.Range(-80.0, 40.0, 0.5, self.tdma_threshold_db, 200)
        self._th_win = qtgui.RangeWidget(self._th_range, self.set_tdma_threshold, 'TDMA Threshold [dB]', "counter_slider", float)

        self._avg_range = qtgui.Range(1, 50, 1, self.tdma_avg_len, 200)
        self._avg_win = qtgui.RangeWidget(self._avg_range, self.set_tdma_avg, 'TDMA Avg Len [samples]', "counter_slider", float)

        # Band-power ROI
        self._band_target_freq_range = qtgui.Range(880e6, 1.0e9, 100e3, self.band_target_freq, 200)
        self._band_target_freq_win = qtgui.RangeWidget(self._band_target_freq_range, self.set_band_target_freq, 'Band Target Freq [Hz]', "counter_slider", float)
        
        self._band_bandwidth_range = qtgui.Range(50e3, 2e6, 50e3, self.band_bandwidth, 200)
        self._band_bandwidth_win = qtgui.RangeWidget(self._band_bandwidth_range, self.set_band_bandwidth, 'Band Bandwidth [Hz]', "counter_slider", float)

        # Layout
        r = 0
        layout.addWidget(self._cf_win,   r, 0, 1, 2); r += 1
        layout.addWidget(self._fs_win,   r, 0, 1, 2); r += 1
        layout.addWidget(self._rfbw_win, r, 0, 1, 2); r += 1
        layout.addWidget(self._rbw_win,  r, 0, 1, 1)
        layout.addWidget(self._vbw_win,  r, 1, 1, 1); r += 1
        layout.addWidget(self._k_win,    r, 0, 1, 2); r += 1
        layout.addWidget(self._pvtdec_win, r, 0, 1, 2); r += 1
        layout.addWidget(self._th_win,   r, 0, 1, 1)
        layout.addWidget(self._avg_win,  r, 1, 1, 1); r += 1
        layout.addWidget(self._band_target_freq_win, r, 0, 1, 1)
        layout.addWidget(self._band_bandwidth_win,   r, 1, 1, 1)
        layout.addWidget(self._rbw_label,     r, 1, 1, 1); r += 1

        # Add to the top
        self.top_layout.insertWidget(0, panel)

    # ---- setters called by widgets ----
    def set_center_freq(self, cf):
        self.center_freq = cf
        self.uhd_usrp_source_0.set_center_freq(self.center_freq, 0)
        self.qtgui_freq_sink_x_0.set_frequency_range(self.center_freq, self.samp_rate)
        self.qtgui_vector_sink_f_0.set_x_axis(self.center_freq - self.samp_rate/2,
                                              self.samp_rate/self.fft_size)
        self.peak_detector.set_center_freq(self.center_freq)
        self.band_power.set_center_freq(self.center_freq)

    def set_samp_rate(self, fs):
        self.samp_rate = fs
        self.uhd_usrp_source_0.set_samp_rate(self.samp_rate)
        
        # Update all displays and processing blocks
        self.qtgui_freq_sink_x_0.set_frequency_range(self.center_freq, self.samp_rate)
        self.qtgui_vector_sink_f_0.set_x_axis(self.center_freq - self.samp_rate/2,
                                              self.samp_rate/self.fft_size)
        
        # Update normalization for new sample rate
        fft_scale = 1.0 / self.samp_rate
        self.blocks_multiply_const_fft = blocks.multiply_const_ff(fft_scale, self.fft_size)

        # Keep PVT scaling chain logic the same (value left as-is)
        power_scale = 1.0
        self.pvt_normalize.set_k(power_scale)
        
        # Update filters
        self.rbw_filter.samp_rate = self.samp_rate
        self._rbw_label.setText(f"{self.rbw/1e3:.1f} kHz")
        self.vbw_filter.samp_rate = self.samp_rate
        self.rbw_filter.update_kernel()
        self.vbw_filter.update_alpha()
        
        # Update detectors
        self.band_power.samp_rate = self.samp_rate
        self.peak_detector.samp_rate = self.samp_rate
        self.band_power.update_freq_bins()
        self.peak_detector.update_freq_bins()
        
        # Recalculate sweep time with new sample rate
        self.sweep_time = self.calculate_sweep_time()
        self.update_rate = self.sweep_time
        self.update_all_display_rates()
        
        # Update time sinks with new decimated rate
        decimated_fs = self.samp_rate / max(1, int(self.pvt_decim))
        self.pvt_sink.set_samp_rate(decimated_fs)
        self.tdma_sink.set_samp_rate(decimated_fs)
        
        # Recalculate expected samples per slot
        expected_samples_per_slot = int(577e-6 * decimated_fs)
        print(f"Updated: Sample rate {fs/1e6:.1f} MHz, decimated {decimated_fs/1e3:.1f} kSa/s")
        print(f"Expected samples per GSM slot: {expected_samples_per_slot}")
        print(f"New sweep time: {self.sweep_time*1000:.1f} ms")

    def set_rf_bw(self, bw):
        self.rf_bw = bw
        self.uhd_usrp_source_0.set_bandwidth(self.rf_bw, 0)

    def set_k_total(self, k):
        """NEW: Set the sweep time factor k_total"""
        self.k_total = k
        self.sweep_time = self.calculate_sweep_time()
        self.update_rate = self.sweep_time
        self.update_all_display_rates()
        print(f"Sweep time factor updated: k={self.k_total:.1f}")
        print(f"New sweep time: {self.sweep_time*1000:.1f} ms")

    def rbw_changed(self, val):
        self.rbw = val
        self.rbw_filter.set_rbw(self.rbw)
        
        # Recalculate sweep time with new RBW
        self.sweep_time = self.calculate_sweep_time()
        self.update_rate = self.sweep_time
        self.update_all_display_rates()
        
        print(f"RBW updated: {self.rbw/1e3:.1f} kHz")
        print(f"New sweep time: {self.sweep_time*1000:.1f} ms")
        print(f"Update rate: {self.update_rate*1000:.1f} ms")

    def vbw_changed(self, val):
        self.vbw = val
        self.vbw_filter.set_vbw(self.vbw)

    def update_all_display_rates(self):
        """Helper method to update all display update rates"""
        self.qtgui_vector_sink_f_0.set_update_time(self.update_rate)
        self.qtgui_freq_sink_x_0.set_update_time(self.update_rate)
        self.pvt_sink.set_update_time(self.update_rate)
        self.tdma_sink.set_update_time(self.update_rate)
        self.band_time_sink.set_update_time(self.update_rate)

    def set_pvt_decim(self, d):
        d = max(0.01, float(d))  # allow float, minimum 0.01 to avoid zero or negative
        self.pvt_decim = d

        # Since keep_one_in_n only supports integer decimation, replace or bypass it
        # For fractional decimation, you need a custom block or resampler

        # For now, disconnect old keep_one_in_n and create a new fractional decimator block or use a rational resampler
        # Example: use a rational resampler block from GNU Radio (if available)
        # self.keep_one_in_n.set_n(int(round(self.pvt_decim)))  # no longer valid

        # Update time sink sample rates accordingly
        decimated_fs = self.samp_rate / self.pvt_decim
        self.pvt_sink.set_samp_rate(decimated_fs)
        self.tdma_sink.set_samp_rate(decimated_fs)

        expected_samples_per_slot = int(577e-6 * decimated_fs)
        print(f"PVT Decimation updated: {decimated_fs/1e3:.1f} kSa/s, ~{expected_samples_per_slot} samples/GSM slot")

    def set_tdma_threshold(self, th):
        self.tdma_threshold_db = th
        self.tdma_slicer.set_threshold(th)

    def set_tdma_avg(self, n):
        self.tdma_avg_len = int(n)
        self.tdma_slicer.set_avg_len(self.tdma_avg_len)

    def set_band_target_freq(self, f):
        self.band_target_freq = f
        self.band_power.set_target(self.band_target_freq, self.band_bandwidth)

    def set_band_bandwidth(self, b):
        self.band_bandwidth = b
        self.band_power.set_target(self.band_target_freq, self.band_bandwidth)

    # ---- housekeeping ----
    def closeEvent(self, event):
        self.settings = Qt.QSettings("gnuradio/flowgraphs", "SpecAnalyser")
        self.settings.setValue("geometry", self.saveGeometry())
        self.stop()
        self.wait()
        event.accept()

    def get_samp_rate(self):
        return self.samp_rate

    def rbw_fraction_changed(self, fraction):
        self.rbw_fraction = fraction
        new_rbw = fraction * self.samp_rate
        self.rbw = new_rbw
        self.rbw_filter.set_rbw(new_rbw)
        
        self._rbw_label.setText(f"{self.rbw/1e3:.1f} kHz")
        
        self.sweep_time = self.calculate_sweep_time()
        self.update_rate = self.sweep_time
        self.update_all_display_rates()

    # ----------------------------
    # TDMA plotting method
    # ----------------------------

    def filter_tdma_frame_starts(self, rising_edges, decimated_fs, gsm_frame_ms=4.615, frame_tolerance_ms=0.5):
        """Filter rising edges to only include those that match TDMA frame timing"""
        if len(rising_edges) < 2:
            return rising_edges

        samples_per_ms = decimated_fs / 1000
        expected_frame_samples = gsm_frame_ms * samples_per_ms
        tolerance_samples = frame_tolerance_ms * samples_per_ms
        min_frame_samples = (gsm_frame_ms - frame_tolerance_ms) * samples_per_ms

        valid_frame_starts = [rising_edges[0]]
        last_valid_frame = rising_edges[0]

        for edge in rising_edges[1:]:
            time_since_last = edge - last_valid_frame
            
            # Accept any burst that's at least one frame duration away
            if time_since_last >= min_frame_samples:
                # Check if it's close to an expected frame boundary
                frame_multiples = round(time_since_last / expected_frame_samples)
                expected_position = last_valid_frame + (frame_multiples * expected_frame_samples)
                
                # More lenient tolerance - accept if within tolerance OR if it's been long enough
                if (frame_multiples >= 1 and 
                    (abs(edge - expected_position) <= tolerance_samples or 
                     time_since_last >= expected_frame_samples * 0.8)):  # Accept if >80% of frame time
                    valid_frame_starts.append(edge)
                    last_valid_frame = edge

        return np.array(valid_frame_starts)

    def collect_tdma_circular_buffer(self, total_samples_needed, snapshot_size=1024, interval_ms=1.5):
        """Collect TDMA data using circular buffer approach - optimized for 4 windows"""
        all_data = []
        snapshots_needed = max(4, total_samples_needed // snapshot_size)  # Ensure at least 4 snapshots
        
        for i in range(snapshots_needed):
            snapshot = self.tdma_probe.get_last(snapshot_size)
            if len(snapshot) > 0:
                all_data.extend(snapshot)
            
            # Smaller delay between snapshots for more diverse data
            if i < snapshots_needed - 1:
                time.sleep(interval_ms / 1000.0)
        
        return np.array(all_data[:total_samples_needed]) if all_data else np.array([])

    def plot_tdma_window(self, duration_ms=None, timeout=3.0):
        """TDMA analysis with combined single plot view showing 2 windows"""

        gnuradio_buffer_size = 8192
        decimated_fs = self.samp_rate / max(1, int(self.pvt_decim))

        # Use circular buffer for data collection
        data = self.collect_tdma_circular_buffer(gnuradio_buffer_size)
        
        # Fallback if circular buffer fails
        if len(data) == 0:
            data = self.tdma_probe.get_last(gnuradio_buffer_size)
        
        if len(data) == 0:
            return

        # GATE 1 REMOVED: Always proceed with analysis regardless of rising edges
        rising_edges = np.where((data[:-1] < 0.5) & (data[1:] >= 0.5))[0] + 1

        gsm_frame_ms = 4.615
        frame_tolerance_ms = 0.2
        samples_per_ms = decimated_fs / 1000
        samples_per_frame = int(gsm_frame_ms * samples_per_ms)

        # Handle both cases: with and without rising edges
        if len(rising_edges) > 0:
            valid_frame_starts = self.filter_tdma_frame_starts(rising_edges, decimated_fs, gsm_frame_ms, frame_tolerance_ms)
            if len(valid_frame_starts) == 0:
                valid_frame_starts = rising_edges
            print(f"DEBUG: Using {len(valid_frame_starts)} edge-based frame starts")
        else:
            # Create synthetic frame boundaries at GSM intervals when no edges detected
            max_frames = max(10, len(data) // samples_per_frame)
            valid_frame_starts = np.arange(0, len(data), samples_per_frame)[:max_frames]
            print(f"DEBUG: No rising edges - created {len(valid_frame_starts)} synthetic frame starts")

        # Start before the first frame to show pre-burst quiet period
        pre_burst_offset = int(0.8 * samples_per_ms)
        alignment_start = max(0, valid_frame_starts[0] - pre_burst_offset)
        aligned_data = data[alignment_start:]

        if len(aligned_data) == 0:
            return

        gsm_slot_ms = 0.577
        slot_tolerance_ms = 0.05
        activity_threshold = 0.6
        peak_threshold = 0.8
        min_mean_level = 0.4

        target_frames_per_window = 10
        minimum_window_duration_ms = target_frames_per_window * gsm_frame_ms
        window_size_samples = int(minimum_window_duration_ms * samples_per_ms)
        total_windows = 2  # Changed from 5 to 2
        total_needed_samples = window_size_samples * total_windows

        # Build 2 windows
        if len(aligned_data) >= total_needed_samples:
            windows = []
            for i in range(total_windows):
                start_idx = i * window_size_samples
                end_idx = (i + 1) * window_size_samples
                windows.append(aligned_data[start_idx:end_idx])
        
        else:
            extended_data = self.tdma_probe.get_last(total_needed_samples)
            if len(extended_data) >= total_needed_samples:
                extended_rising_edges = np.where((extended_data[:-1] < 0.5) & (extended_data[1:] >= 0.5))[0] + 1
                
                # Apply same logic for extended data
                if len(extended_rising_edges) > 0:
                    extended_valid_frames = self.filter_tdma_frame_starts(extended_rising_edges, decimated_fs, gsm_frame_ms, frame_tolerance_ms)
                    start_idx = extended_valid_frames[0] if len(extended_valid_frames) > 0 else extended_rising_edges[0]
                else:
                    # Create synthetic start for extended data too
                    start_idx = 0
                    
                pre_burst_offset = int(0.5 * samples_per_ms)
                alignment_start = max(0, start_idx - pre_burst_offset)
                aligned_data = extended_data[alignment_start:] if len(extended_rising_edges) > 0 else extended_data

                if len(aligned_data) >= total_needed_samples:
                    windows = []
                    for i in range(total_windows):
                        start_idx = i * window_size_samples
                        end_idx = (i + 1) * window_size_samples
                        windows.append(aligned_data[start_idx:end_idx])
                else:
                    # Divide available data into 2 parts
                    part_size = len(aligned_data) // total_windows
                    windows = []
                    for i in range(total_windows):
                        start_idx = i * part_size
                        end_idx = (i + 1) * part_size if i < total_windows - 1 else len(aligned_data)
                        windows.append(aligned_data[start_idx:end_idx])
            elif len(aligned_data) >= window_size_samples:
                # Create overlapping windows from available data
                overlap_offset = max(0, len(aligned_data) - window_size_samples)
                step_size = overlap_offset // (total_windows - 1) if total_windows > 1 else 0
                windows = []
                for i in range(total_windows):
                    start_idx = min(i * step_size, len(aligned_data) - window_size_samples)
                    end_idx = min(start_idx + window_size_samples, len(aligned_data))
                    if end_idx > start_idx:
                        windows.append(aligned_data[start_idx:end_idx])
            else:
                # Fallback: divide whatever data we have into 2 parts
                part_size = len(aligned_data) // total_windows
                windows = []
                for i in range(total_windows):
                    start_idx = i * part_size
                    end_idx = (i + 1) * part_size if i < total_windows - 1 else len(aligned_data)
                    if end_idx > start_idx:
                        windows.append(aligned_data[start_idx:end_idx])

        if not windows:
            windows = [aligned_data]

        total_slot_counts = np.zeros(8)
        total_expected_frames = 0
        all_results = []

        print(f"DEBUG: Processing {len(windows)} windows")

        # Process all windows (now handles 2 windows)
        for window_idx, window_data in enumerate(windows):
            if len(window_data) < samples_per_frame:
                continue

            window_time_ms = len(window_data) / decimated_fs * 1000
            expected_frames = max(10, int(window_time_ms / gsm_frame_ms))

            # Apply same no-gate logic to window-level edge detection
            window_rising_edges = np.where((window_data[:-1] < 0.5) & (window_data[1:] >= 0.5))[0] + 1
            
            if len(window_rising_edges) > 0:
                window_valid_frames = self.filter_tdma_frame_starts(window_rising_edges, decimated_fs, gsm_frame_ms, frame_tolerance_ms)
                if len(window_valid_frames) == 0:
                    window_valid_frames = window_rising_edges
            else:
                # Create synthetic frames for this window too
                synthetic_frame_count = max(1, int(window_time_ms / gsm_frame_ms))
                window_valid_frames = np.arange(0, len(window_data), samples_per_frame)[:synthetic_frame_count]

            window_slot_counts = np.zeros(8)
            
            # Always analyze frames (whether real or synthetic)
            if len(window_valid_frames) >= 1:
                first_frame_start = window_valid_frames[0]
                first_frame_start_ms = first_frame_start / decimated_fs * 1000
                
                expected_frame_positions = []
                for frame_idx in range(expected_frames):
                    expected_time_ms = first_frame_start_ms + (frame_idx * gsm_frame_ms)
                    expected_sample = int(expected_time_ms * samples_per_ms)
                    if expected_sample < len(window_data):
                        expected_frame_positions.append(expected_sample)
                
                frames_analyzed = 0
                for frame_idx in range(10):
                    if frame_idx >= len(expected_frame_positions):
                        break
                        
                    frame_start_sample = expected_frame_positions[frame_idx]
                    frame_end_sample = min(len(window_data), frame_start_sample + samples_per_frame)
                    frame_data = window_data[frame_start_sample:frame_end_sample]
                    
                    if len(frame_data) < int(0.8 * samples_per_frame):
                        continue
                    
                    frames_analyzed += 1
                    
                    for slot_num in range(8):
                        slot_start_idx = int((slot_num * gsm_slot_ms) * samples_per_ms)
                        slot_end_idx = int(((slot_num + 1) * gsm_slot_ms) * samples_per_ms)
                        slot_start_idx = max(0, slot_start_idx)
                        slot_end_idx = min(len(frame_data), slot_end_idx)

                        if slot_end_idx > slot_start_idx:
                            slot_data = frame_data[slot_start_idx:slot_end_idx]
                            
                            if len(slot_data) > 0:
                                max_amplitude = np.max(slot_data)
                                mean_amplitude = np.mean(slot_data)
                                activity_ratio = np.mean(slot_data > 0.5)
                                
                                if (max_amplitude > peak_threshold and 
                                    activity_ratio > activity_threshold and 
                                    mean_amplitude > min_mean_level):
                                    window_slot_counts[slot_num] += 1

                window_frames_analyzed = frames_analyzed
            else:
                window_frames_analyzed = 0

            all_results.append({
                'slot_counts': window_slot_counts,
                'expected_frames': expected_frames,
                'window_idx': window_idx,
                'data': window_data,
                'valid_frames': window_valid_frames if len(window_valid_frames) > 0 else window_rising_edges,
                'first_frame_start': window_valid_frames[0] if len(window_valid_frames) > 0 else 0
            })

            total_slot_counts += window_slot_counts
            total_expected_frames += window_frames_analyzed

        print(f"DEBUG: Analyzed {total_expected_frames} total frames across {len(all_results)} windows")

        # Calculate probabilities (now based on up to 20 frames: 2 windows × 10 frames each)
        slot_probabilities = np.zeros(8)
        for slot_idx in range(8):
            if total_expected_frames > 0:
                slot_probabilities[slot_idx] = (total_slot_counts[slot_idx] / total_expected_frames) * 100
        
        slot_probabilities = np.where(slot_probabilities >= 90, 100, slot_probabilities)
        
        # HOPPING MITIGATION - Enhanced user estimation
        naive_active_users = [i for i, prob in enumerate(slot_probabilities) if prob > 80]
        naive_user_count = len(naive_active_users)
        
        # Method 1: Maximum Simultaneous Detection
        max_simultaneous = 0
        for result in all_results:
            if len(result['valid_frames']) > 0:
                # Check each frame in this window for simultaneous active slots
                window_data = result['data']
                
                for frame_idx in range(min(3, 10)):  # Check first 3 frames per window
                    frame_active_slots = 0
                    frame_start = frame_idx * samples_per_frame
                    frame_end = min(len(window_data), frame_start + samples_per_frame)
                    
                    if frame_end > frame_start:
                        frame_data = window_data[frame_start:frame_end]
                        
                        # Check each slot in this frame
                        for slot_num in range(8):
                            slot_start_idx = int((slot_num * gsm_slot_ms) * samples_per_ms)
                            slot_end_idx = int(((slot_num + 1) * gsm_slot_ms) * samples_per_ms)
                            
                            if slot_end_idx <= len(frame_data):
                                slot_data = frame_data[slot_start_idx:slot_end_idx]
                                
                                if len(slot_data) > 0:
                                    max_amplitude = np.max(slot_data)
                                    mean_amplitude = np.mean(slot_data)
                                    activity_ratio = np.mean(slot_data > 0.5)
                                    
                                    if (max_amplitude > peak_threshold and 
                                        activity_ratio > activity_threshold and 
                                        mean_amplitude > min_mean_level):
                                        frame_active_slots += 1
                    
                    max_simultaneous = max(max_simultaneous, frame_active_slots)
        
        # Method 2: Pattern-based reduction for likely hopping
        estimated_users = naive_user_count
        hopping_detected = False
        
        # If we see high slot usage but low simultaneous, likely hopping
        if naive_user_count >= 3 and max_simultaneous < naive_user_count:
            # Conservative estimate: use max simultaneous + small buffer
            estimated_users = min(max_simultaneous + 1, naive_user_count)
            hopping_detected = True
        elif naive_user_count >= 2 and max_simultaneous <= 1:
            # Likely single user hopping between slots
            estimated_users = 1
            hopping_detected = True
        else:
            estimated_users = max_simultaneous
        
        # Ensure we don't estimate 0 users if we detected activity
        if estimated_users == 0 and naive_user_count > 0:
            estimated_users = 1
        
        active_users = naive_active_users  # Keep original for slot display

        # COMBINED PLOTTING - Single plot with 2 windows
        fig, (ax_main, ax_prob) = plt.subplots(2, 1, figsize=(20, 12), height_ratios=[3, 1])
        
        # Combine all window data with time offsets
        combined_data = []
        combined_time = []
        window_boundaries = []
        current_time_offset = 0
        
        # Color palette for 2 windows
        window_colors = ['blue', 'red']
        
        for i, result in enumerate(all_results[:total_windows]):
            window_data = result['data']
            window_time_ms = np.arange(len(window_data)) / decimated_fs * 1000
            
            # Add time offset for continuous display
            offset_time = window_time_ms + current_time_offset
            combined_time.extend(offset_time)
            combined_data.extend(window_data)
            
            # Plot this window's data
            ax_main.plot(offset_time, window_data, color=window_colors[i], 
                        linewidth=0.8, alpha=0.8, label=f'W{i+1}')
            
            # Add window boundary marker
            if i > 0:
                ax_main.axvline(x=current_time_offset, color='black', linestyle=':', 
                               linewidth=2, alpha=0.7)
                ax_main.text(current_time_offset, 1.3, f'W{i+1}', rotation=0, 
                            ha='center', va='bottom', fontsize=10, fontweight='bold',
                            bbox=dict(boxstyle='round,pad=0.2', facecolor='lightgray', alpha=0.8))
            
            # Add green frame markers (E0, E1, E2, etc.) for this window
            if len(result['valid_frames']) > 0:
                first_frame_start = result['first_frame_start']
                first_frame_start_ms = first_frame_start / decimated_fs * 1000
                
                window_time_span = len(window_data) / decimated_fs * 1000
                expected_frames_count = int(window_time_span / gsm_frame_ms)
                
                for expected_frame in range(expected_frames_count + 1):
                    expected_time_ms = first_frame_start_ms + (expected_frame * gsm_frame_ms)
                    if 0 <= expected_time_ms <= window_time_span:
                        # Add time offset to align with combined plot
                        global_time_ms = expected_time_ms + current_time_offset
                        ax_main.axvline(x=global_time_ms, color='green', linestyle='-', 
                                       alpha=0.8, linewidth=2)
                        ax_main.text(global_time_ms, 1.25, f'E{expected_frame}', rotation=0, 
                                   ha='center', va='bottom', fontsize=8, color='green', 
                                   fontweight='bold',
                                   bbox=dict(boxstyle='round,pad=0.2', facecolor='lightgreen', alpha=0.8))
                
                # Add slot boundaries for first 3 frames of each window
                for frame_idx in range(min(3, expected_frames_count)):
                    frame_start_ms = first_frame_start_ms + (frame_idx * gsm_frame_ms)
                    
                    for slot_num in range(8):
                        slot_start_ms = frame_start_ms + (slot_num * gsm_slot_ms)
                        
                        if 0 <= slot_start_ms <= window_time_span:
                            global_slot_time = slot_start_ms + current_time_offset
                            ax_main.axvline(x=global_slot_time, color='orange', linestyle='-', 
                                           alpha=0.4, linewidth=0.6)
                            # Only label slots in first frame of first window to avoid clutter
                            if i == 0 and frame_idx == 0 and slot_num < 7:
                                ax_main.text(global_slot_time + gsm_slot_ms/2, 0.1, f'S{slot_num}', 
                                           rotation=90, ha='center', va='bottom', fontsize=7, color='orange')
            
            # Update time offset for next window
            current_time_offset += len(window_data) / decimated_fs * 1000
            window_boundaries.append(current_time_offset)
        
        ax_main.set_title(f"Combined TDMA Analysis - 2 Windows ({total_expected_frames} total expected frames)")
        ax_main.set_xlabel("Time (ms)")
        ax_main.set_ylabel("Signal Level")
        ax_main.set_ylim(-0.3, 1.4)
        ax_main.grid(True, alpha=0.3)
        ax_main.legend(loc='upper right')
        
        # Add legend for markers
        ax_main.text(0.02, 0.98, 
                   'Green: Expected frames (E0, E1, E2...)\nOrange: Slot boundaries (S0-S7)\nDotted: Window boundaries', 
                   transform=ax_main.transAxes, verticalalignment='top', fontsize=9,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))
        
        # Probability chart
        slot_names = [f'S{i}' for i in range(8)]
        colors = ['red' if prob > 80 else 'lightblue' for prob in slot_probabilities]

        bars = ax_prob.bar(slot_names, slot_probabilities, color=colors, alpha=0.7, edgecolor='black')
        ax_prob.axhline(y=80, color='red', linestyle='--', linewidth=2, alpha=0.8, label='80% Threshold')

        for i, (bar, prob) in enumerate(zip(bars, slot_probabilities)):
            ax_prob.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                         f'{prob:.0f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
            if prob > 80:
                ax_prob.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2, 
                             'USER', ha='center', va='center', fontweight='bold', color='white', fontsize=10)

        ax_prob.set_title(f'Slot Analysis - Estimated Users: {estimated_users} (Naive: {naive_user_count}) - 2 Windows')
        ax_prob.set_xlabel('Time Slots')
        ax_prob.set_ylabel('Usage (%)')
        ax_prob.set_ylim(0, 110)
        ax_prob.grid(True, alpha=0.3)
        ax_prob.legend()

        info_text = f"""Estimated Users: {estimated_users}
    Naive Count: {naive_user_count}
    Max Simultaneous: {max_simultaneous}
    Active Slots: {active_users}
    Total Frames: {total_expected_frames}
    Hopping: {'Detected' if hopping_detected else 'None'}"""
        ax_prob.text(0.02, 0.98, info_text, transform=ax_prob.transAxes,
                     verticalalignment='top', fontsize=9,
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcyan', alpha=0.8))

        # Store user count for current peak
        if hasattr(self, '_current_peak_freq'):
            op = operator_for(self._saved_center, self._saved_span, self._current_peak_freq)
            self.current_operator_peaks[op] = [peak for peak in self.current_operator_peaks[op] 
                                              if abs(peak['freq'] - self._current_peak_freq) > 1000]
            self.current_operator_peaks[op].append({
                'freq': self._current_peak_freq,
                'users': estimated_users  # Use hopping-aware count instead of len(active_users)
            })

        # DEBUG: Confirm plot generation
        print(f"DEBUG: TDMA plot generated successfully with {len(all_results)} windows, {total_expected_frames} frames")
        
        plt.tight_layout()
        plt.show()
    
    def calculate_operator_totals(self):
        """Calculate and print total users per operator"""
        total_users_all_operators = 0  # Add this line
        
        for operator, peaks in self.current_operator_peaks.items():
            if peaks:
                total_users = sum(peak['users'] for peak in peaks)
                self.operator_users[operator] = total_users
                total_users_all_operators += total_users  # Add this line
                print(f"Operator {operator}: {total_users} users across {len(peaks)} peaks")
                
                # Optional: show breakdown per peak
                for i, peak in enumerate(peaks):
                    print(f"  Peak {i+1}: {peak['freq']/1e6:.6f} MHz - {peak['users']} users")
        
        # Add this line at the end
        print(f"TOTAL USERS DETECTED: {total_users_all_operators}")
                    
def main(top_block_cls=SpecAnalyser, options=None):
    qapp = Qt.QApplication(sys.argv)
    tb = top_block_cls()
    tb.start()
    tb.flowgraph_started.set()
    tb.show()

    def sig_handler(sig=None, frame=None):
        tb.stop()
        
        tb.wait()
        Qt.QApplication.quit()

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    timer = Qt.QTimer()
    timer.start(500)
    timer.timeout.connect(lambda: None)
    qapp.exec_()


if __name__ == '__main__':
    main()