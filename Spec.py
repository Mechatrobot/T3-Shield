#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: GPL-3.0
# GNU Radio Python Flow Graph
# Title: GSM_PVT_Spectrum_Improved
# GNU Radio version: 3.10.12.0

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

# ----------------------------
# Custom helper blocks
# ----------------------------

class RBW_Filter(gr.sync_block):
    """Resolution Bandwidth Filter - Controls frequency domain resolution"""
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
        """Set RBW as a fraction of the current span (samp_rate)"""
        self.rbw = fraction * self.samp_rate
        self.update_kernel()

    def work(self, input_items, output_items):
        data = input_items[0]
        bin_bw = self.samp_rate / self.fft_size
        rbw_hz = len(self.kernel) * bin_bw
        noise_offset_db = 10 * np.log10(rbw_hz / bin_bw)

        for i, vec in enumerate(data):
            if len(self.kernel) > 1:
                smoothed = np.convolve(vec, self.kernel, mode='same')
                edge_cut = len(self.kernel) // 2
                smoothed[:edge_cut] = vec[:edge_cut]
                smoothed[-edge_cut:] = vec[-edge_cut:]
                # apply RBW correction
                output_items[0][i] = smoothed + noise_offset_db
            else:
                output_items[0][i] = vec
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
        max_vbw = 1e6
        min_vbw = 1e3
        normalized_vbw = (self.vbw - min_vbw) / (max_vbw - min_vbw)
        normalized_vbw = np.clip(normalized_vbw, 0.0, 1.0)
        self.alpha = 0.01 + 0.99 * normalized_vbw
        print(f"VBW Filter updated: {self.vbw/1e3:.1f} kHz, alpha={self.alpha:.3f}")

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
    """Peak detector over FFT data (GSM only)."""
    def __init__(self, fft_size, samp_rate, center_freq, p_threshold=-40, output_path="P_Val.csv"):
        gr.sync_block.__init__(self,
            name="Peak Detector",
            in_sig=[(np.float32, fft_size)],
            out_sig=None)
        self.fft_size = fft_size
        self.samp_rate = samp_rate
        self.center_freq = center_freq
        self.p_threshold = p_threshold
        self.output_path = output_path
        self.update_freq_bins()
        if not os.path.isfile(self.output_path):
            with open(self.output_path, "w", newline="") as f:
                csv.writer(f).writerow(["Timestamp", "BinIndex", "PeakPower(dB)", "CenterFreq(MHz)", "Bandwidth(Hz)", "Protocol"])

    def update_freq_bins(self):
        freq_res = self.samp_rate / self.fft_size
        self.freqs = self.center_freq - (self.samp_rate / 2) + np.arange(self.fft_size) * freq_res

    def set_center_freq(self, center_freq):
        self.center_freq = center_freq
        self.update_freq_bins()

    def get_protocol(self, freq_hz):
        # GSM-900 downlink (935–960 MHz) & uplink (890–915 MHz)
        if (935e6 <= freq_hz <= 960e6) or (890e6 <= freq_hz <= 915e6):
            return "GSM"
        return "Unknown"

    def work(self, input_items, output_items):
        power_data = input_items[0]
        rows = []
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
        for frame in power_data:
            frame = np.fft.fftshift(frame)
            peaks, props = find_peaks(frame, height=self.p_threshold)
            if len(peaks) > 0:
                widths = peak_widths(frame, peaks, rel_height=0.1)
                for i, idx in enumerate(peaks):
                    peak_power = props["peak_heights"][i]
                    peak_freq = self.freqs[idx]
                    width_bins = widths[0][i] if i < len(widths[0]) else 1
                    bandwidth_hz = width_bins * (self.samp_rate / self.fft_size)
                    rows.append([timestamp, idx, f"{peak_power:.2f}", f"{peak_freq/1e6:.2f}", f"{bandwidth_hz:.2f}", self.get_protocol(peak_freq)])
        if rows:
            with open(self.output_path, "a", newline="") as f:
                csv.writer(f).writerows(rows)
        return len(power_data)

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
        self.samp_rate = 5e6    # 5 MHz sample rate - good for GSM
        self.fft_size = 4096

        # Spectrum analyzer params
        self.rbw = 10e3         # Initial RBW
        self.vbw = 100e3        # Initial VBW
        self.k_total = 2.5      # Sweep time constant (adjustable)

        # Calculate initial sweep time using improved method
        self.sweep_time = self.calculate_sweep_time()
        self.update_rate = self.sweep_time

        # Default GSM downlink center
        self.center_freq = 947.5e6      # GSM-900 DL example
        self.rf_bw = 10e6               # USRP frontend BW

        # Band-power ROI for GSM channel
        self.band_target_freq = 947.5e6
        self.band_bandwidth = 200e3     # GSM channel bandwidth

        # PVT decimation for TDMA visibility
        self.pvt_decim = 100            # 5MHz/100 = 50kSa/s
        self.tdma_threshold_db = -40.0
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
        self.uhd_usrp_source_0.set_antenna("RX2", 0)
        self.uhd_usrp_source_0.set_bandwidth(self.rf_bw, 0)
        self.uhd_usrp_source_0.set_gain(70, 0)

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
        self.peak_detector = PeakDetector(self.fft_size, self.samp_rate, self.center_freq, p_threshold=-40)
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

        self.qtgui_vector_sink_f_0.set_y_axis(-140, 5)
        self.qtgui_vector_sink_f_0.enable_autoscale(False)
        self.qtgui_vector_sink_f_0.enable_grid(True)
        
        self.qtgui_freq_sink_x_0.set_update_time(self.update_rate)
        self.qtgui_freq_sink_x_0.set_y_axis(-140, 5)
        self.qtgui_freq_sink_x_0.enable_autoscale(False)
        self.qtgui_freq_sink_x_0.enable_grid(True)

        # Time-domain power (PVT) with proper normalization
        self.blocks_complex_to_mag_squared_td = blocks.complex_to_mag_squared(1)
        
        # Power normalization for dB scale
        # normalize to W/Hz to match FFT density
        pvt_scale = 1.0 / self.samp_rate
        self.pvt_normalize = blocks.multiply_const_ff(pvt_scale)

        
        # Minimal averaging to preserve TDMA bursts
        self.pvt_mavg_len = 5
        self.pvt_mavg = blocks.moving_average_ff(self.pvt_mavg_len, 1.0/self.pvt_mavg_len, 4000)
        
        # Convert to dB
        self.pvt_log10 = blocks.nlog10_ff(10, 1, 0)

        # Decimate for display
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

        # Wrap and place widgets
        self.top_layout.addWidget(sip.wrapinstance(self.qtgui_vector_sink_f_0.qwidget(), Qt.QWidget))
        self.top_layout.addWidget(sip.wrapinstance(self.qtgui_freq_sink_x_0.qwidget(), Qt.QWidget))
        self.top_layout.addWidget(sip.wrapinstance(self.pvt_sink.qwidget(), Qt.QWidget))
        self.top_layout.addWidget(sip.wrapinstance(self.tdma_sink.qwidget(), Qt.QWidget))
        self.top_layout.addWidget(sip.wrapinstance(self.band_time_sink.qwidget(), Qt.QWidget))

        # ----------------------------
        # Connections
        # ----------------------------
        # Spectrum: USRP -> vec -> FFT -> |.|^2 -> normalize -> dB -> RBW -> VBW -> displays
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

        # Live FFT view directly (complex)
        self.connect((self.uhd_usrp_source_0, 0), (self.qtgui_freq_sink_x_0, 0))

        # PVT chain with proper normalization and minimal averaging
        self.connect((self.uhd_usrp_source_0, 0), (self.blocks_complex_to_mag_squared_td, 0))
        self.connect((self.blocks_complex_to_mag_squared_td, 0), (self.pvt_normalize, 0))
        self.connect((self.pvt_normalize, 0), (self.pvt_mavg, 0))
        self.connect((self.pvt_mavg, 0), (self.pvt_log10, 0))
        self.connect((self.pvt_log10, 0), (self.keep_one_in_n, 0))
        self.connect((self.keep_one_in_n, 0), (self.pvt_sink, 0))

        # TDMA slicer (threshold on dB power, same decimated stream)
        self.connect((self.keep_one_in_n, 0), (self.tdma_slicer, 0))
        self.connect((self.tdma_slicer, 0), (self.tdma_sink, 0))

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
        self._fs_range = qtgui.Range(200e3, 35e6, 100e3, self.samp_rate, 200)
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
        self._k_range = qtgui.Range(0.5, 10.0, 0.1, self.k_total, 200)
        self._k_win = qtgui.RangeWidget(self._k_range, self.set_k_total, 'Sweep Time Factor (k)', "counter_slider", float)

        # PVT decimation
        self._pvtdec_range = qtgui.Range(1, 1000, 10, self.pvt_decim, 200)
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
        
        # Optional: Use adaptive k_total based on RBW
        # self.k_total = self.get_adaptive_k_total(self.rbw)
        
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
        d = max(1, int(d))
        self.pvt_decim = d
        self.keep_one_in_n.set_n(self.pvt_decim)
        
        # Update time sink sample rates
        decimated_fs = self.samp_rate / self.pvt_decim
        self.pvt_sink.set_samp_rate(decimated_fs)
        self.tdma_sink.set_samp_rate(decimated_fs)
        
        # Show expected samples per GSM slot
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
        
        # Update the RBW label
        self._rbw_label.setText(f"{self.rbw/1e3:.1f} kHz")
        
        # Recalculate sweep time
        self.sweep_time = self.calculate_sweep_time()
        self.update_rate = self.sweep_time
        self.update_all_display_rates()
        
        print(f"RBW fraction updated: {fraction:.4f} -> RBW = {new_rbw/1e3:.1f} kHz")
        print(f"New sweep time: {self.sweep_time*1000:.1f} ms")

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