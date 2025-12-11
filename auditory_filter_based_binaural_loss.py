import torch
from scipy.io import wavfile
import numpy as np
import scipy.special
import time
import torchaudio

import torch.nn.functional as F
from torchlpc import sample_wise_lpc

from scipy.signal import butter
import sys

torch.set_default_dtype(torch.float64)

MAX_WAV_VALUE = 32768.0


class BinauralLoss(torch.nn.Module):
    def __init__(self, device, fs=16000):
        """Initilize binaural loss module."""
        super(BinauralLoss, self).__init__()

        self.adjustment_factor_1 = 10 ** (8 / 20)
        self.adjustment_factor_2 = 10 ** (-15 / 20)

        self.GFB_L = 24.7  # see equation (17) in [Hohmann 2002]
        self.GFB_Q = 9.265  # see equation (17) in [Hohmann 2002]

        self.fs = fs
        self.device = device
        self.fsNew = 6000

        # Gammatone filterbank parameters
        fLow = 50
        fHigh = 6000
        fBase = 700
        filtersPerERB = 1
        self.gammaOrder = 3
        self.bandwidthFactor = 1

        # modulation filter parameters
        self.modGammaCenterFreqHz = 300
        self.modGammaBandwidthHz = 300
        self.modGammaAttenuationDB = 3
        self.modGammaOrder = 1

        # temporal resolution
        self.tauCycles = 5  # in cycles

        self.compressionPower = 0.4


        self.center_freqs_gammatone_full, _ = self.audspacebw(fLow, fHigh, 1 / filtersPerERB,
                                                               fBase)
        self.idxHighestFine = np.where(self.center_freqs_gammatone_full.cpu().numpy() < 1400)[0][-1]
        self.gammatone_filters_full = self.gfb_filters_new(self.center_freqs_gammatone_full, self.gammaOrder, self.bandwidthFactor)
        self.sos_gammatone_filters_full_1, self.sos_gammatone_filters_full_2 = self.create_filter_coeffs(self.gammatone_filters_full)


        bandwidth_hz = self.bandwidthFactor * torch.ones(self.idxHighestFine+1).to(self.device)
        self.gammatone_filters_fine = self.gfb_filters_new(self.center_freqs_gammatone_full[:self.idxHighestFine + 1], self.gammaOrder, bandwidth_hz)
        self.sos_gammatone_filters_fine_1, self.sos_gammatone_filters_fine_2 = self.create_filter_coeffs(self.gammatone_filters_fine)

        center_frequency_hz = self.modGammaCenterFreqHz * torch.ones(len(self.center_freqs_gammatone_full) - self.idxHighestFine - 1).to(
            self.device)
        bandwidth_hz = self.modGammaBandwidthHz * torch.ones(len(self.center_freqs_gammatone_full) - self.idxHighestFine - 1).to(self.device)
        self.gammatone_filters_mod = self.gfb_filters_new(center_frequency_hz, bandwidth_hz, self.modGammaAttenuationDB,
                                            self.modGammaOrder)
        self.sos_gammatone_filters_mod_1, self.sos_gammatone_filters_mod_2 = self.create_filter_coeffs(self.gammatone_filters_mod)


        vMiddleEarThr = [500, 2000]  # Bandpass freqencies for middle ear transfer
        middleEarOrder = 1
        sos = butter(middleEarOrder, vMiddleEarThr, btype='bandpass', fs=self.fs, output='sos')
        self.sos_middle_ear = torch.tensor(sos, dtype=torch.float64).to(self.device)

        # Hair cell low-pass filter parameters
        haircellLowOrder = 5
        haircellLowCutoffHz = 770
        sos = butter(haircellLowOrder, haircellLowCutoffHz, btype='lowpass', fs=self.fs, output='sos')
        self.sos_hair_cell_low = torch.tensor(sos, dtype=torch.float64).to(self.device)

        self.sos_remove_dc = torch.tensor([[1, -1, 1, 0]], dtype=torch.float64).to(self.device)     # modulation filter part 1, derivation to remove DC

        # Modulation lowpass filter #1
        fModLow1 = 500
        orderModLow1 = 2

        sos = butter(orderModLow1, fModLow1, btype='lowpass', fs=self.fs, output='sos')
        self.sos_mod_low1 = torch.tensor(sos, dtype=torch.float64).to(self.device)


        # Modulation lowpass filter #2
        fModLow2 = 600
        orderModLow2 = 2               # orderModLow2 = 6

        sos = butter(orderModLow2, fModLow2, btype='lowpass', fs=self.fs, output='sos')
        self.sos_mod_low2 = torch.tensor(sos, dtype=torch.float64).to(self.device)

        # Modulation ILD low pass
        fModLevelDiffLow = 30
        fModLevelDiffLowOrder = 2

        sos = butter(fModLevelDiffLowOrder, fModLevelDiffLow, btype='lowpass', fs=self.fsNew, output='sos')
        self.sos_mod_ild_low = torch.tensor(sos, dtype=torch.float64).to(self.device)

        self.resample = torchaudio.transforms.Resample(self.fs, self.fsNew, lowpass_filter_width=64, dtype=torch.float64).to(self.device)

        # For IVS calculation (Coherence measure)
        self.sos_coherence_fine = self.sos_coherence_calc()
        self.sos_coherence_mod = self.sos_coherence_calc(fine_structure=False)


    def forward(self, mRef, mTest):
        """Calculate forward propagation.
        Args:
            mRef (Tensor): predicted binaural signal (#time_bins, #channels=2, B).
            mTest (Tensor): groundtruth binaural signal (#time_bins, #channels=2, B).
        Returns:
            Tensor: spatial score loss value.
        """

        mRefTest = torch.cat((mRef[:, 0].unsqueeze(1), mTest[:, 0].unsqueeze(1), mRef[:, 1].unsqueeze(1), mTest[:, 1].unsqueeze(1)), dim=1)
        mRefTest *= self.adjustment_factor_1 * self.adjustment_factor_2
        # avoid NaNs in IVS calculation:
        mRefTest = torch.cat((torch.full((1, 4, mRefTest.shape[2]), 0.00001, dtype=torch.float64, device=self.device), mRefTest[:-1, :]),
                         dim=0)


        mIPD, mIVS, mILD = self.BAMQFrontEnd(mRefTest)

        IPDdiff = torch.abs(mIPD[:, :, 0] - mIPD[:, :, 1]).mean()
        ILDdiff = torch.abs(mILD[:, :, 0] - mILD[:, :, 1]).mean()

        # IC measure parameters
        k = {'r': 2, 's': 0.0012, 't': 666, 'a': 0.023}
        n = {'r': 3.17, 's': -0.0015, 't': 560, 'a': -2.75}

        # IVS sub-measure
        # -------------------------------------------------------------------------
        K = k['r'] / (1 + torch.exp(k['s'] * (self.center_freqs_gammatone_full.unsqueeze(0).unsqueeze(-1).unsqueeze(-1) - k['t']))) + k['a']
        N = n['r'] / (1 + torch.exp(n['s'] * (self.center_freqs_gammatone_full.unsqueeze(0).unsqueeze(-1).unsqueeze(-1) - n['t']))) + n['a']

        dIVS = torch.exp(K + N) - torch.exp(K * mIVS.permute(1, 0, 3, 2) + N)
        IVSdiff = torch.abs(dIVS[:, :, :, 0] - dIVS[:, :, :, 1]).mean()

        return IPDdiff, IVSdiff, ILDdiff


    def BAMQFrontEnd(self, insig):

        mSignalME = torchaudio.functional.lfilter(insig.permute(2, 1, 0), self.sos_middle_ear[0, 3:], self.sos_middle_ear[0, :3]).permute(2, 1, 0).unsqueeze(0)


        mSignalIE = self.apply_torchlpc_filter(self.sos_gammatone_filters_full_1, mSignalME.to(torch.float64), is_complex=False)
        for i in range(0, self.gammatone_filters_full['gamma_order'] - 1):
            mSignalIE = self.apply_torchlpc_filter(self.sos_gammatone_filters_full_2, mSignalIE, is_complex=True)

        del mSignalME

        # Half-Wave Rectification
        mSignalRect = torch.clamp(mSignalIE.real, min=1e-16)

        del mSignalIE
        torch.cuda.empty_cache()  # If using GPU

        # Compression
        mSignalHC = mSignalRect ** self.compressionPower

        del mSignalRect
        torch.cuda.empty_cache()  # If using GPU

        mSignalHCre = self.resample(mSignalHC.permute(0, 3, 2, 1).contiguous()).permute(0, 3, 2, 1).contiguous()

        for section in range(self.sos_hair_cell_low.shape[0]):
            mSignalHC = torchaudio.functional.lfilter(mSignalHC.permute(2, 3, 0, 1),
                             self.sos_hair_cell_low[section, 3:], self.sos_hair_cell_low[section, :3]).permute(2, 3, 0, 1)


        # Call the fine structure filter function
        mSignalFine = self.apply_torchlpc_filter(self.sos_gammatone_filters_fine_1, mSignalHC[:self.idxHighestFine + 1].to(torch.float64), is_complex=False)
        for i in range(0, self.gammatone_filters_fine['gamma_order'] - 1):
            mSignalFine = self.apply_torchlpc_filter(self.sos_gammatone_filters_fine_2, mSignalFine, is_complex=True)

        mSignalModNoDC = torchaudio.functional.lfilter(mSignalHC[self.idxHighestFine + 1:].permute(2, 3, 0, 1),
                                                       self.sos_remove_dc[0, 2:], self.sos_remove_dc[0, :2]).permute(2, 3, 0, 1)

        del mSignalHC
        torch.cuda.empty_cache()  # If using GPU

        mSignalFine_real_resampled = self.resample(mSignalFine.real.permute(0, 3, 2, 1).contiguous()).permute(0, 3, 2, 1).contiguous()
        mSignalFine_imag_resampled = self.resample(mSignalFine.imag.permute(0, 3, 2, 1).contiguous()).permute(0, 3, 2, 1).contiguous()
        s = torch.complex(mSignalFine_real_resampled, mSignalFine_imag_resampled)

        mITF = s[:, :, 2:] * torch.conj(s[:, :, :2])  # change

        mIVS1 = self.ivs_calc(mITF)
        mIPD1 = torch.angle(mITF + 1e-16 + 1e-16j).real.to(torch.float64)


        mSignalModLow = torchaudio.functional.lfilter(
            mSignalModNoDC.permute(2, 3, 0, 1), self.sos_mod_low1[0, 3:], self.sos_mod_low1[0, :3]).permute(2, 3, 0, 1)

        del mSignalModNoDC
        torch.cuda.empty_cache()  # If using GPU

        for section in range(self.sos_mod_low2.shape[0]):
            mSignalModLow = torchaudio.functional.lfilter(
                mSignalModLow.permute(2, 3, 0, 1), self.sos_mod_low2[section, 3:], self.sos_mod_low2[section, :3]).permute(2, 3, 0, 1)


        mSignalMod= self.apply_torchlpc_filter(self.sos_gammatone_filters_mod_1, mSignalModLow.to(torch.float64), is_complex=False)
        for i in range(0, self.gammatone_filters_mod['gamma_order'] - 1):
            mSignalMod = self.apply_torchlpc_filter(self.sos_gammatone_filters_mod_2, mSignalMod, is_complex=True)

        del mSignalModLow
        torch.cuda.empty_cache()  # If using GPU


        mSignalMod_real_resampled = self.resample(mSignalMod.real.permute(0, 3, 2, 1).contiguous()).permute(0, 3, 2,
                                                                                                   1).contiguous()
        mSignalMod_imag_resampled = self.resample(mSignalMod.imag.permute(0, 3, 2, 1).contiguous()).permute(0, 3, 2,
                                                                                                   1).contiguous()
        s = torch.complex(mSignalMod_real_resampled, mSignalMod_imag_resampled)

        mITF = s[:, :, 2:] * torch.conj(s[:, :, :2])  # change

        mIVS2 = self.ivs_calc(mITF, fine_structure=False)
        mIPD2 = torch.angle(mITF + 1e-16 + 1e-16j).real.to(torch.float32)

        del mSignalMod
        torch.cuda.empty_cache()  # If using GPU

        mIPD = torch.cat((mIPD1, mIPD2), dim=0)
        mIVS = torch.cat((mIVS1, mIVS2), dim=0)


        mSignalLP = torchaudio.functional.lfilter(
            mSignalHCre.permute(2, 3, 0, 1), self.sos_mod_ild_low[0, 3:], self.sos_mod_ild_low[0, :3]).permute(2, 3, 0, 1)

        mILD = 20 * torch.log10(
            torch.maximum(mSignalLP[:, :, 2:], torch.tensor(1e-4).to(self.device)) / torch.maximum(mSignalLP[:, :, :2],
                                                                                              torch.tensor(1e-4).to(
                                                                                                  self.device))) / 0.4
        return mIPD1, mIVS, mILD

    def create_filter_coeffs(self, filters):
        factor = filters['normalization_factor']
        zeros = torch.zeros((len(factor), 1), dtype=torch.complex128, device=self.device)
        ones = torch.ones((len(factor), 1), dtype=torch.complex128, device=self.device)
        coef = -filters['coefficient'].unsqueeze(1).to(self.device)

        sos1 = torch.cat((factor.unsqueeze(1).to(self.device), zeros, ones, coef), dim=1)
        sos2 = torch.cat((ones, zeros, ones, coef), dim=1)

        return sos1, sos2

    def sos_coherence_calc(self, fine_structure = True):
        if fine_structure:
            tau = self.tauCycles / self.center_freqs_gammatone_full[:self.idxHighestFine + 1]
        else:
            tau = self.tauCycles / (
                        self.modGammaCenterFreqHz + torch.zeros(len(self.center_freqs_gammatone_full) - self.idxHighestFine - 1).to(self.device))

        c_coherence = torch.exp(-1 / (self.fsNew * tau)).to(torch.float64)
        c_coherence_unsqueezed = c_coherence.unsqueeze(1)
        one_minus_c_coherence_unsqueezed = 1 - c_coherence_unsqueezed
        minus_c_coherence_unsqueezed = -c_coherence_unsqueezed
        zeros_tensor = torch.zeros((len(c_coherence), 1), dtype=torch.float64).to(self.device)
        ones_tensor = torch.ones((len(c_coherence), 1), dtype=torch.float64).to(self.device)

        sos_coherence = torch.cat((one_minus_c_coherence_unsqueezed, zeros_tensor, ones_tensor, minus_c_coherence_unsqueezed),
                            dim=1).to(self.device)
        return sos_coherence


    def ivs_calc(self, mITF, fine_structure = True):
        if fine_structure:
            sos_coherence = self.sos_coherence_fine
        else:
            sos_coherence = self.sos_coherence_mod

        abs_itf_col = torch.abs(mITF)
        real_filtered_itf = torchaudio.functional.lfilter(mITF.real.permute(2, 3, 0, 1), sos_coherence[:, 2:],
                                                          sos_coherence[:, :2]).permute(2, 3, 0, 1)  # change
        imag_filtered_itf = torchaudio.functional.lfilter(mITF.imag.permute(2, 3, 0, 1), sos_coherence[:, 2:],
                                                          sos_coherence[:, :2]).permute(2, 3, 0, 1)  # change
        filtered_itf = torch.complex(real_filtered_itf, imag_filtered_itf)
        filtered_abs_itf = torchaudio.functional.lfilter(abs_itf_col.permute(2, 3, 0, 1), sos_coherence[:, 2:],
                                                         sos_coherence[:, :2]).permute(2, 3, 0, 1)  # change

        return torch.abs(filtered_itf) / torch.abs(filtered_abs_itf)


    def audspacebw(self, flow, fhigh, *args):
        bw = args[0]
        hitme = args[1]

        aud_to_freq_scale = (1 / 0.00437)
        aud_to_freq_offset = 9.2645

        audlimits = np.sign([flow, fhigh, hitme]) * aud_to_freq_offset * np.log(
            1 + np.abs([flow, fhigh, hitme]) * 0.00437)

        audrangelow = audlimits[2] - audlimits[0]
        audrangehigh = audlimits[1] - audlimits[2]

        nlow = int(np.floor(audrangelow / bw))
        nhigh = int(np.floor(audrangehigh / bw))

        audpoints = np.arange(-nlow, nhigh + 1) * bw + audlimits[2]
        n = nlow + nhigh + 1

        y = aud_to_freq_scale * np.sign(audpoints) * (np.exp(np.abs(audpoints) / aud_to_freq_offset) - 1)
        y = torch.tensor(y, dtype=torch.float64).to(self.device)

        return y, n


    def apply_torchlpc_filter(self, sos, input_signal, is_complex=False):
        a = sos[:, 3:].unsqueeze(0).unsqueeze(0).expand(input_signal.shape[2] * input_signal.shape[3],
                                                                        input_signal.shape[1], -1, -1).to(torch.complex128)
        b = sos[:, :2].real.view(sos.shape[0], 1, -1)

        insig_real = input_signal.real.permute(3, 2, 0, 1).reshape(-1, input_signal.shape[0], input_signal.shape[1]).expand(-1,
                            a.shape[2], -1)  # Shape: (batch, num_filters, time_steps)

        filtered_real = F.conv1d(insig_real, b, groups=a.shape[2],
                                 padding=1)  # Shape: (batch, num_filters, time_steps)
        filtered_real = filtered_real[:, :, 1:].permute(1, 0, 2)  # Shape: (batch, time_steps, num_filters) [num_filters, batch_size, seq_len]

        if is_complex:
            insig_imag = input_signal.imag.permute(3, 2, 0, 1).reshape(-1, input_signal.shape[0], input_signal.shape[1]).expand(-1,
                                            a.shape[2], -1)  # Shape: (batch, num_filters, time_steps)
            filtered_imag = F.conv1d(insig_imag[:, :, :-1], b, groups=a.shape[2],
                                     padding=b.shape[2] - 1)  # Shape: (batch, num_filters, time_steps)
            filtered_imag = filtered_imag.permute(1, 0, 2)  # Shape: (batch, time_steps, num_filters) [num_filters, batch_size, seq_len]
            filtered_signal = filtered_real + 1j * filtered_imag
        else:
            filtered_signal = filtered_real.to(torch.complex128)


        filtered_signal = filtered_signal.reshape(-1, filtered_signal.size(-1))
        a = a.permute(2, 0, 1, 3).reshape(-1, a.size(-3), a.size(-1))

        output_signal = sample_wise_lpc(filtered_signal, a)
        output_signal = output_signal.view(b.shape[0], -1, output_signal.size(-1))  # [num_filters, batch_size, seq_len]
        output_signal = output_signal.permute(1, 2, 0)  # [batch_size, seq_len, num_filters]
        output_signal = output_signal.reshape(input_signal.shape[3], input_signal.shape[2], output_signal.shape[1], output_signal.shape[2]
                                      ).permute(3, 2, 1, 0)

        return output_signal

    def gfb_filters_new(self, *args):
        filter = {}
        if len(args) == 3:
            sampling_rate_hz = self.fs
            center_frequency_hz = args[0]
            filter['gamma_order'] = args[1]
            bandwidth_factor = args[2]

            # Equation (13) [Hohmann 2002]
            audiological_erb = (self.GFB_L + center_frequency_hz / self.GFB_Q) * bandwidth_factor

            # Equation (14), line 3 [Hohmann 2002]
            a_gamma = (np.pi * scipy.special.factorial(2 * filter['gamma_order'] - 2) *
                       2 ** -(2 * filter['gamma_order'] - 2) /
                       scipy.special.factorial(filter['gamma_order'] - 1) ** 2)

            # Equation (14), line 2 [Hohmann 2002]
            b = audiological_erb / a_gamma

            lambda_ = torch.exp(-2 * torch.pi * b / sampling_rate_hz)
            # Equation (10) [Hohmann 2002]
            beta = 2 * torch.pi * center_frequency_hz / sampling_rate_hz
            # Equation (1), line 2 [Hohmann 2002]
            filter['coefficient'] = lambda_ * torch.exp(1j * beta)

        elif len(args) == 4:
            sampling_rate_hz = self.fs
            center_frequency_hz = args[0]
            bandwidth_hz = args[1]
            attenuation_db = args[2]
            filter['gamma_order'] = args[3]

            # Equation (12), line 4 [Hohmann 2002]
            phi = torch.pi * bandwidth_hz / sampling_rate_hz
            # Equation (12), line 3 [Hohmann 2002]
            u = -attenuation_db / filter['gamma_order']
            # Equation (12), line 2 [Hohmann 2002]
            p = (-2 + 2 * 10 ** (u / 10) * torch.cos(phi)) / (1 - 10 ** (u / 10))
            # Equation (12), line 1 [Hohmann 2002]
            lambda_ = -p / 2 - torch.sqrt(p * p / 4 - 1)
            # Equation (10) [Hohmann 2002]
            beta = 2 * torch.pi * center_frequency_hz / sampling_rate_hz
            # Equation (1), line 2 [Hohmann 2002]
            filter['coefficient'] = lambda_ * torch.exp(1j * beta)

        # Normalization factor from section 2.2 (text) [Hohmann 2002]
        filter['normalization_factor'] = 2 * (1 - abs(filter['coefficient'])) ** filter['gamma_order']

        return filter



if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # dor added
    binaural_loss = BinauralLoss(device=device)
    start_time = time.time()
    print(device)

    # Read reference signal (clean)
    fsRef, RefSig_np = wavfile.read('bsm_magls_array_rot_90.wav')
    RefSig = torch.tensor(RefSig_np, dtype=torch.float64).to(device)
    RefSig = RefSig.unsqueeze(2).expand(-1, -1, 16)
    RefSig = RefSig / MAX_WAV_VALUE

    # Read test signal (processed)
    fsTest, TestSig_np = wavfile.read('bsm_magls_head_rot_90.wav')
    TestSig = torch.tensor(TestSig_np, dtype=torch.float64).to(device)
    TestSig = TestSig.unsqueeze(2).expand(-1, -1, 16)
    TestSig = TestSig / MAX_WAV_VALUE

    # Compare sampling frequencies
    if fsTest != fsRef:
        raise ValueError('Signals have different sampling frequencies')
    else:
        fs = fsTest


    IPDdiff, IVSdiff, ILDdiff = binaural_loss(RefSig, TestSig)

    print('***********************')
    print('***binaural measures***')
    print('***********************')
    print('ILDdiff: {:4.3f}'.format(ILDdiff.item()))
    print('IPDdiff: {:4.7f}'.format(IPDdiff.item()))
    print('IVSdiff: {:4.4f}'.format(IVSdiff.item()))

    end_time = time.time()

    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} seconds")
