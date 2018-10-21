function [Power, freq, time ] = lofar(data, fs, n_pts_fft, num_overlap, decimation_rate, spectrum_bins_left)
%LOFAR Function that performs Lofar Analysis
%   [Power, freq, time ] = lofar(data, n_pts_fft, num_overlap, decimation_rate, spectrum_bins_left, show_plot)
%   
%   Output arguments: 
%       Power:
%       freq:
%       time:
%
%   Input arguments:
%       data:
%       fs:
%       n_pts_fft:
%       decimation_rate:
%       spectrum_bins_left:
%       show_plot:

if nargin<2, help lofar; return; end
if nargin<3, n_pts_fft = 1024; num_overlap = 0; decimation_rate = 3; spectrum_bins_left=400; end
if nargin<4 num_overlap = 0; decimation_rate = 3; spectrum_bins_left=400; end
if nargin<5, decimation_rate = 3; spectrum_bins_left=400; end
if nargin<6, spectrum_bins_left=400; end

% Default LOFAR Parameters
%num_overlap = 0;

norm_parameters.lat_window_size = 10;
norm_parameters.lat_gap_size = 1;
norm_parameters.threshold = 1.3;

if decimation_rate >=1
    dec_data = decimate(data,decimation_rate,10,'FIR');
    Fs=fs/decimation_rate;
else
    dec_data = data;
    Fs=fs;
end

[Power, freq, time]=spectrogram(data, hanning(n_pts_fft), num_overlap, n_pts_fft, Fs);
Power = abs(Power);
Power=Power./tpsw(Power);
Power=log10(Power);
Power(Power<-.2)=0;

Power = Power(1:spectrum_bins_left,:); % William
freq  = freq(1:spectrum_bins_left);

end

