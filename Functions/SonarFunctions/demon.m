function [Power, freq, time] = demon(data, fs, time_between_show, n_pts_fft, decimation_rate1, decimation_rate2)
%DEMON -  Demon analysis in Matlab
%   [Power, freq, time ] = demon(data, fs, n_pts_fft, decimation_rate1, decimation_rate2)
%   
%   Output arguments: 
%       Power:
%       freq:
%       time:
%
%   Input arguments:
%       data:
%       fs:
%       time_between_show:
%       n_pts_fft:
%       decimation_rate1: 
%       decimation_rate2: 
%       show_plot:


if nargin<2, help lofar; return; end
if nargin<3, time_between_show = 0.5;  n_pts_fft = 1024; decimation_rate1 = 25; decimation_rate2 = 25; end
if nargin<4, n_pts_fft = 1024; decimation_rate1 = 25; decimation_rate2 = 25; end
if nargin<5, decimation_rate1 = 25; decimation_rate2 = 25; end
if nargin<6, decimation_rate2 = 25;end

% AM demodulation
data_demod = abs(data);

% first decimation
if decimation_rate1 >=1
    data_demod_d1 = decimate(data_demod,decimation_rate1);
    fs_d1=fs/decimation_rate1;
else
    data_demod_d1 = data_demod;
    fs_d1=fs;
end

% second decimation
if decimation_rate2 >=1
    data_demod_d2 = decimate(data_demod_d1,decimation_rate2);
    fs_d2=fs_d1/decimation_rate2;
else
    data_demod_d2 = data_demod_d1;
    fs_d2=fs_d1;
end

% Nyquist
Fmax = fs_d2/2;

% Calculo dos espectros DEMON, com overlap novr3
novr=floor(n_pts_fft-2*Fmax*time_between_show);	% Calcula overlap para calculo da FFT

[Power, freq, time] = spectrogram(data_demod_d2-mean(data_demod_d2),hanning(n_pts_fft),novr,n_pts_fft,fs_d2);

Power_abs=abs(Power);
ind=(1:8);Power_abs(ind,:)=repmat(Power_abs(length(ind),:),[length(ind) 1]); % Descarta 8 primeiros bins ???
Power_abs=Power_abs./tpsw(Power_abs); % Normaliza usando TPSW

Power = Power_abs;

end

