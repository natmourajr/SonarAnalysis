inputpath = getenv('INPUTDATAPATH');

database = '4classes';
sprintf('%s/%s/ClassA/navio10.mat',inputpath, database);
[run, fs] = audioread(sprintf('%s/%s/ClassA/navio10.wav',inputpath, database));

num_overlap = 0;
n_pts_fft = 1024;
[power, f, time] = spectrogram(run, hanning(n_pts_fft), num_overlap, n_pts_fft, fs);
power = abs(power);
% power = power ./ tpsw(power);
% power=log10(power);
% power(power<-.2)=0;
save('./lofar_navio10_no_tpsw.mat', 'power')
