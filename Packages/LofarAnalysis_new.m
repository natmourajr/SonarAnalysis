% Projeto Marinha do Brasil

% Autor: Natanael Junior (natmourajr@gmail.com)
% Laboratorio de Processamento de Sinais - UFRJ
% Laboratorio de Tecnologia Sonar - UFRJ/Marinha do Brasil

% iniciando o script
clear all;
close all;
clc;

fprintf('Starting %s.m\n',mfilename('fullpath'));


fprintf('\n\n')
fprintf('\t\t Projeto Marinha do Brasil\n');
fprintf('\t Laboratório de Processamento de Sinais (LPS)\n');
fprintf('\t   Laboratório de Tecnologia Sonar (LabSONAR)\n');
fprintf('\nAutor: Natanael Junior (natmourajr@gmail.com)\n');

% System var. point to external folders
outputpath = getenv('OUTPUTDATAPATH');

% selected version of data
subfolder = '8classes';

% check raw data existence
if(exist(sprintf('%s/%s/raw_data_info.mat',outputpath,subfolder),'file'))
	fprintf('File %s/%s/raw_data_info.mat exists\n',outputpath,subfolder);
	load(sprintf('%s/%s/raw_data_info.mat',outputpath,subfolder));
	
else
	fprintf('File %s/%s/raw_data_info.mat does not exist, please create it\n',outputpath,subfolder);
	clear all;
	close all;
	clc;
	return;
end

% add functions
addpath(genpath('../Functions/SonarFunctions'));


% get user information
user_loop = true;

while(user_loop)
    n_pts_fft = 0;
    n_pts_fft = input(sprintf('Number of FFT Points [default: 1024]: '));
    
    if isempty(n_pts_fft)
        n_pts_fft = 1024;
    end
    
    decimation_rate = 0;
    decimation_rate = input(sprintf('Decimation Ratio [default: 3]: '));
    
    if isempty(decimation_rate)
        decimation_rate = 3;
    end
    
    spectrum_bins_left = 0;
    spectrum_bins_left = input(sprintf('Spectrum Bins left for Analysis [default: 400]: '));
    
    if isempty(spectrum_bins_left)
        spectrum_bins_left = 400;
    end
    
    show_plot = false;
    answer = input(sprintf('Show Plots? [Y,n] '),'s');
    
    if strcmp(answer,'Y')
        show_plot = true;
    end
    
    stat_anal = false;
    answer = input(sprintf('Statisctical Analysis? [Y,n] '),'s');
    
    if strcmp(answer,'Y')
        stat_anal = true;
    end
    
    
    % show user params
    fprintf('\n\n');
    fprintf('Number of FFT points %i\n',n_pts_fft);
    fprintf('Decimation Rate %i\n',decimation_rate);
    fprintf('Spectrum Bins %i\n',spectrum_bins_left);
    show_str = '';
    
    if show_plot
        show_str = 'true';
    else
        show_str = 'false';
    end
    
    fprintf('Show Plots: %s\n',show_str);
    
    stat_str = '';
    
    if stat_anal
        stat_str = 'true';
    else
        stat_str = 'false';
    end
    
    fprintf('Statistical Analysis: %s\n\n',stat_str);
    
    answer = input(sprintf('The User params are correct? [Y,n] '),'s');
    
    if strcmp(answer,'Y')
        user_loop = false;
    end
end

% Default LOFAR Parameters
num_overlap = 0;

norm_parameters.lat_window_size = 10;
norm_parameters.lat_gap_size = 1;
norm_parameters.threshold = 1.3;


lofar_info = [];
lofar_info.n_pts_fft = n_pts_fft;
lofar_info.decimation_rate = decimation_rate;
lofar_info.spectrum_bins_left = spectrum_bins_left;
lofar_info.show_plot = show_plot;
lofar_info.stat_anal = stat_anal;
lofar_info.num_overlap = num_overlap;
lofar_info.norm_parameters = norm_parameters;

% loop over all class
for iclass = 1:numel(class_labels) % All Classes
	fprintf('Processing %s\n',class_labels{iclass});
	
	data_lofar = [];
	
	aux = [];
	
	load(sprintf('%s/%s/raw_data_class%i.mat',outputpath,subfolder,iclass));
	
	% loop over all runs
    for irun = 1:length(sonar_data.run)
        fprintf('Processing %s - run %i\n',class_labels{iclass},irun);
        
        if decimation_rate >=1
            data_lofar.run{irun} = decimate(sonar_data.run{irun},decimation_rate,10,'FIR');
            Fs=fs/decimation_rate;
        else
            data_lofar.run{irun} = sonar_data.run{irun};
            Fs=fs;
        end
        
        [intensity,f,t]=spectrogram(data_lofar.run{irun},hanning(n_pts_fft),num_overlap,n_pts_fft,Fs);
        intensity = abs(intensity);
        intensity=intensity./tpsw(intensity);
        intensity=log10(intensity);
        intensity(intensity<-.2)=0;
        
        data_lofar.run{irun} = intensity(1:spectrum_bins_left,:); % William
        
        
        % Stationarity Analysis
        
        
        buffer = sonar_data.run{irun};
        windownazed_data = zeros(floor(length(buffer)/n_pts_fft),n_pts_fft);
        
        for id = 1:floor(length(buffer)/n_pts_fft)
            windownazed_data(id,:) = buffer(((id-1)*n_pts_fft)+1:id*n_pts_fft,:);
        end
        
        data_lofar.windonazed_run{irun} = windownazed_data;
        aux = [aux; windownazed_data];
        
        
        data_lofar.adftest{irun} = [];
        data_lofar.adftest{irun}.result = zeros(size(windownazed_data,1),1);
        data_lofar.adftest{irun}.pValue = zeros(size(windownazed_data,1),1);
        
        data_lofar.pptest{irun} = [];
        data_lofar.pptest{irun}.result = zeros(size(windownazed_data,1),1);
        data_lofar.pptest{irun}.pValue = zeros(size(windownazed_data,1),1);
        
        data_lofar.kpsstest{irun} = [];
        data_lofar.kpsstest{irun}.result = zeros(size(windownazed_data,1),1);
        data_lofar.kpsstest{irun}.pValue = zeros(size(windownazed_data,1),1);
        
        if stat_anal
            warning OFF
            for id = 1:size(windownazed_data,1)
                [data_lofar.(class_labels{iclass}).adftest{irun}.result(id,:),data_lofar.adftest{irun}.pValue(id,:)] = adftest(windownazed_data(id,:),'model','AR');
                [data_lofar.(class_labels{iclass}).pptest{irun}.result(id,:),data_lofar.pptest{irun}.pValue(id,:)] = pptest(windownazed_data(id,:),'model','AR');
                [data_lofar.(class_labels{iclass}).kpsstest{irun}.result(id,:),data_lofar.kpsstest{irun}.pValue(id,:)] = kpsstest(windownazed_data(id,:));
            end
            warning ON
        end 
    end
    
    data_lofar.windownazed_data = aux;
    
	save(sprintf('%s/%s/lofar_data_class%i_%i_fft_pts_%i_decimation_rate.mat',outputpath,subfolder,iclass,n_pts_fft,decimation_rate),'data_lofar','lofar_info','Fs');
end


fprintf('Finishing %s.m\n',mfilename('fullpath'));
