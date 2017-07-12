% Projeto Marinha do Brasil

% Autor: Natanael Junior (natmourajr@gmail.com)
% Laboratorio de Processamento de Sinais - UFRJ
% Laboratorio de Tecnologia Sonar - UFRJ/Marinha do Brasil

% iniciando o script
clear all;
close all;
clc;

addpath('../Functions/SonarFunctions');

fprintf('Starting %s.m\n',mfilename('fullpath'));

% System var. point to external folders
outputpath = getenv('OUTPUTDATAPATH');
inputpath  = getenv('INPUTDATAPATH');

%inputpath  = '/Users/natmourajr/Workspace/Doutorado/Data/SONAR/Classification';
%outputpath = '/Users/natmourajr/Workspace/Doutorado/SonarAnalysis/Results';

% selected version of data
database = '4classes';

fprintf('Performing Lofar Analysis in %s database\n',database);

if(~exist(sprintf('%s/%s/data_info_file.mat',outputpath,database),'file'))
    fprintf('File Struct %s/%s not found, perform ReadRawData.m',outputpath,database);
    return;
end

dir_info = dir(sprintf('%s/%s',outputpath,database));
load(sprintf('%s/%s/data_info_file.mat',outputpath,database))

m_matlab_version = version('-release');
matlab_year = str2num(m_matlab_version(1:end-1));
if(matlab_year < 2016)
    data_info.lofar_data_date = datetime('now','TimeZone','local','Format','dd-MMM-yyyy HH:mm:ss');
else
    data_info.lofar_data_date = datestr(now,'dd-mm-yyyy HH:MM:SS');
end

% get user information
user_loop = true;

while(user_loop)
    data_info.n_pts_fft = 0;
    data_info.n_pts_fft = input(sprintf('Number of FFT Points [default: 1024]: '));

    if isempty(data_info.n_pts_fft)
        data_info.n_pts_fft = 1024;
    end

    data_info.decimation_rate = 0;
    data_info.decimation_rate = input(sprintf('Decimation Ratio [default: 3]: '));

    if isempty(data_info.decimation_rate)
        data_info.decimation_rate = 3;
    end

    data_info.spectrum_bins_left = 0;
    data_info.spectrum_bins_left = input(sprintf('Spectrum Bins left for Analysis [default: 400]: '));

    if isempty(data_info.spectrum_bins_left)
        data_info.spectrum_bins_left = 400;
    end

    % show user params
    fprintf('\n\n');
    fprintf('Number of FFT points: %i\n',data_info.n_pts_fft);
    fprintf('Decimation Rate: %i\n',data_info.decimation_rate);
    fprintf('Spectrum Bins: %i\n',data_info.spectrum_bins_left);
    
    answer = input(sprintf('Are User params correct? [Y,n] '),'s');

    if strcmp(answer,'Y')
        user_loop = false;
    end

end


%data_info.n_pts_fft = 1024;
%data_info.decimation_rate = 3;
%data_info.spectrum_bins_left = 400;

aux_class = 1;

for iclass = 1:length(dir_info)
    % for mac
    if ~isempty(strfind(dir_info(iclass).name,'.DS_Store'))
        %fprintf('Wav File %s\n',dir_info(iclass).name);
        continue
    end
    % for info file
    if ~isempty(strfind(dir_info(iclass).name,'.mat'))
        %fprintf('Wav File %s\n',dir_info(iclass).name);
        continue
    end
    if strcmp(dir_info(iclass).name,'.')==1 
        continue
    end
    if strcmp(dir_info(iclass).name,'..')==1
        continue
    end
    
    fprintf('Reading %s\n',dir_info(iclass).name);
    load(sprintf('%s/%s/%s/raw_data_file.mat',outputpath,database,dir_info(iclass).name));
    lofar_data = [];
    for irun = 1:data_info.n_runs{aux_class}
        % problemas de conversao de dados (2 canais?!)
        if size(raw_data{irun},2)>1
            raw_data{irun} = raw_data{irun}(:,1);
        end
        [power, freq, time] = lofar(raw_data{irun}', data_info.fs{aux_class}(irun), ...
            data_info.n_pts_fft, data_info.decimation_rate, data_info.spectrum_bins_left);
        lofar_data{irun} = power;
    end
    save(sprintf('%s/%s/%s/lofar_data_file_fft_%i_decimation_%i_spectrum_left_%i.mat', ...
        outputpath,database,dir_info(iclass).name,data_info.n_pts_fft, ...
        data_info.decimation_rate,data_info.spectrum_bins_left),'lofar_data');
    aux_class = aux_class+1;
end

save(sprintf('%s/%s/data_info_file.mat',outputpath,database),'data_info');

rmpath('../Functions/SonarFunctions');
% % selected version of data
% subfolder = '4classes_old';
%
% % if raw_data has created...
% if(~exist(sprintf('%s/RawData_%s.mat',outputpath,subfolder),'file'))
%     fprintf('File OUTPUTPATH/RawData_%s.mat not found, PERFORM ReadRawData.m\n',subfolder);
%     clear all;
%     close all;
%     %clc;
%     return;
% else
%     load(sprintf('%s/RawData_%s.mat',outputpath,subfolder));
% end
%
% % add functions
% addpath(genpath('../Functions/SonarFunctions'));
%
% % get user information
% user_loop = true;
%
% while(user_loop)
%     n_pts_fft = 0;
%     n_pts_fft = input(sprintf('Number of FFT Points [default: 1024]: '));
%
%     if isempty(n_pts_fft)
%         n_pts_fft = 1024;
%     end
%
%     decimation_rate = 0;
%     decimation_rate = input(sprintf('Decimation Ratio [default: 3]: '));
%
%     if isempty(decimation_rate)
%         decimation_rate = 3;
%     end
%
%     spectrum_bins_left = 0;
%     spectrum_bins_left = input(sprintf('Spectrum Bins left for Analysis [default: 400]: '));
%
%     if isempty(spectrum_bins_left)
%         spectrum_bins_left = 400;
%     end
%
%     show_plot = false;
%     answer = input(sprintf('Show Plots? [Y,n] '),'s');
%
%     if strcmp(answer,'Y')
%         show_plot = true;
%     end
%
%     stat_anal = false;
%     answer = input(sprintf('Statisctical Analysis? [Y,n] '),'s');
%
%     if strcmp(answer,'Y')
%         stat_anal = true;
%     end
%
%
%     % show user params
%     fprintf('\n\n');
%     fprintf('Number of FFT points %i\n',n_pts_fft);
%     fprintf('Decimation Rate %i\n',decimation_rate);
%     fprintf('Spectrum Bins %i\n',spectrum_bins_left);
%     show_str = '';
%
%     if show_plot
%         show_str = 'true';
%     else
%         show_str = 'false';
%     end
%
%     fprintf('Show Plots: %s\n',show_str);
%
%     stat_str = '';
%
%     if stat_anal
%         stat_str = 'true';
%     else
%         stat_str = 'false';
%     end
%
%     fprintf('Statistical Analysis: %s\n\n',stat_str);
%
%     answer = input(sprintf('The User params are correct? [Y,n] '),'s');
%
%     if strcmp(answer,'Y')
%         user_loop = false;
%     end
%
% end
%
% % if raw_data has created...
% if(exist(sprintf('%s/LofarData_%s_%i_fft_pts_%i_decimation_rate.mat',outputpath,subfolder,n_pts_fft,decimation_rate),'file'))
%     answer = input(sprintf('File %s.mat exists, perform ReadRawData.m? [Y,n]',sprintf('%s/LofarData_%s_%i_fft_pts_%i_decimation_rate.mat',outputpath,subfolder,n_pts_fft,decimation_rate)),'s');
%     if strcmp(answer,'Y')
%     else if strcmp(answer,'n')
%             clear all;
%             close all;
%             clc;
%             return;
%         end
%     end
% end
%
%
% % Default LOFAR Parameters
% num_overlap = 0;
%
% norm_parameters.lat_window_size = 10;
% norm_parameters.lat_gap_size = 1;
% norm_parameters.threshold = 1.3;
%
% fprintf('\nLOFAR Computing\n');
% data_lofar = [];
% all_time = [];
% stat_test_results = {};
%
% % loop over all class
% for iclass = 1:numel(class_labels) % All Classes
%     fprintf('%s - All Runs in RawData_%s file\n',class_labels{iclass},subfolder);
%
%     aux = [];
%     all_data = [];
%
%     % loop over all runs
%     for irun = 1:length(sonar_data.(class_labels{iclass}).run)
%         fprintf('Processing %s - run %i\n',class_labels{iclass},irun);
%
%         if decimation_rate >=1
%             data_lofar.(class_labels{iclass}).run{irun} = decimate(sonar_data.(class_labels{iclass}).run{irun},decimation_rate,10,'FIR');
%             Fs=fs/decimation_rate;
%         else
%             data_lofar.(class_labels{iclass}).run{irun} = sonar_data.(class_labels{iclass}).run{irun};
%             Fs=fs;
%         end
%
%         [intensity,f,t]=spectrogram(data_lofar.(class_labels{iclass}).run{irun},hanning(n_pts_fft),num_overlap,n_pts_fft,Fs);
%         intensity = abs(intensity);
%         intensity=intensity./tpsw(intensity);
%         intensity=log10(intensity);
%         intensity(intensity<-.2)=0;
%         data_lofar.(class_labels{iclass}).run{irun} = intensity(1:spectrum_bins_left,:); % William
%
%         if size(all_data,1)==0
%             all_data = intensity(1:spectrum_bins_left,:);
%             all_time = t;
%         else
%             all_data = [all_data intensity(1:spectrum_bins_left,:)];
%             all_time = [all_time t];
%         end
%
%         if show_plot
%             h = figure('visible','on','Position', [100, 100, 1049, 895]);
%
%             imagesc(f(1:spectrum_bins_left),t,data_lofar.(class_labels{iclass}).run{irun}');
%
%             if decimation_rate >=1
%                 title(sprintf('LOFARgram for %s (Run %i) with Decimation Ratio: %d and %d FFT Points',class_labels{iclass},irun,decimation_rate,n_pts_fft),'FontSize', 15,'FontWeight', 'bold');
%             else
%                 title(sprintf('LOFARgram for %s (Run %i) with %d FFT Points',class_labels{iclass},irun,n_pts_fft),'FontSize', 15,'FontWeight', 'bold');
%             end
%
%             ylabel('Time (seconds)','FontSize', 15,'FontWeight', 'bold');
%             xlabel('Frequency (Hz)','FontSize', 15,'FontWeight', 'bold');
%             colorbar;
%             saveas(h,sprintf('%s/lofargram_%s_%s_run_%i_fftpoints_%i_decimation_%i.png',outputpath,subfolder,class_labels{iclass},irun,n_pts_fft,decimation_rate));
%             close(h);
%
%         end
%
%         % Stationarity Analysis
%         buffer = sonar_data.(class_labels{iclass}).run{irun};
%         windownazed_data = zeros(floor(length(buffer)/n_pts_fft),n_pts_fft);
%
%         for id = 1:floor(length(buffer)/n_pts_fft)
%             windownazed_data(id,:) = buffer(((id-1)*n_pts_fft)+1:id*n_pts_fft,:);
%         end
%
%         data_lofar.(class_labels{iclass}).windonazed_run{irun} = windownazed_data;
%
%         data_lofar.(class_labels{iclass}).adftest{irun} = [];
%         data_lofar.(class_labels{iclass}).adftest{irun}.result = zeros(size(windownazed_data,1),1);
%         data_lofar.(class_labels{iclass}).adftest{irun}.pValue = zeros(size(windownazed_data,1),1);
%
%         data_lofar.(class_labels{iclass}).pptest{irun} = [];
%         data_lofar.(class_labels{iclass}).pptest{irun}.result = zeros(size(windownazed_data,1),1);
%         data_lofar.(class_labels{iclass}).pptest{irun}.pValue = zeros(size(windownazed_data,1),1);
%
%         data_lofar.(class_labels{iclass}).kpsstest{irun} = [];
%         data_lofar.(class_labels{iclass}).kpsstest{irun}.result = zeros(size(windownazed_data,1),1);
%         data_lofar.(class_labels{iclass}).kpsstest{irun}.pValue = zeros(size(windownazed_data,1),1);
%
%
%         if stat_anal
%             warning OFF
%             for id = 1:size(windownazed_data,1)
%                 [data_lofar.(class_labels{iclass}).adftest{irun}.result(id,:),data_lofar.(class_labels{iclass}).adftest{irun}.pValue(id,:)] = adftest(windownazed_data(id,:),'model','AR');
%                 [data_lofar.(class_labels{iclass}).pptest{irun}.result(id,:),data_lofar.(class_labels{iclass}).pptest{irun}.pValue(id,:)] = pptest(windownazed_data(id,:),'model','AR');
%                 [data_lofar.(class_labels{iclass}).kpsstest{irun}.result(id,:),data_lofar.(class_labels{iclass}).kpsstest{irun}.pValue(id,:)] = kpsstest(windownazed_data(id,:));
%             end
%             warning ON
%         end
%         aux = [aux; windownazed_data];
%     end
%     data_lofar.(class_labels{iclass}).windownazed_data = aux;
%
%     if show_plot
%         h = figure('visible','on', 'Position', [100, 100, 1049, 895]);
%
%         imagesc(f(1:spectrum_bins_left),all_time,all_data');
%
%         if decimation_rate >=1
%             title(sprintf('LOFARgram for %s, Decimation Ratio: %d, %d FFT Points',class_labels{iclass},decimation_rate,n_pts_fft),'FontSize', 15,'FontWeight', 'bold');
%         else
%             title(sprintf('LOFARgram for %s, %d FFT Points',class_labels{iclass},n_pts_fft),'FontSize', 15,'FontWeight', 'bold');
%         end
%
%         ylabel('Time (seconds)','FontSize', 15,'FontWeight', 'bold');
%         xlabel('Frequency (Hz)','FontSize', 15,'FontWeight', 'bold');
%         colorbar;
%         saveas(h,sprintf('%s/lofargram_%s_%s_fftpoints_%i_decimation_%i.png',outputpath,subfolder,class_labels{iclass},n_pts_fft,decimation_rate));
%         close(h);
%
%     end
%
% end
%
%
%
% fprintf('\nCreating LOFAR Data File\n');
% save(sprintf('%s/LofarData_%s_%i_fft_pts_%i_decimation_rate.mat',outputpath,subfolder,n_pts_fft,decimation_rate),'decimation_rate','Fs','num_overlap','norm_parameters','data_lofar','n_pts_fft');

fprintf('Finishing %s.m\n',mfilename('fullpath'));
