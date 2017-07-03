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
%outputpath = getenv('OUTPUTDATAPATH');

inputpath  = '/Users/natmourajr/Workspace/Doutorado/Data/SONAR/Classification';
outputpath = '/Users/natmourajr/Workspace/Doutorado/SonarAnalysis/Results';

% selected version of data
database = 'DemonSimulatedData';

fprintf('Performing BSS on Time Analysis in %s database\n',database);

if(~exist(sprintf('%s/%s/data_info_file.mat',outputpath,database),'file'))
    fprintf('File Struct %s/%s not found, perform ReadRawData.m',outputpath,database);
    return;
end


dir_info = dir(sprintf('%s/%s',outputpath,database));
load(sprintf('%s/%s/data_info_file.mat',outputpath,database))

data_info.demon_data_date = datetime('now','TimeZone','local','Format','dd-MMM-yyyy HH:mm:ss');

data_info.n_pts_fft = 1024;
data_info.time_between_show = 0.5;
data_info.decimation_rate1 = 25;
data_info.decimation_rate2 = 25;

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
    demon_data = [];
    for irun = 1:data_info.n_runs{aux_class}
        % problemas de conversao de dados (2 canais?!)
        if size(raw_data{irun},2)>1
            raw_data{irun} = raw_data{irun}(:,1);
        end
        [power, freq, time] = demon(raw_data{irun}', data_info.fs{aux_class}(irun), ...
            data_info.time_between_show, data_info.n_pts_fft, data_info.decimation_rate1, data_info.decimation_rate2);
        demon_data{irun} = power;
    end
    save(sprintf('%s/%s/%s/demon_data_file_fft_%i_decimation1_%i_decimation2_%i.mat', ...
        outputpath,database,dir_info(iclass).name,data_info.n_pts_fft, ...
        data_info.decimation_rate1,data_info.decimation_rate2),'demon_data');
    aux_class = aux_class+1;
end

rmpath('../Functions/SonarFunctions');
fprintf('Finishing %s.m\n',mfilename('fullpath'));

