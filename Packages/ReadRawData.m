% Projeto Marinha do Brasil

%Autor: Natanael Junior (natmourajr@gmail.com)
% Laboratorio de Processamento de Sinais - UFRJ

% iniciando o script
clear all;
close all;
clc;

fprintf('Starting %s.m\n',mfilename('fullpath'));

% System var. point to external folders
inputpath = getenv('INPUTDATAPATH');
outputpath = getenv('OUTPUTDATAPATH');


% if raw_data has created...
if(exist(sprintf('%s/RawData.mat',outputpath),'file'))
    answer = input(sprintf('File OUTPUTPATH/RawData.mat exists, perform CreateRawData.m? [Y,n]'),'s');
    if strcmp(answer,'Y')
    else if strcmp(answer,'n')
            clear all;
            close all;
            clc;
            return;
        end
    end
end

subfolder = '8classes';


class_labels = {};
class_info = {};
sonar_data = {};
iclass = 1;

% read all class and save run data
% Automatic detect the next class
for iletter = 'A':'Z'
    aux = [];
    irun = 0;
    if(exist(sprintf('%s/%s/Class%s',inputpath,subfolder,iletter),'dir'))
        %fprintf('Filename: %s/%s/Class%s\n',inputpath,subfolder,iletter);
        %fprintf('Subfolder %s exists\n',sprintf('%s/%s/Class%s',inputpath,subfolder,iletter))
        dir_info = dir(sprintf('%s/%s/Class%s',inputpath,subfolder,iletter));
        % get files in folder
        class_labels{iclass} = sprintf('Class%s',iletter);
        
        for ifile = 3:length(dir_info) 
           irun = irun +1;
           fprintf('File %s exists\n',sprintf('%s/%s/Class%s/%s',inputpath,subfolder,iletter,dir_info(ifile).name)); 
           [aux{irun},fs] = wavread(sprintf('%s/%s/Class%s/%s',inputpath,subfolder,iletter,dir_info(ifile).name));
        end
        class_info.(class_labels{iclass}).n_runs = irun;
        sonar_data.(class_labels{iclass}).run_data = aux;
        iclass = iclass +1;
    else
        continue;
    end    
end

save(sprintf('%s/RawData.mat',outputpath),'fs','sonar_data','class_labels','class_info');

fprintf('Finishing %s.m\n',mfilename('fullpath'));
