% Projeto Marinha do Brasil

% Autor: Natanael Junior (natmourajr@gmail.com)
% Laboratorio de Processamento de Sinais - UFRJ
% Laboratorio de Tecnologia Sonar - UFRJ/Marinha do Brasil

% iniciando o script
clear all;
close all;
clc;

fprintf('Starting %s.m\n',mfilename('fullpath'));

% System var. point to external folders
inputpath = getenv('INPUTDATAPATH');
outputpath = getenv('OUTPUTDATAPATH');

%inputpath  = '/Users/natmourajr/Workspace/Doutorado/Data/SONAR/Classification';
%outputpath = '/Users/natmourajr/Workspace/Doutorado/SonarAnalysis/Results';

% selected version of data
database = '4classes';


fprintf('Reading Raw data in %s database\n',database);

if(exist(sprintf('%s/%s',outputpath,database),'dir'))
    answer = input(sprintf('Folder %s/%s exists, perform ReadRawData.m? [Y,n]',outputpath,database),'s');
    if ~strcmp(answer,'Y')
        clear all;
        close all;
        clc;
        return
    end
end

if(exist(sprintf('%s/%s',outputpath,database),'dir'))
    rmdir(sprintf('%s/%s',outputpath,database), 's')
end

mkdir(sprintf('%s/%s',outputpath,database));

data_info.class_labels = {};
data_info.fs = {};
data_info.n_runs = {};
data_info.database = database;

m_matlab_version = version('-release');
matlab_year = str2num(m_matlab_version(1:end-1));
if(matlab_year < 2016)
    data_info.raw_data_date = datetime('now','TimeZone','local','Format','dd-MMM-yyyy HH:mm:ss');
else
    data_info.raw_data_date = datestr(now,'dd-mm-yyyy HH:MM:SS');
end

data_info.username = char(java.lang.System.getProperty('user.name'));
data_info.computername = char(java.net.InetAddress.getLocalHost.getHostName);

dir_info = dir(sprintf('%s/%s',inputpath,database));
aux_class = 1;

for iclass = 1:length(dir_info)
    if ~isempty(strfind(dir_info(iclass).name,'.wav'))
        %fprintf('Wav File %s\n',dir_info(iclass).name);
        continue
    end
    if strcmp(dir_info(iclass).name,'.')==1 
        continue
    end
    if strcmp(dir_info(iclass).name,'..')==1
        continue
    end
    % for mac
    if ~isempty(strfind(dir_info(iclass).name,'.DS_Store'))
        %fprintf('Wav File %s\n',dir_info(iclass).name);
        continue
    end
    
    fprintf('Reading %s\n',dir_info(iclass).name);
    mkdir(sprintf('%s/%s/%s',outputpath,database,dir_info(iclass).name));
    
    data_info.class_labels{aux_class} = dir_info(iclass).name;
    
    dir_info_classes = dir(sprintf('%s/%s/%s',inputpath,database,dir_info(iclass).name));
    data_info.fs{aux_class} = {};
    
    raw_data = [];
    aux_run = 1;
    
    
    for irun = 1:length(dir_info_classes)
        if isempty(strfind(dir_info_classes(irun).name,'.wav'))
            continue
        end
        
        if strcmp(dir_info_classes(irun).name,'.')==1
            continue
        end
        if strcmp(dir_info_classes(irun).name,'..')==1
            continue
        end
        % for mac
        if ~isempty(strfind(dir_info_classes(irun).name,'.DS_Store'))
            %fprintf('Wav File %s\n',dir_info(iclass).name);
            continue
        end
        
        fprintf('Reading run: %s \n', dir_info_classes(irun).name);
        [data,fs] = audioread(sprintf('%s/%s/%s/%s',inputpath,database,dir_info(iclass).name,dir_info_classes(irun).name));
        raw_data{aux_run} = data;
        if aux_run == 1
            data_info.fs{aux_class} = fs;
        else
            data_info.fs{aux_class} = [data_info.fs{aux_class} fs];
        end
        aux_run = aux_run+1;
    end
    data_info.n_runs{aux_class} = aux_run-1;
    
    aux_class = aux_class+1;
    save(sprintf('%s/%s/%s/raw_data_file.mat',outputpath,database,dir_info(iclass).name),'raw_data');
end
save(sprintf('%s/%s/data_info_file.mat',outputpath,database),'data_info');

% % selected version of data
% subfolder = '4classes_old';
%
% % if raw_data has created...
% if(exist(sprintf('%s/RawData_%s.mat',outputpath,subfolder),'file'))
%     answer = input(sprintf('File OUTPUTPATH/RawData_%s.mat exists, perform ReadRawData.m? [Y,n]',subfolder),'s');
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
% class_labels = {};
% class_info = {};
% sonar_data = {};
% iclass = 1;
%
% % read all class and save run data
% % Automatic detect the next class
% for iletter = 'A':'Z'
%     aux = [];
%     irun = 0;
%     if(exist(sprintf('%s/%s/Class%s',inputpath,subfolder,iletter),'dir'))
%         %fprintf('Filename: %s/%s/Class%s\n',inputpath,subfolder,iletter);
%         %fprintf('Subfolder %s exists\n',sprintf('%s/%s/Class%s',inputpath,subfolder,iletter))
%         dir_info = dir(sprintf('%s/%s/Class%s',inputpath,subfolder,iletter));
%         % get files in folder
%         class_labels{iclass} = sprintf('Class%s',iletter);
%
%         for ifile = 3:length(dir_info)
%            irun = irun +1;
%            fprintf('File %s exists\n',sprintf('%s/%s/Class%s/%s',inputpath,subfolder,iletter,dir_info(ifile).name));
%            [aux{irun},fs] = wavread(sprintf('%s/%s/Class%s/%s',inputpath,subfolder,iletter,dir_info(ifile).name));
%         end
%         class_info.(class_labels{iclass}).n_runs = irun;
%         sonar_data.(class_labels{iclass}).run = aux;
%         iclass = iclass +1;
%     else
%         continue;
%     end
% end
%
% save(sprintf('%s/RawData_%s.mat',outputpath,subfolder),'fs','sonar_data','class_labels','class_info');
%
fprintf('Finishing %s.m\n',mfilename('fullpath'));
