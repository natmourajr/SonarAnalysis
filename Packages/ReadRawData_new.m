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
inputpath = getenv('INPUTDATAPATH');
outputpath = getenv('OUTPUTDATAPATH');

% selected version of data
subfolder = '8classes';


% if RawData struct has created...do nothing to save time!!!
if(exist(sprintf('%s/%s',outputpath,subfolder),'dir'))
	fprintf('Subfolder %s exists\n',sprintf('%s/%s',outputpath,subfolder));
	answer = input(sprintf('Folder %s/%s exists, perform ReadRawData.m? [Y,n]',outputpath,subfolder),'s');
	if strcmp(answer,'Y')
		rmdir(sprintf('%s/%s',outputpath,subfolder),'s');
		% create subfolder to read data
		mkdir(sprintf('%s/%s',outputpath,subfolder));
	else
		clear all;
		close all;
		clc;
		return;
	end
else 
	fprintf('Subfolder %s doesnt exist, I will create it...\n',sprintf('%s/%s',outputpath,subfolder));
	% create subfolder to read data
	mkdir(sprintf('%s/%s',outputpath,subfolder));
end


class_labels = {};
class_info = {};

% read all class and save run data
% Automatic detect the next class
did_once = false;
for iclass = 1:100
	aux = [];
    	irun = 0;

	if(exist(sprintf('%s/%s/Class%i',inputpath,subfolder,iclass),'dir'))
		fprintf('Reading folder %s/%s/Class%i\n',inputpath,subfolder,iclass);
		dir_info = dir(sprintf('%s/%s/Class%i',inputpath,subfolder,iclass));
		% get files in folder
        	class_labels{iclass} = sprintf('Class%i',iclass);

		for ifile = 3:length(dir_info) 
           		irun = irun +1;
           		fprintf('File %s exists\n',sprintf('%s/%s/Class%i/%s',inputpath,subfolder,iclass,dir_info(ifile).name)); 
           		[aux{irun},fs] = wavread(sprintf('%s/%s/Class%i/%s',inputpath,subfolder,iclass,dir_info(ifile).name));
        	end
		class_info.(class_labels{iclass}).n_runs = irun;
		sonar_data.run = aux;
		save(sprintf('%s/%s/raw_data_class%i.mat',outputpath,subfolder,iclass),'sonar_data');
		did_once = true;
	else
		fprintf('Folder %s/%s/Class%i does not exist\n',inputpath,subfolder,iclass);
		continue;
	end
end

if did_once
	save(sprintf('%s/%s/raw_data_info.mat',outputpath,subfolder),'class_labels','class_info','fs');
end
fprintf('Finishing %s.m\n',mfilename('fullpath'));
