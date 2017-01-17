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
subfolder = '4classes';

% check raw data existence
if(exist(sprintf('%s/%s/raw_data_info.mat',outputpath,subfolder),'file'))
	fprintf('File %s/%s/raw_data_info.mat exists\n',outputpath,subfolder);
else
	fprintf('File %s/%s/raw_data_info.mat does not exist, please create it\n',outputpath,subfolder);
end
