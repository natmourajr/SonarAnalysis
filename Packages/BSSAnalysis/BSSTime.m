% Projeto Marinha do Brasil

% Autor: Natanael Junior (natmourajr@gmail.com)
% Laboratorio de Processamento de Sinais - UFRJ
% Laboratorio de Tecnologia Sonar - UFRJ/Marinha do Brasil

% iniciando o script
clear all;
close all;
clc;

addpath('../../Functions/SonarFunctions');

fprintf('Starting %s.m\n',mfilename('fullpath'));

inputpath = '/Users/natmourajr/Workspace/Doutorado/SonarAnalysis/Results';

% selected version of data
%database = 'DemonSimulatedData';
database = 'DemonRealData';

% fixed params (hard coded)

window_size = 60; % window size in seconds

if strcmp(database,'DemonSimulatedData')
    m_ylim = [-30 5];
    m_yticks = [-30:5:5];
    m_interesting_freq = [386.7 471.7];
    m_xmax = 560;
    m_xmin = 340;
end

if strcmp(database,'DemonRealData')
    m_ylim = [-10 5];
    m_yticks = [-10:2.5:5];
    m_interesting_freq = [120.1 146.5 293.0 304.7];
    m_xmax = 350;
    m_xmin = 50;
end

fprintf('Performing Demon Analysis in %s database\n',database);

if(~exist(sprintf('%s/%s/data_info_file.mat',inputpath,database),'file'))
    fprintf('File Struct %s/%s not found, perform ReadRawData.m',inputpath,database);
    return;
end

dir_info = dir(sprintf('%s/%s',inputpath,database));
load(sprintf('%s/%s/data_info_file.mat',inputpath,database))


data = {};
fs = {};
labels = {};

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
    load(sprintf('%s/%s/%s/raw_data_file.mat',inputpath,database,dir_info(iclass).name));
    
    data{aux_class} = [];
    fs{aux_class} = [];
    labels{aux_class} = dir_info(iclass).name;
    for irun = 1:data_info.n_runs{aux_class}
        data{aux_class} = [data{aux_class} raw_data{irun}];
        fs{aux_class} = [fs{aux_class} data_info.fs{aux_class}];
    end
    aux_class = aux_class + 1;
end

% show demongram
if false
    for ibearing = 1:length(data)
        [power, freq, time] = demon(data{ibearing}', fs{ibearing});
        figure;
        % Apresenta espectros DEMON
        figure(ibearing)					% Abre nova janela
        subplot(2,1,1)				% Seleciona parte superior da figura
        imagesc(freq*60,time,power')			% Desenha DEMONgrama
        xlabel('Rotation (rpm)','fontsize',14)		% Legenda do Eixo x
        ylabel('Time (s)','fontsize',14)         % Legenda do Eixo y
        title(sprintf('DEMONgram - %s',labels{ibearing}),'fontsize',14);
        colormap(1-gray)			% Escala de cores
        
        subplot(2,1,2)				% Seleciona parte inferior da figura
        plot(freq*60,10*log10(normaliza(mean(power'),0))),grid			% Plota espectro DEMON medio
        
        axis tight					% Ajusta eixos
        xlabel('Rotation (rpm)','fontsize',14)		% Legenda do Eixo x
        ylabel('Amplitude','fontsize',14)
        
    end
end

% shows mean spectrum (enf in maximum points)
if false
    for ibearing = 1:length(data)
        [power, freq, time] = demon(data{ibearing}', fs{ibearing});
        figure;
        plot(60*freq,10*log10(normaliza(mean(power'),0)),'b')
        grid on;
        
        axis tight					% Ajusta eixos
        xlabel('Rotation (rpm)','fontsize',14)		% Legenda do Eixo x
        ylabel('Amplitude','fontsize',14)
        title(sprintf('Mean Spectrum - %s',labels{ibearing}),'fontsize',14);
        set(gca,'XTick',[60*min(freq):50:60*max(freq)]);
        set(gca,'YTick',m_yticks);
        ylim(m_ylim);
        set(gca,'XTickLabelRotation',45);
        
        [sorted,sorted_ids] = sort(10*log10(normaliza(mean(power'),0)),'descend');
        
        n_first_max = 10;
        min_diff = 10;
        
        sorted = sorted(1:n_first_max);
        sorted_ids = sorted_ids(1:n_first_max);
        
        ref_sorted_id = sorted_ids(1);
        delete_ids = [];
        for i = 2:length(sorted)
            for j = 1:length(ref_sorted_id)
                if abs(sorted_ids(i)-ref_sorted_id(j)) < min_diff
                    if ~(ismember(i,delete_ids))
                        delete_ids = [delete_ids i];
                    end
                else
                    if ~(ismember(sorted_ids(i),ref_sorted_id))
                        ref_sorted_id = [ref_sorted_id sorted_ids(i)];
                    end
                end
            end
        end
        
        sorted_ids(delete_ids) = [];
        sorted(delete_ids) = [];
        
        hold on;
        plot(60*freq(sorted_ids),sorted,'ro');
        
        for iannot = 1:length(sorted_ids)
            h = text(60*freq(sorted_ids(iannot))+10,sorted(iannot)+0.1,...
                sprintf('%1.1f rpm, \n %1.1f dB',60*freq(sorted_ids(iannot)),sorted(iannot)));
            set(h, 'rotation', 30,'FontWeight','bold')
        end
        
        hold off;
    end
end

min_len_data = 9999999999999;
for ibearing = 1:length(data)
    if length(data{ibearing}) < min_len_data
        min_len_data = length(data{ibearing});
    end
end

all_data = zeros(length(data),min_len_data);
for ibearing = 1:length(data)
    all_data(ibearing,:) = data{ibearing}(1:min_len_data);
end

% shows mean spectrum for dif bearings in a pict (enf in maximum points)
if false
    figure;
    for ibearing = 1:length(data)
        subplot(size(all_data,1),1,ibearing);
        [power, freq, time] = demon(all_data(ibearing,:)', fs{ibearing});
        plot(60*freq,10*log10(normaliza(mean(power'),0)),'b')
        grid on;
        
        axis tight					% Ajusta eixos
        xlabel('Rotation (rpm)','fontsize',14)		% Legenda do Eixo x
        ylabel('Amplitude','fontsize',14)
        title(sprintf('Mean Spectrum - %s',labels{ibearing}),'fontsize',14);
        set(gca,'XTick',[60*min(freq):50:60*max(freq)]);
        set(gca,'YTick',m_yticks);
        ylim(m_ylim);
        set(gca,'XTickLabelRotation',45);
        
        [sorted,sorted_ids] = sort(10*log10(normaliza(mean(power'),0)),'descend');
        
        n_first_max = 10;
        min_diff = 10;
        
        sorted = sorted(1:n_first_max);
        sorted_ids = sorted_ids(1:n_first_max);
        
        ref_sorted_id = sorted_ids(1);
        delete_ids = [];
        for i = 2:length(sorted)
            for j = 1:length(ref_sorted_id)
                if abs(sorted_ids(i)-ref_sorted_id(j)) < min_diff
                    if ~(ismember(i,delete_ids))
                        delete_ids = [delete_ids i];
                    end
                else
                    if ~(ismember(sorted_ids(i),ref_sorted_id))
                        ref_sorted_id = [ref_sorted_id sorted_ids(i)];
                    end
                end
            end
        end
        
        sorted_ids(delete_ids) = [];
        sorted(delete_ids) = [];
        
        hold on;
        plot(60*freq(sorted_ids),sorted,'ro');
        
        for iannot = 1:length(sorted_ids)
            h = text(60*freq(sorted_ids(iannot))+10,sorted(iannot)+0.1,...
                sprintf('%1.1f rpm, \n %1.1f dB',60*freq(sorted_ids(iannot)),sorted(iannot)));
            set(h, 'rotation', 30,'FontWeight','bold')
        end
        
        hold off;
    end
end

% shows mean spectrum (enf in intere. freq)
if false
    for ibearing = 1:length(data)
        [power, freq, time] = demon(data{ibearing}', fs{ibearing});
        y_values = 10*log10(normaliza(mean(power'),0));
        x_values = 60*freq;
        figure;
        plot(x_values,y_values,'b')
        grid on;
        
        axis tight					% Ajusta eixos
        xlabel('Rotation (rpm)','fontsize',14)		% Legenda do Eixo x
        ylabel('Amplitude','fontsize',14)
        title(sprintf('Mean Spectrum - %s',labels{ibearing}),'fontsize',14);
        set(gca,'XTick',[60*min(freq):50:60*max(freq)]);
        set(gca,'YTick',m_yticks);
        ylim(m_ylim);
        set(gca,'XTickLabelRotation',45);
        
        % mark import freqs
        selector = zeros(2,length(m_interesting_freq));
        for i = 1:length(m_interesting_freq)
            selector(1,i) = floor(m_interesting_freq(i)-0.1);
            selector(2,i) = ceil(m_interesting_freq(i)+0.1);
        end
        
        selected_ids = [];
        for i = 1:length(m_interesting_freq)
            [~,aux] = max(60*freq > selector(1,i) & 60*freq < selector(2,i));
            selected_ids = [selected_ids aux];
        end
        hold on;
        plot(x_values(selected_ids),y_values(selected_ids),'ro');
        
        for iannot = 1:length(selected_ids)
            h = text(x_values(selected_ids(iannot))+10,y_values(selected_ids(iannot))+0.1,...
                sprintf('%1.1f rpm, \n %1.1f dB',x_values(selected_ids(iannot)),...
                y_values(selected_ids(iannot))));
            set(h, 'rotation', 30,'FontWeight','bold')
        end
        
        hold off;
    end
end


% show spectrum per window
choose_window = 3;
num_windows = floor(size(all_data,2)/fs{1}/window_size);


if true
    window_begin = (choose_window-1)*window_size*fs{1}+1;
    window_end   = choose_window*window_size*fs{1};
    
    window_data = all_data(:,window_begin:window_end);
    
    mag_matrix = zeros(length(m_interesting_freq),length(data));
    
    % shows mean spectrum for dif bearings in a pict (enf in maximum points)
    figure;
    for ibearing = 1:length(data)
        subplot(size(window_data,1),1,ibearing);
        [power, freq, time] = demon(window_data(ibearing,:)', fs{ibearing});
        xvalues = 60*freq;
        yvalues = 10*log10(normaliza(mean(power'),0));
        
        plot(xvalues(xvalues>m_xmin & xvalues<m_xmax),yvalues(xvalues>m_xmin & xvalues<m_xmax),'b');
        grid on;
        
        axis tight					% Ajusta eixos
        xlabel('Rotation (rpm)','fontsize',14)		% Legenda do Eixo x
        ylabel('Amplitude','fontsize',14)
        title(sprintf('Spectrum (window %i) - %s',choose_window, labels{ibearing}),'fontsize',14);
        set(gca,'XTick',[60*min(freq):1:60*max(freq)]);
        set(gca,'YTick',m_yticks);
        ylim(m_ylim);
        set(gca,'XTickLabelRotation',45);
        
%         [sorted,sorted_ids] = sort(yvalues,'descend');
%         n_first_max = 100;
%         min_diff = 5;
%         
%         sorted = sorted(1:n_first_max);
%         sorted_ids = sorted_ids(1:n_first_max);
%         
%         % filter max points
%         ref_sorted_id = sorted_ids(1);
%         delete_ids = [];
%         for i = 2:length(sorted)
%             for j = 1:length(ref_sorted_id)
%                 if abs(sorted_ids(i)-ref_sorted_id(j)) < min_diff
%                     if ~(ismember(i,delete_ids))
%                         delete_ids = [delete_ids i];
%                     end
%                 else
%                     if ~(ismember(sorted_ids(i),ref_sorted_id))
%                         ref_sorted_id = [ref_sorted_id sorted_ids(i)];
%                     end
%                 end
%             end
%         end
%         
%         sorted_ids(delete_ids) = [];
%         sorted(delete_ids) = [];
%         
%         ref_sorted_id = sorted_ids(1);
%         selected_ids = [];
%         for i = 1:length(sorted)
%             for j = 1:length(m_interesting_freq)
%                 if abs(60*freq(sorted_ids(i))-m_interesting_freq(j)) < 12.0
%                     if ~(ismember(i,selected_ids))
%                         selected_ids = [selected_ids i];
%                     end
%                 end
%             end
%         end
%         
%         hold on;
%         plot(xvalues(sorted_ids(selected_ids)),yvalues(sorted_ids(selected_ids)),'ro');
%         
%         for iannot = 1:length(selected_ids)
%             for j = 1:length(m_interesting_freq)
%                 if abs(xvalues(sorted_ids(selected_ids(iannot)))-m_interesting_freq(j)) < 2.0
%                     h = text(xvalues(sorted_ids(selected_ids(iannot)))+10,...
%                         yvalues(sorted_ids(selected_ids(iannot)))+0.1,...
%                         sprintf('%1.1f rpm, \n %1.1f dB',m_interesting_freq(j),...
%                         yvalues(sorted_ids(selected_ids(iannot)))));
%                     set(h, 'rotation', 30,'FontWeight','bold')
%                     mag_matrix(j,ibearing) = yvalues(sorted_ids(selected_ids(iannot)));
%                 end
%             end
%         end
%         hold off;
    end
end


rmpath('../../Functions/SonarFunctions');
