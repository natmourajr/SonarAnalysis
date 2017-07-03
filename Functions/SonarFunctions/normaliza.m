function [yout]=normaliza(y,tipo)
%
% Normaliza dados de acordo com tipo
%
% yout=normaliza(y,tipo)
%
% y = Dados de entrada dispostos em colunas
% tipo = Tipo de normalizacao
%   0 = Normaliza pelo maximo
%   1 = Normaliza pela media
%   2 = Nao normaliza
%   3 = Normaliza pela norma
 
y=y';%'
if tipo==0
    y=y./(ones(size(y,1),1)*max(abs(y)));
elseif tipo==1
    y=y./(ones(size(y,1),1)*mean(y));
elseif tipo==2
    y=y;
elseif tipo==3
    y=y./(ones(size(y,1),1)*sqrt(sum(y.^2)));
end
 
yout=y';
	
