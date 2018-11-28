function mx=tpsw(x,npts,n,p,a)
% |
% |[mx] = tpsw(x,npts,n,p,a)
% |
% |calcula a media local para o vetor x usando o algoritmo
% |"two-pass split window"
% |
% |onde:	x	= vetor de entrada
% |		npts	= numero de pontos em x a ser usado
% |		n	= tamanho da janela lateral
% |		p	= tamanho do gap lateral
% |		a	= constante para calculo do limiar
% |
% |		mx	= vetor de saida com a media local
% |________________________________________________________
% |
% |William Soares-Filho
% |Instituto de Pesquisas da Marinha
% |
% |Modificado em 20-08-1999
% |
% |Esta versao esta' funcionando igual `a versao em FORTRAN %'
% |quando n=17, p=3 e a=2.0
% |________________________________________________________


global h
% Verificacao dos argumentos de entrada
%
if nargin<1,help tpsw;return;end		% apresenta o help (sem argumentos)
if min(size(x))==1,x=x(:);end			% Se vetor, coloca em coluna
if nargin<2,npts=size(x,1);end			% calcula tamanho do vetor x
x=x(1:npts,:);					% limita entrada a npts pontos
if nargin<3,n=round(npts*.04/2+1);end		% tamanho da janela lateral
if nargin<4,p=round(n/8+1);end			% tamanho do gap lateral
if nargin<5,a=2.0;end				% constante para calculo do limiar

%[npts n p a]


% Calcula media local usando janela com gap central
%
if p>0
    h=[ones(1,n-p+1) zeros(1,2*p-1) ones(1,n-p+1)];
	% Filtro com gap central ...
else
    h=[ones(1,2*n+1)];
    p=1;
end
h=h/norm(h,1);					% ... normalizado
mx=conv2(h,1,x);				% Filtra sinal
ix=fix((length(h)+1)/2);			% Defasagem do filtro
mx=mx(ix:npts+ix-1,:);				% Corrige da defasagem

% Corrige pontos extremos do espectro
% 
ixp=ix-p;					% Variavel auxiliar
mult=2*ixp./[ones(1,p-1)*ixp ixp:2*ixp]';%'	% Correcao dos pontos extremos
mx(1:ix,:)=mx(1:ix,:).*(mult*ones(1,size(x,2)));% Pontos iniciais
mx(npts-ix+1:npts,:)=mx(npts-ix+1:npts,:).*...
	    (flipud(mult)*ones(1,size(x,2)));	% Pontos finais
% 
% %return
% % Elimina picos para a segunda etapa da filtragem
% %
ind1=find((x-a*mx)>0);				% Pontos maiores que a*mx
x(ind1)=mx(ind1);				% Substitui pontos por mx
mx=conv2(h,1,x);				% Filtra sinal
mx=mx(ix:npts+ix-1,:);				% Corrige defasagem
% 
% % Corrige pontos extremos do espectro
% % 
mx(1:ix,:)=mx(1:ix,:).*(mult*ones(1,size(x,2)));% Pontos iniciais
mx(npts-ix+1:npts,:)=mx(npts-ix+1:npts,:).*...
	    (flipud(mult)*ones(1,size(x,2)));	% Pontos finais
