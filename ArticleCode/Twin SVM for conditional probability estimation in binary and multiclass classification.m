function [PredictY,model] = F_CPTWSVM(ValX, Trn, Para)
% function [Fval,model] = F_CPTWSVM(ValX, Trn, Para)
%%
%Parameter
%epsilon: 2^-8 : 2^-1
%C1 :2^-8 : 2^8
%C2 :2^-8 : 2^8
%% Initialization & Data Input
C1 = Para.p1;
C2 = Para.p2;
C11 = Para.p1;
C22 = Para.p2;
% kerfPara.pars = Para.kpar.par1;
% kerfPara.type = Para.kpar.type;
kpar = Para.kpar;
X = Trn.X; Y = Trn.Y;
clear Trn;
[ m, ~ ] = size(X);
X = [X, ones(m,1)];
X1 = X( Y==1 , : );X2 = X( Y == -1 , : );
m1=size(X1 , 1);%Number of +1 samples
m2=size(X2 , 1);%Number of -1 samples
X=[ X1 ; X2 ];
mt = size(ValX, 1);
ValX = [ValX, ones(mt,1)];
Y=[ones(m1,1);-ones(m2,1)];
e1 = ones(m1,1);
e2 = ones(m2,1);
e = ones(m,1);
oe = zeros(m,1);
%
alp1 = zeros(m1,1);
alp2 = zeros(m2,1);
beta1 = zeros(m,1);
beta2 = zeros(m,1);
gamma1 = zeros(m,1);
gamma2 = zeros(m,1);
options = optimoptions('quadprog',   'Display', 'off'); %
K= KerF(X,kpar, X); %KerX：mXm
K1= KerF(X1,kpar, X1); %KerX_1：m1Xm1
K2= KerF(X1,kpar, X); %KerX_2：m1Xm

% K= Ker(X,kerfPara, X); %KerX：mXm
K_1= KerF(X2,kpar, X2); %KerX_1：m2Xm2
K_2= KerF(X2,kpar, X); %KerX_2：m2Xm
abg_old = ones(5*m,1); Fval_old=[-1;-1]; Fval1_old=0;Fval2_old=0;
%%  Model
tr = tic;
ite = 0;
while ite < 100
    f1 = C2  * K2 * Y - (e1/2) + K2 * beta1 - K2 * gamma1;
    f2 = -C22  * K_2 * Y - (e2/2) + K_2 * beta2 - K_2 * gamma2;
    lb1 = zeros(m1,1);
    ub1 = C1 * e1;
    lb2 = zeros(m2,1);
    ub2 = C11*e2;
    ite = ite + 1;
    %     alp1_old = alp1; alp2_old = alp2;
    %     beta1_old = beta1; beta2_old = beta2;
    %     gamma1_old = gamma1; gamma2_old = gamma2;
    
    % alp
    [alp1, ~] = quadprog(K1, f1, [], [], [], [], lb1, ub1, [], options);
    [alp2, ~] = quadprog(K_1, f2, [], [], [], [], lb2, ub2, [], options);
    %     alpha1 = qpSOR(K1,0.5,cpos,0.05);
    %     alpha2 = qpSOR(K_1,0.5,cpos,0.05);
    
    % beta
    f_b1 = K2' * alp1 - K * gamma1 + C2 * K * Y; f_b2 =  K_2' * alp2  - K * gamma2 - C22 * K * Y;
    [beta1, ~] = quadprog(K, f_b1, [], [], [], [], oe, [], [], options);
    [beta2, ~] = quadprog(K, f_b2, [], [], [], [], oe, [], [], options);
    
    
    % gamma
    f_g1 = -K2' * alp1 - K * beta1 - C2 * K * Y + e; f_g2 = - K_2' * alp2 - K * beta2 + C22 * K * Y + e;
    [gamma1, ~] = quadprog(K, f_g1, [], [], [], [], oe, [], [], options);
    [gamma2, ~] = quadprog(K, f_g2, [], [], [], [], oe, [], [], options);
    %  if norm(alp1 - alp1_old) < 1e-3 && norm(alp2 - alp2_old) < 1e-3
    abg = [alp1; beta1; gamma1; alp2; beta2; gamma2];  abg(abg<1e-9)=0;
    %     Tnorm(ite) = [norm(abg - abg_old) / (1+norm(abg_old))];
    Tnorm(ite) = [norm(abg - abg_old) / ((1+norm(abg_old)))];
    
    %     abge1=  [alp1; beta1 ; gamma1];
    %     abge2=  [alp2; beta2 ; gamma2 ];
    %     Fval1 = 0.5 * abge1' * H1 * abge1 + [(C2 * K2 * Y - 0.5* e1); C2 * K * Y ; (-C2 * K * Y +e)]'*abge1;
    %     Fval2 = 0.5 * abge2' * H2 * abge2 + [(-C22 * K_2 * Y -0.5* e2); -C22 * K * Y ; (C22 * K * Y +e)]' * abge2 ;
    %     Fval = [Fval1; Fval2];
    % %     TnF(ite) = [log(norm(Fval))];
    %   TnF1(ite) = [log(norm(Fval1- Fval1_old))]; TnF2(ite) = [log(norm(Fval2 - Fval2_old))];
    if  norm(abg - abg_old) / ((1+norm(abg_old))) <1e-3, break; end
    % if  norm(Fval - Fval_old) / norm(Fval_old) < 1e-3, break; end
    abg_old = abg;
    %     Fval_old = Fval;
    %     Fval1_old = Fval1;
    %     Fval2_old = Fval2;
end
tr_time = toc(tr);
%  s =1: ite;
%  subplot(3,1,1);
% plot (s, Tnorm, '-b');
% legend('变量误差精度');
% subplot(3,1,2);
%  plot(s, TnF1, '-r');
%  legend('函数值1的差值');
%  subplot(3,1,3);
%  plot(s, TnF2, '-g');
%  legend('函数值2的差值');


alp1(alp1<1e-9) = 0;  alp2(alp2<1e-9) = 0;
alp1(C1 - alp1 <1e-9) = C1; alp2(C11 - alp2 <1e-9) = C11;
beta1(beta1<1e-9) = 0; beta2(beta2<1e-9) = 0;
gamma1(gamma1<1e-9) = 0; gamma2(gamma2<1e-9) = 0;
abge1=  [alp1; beta1 ; gamma1];
abge2=  [alp2; beta2 ; gamma2 ];
H1 = [K1     K2       -K2;
    K2'       K          -K;
    -K2'     -K'         K];
H1=( H1 + H1' ) / 2 ;
H2 = [K_1      K_2       -K_2;
    K_2'      K          -K;
    -K_2'     -K'         K];
H2=( H2 + H2' ) / 2 ;
Fval1 = 0.5 * abge1' * H1 * abge1 + [(C2 * K2 * Y - 0.5* e1); C2 * K * Y ; (-C2 * K * Y +e)]'*abge1;
Fval2 = 0.5 * abge2' * H2 * abge2 + [(-C22 * K_2 * Y -0.5* e2); -C22 * K * Y ; (C22 * K * Y +e)]' * abge2 ;
Fval.Fval1 = Fval1;
Fval.Fval2 = Fval2;
%%
N_SV1 = nnz(alp1);
N_SV2 = nnz(alp2);
N_SVs = N_SV1 + N_SV2;
% w1b1 = Q1' * alp1e;
% w2b2 = Q2' * alp2e;
%% CPTWSVM Output
alp1e = [abge1; e];
alp2e = [abge2; e];
KerTeX = KerF(ValX,kpar,X);
KerTeX1 = KerF(ValX,kpar,X1);
KerTeX2 = KerF(ValX,kpar,X2);
KerTeX = KerTeX';KerTeX1=KerTeX1';KerTeX2=KerTeX2';
G_1 = [ KerTeX1 ; KerTeX ;  -KerTeX ; C2*diag(Y)* KerTeX];
G_2 = [ KerTeX2 ; KerTeX ;  -KerTeX ; -C22*diag(Y)* KerTeX];
Fun1 = G_1' * alp1e ;
Fun2 = G_2' * alp2e ;
%  F = [Fun1, Fun2];
%  [ Fmax , PredictY ] = max( F , [] , 2 );
Pt1 = Fun1 - 0.5;
%  y1n = abs(Fun1-1)/ (norm(w1b1));
Pt2 = Fun2 - 0.5;
%  y2n = abs(Fun2-1)/(norm(w2b2));Ptn = y2n - y1n ;
Pt = Pt1 - Pt2;
model.tr_time = tr_time;
model.n_SV= N_SVs;
% model.val = Ptn;

PredictY = sign(Pt);

if Para.drw == 1
    drw.ds = Pt;
    drw.ss1 = Pt1;
    drw.ss2 = Pt2;
    model.twin = 1;
    drw.dss1r = Fun1 - model.twin;
    drw.dss1l = Fun1 ;
    drw.dss2r = Fun2 ;
    drw.dss2l = Fun2 - model.twin;
    model.drw = drw;
end
if Para.drw == 2
    model.Fun1 = Fun1;
    model.Fun2 = Fun2;
    %      model.w1 = w1b1;
    %      model.w2 = w2b2;
    %      model.drw.ds = Pt;
end


end