function [PredictY,model]  = GRMM(ValX, Trn, Para)
%% Parameter
%C ,D,V : 2^-8 : 2^8
%kerfPara.pars 
% rng(1)
%%
% % 测试代码
% load('./Data/UCI/Echo.mat');
% Para.p1=2^4;Para.p2=2^4; Para.p3 = 2^4;
%%
    C1 = Para.p1;
    C2 = Para.p2;
    kpar = Para.kpar;
    X = Trn.X; Y = Trn.Y; clear Trn;
%     ValX = X; %测试样本时打开
    XA = X( Y==1 , : ); XB = X( Y == -1 , : );
    X=[ XA ; XB ];
    m1=size(XA , 1);%Number of +1 samples
    m2=size(XB , 1);%Number of -1 samples
    Y1 = ones(m1,1); Y2 =- ones(m2,1);
    Y=[Y1;Y2];
    [ m, ~ ] = size(X);  %m:训练样本个数，n:训练样本维度
    mv = size(ValX,1); ev = ones(mv,1);
    options = optimoptions('quadprog',   'Display', 'off'); %将quadprog显示的命令行信息关闭
  %                 options = optimoptions('quadprog','Display','iter');
  e = ones(m,1);%m个全为1的向量1
  e1 = ones(m1,1); e2 = ones(m2,1);
  o = zeros(m,1);%o为全为0的向量
  %% ------------------QP optimization model ----------------------
   tr = tic;
   Km = KerF(X, kpar, X); Km1 = KerF(X, kpar, XA); Km2 = KerF(X, kpar, XB);
   K11 = diag(Y)*Km*diag(Y);
   K12 = diag(Y)*Km1;
   K22 = KerF(XA, kpar, XA);
   K13 = diag(Y)*Km2;
   K23 = KerF(XA, kpar, XB);
   K33 = KerF(XB, kpar, XB);
   d1 = diag([e1 * (m1/(2*C1)); e2 * (m2/(2*C2))]);
   d2 = diag(e1 * (m1/(2*C1))) ;
   d3 = diag(e2 * (m2/(2*C2))) ;
%    d2 = diag(e1 * (1/(2*D*V))) ;
%    d3 = diag(e2 * (1/(2*D*V))) ;
   H = [K11+d1     -K12         K13;
           -K12'       K22+d2      -K23;
            K13'          -K23'     K33+d3];
   H=( H + H' ) / 2 ;   
   f =[-e; e1; e2];     
  A = [o; e1; e2]'; %A是模型中的向量z
  b = m1/(2*C1)+m2/(2*C2);%b为模型中的
  Aeq = [Y ; -e1 ; e2]';
  beq = 0;
  lb = zeros(2*m,1);
   [alp, ~] = quadprog(H, f ,A,b,Aeq,beq,lb,[],[],options); %alpha为变量
   tr_time = toc(tr);
%    alp(alp<1e-9) = 0;  % 计算精度问题
   alphA = alp(1: m1); alphB = alp(m1+1: m);
   alpha = [alphA; alphB];
   beta = alp(m+1: m+m1);
   gama = alp(m+m1+1: 2*m);
   alp = [ alpha ; beta ; gama ]; 
   N_SVs = nnz(alp);
   G = [Y.* Km ; -Km1' ; Km2'];
%    Q = [ Y.* X ; -XA ;  XB ] ;
%    w =Q' * alp;
   L1A = find(alphA>0); L1B = find(alphB>0);L2 = find(beta>0); L3 = find(gama>0)+m1;
   bA = Y(L1A) - G(: , L1A)' * alp - ((Y(L1A)' * alphA))*(m1/(2*C1)) ;
   bB = Y(L1B) - G(: , L1B)' * alp - ((Y(L1B)' * alphB))*(m2/(2*C2)) ;
   bb = mean([bA; bB]);
   sig1 = abs( mean(G(: , L2)' * alp  - m1/(2*C1)*beta) );
   sig2 = abs( mean(-G(: , L3)' * alp - m2/(2*C2)*gama) )  ;
%    TrVaA =XA * w+ b;
%    TrVaB = XB * w + b;
%    muA = mean(TrVaA);
%    muB = mean(TrVaB);
%    sigA = cov(TrVaA);
%    sigB = cov(TrVaB);
   %% output
   KerVaX = KerF(X, kpar, ValX);
   KerVaX1 = KerF(XA, kpar, ValX);
   KerVaX2 = KerF(XB, kpar, ValX);
   Gv = [diag(Y) * KerVaX ; -KerVaX1 ;  KerVaX2];
   Val = Gv' * alp + bb;
   %%
   %-----------------------------Fisher Decsion Move------------------------------------%
%    mu1 = sig1/2; mu2 =sig2/2;
% %    y_1 = (Val - ev*mu1) .* (Val- ev*mu1);
% %    y_2 = (Val + ev*mu2) .* (Val+ ev*mu2);
% % %    M_1=length(DataTrain.Y(DataTrain.Y == 1)); %正类样本个数
% % %    M_2=length(DataTrain.Y(DataTrain.Y == -1));%负类样本个数
% %    Dec_Val= - y_1 ./ ( ev*abs(sig1)) + y_2 ./ ( ev*abs(sig2))  + ...
% %        ev*log(abs((sig2) /(sig1)) ) + ev*2*log(m1/ m2);
%     Dec_Val= - diag( (Val - ev*mu1) *inv(sqrt(sig1))* (Val- ev*mu1)' ) ...
%         +  diag( (Val + ev*mu2) *inv(sqrt(sig2))* (Val+ ev*mu2)' ) ...
%         +ev* log(det(sig2)/det(sig1)) +ev*2*log(m1/ m2);
%     +ev*2*log(m1/ m2);
 
   %----------------------------------------------------------------------------------%
   %----------------------------- Geometry Decsion Move--------------------------%
     sig = sig1 + sig2;
     sigd1  = (2* sig1)/sig;
     sigd2  = (2* sig2)/sig;
    Dec_Val = Val + 1 - sigd1;
%    model.sigma_12 = 2 + model.sigma_1 + model.sigma_2;
%    sigd1  = (2* model.sigma_1 +2 )/ (model.sigma_12);
%    sigma_2_bar  = (2* model.sigma_2 +2 )/ (model.sigma_12);
%    Dec_Val_2= model.w * ValX' + one*model.bb +one - one*model.sigma_1_bar;
   %----------------------------------------------------------------------------------%

   %    [Dec_Val , sigma_1, sigma_2] = FisherDecision(DataTrain,TestX,Para, '3');%用fisher决策，1为分布间隔代表方差，2为ywx方差
   
   PredictY = sign(Dec_Val);  
%    PredictY = sign(Val);  
%    Predict.Y = PredictY'; 
%    Predict.Y1 = sign(F1);  Predict.Y1 = Predict.Y1';
%    Predict.Y2 = sign(Dec_Val_2);  Predict.Y2 = Predict.Y2';
   model .n_SV = N_SVs;
   model.tr_time = tr_time;
    model.dis = 1;
   drw.ds = Val;
   drw.ss1 = (Val + 1);
   drw.ss2 = (Val - 1);
   
   model.fier =1;
   drw.ds1 = Dec_Val;
   drw.sg1 = (Val + sig1+1);
   drw.sg2 = (Val - sig2-1);
   model.drw = drw;
 
   model.twin = 0;
    

end