% Three classes data, each has 4 samples and 5000 variables.
clear all;
load OctPat1;
x1=p1018A(401:5400,2:5)';
x2=p1018A(401:5400,12:15)';
x3=p1018A(401:5400,8:11)';
%Split data set into training (1:25) and testing (26:50)
idxTrain = 1:2;
idxTest = 3:4;

size_train=size(idxTrain,2);
size_test=size(idxTest,2);

% Combine training data with normalization
X = [x1(idxTrain,:);x2(idxTrain,:);x3(idxTrain,:)];
% Define class indicator as Y
Y = kron(eye(3),ones(size_train,1));
% Normalization
xmean = mean(X);
xstd = std(X);
ymean = mean(Y);
ystd = std(Y);

X = (X - xmean(ones(3*size_train,1),:))./xstd(ones(3*size_train,1),:);
Y = (Y - ymean(ones(3*size_train,1),:))./ystd(ones(3*size_train,1),:);

% Tolerance for 90 percent score
tol = (1-0.8) ;
% Perform PLS
[T,P,U,Q,B] = pls(X,Y,tol);

% Results
fprintf('Number of components retained: %i\n',size(B,1))
% Predicted classes
X1 = (x1(idxTest,:) - xmean(ones(size_test,1),:))./xstd(ones(size_test,1),:);
X2 = (x2(idxTest,:) - xmean(ones(size_test,1),:))./xstd(ones(size_test,1),:);
X3 = (x3(idxTest,:) - xmean(ones(size_test,1),:))./xstd(ones(size_test,1),:);
Y1 = X1 * (P*B*Q');
Y2 = X2 * (P*B*Q');
Y3 = X3 * (P*B*Q');

Y1 = Y1 .* ystd(ones(size_test,1),:) + ymean(ones(size_test,1),:);
Y2 = Y2 .* ystd(ones(size_test,1),:) + ymean(ones(size_test,1),:);
Y3 = Y3 .* ystd(ones(size_test,1),:) + ymean(ones(size_test,1),:);

% Class is determined from the cloumn which is most close to 1
[dum,classid1]=min(abs(Y1-1),[],2);
[dum,classid2]=min(abs(Y2-1),[],2);
[dum,classid3]=min(abs(Y3-1),[],2);
bar(1:size_test,classid1,'b');
hold on
bar(size_test+1:size_test*2,classid2,'r');
bar(size_test*2+1:size_test*3,classid3,'g');
hold off

% The results show that most samples are classified correctly.
legend('1018A control','1026A Pneumonia','1026B ARDS');

regcoeff=P*B*Q';
figure;
plot(1:size(x1,2),regcoeff(:,1),1:size(x1,2),regcoeff(:,2),1:size(x1,2),regcoeff(:,3));
























function [T,P,U,Q,B,W] = pls (X,Y,tol2)
% PLS   Partial Least Squares Regrassion
%
% [T,P,U,Q,B,Q] = pls(X,Y,tol) performs particial least squares regrassion
% between the independent variables, X and dependent Y as
% X = T*P' + E;
% Y = U*Q' + F = T*B*Q' + F1;
%
% Inputs:
% X     data matrix of independent variables
% Y     data matrix of dependent variables
% tol   the tolerant of convergence (defaut 1e-10)
% 
% Outputs:
% T     score matrix of X
% P     loading matrix of X
% U     score matrix of Y
% Q     loading matrix of Y
% B     matrix of regression coefficient
% W     weight matrix of X
%
% Using the PLS model, for new X1, Y1 can be predicted as
% Y1 = (X1*P)*B*Q' = X1*(P*B*Q')
% or
% Y1 = X1*(W*inv(P'*W)*inv(T'*T)*T'*Y)
%
% Without Y provided, the function will return the principal components as
% X = T*P' + E
%
% Example: taken from Geladi, P. and Kowalski, B.R., "An example of 2-block
% predictive partial least-squares regression with simulated data",
% Analytica Chemica Acta, 185(1996) 19--32.
%{
x=[4 9 6 7 7 8 3 2;6 15 10 15 17 22 9 4;8 21 14 23 27 36 15 6;
10 21 14 13 11 10 3 4; 12 27 18 21 21 24 9 6; 14 33 22 29 31 38 15 8;
16 33 22 19 15 12 3 6; 18 39 26 27 25 26 9 8;20 45 30 35 35 40 15 10];
y=[1 1;3 1;5 1;1 3;3 3;5 3;1 5;3 5;5 5];
% leave the last sample for test
N=size(x,1);
x1=x(1:N-1,:);
y1=y(1:N-1,:);
x2=x(N,:);
y2=y(N,:);
% normalization
xmean=mean(x1);
xstd=std(x1);
ymean=mean(y1);
ystd=std(y);
X=(x1-xmean(ones(N-1,1),:))./xstd(ones(N-1,1),:);
Y=(y1-ymean(ones(N-1,1),:))./ystd(ones(N-1,1),:);
% PLS model
[T,P,U,Q,B,W]=pls(X,Y);
% Prediction and error
yp = (x2-xmean)./xstd * (P*B*Q');
fprintf('Prediction error: %g\n',norm(yp-(y2-ymean)./ystd));
%}
%
% By Yi Cao at Cranfield University on 2nd Febuary 2008
%
% Reference:
% Geladi, P and Kowalski, B.R., "Partial Least-Squares Regression: A
% Tutorial", Analytica Chimica Acta, 185 (1986) 1--7.
%

% Input check
error(nargchk(1,3,nargin));
error(nargoutchk(0,6,nargout));
if nargin<2
    Y=X;
end
tol = 1e-10;
if nargin<3
    tol2=1e-10;
end

% Size of x and y
[rX,cX]  =  size(X);
[rY,cY]  =  size(Y);
assert(rX==rY,'Sizes of X and Y mismatch.');

% Allocate memory to the maximum size 
n=max(cX,cY);
T=zeros(rX,n);
P=zeros(cX,n);
U=zeros(rY,n);
Q=zeros(cY,n);
B=zeros(n,n);
W=P;
k=0;
% iteration loop if residual is larger than specfied
while norm(Y)>tol2 && k<n
    % choose the column of x has the largest square of sum as t.
    % choose the column of y has the largest square of sum as u.    
    [dummy,tidx] =  max(sum(X.*X));
    [dummy,uidx] =  max(sum(Y.*Y));
    t1 = X(:,tidx);
    u = Y(:,uidx);
    t = zeros(rX,1);

    % iteration for outer modeling until convergence
    while norm(t1-t) > tol
        w = X'*u;
        w = w/norm(w);
        t = t1;
        t1 = X*w;
        q = Y'*t1;
        q = q/norm(q);
        u = Y*q;
    end
    % update p based on t
    t=t1;
    p=X'*t/(t'*t);
    pnorm=norm(p);
    p=p/pnorm;
    t=t*pnorm;
    w=w*pnorm;
    
    % regression and residuals
    b = u'*t/(t'*t);
    X = X - t*p';
    Y = Y - b*t*q';
    
    % save iteration results to outputs:
    k=k+1;
    T(:,k)=t;
    P(:,k)=p;
    U(:,k)=u;
    Q(:,k)=q;
    W(:,k)=w;
    B(k,k)=b;
    % uncomment the following line if you wish to see the convergence
%     disp(norm(Y))
end
T(:,k+1:end)=[];
P(:,k+1:end)=[];
U(:,k+1:end)=[];
Q(:,k+1:end)=[];
W(:,k+1:end)=[];
B=B(1:k,1:k);
