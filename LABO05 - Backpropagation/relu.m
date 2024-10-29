clear close()
P = -4 :0.1: 4 
T = P.^2 + 1*(rand(size(P))-0.5)

%siec 
S1 = 2;
W1 = rand(S1, 1)- 0.5; B1 = rand(S1, 1)- 0.5;
W2 = rand(1, S1) -0,5; B2 = rand(1,1) -0.5;
lr = 0.001


for  epoka = 1 : 200
%odpowiedz sieci
X = W1*P + B1*ones(size(P));
A1 = max(X,0);   %ReLu
A2 = W2*A1 + B2;

%propagacja wsteczna
E2 = T - A2;
E1 = W2'*E2;

dW2 = lr* E2 * A1';
dB2 = lr *E2 * ones(size(E2))';
dW1 = lr * (exp(X)./(exp(X)+1)) .* E1 * P';
dB1 = lr * (exp(X)./(exp(X)+1)) .* E1 * ones(size(P))';

W2 = W2 + dW2; B2 =  B2 + dB2;
W1 = W1 + dW1; B1 = B1 + dB1;


if (mod(epoka, 5)==0)
clf();
plot(P,T, 'r*')
hold on
plot(P,A2)
pause(0.1);
endif

end
