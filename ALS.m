function [U, V_T] = Offline3ALS(i, j, s, K, lambda_u, lambda_v)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
%r = rng;
N = max(i);
M = max(j);

%initialize U and V^T
U = rand(N, K) * sqrt(5);
V_T = rand(K, M) * sqrt(5);

%Terminating condition
iter = 100;
threshold = 10^-1;

cnt = 0;
lossOld = Inf;
diffLoss = 10.0;

%ALS
%while(cnt < iter)
while(diffLoss > threshold)    
    
    if(cnt > 0)
        lossOld = lossNew;
    end
    for m = 1 : M
        temp_i = i(j == m);
        temp_s = s(j == m);
        temp_U = U(temp_i, :);
        temp_vm = inv(temp_U' * temp_U  + lambda_v * eye(K)) * temp_U' * temp_s;
        V_T(:, m) = temp_vm;
    end
    
    for n = 1 : N
        temp_j = j(i == n);
        temp_s = s(i == n);
        temp_V_T = V_T(:, temp_j);
        temp_un = inv(temp_V_T * temp_V_T'  + lambda_u * eye(K)) * temp_V_T * temp_s;
        U(n, :) = temp_un';
    end
    
    T = U * V_T;
    ind = sub2ind(size(T), i, j);
    elements = T(ind);
    diff = elements - s;
    partial_sum1 = dot(diff, diff);
    
    partial_sum2 = 0;
    for n = 1 : N
        partial_sum2 = partial_sum2 + lambda_u * norm(U(n, :))^2;
    end
    
    partial_sum3 = 0;
    for m = 1 : M
        partial_sum3 = partial_sum3 + lambda_v * norm(V_T(:, m))^2;
    end

    lossNew = partial_sum1 + partial_sum2 + partial_sum3;
    %disp(lossNew);
    if(cnt > 0)
        diffLoss = lossOld - lossNew;    
    end
    cnt = cnt + 1;
    
end
%disp(cnt);
%rng(r);
