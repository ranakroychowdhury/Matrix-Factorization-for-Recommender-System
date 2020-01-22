A = importdata('train.csv');
num = 10000;
rating = [];
reviewerID = [];
itemID = [];
sizeK = 3;
sizeLambdaU = 2;
sizeLambdaV = 2;

for idx = 1 : num
    rating = [rating, A.data(idx)];
    i = convertStringsToChars(string(A.textdata(idx + 1)));
    itemID = [itemID, str2num(i(1, 2 : 10))];
    r = convertStringsToChars(string(A.textdata(idx + 1, 2)));
    reviewerID = [reviewerID, str2num(r(1, 2 : 10))];
end

%finding repetitive users
U = unique(reviewerID);
user = U(1<histc(reviewerID,unique(reviewerID)));
%disp(size(user));
%preparing test and validation set
c = size(user);
val = [];
test = [];
valTest = [];
valItem = [];
item = []; %finding the set of products from validation and test sets
for n = 1 : ceil(c(2)/2)
    k = find(reviewerID == user(n));
    val = [val, k];
    valTest = [valTest, k];
end
for n = ceil(c(2)/2) + 1 : c(2)
    k = find(reviewerID == user(n));
    test = [test, k];
    valTest = [valTest, k];
end
c = size(valTest);
%disp(c);
for n = 1 : c(2)
    item = [item, itemID(valTest(n))];
end
%disp(item); 
%disp(user); 

%preparing training set
train = []; 
reviewerTrain = [];
itemTrain = [];
ratingTrain = [];
for i = 1 : num
    rID = reviewerID(i);
    iID = itemID(i);
    if(ismember(rID, user) == 0 && ismember(iID, item) == 1)
        train = [train, i];
    end
end

c = size(train);
for i = 1 : c(2)
    reviewerTrain = [reviewerTrain, reviewerID(train(i))];
    itemTrain = [itemTrain, itemID(train(i))];
    ratingTrain = [ratingTrain, rating(train(i))];
end

reviewerTrain = sort(reviewerTrain);
itemTrain = sort(itemTrain);
uniqueItem = unique(itemTrain);
uniqueSize = size(uniqueItem);

revTrain = [];
itTrain = [];
for i = 1 : c(2)
    revTrain = [revTrain, i];
end
for i = 1 : uniqueSize(2)
    k = find(itemTrain == uniqueItem(i));
    itTrain(k) = i;
end

%do validation
S = sparse(revTrain, itTrain, ratingTrain);
[row, column, s] = find(S);
K = [8, 10, 12];
lambda_u = [0.01, 0.1, 1.0, 10.0];
lambda_v = [0.01, 0.1, 1.0, 10.0];
minRMSE = Inf;

for i = 1 : sizeK
    for j = 1 : sizeLambdaU
        for k = 1 : sizeLambdaV
            [U, V_T] = Offline3ALS(row, column, s, K(i), lambda_u(j), lambda_v(k));
            c = size(user);
            counter = 0;
            RMSE = 0.0;
            for u = 1 : ceil(c(2)/2)
                userArr = find(reviewerID == user(u)); %userArr stores the indices of the repeated user in reviewerID[]
                userArrSize = size(userArr);
                validationUser = user(u); %take the u-th user from the set of repeated users
                %if a product in validation set doesnt exist in training
                %set, look for the next product by the user in the
                %validation set
                countProd = 1;
                while true
                    validationItem = itemID(userArr(countProd)); 
                    useRating = rating(userArr(countProd));
                    indexItem = find(itemTrain == validationItem); 
                    rho = size(indexItem);
                    countProd = countProd + 1;
                    if(rho(2) == 0)
                        if(countProd > userArrSize(2))
                            break;
                        end
                        continue;
                    end
                    col = itTrain(indexItem(1));
                    break;
                end
                if(col < max(itTrain))
                    temp_V_T = V_T(:, col);
                    temp_un = inv(temp_V_T * temp_V_T'  + lambda_u(j) * eye(K(i))) * temp_V_T * useRating;
                    validationRatingSet = temp_un' * V_T; %validationRatingSet stores the rating by this user for all products 
                end
                for v = countProd : userArrSize(2)
                    validationItem = itemID(userArr(v)); %take the item corresponding to the v-th index of the repeated user from itemID[]
                    actualRating = rating(userArr(v)); %take the rating corresponding to the v-th index of the repeated user from rating[]
                    indexItem = find(itemTrain == validationItem); %find all the pos in itemTrain, where this item occurs
                    sh = size(indexItem);
                    if(sh(2) == 0)
                        continue;
                    end
                    col = itTrain(indexItem(1)); %take the first appearance of this item in itemTrain, and use this as the column number
                    RMSE = RMSE + (actualRating - validationRatingSet(col))^2; %col-th index from validationRatingSet gives the predicted rating
                    counter = counter + 1;
                end
            end
            RMSE = sqrt(RMSE / counter);
            if(RMSE < minRMSE)
                minRMSE = RMSE;
                optLambdaU = lambda_u(j);
                optLambdaV = lambda_v(k);
                optK = K(i);
                optU = U;
                optV_T = V_T;
            end
        end
    end
end

disp(optK);
disp(optLambdaU);
disp(optLambdaV);
disp(minRMSE);


%do testing
c = size(user);
RMSE = 0.0;
counter = 0;
for u = ceil(c(2)/2) + 1 : c(2)
    userArr = find(reviewerID == user(u)); %userArr stores the indices of the repeated user in reviewerID[]
    userArrSize = size(userArr);
    validationUser = user(u); %take the u-th user from the set of repeated users
    %if a product in validation set doesnt exist in training
    %set, look for the next product by the user in the
    %validation set
    countProd = 1;
    while true
        validationItem = itemID(userArr(countProd)); 
        useRating = rating(userArr(countProd));
        indexItem = find(itemTrain == validationItem); 
        rho = size(indexItem);
        countProd = countProd + 1;
        if(rho(2) == 0)
            if(countProd > userArrSize(2))
                break;
            end
            continue;
        end
        col = itTrain(indexItem(1));
        break;
    end
    if(col < max(itTrain))
       temp_V_T = optV_T(:, col);
       temp_un = inv(temp_V_T * temp_V_T'  + optLambdaU * eye(optK)) * temp_V_T * useRating;
       validationRatingSet = temp_un' * optV_T; %validationRatingSet stores the rating by this user for all products 
    end
    for v = countProd : userArrSize(2)
        validationItem = itemID(userArr(v)); %take the item corresponding to the v-th index of the repeated user from itemID[]
        actualRating = rating(userArr(v)); %take the rating corresponding to the v-th index of the repeated user from rating[]
        indexItem = find(itemTrain == validationItem); %find all the pos in itemTrain, where this item occurs
        sh = size(indexItem);
        if(sh(2) == 0)
            continue;
        end
        col = itTrain(indexItem(1)); %take the first appearance of this item in itemTrain, and use this as the column number
        RMSE = RMSE + (actualRating - validationRatingSet(col))^2; %col-th index from validationRatingSet gives the predicted rating
        counter = counter + 1;
    end
end
RMSE = sqrt(RMSE / counter);
disp(RMSE);