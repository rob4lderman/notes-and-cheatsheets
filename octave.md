
# Octave cheatsheet: 

* [Starting octave](#Starting-octave)
* [Built-in vars ](#Built-in-vars-)
* [Matrices and Vectors](#Matrices-and-Vectors)
* [Loading data, Saving Data, ETL](#Loading-data,-Saving-Data,-ETL)
* [Misc Functions](#Misc-Functions)
* [Plots](#Plots)
* [Control statements: if, for, while](#Control-statements--if,-for,-while)
* [Functions:](#Functions-)
* [Vectorization](#Vectorization)


## <a name="Starting-octave"></a>Starting octave

    $ octave --no-gui
    $ octave


## <a name="Built-in-vars-"></a>Built-in vars 

    pi
    i
    ans     % result of previous command


## <a name="Matrices-and-Vectors"></a>Matrices and Vectors

    A = [3, 4; 5, 6; 7, 8]      % 3x2 matrix
    A = [3 4; 5 6; 7 8;]
    A = [3 4;
         5 6;
         7 8]

    A = reshape([3,5,7,4,6,8], 3,2 )    % 3x2, cols filled first

    size(A)         % [num rows, num cols]
    size(A,1)       % num rows
    size(A,2)       % num cols
    length(A)       % prints out LONGER dim
    
    A(3,2)          % subscript row=3,col=2
    A(1:3)          % subscript rows=1:3,col=1
    A(1:3,2)        % subscript rows=1:3,col=2
    A(1:3,1:2)      % subscript rows=1:3,cols=1:2
    A(1,:)          % subscript rows=1, cols=ALL
    A(:,2)          % subscript rows=ALL, cols=2
    A(:,:)          % subscript rows=ALL, cols=ALL  (entire matrix)
    A(:)            % subscript rows=ALL, cols=ALL, returned as single-column matrix
    A(3:end,:)      $ rows 3 thru n

    % random sampling
    randperm(10,5)                  % randomly sample from 1:10, sample size = 5
    A(randperm(size(A,1),3), :)     % randomly sample 3 rows from matrix A



    I = eye(2)      % 2x2 identity matrix

    

    V = [1;2;3;4;5;6;7;8;9]
    X = V(1:5)      % subscript/subset/select rows 1:5 (cols 1)

    V = 1:9         % sequence, by 1
    V = 1:0.1:9     % sequence, by 0.1

    A([1,3],:)                  % rows 1 and 3, all cols
    A(:,2) = [10; 11; 12]
    
    A = [ A, [100; 101; 102]]   % add a column
    
    A = [1 2; 3 4; 5 6]
    B = [11 12; 13 14; 15 16]
    
    C = [A, B]                  % B added as new columns
    D = [A; B]                  % B added as new rows


    A = [1 2; 3 4; 5 6]
    B = [11 12; 13 14; 15 16]
    C = [1 1; 2 2]
    
    A * C   % cross product
    A .* B  % dot product (element-wise mult)
    
    A .^ 2  % element-wise squaring
    1 ./ A  % element-wise 
    
    log(A)
    exp(A)
    abs(A)
    -A
    -1 * A
    
    A'  % transpose
    pinv(A)
    
    a = [ 1 15 2 0.5 ]
    max(a)
    [ val, ind ] = max(a)
    a < 3
    a( a < 3 )

    sum(a)
    prod(a)
    floor(a)
    ceil(a)
    
    find(a < 3)
    
    A = magic(3)        % each row, col, and diag add to the same thing
    help magic
    sum(A, 1)           % sum cols
    sum(A, 2)           % sum rows
    
    A .* eye(9)
    sum( sum( A.* eye(9)))
    
    flipud              % flips the matrix up-down
    
    [r, c] = find(A >= 7)
    help find
    
    rand(3)
    max(A,[],1)         % per col max's
    max(A,[],2)         % per row max's

    X = [1 3 4500;
         1 2 2100;
         1 2 1500;
         1 4 2300;
         1 2 1200;
         1 1 900;
         1 3 1400 ];
    mu_x = mean(X)
    sigma_x = std(X)


### Cell Arrays

* A cell-array is just like a normal array/vector,
* except that its elements can also be strings 
    * which they can't in a normal vector,
* and you index into them using curly braces 
    * instead of square brackets
    * e.g. `wordList{3}`

Example: [machine-learning-ex6/ex6/getVocabList.m](machine-learning-ex6/ex6/getVocabList.m)

    %% Read the fixed vocabulary list
    fid = fopen('vocab.txt');
    
    % Store all dictionary words in cell array vocab{}
    n = 1899;  % Total number of words in the dictionary
    
    % For ease of implementation, we use a struct to map the strings => integers
    % In practice, you'll want to use some form of hashmap
    vocabList = cell(n, 1);
    for i = 1:n
        % Word Index (can ignore since it will be = i)
        fscanf(fid, '%d', 1);
        % Actual Word
        vocabList{i} = fscanf(fid, '%s', 1);
    end
    fclose(fid);



## <a name="Loading-data,-Saving-Data,-ETL"></a>Loading data, Saving Data, ETL

    load featuresX.dat      % loads data into var named featuresX
    load('featuresX.dat')   

    who             % show vars in workspace

    V = [1;2;3;4;5;6;7;8;9]
    save hello.mat V;           % save data to file in BINARY format
    save hello.txt V -ascii;    % save data in ASCII format

    % Load character file
    fid = fopen(filename);
    if fid
        file_contents = fscanf(fid, '%c', inf);     % read the whole file
        one_string = fscanf(fid, '%s', 1);          % read one string
        fclose(fid);
    else
        file_contents = '';
        fprintf('Unable to open %s\n', filename);
    end



## <a name="Misc-Functions"></a>Misc Functions

    format long                     % show more decimal digits 
    format short                    % show less decimal digits
    disp(x)                         % print to terminal
    disp("hello"), disp("world")

    rand(3,4)       % 3x4 matrix of uniform-dist randoms (0:1)
    randn(1,1000)   % 1x1000 matrix of normal-dist randoms

    V = [1 2 3 4];
    length(V)

    pwd;            % filesystem
    cd '{dir}'
    ls

    who             % show vars in workspace
    whos            % show vars in workspace, with more details
    clear featuresX % clear var
    clear           % clear all vars

    class(x)        % type of var x



## <a name="Plots"></a>Plots


    x=1:10; y=1:10;
    plot(x,y)           % line plot
    plot(x,y,"*")       % scatter plot
    plot(x,y,":*")      % linestyle=dashed, marker=*  
    plot(x,y,":*r")     % linestyle=dashed, marker=*, color=red

    plot(x1, y1, "+;Admitted;", x2, y2, "o;Not admitted;")

    w = randn(1,1000);  
    hist(w)
    hist(w,50)          % 50 buckets


    t = [0:0.01:0.98];
    y1 = sin(2 * pi * 4 * t);
    plot(t, y1);
    
    y2 = cos(2 *pi*4*t);
    plot(t, y2);
    
    
    plot(t, y1);
    hold on;                    % subsequent command apply to same figure
    
    plot(t, y2, 'r');

    xlabel('time');
    xlabel('value');
    legend('sin','cos');
    title('my plot');
    hold off;

    print -dpng 'myplot.png'
    
    help plot;
    close   % close the plot window
    figure(1); plot(t, y1);
    figure(2); plot(t, y2);
    subplot(1,2,1);             % divide figure into 1x2 grid, access first (1) element
    plot(t,y1);
    subplot(1,2,2);             % divide figure into 1x2 grid, access first (2) element
    plot(t,y2);
    
    axis([0.5 1 -1 1])          % x, y range
    clf;                        % clear figure
    
    A = magic(5)
    imagesc(A)                  % visualize matrix
    imagesc(A), colorbar, colormap gray;



## <a name="Control-statements--if,-for,-while"></a>Control statements: if, for, while

    v = zeros(10,1)         % 10 rows, 1 col
    
    for i=1:10,
        v(i) = 2^i;
    end;
    
    indices = 1:10
    for i = indices,
        disp(i);
    end;
    
    
    i = 1;
    while i <= 5,
        v(i) = 500;
        i = i + 1;
    end;
    
    while true,
        v(i) = 999;
        if i == 6,
            break;
        end;
    end;
    
    if v(1) == 1,
       disp('value is 1');
    elseif ,
    else 
    end;



## <a name="Functions-"></a>Functions:

Functions are defined in a file named after the function.   
Functions are invoked by calling the file by name (without the '.m' suffix)

    addpath('{dir}')        % add dir to path, used when searching for functions


Example: FILE squareThisNumber.m:

    function y = squareThisNumber(x)
    y = x^2



Example: FILE squareAndCubeThisNumber.m:

    % returns more than 1 value
    function [y1,y2] = squareAndCubeThisNumber(x)
    y1 = x^2
    y2 = x^2

Called like so:

    [a,b] = squareAndCubeThisNumber(5)


Example: define cost function

    X = [ 1 1; 1 2; 1 3]
    Y = [ 1; 2; 3]

FILE costFunctionJ.m:

    function J = costFunctionJ(X, y, theta)
    
    % X is the "design matrix" - i.e. regressor variables
    % y is the "class labels" - i.e. response variable
    
    m = size(X,1)               % number of training samples
    predictions = X * theta     % predictions of hypothesis (theta) on all m training samples
    sqrErrors = (predictions - y).^2
    
    J = 1/(2*m) * sum(sqrErrors)

Called like so:

    theta = [0; 1]
    costFunctionJ(X, Y, theta)
    
    theta = [0; 0]
    costFunctionJ(X, Y, theta)



## <a name="Vectorization"></a>Vectorization

take advantage of optimized numerical methods for matrix calculations

    % unvectorized
    pred = 0.0
    for j = 1:n+1,
        pred = pred + theta(j) * x(j)
    end;
    
    % vectorized  - simpler and more efficient
    pred = theta' * x;


    % vectorized gradient descent:
    
    %    for n >= 1
    %    
    %    repeat {
    %
    %        theta_j = theta_j - alpha * 1/m SUM_i=1..m [ h0(X_i) - y_i ] * Xj_i
    %
    %    }  (simultaneous update theta_j, for j=0..n)
    
    
    % theta = (n+1)x1 column vector
    % alpha is scalar
    % X = m x (n+1)
    % m = number of training samples
    % n = number of features
    % n+1 : includes the y-intercept (x0) term
    % X_i = ith training sample
    % y_i = ith response variable value
    % Xj_i = xj in the ith training sample
    
    % theta = theta - alpha * delta           
    %
    % delta is (n+1)x1 vector
    %
    %        delta_j = 1/m SUM_i=1..m [ h0(X[i]) - y[i]] * X[i,j]
    %
    %        delta_1 = 1/m SUM_i=1..m [ h0(X[i]) - y[i]] * X[i,1]
    %                                 [ h0(X[1]) - y[1]] * X[1,1]
    %                                 [ h0(X[2]) - y[2]] * X[2,1]
    %                                 [ h0(X[3]) - y[3]] * X[3,1]
    %
    %        delta_2 = 1/m SUM_i=1..m [ h0(X[i]) - y[i]] * X[i,2]
    %

    predictions = X * theta             % mx1 vector
    errors = predictions - y            % mx1 vector

                                        % need a (n+1)xm vector: X'
    % X' * errors                       % (n+1)xm  * mx1 = (n+1)x1
    %
    % X = [ x0 x1 x2 x3 ...;
    %       x0 x1 x2 x3 ...;
    %       ... ]
    %
    % X' = [ x0 x0 x0 x0 x0 ...;
    %        x1 x1 x1 x1 x1 ...;
    %        x2 x2 x2 x2 x2 ...;
    %
    % errors = [ error_1;
    %            error_2;
    %            ... ]
    %
    % X' * errors = [ x0 * error_1 + x0 * error_2 + x0 * error_3 ... ;
    %                 x1 * error_1 + x1 * error_2 + x1 * error_3 ... ;
    %                 ... ]


    delta = 1 / m * (X' * errors)

    theta = theta - alpha * delta           
    




