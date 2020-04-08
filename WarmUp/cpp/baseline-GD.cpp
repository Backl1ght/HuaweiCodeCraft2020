#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <pthread.h>
#include <sched.h>
using namespace std;

#define L1_CACHE_SIZE 64

#define DEBUG

#ifdef DEBUG
#include <cstdio>
#include <ctime>
#include <stdarg.h>
int _dbg(const char *format, ...) {
    va_list argPtr;
    int cnt = 0;
 
    va_start(argPtr, format);
    fflush(stdout);
    cnt = vfprintf(stderr, format, argPtr);
    va_end(argPtr);
    return cnt;
}
#else
inline void _dbg(const char *format, ...) {}
#endif


/**
 * calculate g = \frac{1}{1+e^{-x}}
 * */
inline double sigmoid(double x) { return 1.0 / (1 + exp(-x)); }


/**
 * method : based on fread, single thread
 * time : read (160000, 1000) use about 4.7 seconds
 * */
class BacklightFreadReader {
public:
    BacklightFreadReader(string _filename){
        filename = _filename;
        file = fopen(filename.c_str(), "r");
        REOF = 1;
        p1 = buf;
        p2 = buf;
    }
    BacklightFreadReader(string filename, char _col_separator, char _row_separator){
        col_separator = _col_separator;
        row_separator = _row_separator;
        file = fopen(filename.c_str(), "r");
        REOF = 1;
        p1 = buf;
        p2 = buf;
    }

    int loadcsv(vector<vector<double> > &X){
        col_separator = ',';
        row_separator = '\n';
        return load(X);
    }

    int loadcsv(vector<vector<double> > &X, int row){
        col_separator = ',';
        row_separator = '\n';
        return load(X, row);
    }

    int loadtxt(vector<vector<double> > &X){
        col_separator = ' ';
        row_separator = '\n';
        return load(X);
    }

    int load(vector<vector<double> > &X) {
        double x;
        vector<double> line;
        while(true) {
            int tag = read_double(x);
            if(tag==3){
                line.push_back(x);
                X.push_back(line);
                return 1;
            }
            else if(tag==4){
                return 1;
            }
            else if(tag==1) {
                line.push_back(x);
            }
            else if(tag==2) {
                line.push_back(x);
                X.push_back(line);
                line.clear();
            }
            else {
                return -1;
            }
        }
    }

    
    int load(vector<vector<double> > &X, int row) {
        double x;
        vector<double> line;
        while(true) {
            int tag = read_double(x);
            if(tag==3){
                line.push_back(x);
                X.push_back(line);
                return 1;
            }
            else if(tag==4){
                return 1;
            }
            else if(tag==1) {
                line.push_back(x);
            }
            else if(tag==2) {
                line.push_back(x);
                X.push_back(line);
                line.clear();
                if((int)X.size()==row)return 1;
            }
            else {
                return -1;
            }
        }
    }
private:
    #define BUFFER_SIZE 1<<20
    string filename;
    char col_separator;
    char row_separator;
    int REOF; // 0表示读到文件结尾
    FILE *file;
    char buf[BUFFER_SIZE], *p1, *p2;
    inline char nc(){
        return (p1 == p2) && REOF 
            && (p2 = (p1 = buf) + fread(buf, 1, BUFFER_SIZE, file), p1 == p2) ? (REOF = 0, EOF) : *p1++;
    }

    int read_double(double& x){
        if(REOF==0) return 4;
        x = 0; bool f = false;
        register char ch=nc();
        while(ch<'0' || ch>'9'){f|=(ch=='-');ch=nc();}
        while(ch>='0'&&ch<='9'){x=x*10+(ch^48);ch=nc();}
        if(ch == '.'){
            double tmp=1; ch=nc();
            while(ch>='0' && ch<='9'){tmp=tmp/10.0;x=x+tmp*(ch^48);ch=nc();}
        }
        if(f)x=-x;
        
        if(ch==col_separator) return 1;
        else if(ch==row_separator) return 2;
        else if(REOF==0) return 3;
        return -1;
    }
    #undef BUFFER_SIZE
};


/**
 * use gradient descent to optimize model
 * */
class GradientDescentOptimizer {
public:
    void optimize(const vector<vector<double> > &X, const vector<int> &Y, vector<double> &theta, const double learning_rate){
        int n = (int)X.size(), m = (int)X[0].size();

        // predict
        vector<double> h(n);
        for(int i = 0; i < n; ++i) {
            double score=0;
            for(int j = 0; j < m; ++j)
                score = score + theta[j] * X[i][j];
            h[i] = sigmoid(score);
        }

        // compute gradient
        vector<double> gradient(m);
        for(int i = 0; i < n; ++i) 
            for(int j = 0; j < m; ++j) 
                gradient[j] += (h[i] - Y[i]) * X[i][j];

        // update parameter
        for(int j = 0; j < m; ++j) {
            gradient[j] /= m;
            theta[j] = theta[j] - gradient[j] * learning_rate;
        }
    }

    void minibatch_optimize(const vector<vector<double> > &X, const vector<int> &Y, vector<double> &theta, const double learning_rate, const int batch_size){
        int n = (int)X.size(), m = (int)X[0].size();
        vector<int> posx(n), posy(m);
        for(int i = 0; i < n; ++i) posx[i] = i;
        random_shuffle(posx.begin(), posx.end());

        vector<double> h(batch_size);
        vector<vector<double> > x_bt(batch_size, vector<double>(batch_size));
        vector<int> y_bt(batch_size);

        int st = 0;
        for(int epoch = 0; epoch < n / batch_size; ++epoch) {
            for(int i = 0; i < batch_size; ++i) {
                x_bt[i] = X[posx[st + i]];
                y_bt[i] = Y[posx[st + i]];
            }
            optimize(x_bt, y_bt, theta, learning_rate);
            st += batch_size;
        }
    }
};


/**
 * calculate accuracy or cross entrophy loss
 * */
class CrossEntrophyCalculater {
public:
    // calculater the accuracy
    double accuracy(const vector<int>& Y, const vector<int>& h){
        int n = Y.size(), correct = 0;
        for(int i = 0; i < n; ++i)
            if(Y[i] == h[i])
                correct++;
        return (double) correct / n;
    }

    // calculater the loss
    double loss(const vector<int>& Y, const vector<double>& h) {
        int n = Y.size();
        double loss = 0;
        for(int i = 0; i < n; ++i) 
            loss -= Y[i] * log(h[i]) + (1 - Y[i]) * log(1 - h[i]);
        return loss;
    }
};


/**
 * Model Template
 * Logistic Regression
 * TODO : use extend
 * */
template <class Calculater, class Optimizer>
class Model {
public:
    Model(){};
    Model(int _max_iters, double _learning_rate, int _batch_size = -1) {
        max_iters = _max_iters;
        learning_rate = _learning_rate;
        batch_size = _batch_size;
    }

    void train(vector<vector<double> > &X, vector<int> &Y){
        #ifdef DEBUG
        _dbg("Training...\n");
        clock_t start = clock();
        #endif


        int m = X[0].size();
        theta = vector<double>(m, 0);

        for(int iter = 0; iter < max_iters; ++iter) {
            if(batch_size != -1)
                optimizer.minibatch_optimize(X, Y, theta, learning_rate, batch_size);
            else
                optimizer.optimize(X, Y, theta, learning_rate);


            #ifdef DEBUG
            if(iter % 10 == 0){
                int n = X.size();
                vector<double> h(n);
                for(int i = 0; i < n; ++i) {
                    double score=0;
                    for(int j = 0; j < m; ++j)
                        score = score + theta[j] * X[i][j];
                    h[i] = sigmoid(score);
                }
                vector<int> _h(n);
                for(int i = 0; i < n; ++i) _h[i] = h[i] >= 0.5 ? 1 : 0; 
                double loss = calculater.loss(Y, h);
                double accuracy = calculater.accuracy(Y, _h);
                _dbg("Iter %d : loss = %.2f, accuracy = %.2f\n", iter, loss, accuracy);
                // for(int i = 0; i < (int)theta.size(); ++i) cout << theta[i] << " "; cout << endl;
            }
            #endif
        }


        #ifdef DEBUG
        clock_t end = clock();
        _dbg("Train success, cost %.2f s\n", (double) (end - start) / CLOCKS_PER_SEC);
        #endif
    }

    void predict(vector<vector<double> > &X, vector<int> &Y){
        #ifdef DEBUG
        _dbg("Predict...\n");
        clock_t start = clock();
        #endif
        int n = X.size(), m = X[0].size();
        for(int i = 0; i < n; ++i) {
            double score=0;
            for(int j=0; j < m; ++j)
                score = score + theta[j] * X[i][j];
            Y[i]=sigmoid(score) >= 0.5 ? 1 : 0;
        }
        #ifdef DEBUG
        clock_t end = clock();
        _dbg("Predict success, cost %.2f s\n", (double) (end - start) / CLOCKS_PER_SEC);
        #endif
    }

private:
    int max_iters;
    double learning_rate;
    int batch_size;
    vector<double> theta;
    Calculater calculater;
    Optimizer optimizer;
};


bool loadAnswerData(string awFile, vector<int>& awVec)
{
    ifstream infile(awFile.c_str());
    if (!infile) {
        _dbg("Open file error\n");
        exit(0);
    }

    while (infile) {
        string line;
        int aw;
        getline(infile, line);
        if (line.size() > 0) {
            stringstream sin(line);
            sin >> aw;
            awVec.push_back(aw);
        }
    }

    infile.close();
    return true;
}

// load train data
void loadData(string filename, vector<vector<double> > &X, int row = -1){
    #ifdef DEBUG
    _dbg("Loading data from [ %s ]\n", filename.c_str());
    clock_t start = clock();
    #endif
    BacklightFreadReader in(filename);
    int tag = in.loadcsv(X, row);
    if(tag == -1) {
        _dbg("Load data Error\n");
        exit(0);
    }
    #ifdef DEBUG
    clock_t end = clock();
    _dbg("Load data (%d, %d) succcess, cost %.2f s\n", (int)X.size(), (int)X[0].size(), (double)(end-start) / CLOCKS_PER_SEC);
    #endif
}


/**
 * 归一化处理
 * after cache optimization
 * before : about 3.0 s
 * now : 0.2 s
 * */
void normalize(vector<vector<double> > &X){
    #ifdef DEBUG
    _dbg("Normalizing...\n");
    clock_t start = clock();
    #endif
    int n = X.size(), m = X[0].size();
    vector<double> mean(m), mean2(m), std(m);
    for(int i = 0; i < n; ++i) {
        for(int j = 0; j < m; ++j) {
            mean[j] += X[i][j];
            mean2[j] += X[i][j] * X[i][j];
        }
    }
    for(int j = 0; j < m; ++j) {
        mean[j] /= n;
        mean2[j] /= n;
        std[j] = sqrt(mean2[j] - mean[j] * mean[j]);
    }
    for(int i = 0; i < n; ++i) 
        for(int j = 0; j < m; ++j) 
            X[i][j] = (X[i][j] - mean[j]) / std[j];

    #ifdef DEBUG
    clock_t end = clock();
    _dbg("Normalize (%d, %d) success, cost %.2f s\n", n, m, (double)(end - start) / CLOCKS_PER_SEC);
    #endif
}

// save predict result
void savePredictResult(const string filename, vector<int> &pred){
    #ifdef DEBUG
    _dbg("Saving predict result\n");
    clock_t start = clock();
    #endif
    FILE *output = fopen(filename.c_str(), "w");
    if(output == NULL) {
        _dbg("Open predictFile Error\n");
        return;
    }

    for(int i=0; i < (int)pred.size(); ++i) {
        fputc(pred[i]+'0', output);
        fputc('\n', output);
    }

    fclose(output);

    #ifdef DEBUG
    clock_t end = clock();
    _dbg("Save predict result success, cost %.2f s\n", (double)(end - start) / CLOCKS_PER_SEC);
    #endif
}

int main(int argc, char* argv[])
{
#ifdef DEBUG
    string trainFile = "./data/train_data.txt";
    string testFile = "./data/test_data.txt";
    string predictFile = "./projects/student/result.txt";
    string answerFile = "./projects/student/answer.txt";
    clock_t start = clock();
#else
    string trainFile = "/data/train_data.txt";
    string testFile = "/data/test_data.txt";
    string predictFile = "/projects/student/result.txt";
    string answerFile = "/projects/student/answer.txt";
#endif
    vector<vector<double> > X, X_test;
    vector<int> Y, Y_test;

    loadData(trainFile, X);
    int n = X.size();
    Y.resize(n);
    for(int i = 0; i < n; ++i) {
        Y[i] = X[i].back();
        X[i].pop_back();
    }
    // normalize(X);

    loadData(testFile, X_test);
    int _n = X_test.size();
    Y_test.resize(_n);
    // normalize(X_test);

    // clock_t s1 = clock();
    // for(int i = 0; i < n; ++i) X[i].insert(X[i].begin(), 1);
    // for(int i = 0; i < _n; ++i) X_test[i].insert(X_test[i].begin(), 1);
    // clock_t s2 = clock();
    // _dbg("pb time : %.2f s\n", (double)(s2 - s1) / CLOCKS_PER_SEC);

    Model<CrossEntrophyCalculater, GradientDescentOptimizer>
    model(1, 0.001, 64);

    model.train(X, Y);
    model.predict(X_test, Y_test);
    savePredictResult(predictFile, Y_test);

#ifdef DEBUG
    clock_t end = clock();
    vector<int> answerVec, predictVec;
    _dbg("loading answer\n");
    loadAnswerData(answerFile, answerVec);

    _dbg("loading predict\n");
    loadAnswerData(predictFile, predictVec);

    _dbg("answer size : %d\n", (int)answerVec.size());
    _dbg("predict size : %d\n", (int)predictVec.size());

    int correctCount = 0;
    double accuracy;
    for (int j = 0; j < (int)predictVec.size(); ++j)
        if (answerVec[j] == predictVec[j])
            correctCount++;

    accuracy = ((double)correctCount) / answerVec.size();
    _dbg("accuracy : %.2f\n", accuracy);
    _dbg("total time : %.2f s\n", (double)(end - start) / CLOCKS_PER_SEC);
#endif
    return 0;
}