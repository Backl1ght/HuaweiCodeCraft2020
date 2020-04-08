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
#include <sched.h>
#include <sys/wait.h>
#include <sys/mman.h>
#include <arm_neon.h>
using namespace std;

// #define BACKLIGHT

#ifdef BACKLIGHT
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

const int N = 1600;
const int M = 1000;
const int _N = 20000;
const int PROCESSES_NUM = 4;
const int BATCH_SIZE = _N / PROCESSES_NUM;
const int FILE_SIZE = 2 * _N * sizeof(char);

int32_t read_result[M], mean[2][M], cnt[2];

inline void Train(const char* filename) {
    #ifdef BACKLIGHT
    clock_t start = clock();
    #endif

    uint8_t label;

    int fd = open(filename, O_RDONLY);
    struct stat fs;
    fstat(fd, &fs);

    uint8_t *p = (uint8_t*)mmap(NULL, fs.st_size, PROT_READ, MAP_PRIVATE, fd, 0);

    for(int i = 0; i < N; ++i) {
        for(int j = 0; j < M; ++j){
            while(*p ^ '.')++p;
            // read_result[j] = ((*(p+1) ^ 48) << 3 ) + ((*(p+1) ^ 48) << 1 ) + (*(p+2) ^ 48);
            read_result[j] = (int32_t)*(p+1) << 5;
            p = p + 5;
        }
        label = *(p) ^ 48;
        ++cnt[label];
        for(int j = 0; j < M; j++) mean[label][j] += read_result[j];
    }

    for(int j = 0; j < M; ++j) mean[0][j] /= cnt[0];
    for(int j = 0; j < M; ++j) mean[1][j] /= cnt[1];

    #ifdef BACKLIGHT
    clock_t end = clock();
    _dbg("Train (%d, %d) success, cost %.5f s\n", N, M, (float)(end - start) / CLOCKS_PER_SEC);
    #endif
}

bool loadAnswerData(const char* awFile, vector<int>& awVec) {
    ifstream infile(awFile);
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

// save predict result
inline void savePredictResult(const char* filename, char* Y_test) {
    #ifdef BACKLIGHT
    _dbg("Saving predict result\n");
    clock_t start = clock();
    #endif
    FILE *output = fopen(filename, "w");
    if(output == NULL) {
        _dbg("Open predictFile Error\n");
        return;
    }

    for(int i=0; i < _N; ++i) {
        fputc(Y_test[i], output);
        fputc('\n', output);
    }

    fclose(output);

    #ifdef BACKLIGHT
    clock_t end = clock();
    _dbg("Save predict result success, cost %.5f s\n", (float)(end - start) / CLOCKS_PER_SEC);
    #endif
}

int main(int argc, char* argv[])
{
#ifdef BACKLIGHT
    const char* trainfile = "./data/train_data.txt";
    const char* testfile = "./data/test_data.txt";
    const char* predictfile = "./projects/student/result.txt";
    const char* answerfile = "./projects/student/answer.txt";
    clock_t start = clock();
#else
    const char* trainfile = "/data/train_data.txt";
    const char* testfile = "/data/test_data.txt";
    const char* predictfile = "/projects/student/result.txt";
    const char* answerfile = "/projects/student/answer.txt";
#endif

    Train(trainfile);

    int wfd = open(predictfile, O_CREAT|O_WRONLY, 0666);
    ftruncate(wfd, FILE_SIZE);

    uint8_t* w = (uint8_t*)mmap(NULL, FILE_SIZE, PROT_WRITE, MAP_PRIVATE, wfd, 0);

    // Predict(testfile);
    int fd = open(testfile, O_RDONLY);

    struct stat fs;
    fstat(fd, &fs);

    uint8_t *p = (uint8_t*)mmap(NULL, fs.st_size, PROT_READ, MAP_PRIVATE, fd, 0);

    pid_t p0 = fork();
    if(p0 == 0) {

        uint8_t* q0 = p;
        int pos0 = 0;
        for(int i = 0; i < BATCH_SIZE; ++i) {
            int32_t dist0 = 0, dist1 = 0;

            for(int j = 0; j < M ; ++j){
                int32_t x = (int32_t)*(q0 + 2) << 5;
                q0 += 6;
                dist0 += (int32_t)(x - mean[0][j]) * (x - mean[0][j]);
                dist1 += (int32_t)(x - mean[1][j]) * (x - mean[1][j]);
            }

            w[ pos0 ] = (dist0 < dist1 ? '0' : '1');
            w[ pos0 + 1 ] = '\n';
            pos0 += 2;
        }

        exit(0);
    }

    pid_t p1 = fork();
    if(p1 == 0) {

        uint8_t* q1 = p + BATCH_SIZE * 6000;
        int pos1 = 2 * BATCH_SIZE;
        for(int i = 0; i < BATCH_SIZE; ++i) {
            int32_t dist0 = 0, dist1 = 0;

            for(int j = 0; j < M ; ++j){
                int32_t x = (int32_t)*(q1 + 2) << 5;
                q1 += 6;
                dist0 += (int32_t)(x - mean[0][j]) * (x - mean[0][j]);
                dist1 += (int32_t)(x - mean[1][j]) * (x - mean[1][j]);
            }

            w[ pos1 ] = (dist0 < dist1 ? '0' : '1');
            w[ pos1 + 1 ] = '\n';
            pos1 += 2;
        }

        exit(0);
    }

    pid_t p2 = fork();
    if( p2 == 0) {

        uint8_t* q2 = p + 2 * BATCH_SIZE * 6000;
        int pos2 = 4 * BATCH_SIZE;
        for(int i = 0; i < BATCH_SIZE; ++i) {
            int32_t dist0 = 0, dist1 = 0;

            for(int j = 0; j < M ; ++j){
                int32_t x = (int32_t)*(q2 + 2) << 5;
                q2 += 6;
                dist0 += (int32_t)(x - mean[0][j]) * (x - mean[0][j]);
                dist1 += (int32_t)(x - mean[1][j]) * (x - mean[1][j]);
            }

            w[ pos2 ] = (dist0 < dist1 ? '0' : '1');
            w[ pos2 + 1 ] = '\n';
            pos2 += 2;
        }

        exit(0);
    }

    pid_t p3 = fork();
    if( p3 == 0) {

        uint8_t* q3 = p + 3 * BATCH_SIZE * 6000;
        int pos3 = 6 * BATCH_SIZE;
        for(int i = 0; i < BATCH_SIZE; ++i) {
            int32_t dist0 = 0, dist1 = 0;

            for(int j = 0; j < M ; ++j){
                int32_t x = (int32_t)*(q3 + 2) << 5;
                q3 += 6;
                dist0 += (int32_t)(x - mean[0][j]) * (x - mean[0][j]);
                dist1 += (int32_t)(x - mean[1][j]) * (x - mean[1][j]);
            }

            w[ pos3 ] = (dist0 < dist1 ? '0' : '1');
            w[ pos3 + 1 ] = '\n';
            pos3 += 2;
        }
        
        exit(0);
    }


    int st0, st1, st2, st3;
    waitpid(p0, &st0, 0);
    waitpid(p1, &st1, 0);
    waitpid(p2, &st2, 0);
    waitpid(p3, &st3, 0);


    msync((void*)w, FILE_SIZE, MS_ASYNC);
    munmap((void*)w, FILE_SIZE);

#ifdef BACKLIGHT
    clock_t end = clock();
    vector<int> answerVec, predictVec;
    _dbg("loading answer\n");
    loadAnswerData(answerfile, answerVec);

    _dbg("loading predict\n");
    loadAnswerData(predictfile, predictVec);

    _dbg("answer size : %d\n", (int)answerVec.size());
    _dbg("predict size : %d\n", (int)predictVec.size());

    int correctCount = 0;
    float accuracy;
    for (int j = 0; j < (int)predictVec.size(); ++j)
        if (answerVec[j] == predictVec[j])
            correctCount++;

    accuracy = ((float)correctCount) / answerVec.size();
    _dbg("accuracy : %.5f\n", accuracy);
    _dbg("total time : %.5f s\n", (float)(end - start) / CLOCKS_PER_SEC);
#endif

    return 0;
}