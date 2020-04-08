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
#include <cstring>
#include <sys/mman.h>
#include <arm_neon.h>
using namespace std;

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

const int N = 1100;
const int M = 1000;
const int _N = 20000;

const uint8_t T[16] = {0, 10, 0, 10, 0, 10, 0, 10, 0, 10, 0, 10, 0, 10, 0, 10};
const uint8_t T2[16] = {100, 1, 100, 1, 100, 1, 100, 1, 100, 1, 100, 1, 100, 1, 100, 1};
const uint8_t S1[16] = {64, 240, 64, 240, 64, 240, 64, 240, 64, 240, 64, 240, 64, 240, 64, 240};

short read_result[M+5];
// 一次读入8个浮点数保留两位小数
// p要指向前一行的'\n'
void read(uint8_t *p) {
    uint8x16x3_t cval;
    int16x8_t ca;
    uint8x16_t M1 = vld1q_u8(T),M2 = vld1q_u8(T2),M3 = vld1q_u8(S1);
    for(int i = 0; i < M; i += 8){
        cval = vld3q_u8(p);
        ca = vreinterpretq_s16_u16(vpaddlq_u8(vaddq_u8(M3, vaddq_u8(vmulq_u8(cval.val[0],M1) , vmulq_u8(cval.val[1],M2)))));
        vst1q_s16(read_result+i,ca);
        p += 48;
    }
}

char Y_test[_N];
int32_t mean[2][M], cnt[2];

void Train(const char* filename) {
    #ifdef DEBUG
    clock_t start = clock();
    #endif

    int fd = open(filename, O_RDONLY);

    struct stat fs;
    fstat(fd, &fs);

    uint8_t *p = (uint8_t*)mmap(NULL, fs.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    uint8_t *q = new uint8_t[7000];
    short label;

    q[0]='\n';
    short sz = 1;
    while(sz < 6000){
        if((*p) ^ '-') q[sz++] = *p;
        ++p;
    }
    read(q);

    label = (*(p+1)) ^ 48;

    ++cnt[label];
    // for(int j = 0; j < M; ++j) {
    //     mean[label][j] += read_result[j];
    // }

    for(int j = 0; j < M; j+=4) {
        int32x4_t a = vld1q_s32(mean[label] + j);
        int32x4_t b = vmovl_s16(vld1_s16(read_result + j));
        vst1q_s32(mean[label]+j, vqaddq_s32(a, b));
    }

    p = p + 2;

    for(int i= 1; i ^ N; ) {
        if(*(p + 6002) ^ '\n') {
            p = p + 6002;
            while((*p) ^ '\n')++p;
        }
        else{
            read(p);
            p = p + 6002;
            label = (*(p-1)) ^ 48;

            ++cnt[label];
            // for(int j = 0, st = 0; j < M; ++j) {
            //     mean[label][j] += read_result[j];
            // }
            for(int j = 0; j < M; j+=4) {
                int32x4_t a = vld1q_s32(mean[label] + j);
                int32x4_t b = vmovl_s16(vld1_s16(read_result + j));
                vst1q_s32(mean[label]+j, vqaddq_s32(a, b));
            }

            ++i;
        }
    }

    for(int j = 0; j < M; ++j) {
        mean[0][j] /= cnt[0];
        mean[1][j] /= cnt[1];
    }

    
    #ifdef DEBUG
    clock_t end = clock();
    _dbg("Train (%d, %d) success, cost %.5f s\n", N, M, (float)(end - start) / CLOCKS_PER_SEC);
    #endif
}




void Predict(const char* filename) {
    #ifdef DEBUG
    clock_t start = clock();
    #endif

    int fd = open(filename, O_RDONLY);

    struct stat fs;
    fstat(fd, &fs);

    uint8_t *p = (uint8_t*)mmap(NULL, fs.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    uint8_t *q = new uint8_t[7000];
    uint32_t dist0, dist1;

    // read first line
    q[0]='\n';
    short sz = 1;
    while(sz<6000){
        if(*p^'-') q[sz++] = *p;
        ++p;
    }
    read(q);
    dist0 = 0; dist1 = 0;
    for(int j = 0; j < M; ++j) {
        dist0 += (read_result[j] - mean[0][j]) * (read_result[j] - mean[0][j]);
        dist1 += (read_result[j] - mean[1][j]) * (read_result[j] - mean[1][j]);
    }
    
    // for(int j = 0; j < M; j += 4){
    //     int32x4_t a = vmovl_s16(vld1_s16(read_result + j));

    //     int32x4_t b = vld1q_s32(mean[0] + j);
    //     int32x4_t c = vsubq_s32(a, b);
    //     int32x4_t d = vmulq_s32(c, c);
    //     dist0 += vgetq_lane_s32(d, 0);
    //     dist0 += vgetq_lane_s32(d, 1);
    //     dist0 += vgetq_lane_s32(d, 2);
    //     dist0 += vgetq_lane_s32(d, 3);

    //     b = vld1q_s32(mean[1] + j);
    //     c = vsubq_s32(a, b);
    //     d = vmulq_s32(c, c);
    //     dist1 += vgetq_lane_s32(d, 0);
    //     dist1 += vgetq_lane_s32(d, 1);
    //     dist1 += vgetq_lane_s32(d, 2);
    //     dist1 += vgetq_lane_s32(d, 3);
    // }
    
    Y_test[0] = (dist0 < dist1 ? '0' : '1');

    for(int i= 1; i ^ _N; ++i) {
        read(p);
        p = p + 6000;

        dist0 = 0; dist1 = 0;
        for(int j = 0; j < M; ++j) {
            dist0 += (read_result[j] - mean[0][j]) * (read_result[j] - mean[0][j]);
            dist1 += (read_result[j] - mean[1][j]) * (read_result[j] - mean[1][j]);
        }

        // for(int j = 0; j < M; j += 4){
        //     int32x4_t a = vmovl_s16(vld1_s16(read_result + j));

        //     int32x4_t b = vld1q_s32(mean[0] + j);
        //     int32x4_t c = vsubq_s32(a, b);
        //     int32x4_t d = vmulq_s32(c, c);
        //     dist0 += vgetq_lane_s32(d, 0);
        //     dist0 += vgetq_lane_s32(d, 1);
        //     dist0 += vgetq_lane_s32(d, 2);
        //     dist0 += vgetq_lane_s32(d, 3);

        //     b = vld1q_s32(mean[1] + j);
        //     c = vsubq_s32(a, b);
        //     d = vmulq_s32(c, c);
        //     dist1 += vgetq_lane_s32(d, 0);
        //     dist1 += vgetq_lane_s32(d, 1);
        //     dist1 += vgetq_lane_s32(d, 2);
        //     dist1 += vgetq_lane_s32(d, 3);
        // }
        Y_test[i] = (dist0 < dist1 ? '0' : '1');
    }

    #ifdef DEBUG
    clock_t end = clock();
    _dbg("Predict (%d, %d) success, cost %.5f s\n", _N, M, (float)(end - start) / CLOCKS_PER_SEC);
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
void savePredictResult(const char* filename) {
    #ifdef DEBUG
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

    #ifdef DEBUG
    clock_t end = clock();
    _dbg("Save predict result success, cost %.5f s\n", (float)(end - start) / CLOCKS_PER_SEC);
    #endif
}

int main(int argc, char* argv[])
{
#ifdef DEBUG
    const char* trainfile = "./data/train_data.txt";
    const char* testfile = "./data/test_data.txt";
    const char* predictfile = "./projects/student/result.txt";
    const char* answerfile = "./projects/student/answer.txt";
    clock_t start = clock();
#else
    const char* trainfile = "./data/train_data.txt";
    const char* testfile = "./data/test_data.txt";
    const char* predictfile = "./projects/student/result.txt";
    const char* answerfile = "./projects/student/answer.txt";
#endif

    Train(trainfile);
    
    Predict(testfile);

    savePredictResult(predictfile);

#ifdef DEBUG
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