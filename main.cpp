#include <iostream>
#include <thread>
#include <mutex>
#include <vector>
#include <omp.h>
#include <atomic>

#define  STEPS 100000000
#define CACHE_LINE 64u
#define MIN 1
#define MAX 300
#define SEED 100

using namespace std;

typedef double (*R_t)(unsigned*, size_t);
typedef struct experiment_result {
    double result;
    double time_ms;
} experiment_result;

experiment_result run_experiment_random(R_t R) {
    size_t len = 100000;
    unsigned arr[len];
    unsigned seed = SEED;
    double t0 = omp_get_wtime();
    double Res = R((unsigned *)&arr, len);
    double t1 = omp_get_wtime();
    return {
        Res,
        t1 - t0
    };
}

unsigned num_threads = std::thread::hardware_concurrency();

void set_num_threads(unsigned T) {
    omp_set_num_threads(T);
    num_threads = T;
}
unsigned get_num_threads()
{
    return num_threads;
}

double randomize_arr_single(unsigned* V, size_t n){
    uint64_t a = 6364136223846793005;
    unsigned b = 1;
    uint64_t prev = SEED;
    uint64_t sum = 0;

    for (unsigned i=0; i<n; i++){
        uint64_t cur = a*prev + b;
        V[i] = (cur % (MAX - MIN + 1)) + MIN;
        prev = cur;
        sum +=V[i];
    }

    return (double)sum/(double)n;
}

uint64_t getA(unsigned size, uint64_t a){
    uint64_t res = 1;
    for (unsigned i=1; i<=size; i++) res = res * a;
    return res;
}

uint64_t getB(unsigned size, uint64_t a){
    uint64_t* acc = new uint64_t(size);
    uint64_t res = 1;
    acc[0] = 1;
    for (unsigned i=1; i<=size; i++){
        for (unsigned j=0; j<i; j++){
            acc[i] = acc[j] * a;
        }
        res += acc[i];
    }
    free(acc);
    return res;
}

double randomize_arr_fs(unsigned* V, size_t n){
    uint64_t a = 6364136223846793005;
    unsigned b = 1;
    unsigned T;
    uint64_t myA;
    uint64_t myB;
    uint64_t sum = 0;

#pragma omp parallel shared(V, T, myA, myB)
    {
        unsigned t = (unsigned) omp_get_thread_num();
#pragma omp single
        {
            T = (unsigned) get_num_threads();
            myA = getA(T, a);
            myB = getB((T - 1), a)*b;
        }
        uint64_t prev = SEED;
        uint64_t cur;

        for (unsigned i=t; i<n; i += T){
            if (i == t){
                cur = getA(i+1, a)*prev + getB(i, a) * b;
            } else {
                cur = myA*prev + myB;
            }
            V[i] = (cur % (MAX - MIN + 1)) + MIN;
            prev = cur;
        }
    }

    for (unsigned i=0; i<n;i++)
        sum += V[i];

    return (double)sum/(double)n;
}

void show_experiment_result_Rand(R_t Rand) {
    double T1;
    uint64_t a = 6364136223846793005;
    unsigned b = 1;

    double dif = 0;
    double avg = (MAX + MIN)/2;

    printf("%10s\t%10s\t%10s\t%10s\t%10s\n", "Threads", "Result", "Avg", "Difference", "Acceleration");
    for (unsigned T = 1; T <= omp_get_num_procs(); ++T) {
        set_num_threads(T);
        experiment_result R = run_experiment_random(Rand);
        if (T == 1) {
            T1 = R.time_ms;
        }
        dif = avg - R.result;
        printf("%10u\t%10g\t%10g\t%10g\t%10g\n", T, R.result, avg, dif, T1/R.time_ms);
    };
}

int main() {
    printf("Rand omp fs\n");
    show_experiment_result_Rand(randomize_arr_fs);
    printf("Rand single\n");
    show_experiment_result_Rand(randomize_arr_single);

    size_t len = 20;
    unsigned arr[len];

    cout << randomize_arr_single(arr, len) << endl;
    cout << randomize_arr_fs(arr, len) << endl;
}
