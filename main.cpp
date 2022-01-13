#include <iostream>
#include <thread>
#include <mutex>
#include <vector>
#include <omp.h>
#include <atomic>

#define SEED 69

static unsigned num_treads = std::thread::hardware_concurrency();
unsigned get_num_threads()
{
    return num_treads;
}

void set_num_threads(unsigned t)
{
    num_treads = t;
    omp_set_num_threads(t);
}
struct experiment_result
{
    double Result;
    double TimeMS;
};

double Quadratic(double x)
{
    return x*x;
}


typedef double (*randomize_function) (unsigned, unsigned*, size_t, unsigned, unsigned);

double RandomizeArraySingle(unsigned seed, unsigned* V, size_t n, unsigned min, unsigned max)
{
    uint64_t A = 6364136223846793005;
    unsigned B = 1;

    uint64_t prev = seed;
    uint64_t Sum = 0;

    for(unsigned int i = 0; i < n; i++)
    {
        uint64_t next = A*prev + B;
        V[i] = (next % (max - min + 1)) + min;
        prev = next;
        Sum += V[i];
        std::cout<<V[i]<<" ";
    }

    return (double)Sum/(double)n;
}

uint64_t pow(uint64_t base, uint64_t exp)
{
    uint64_t res = 1;
    for (unsigned i = 0; i < exp; i++)
    {
        res *= base;
    }
    return res;
}

uint64_t getB(unsigned exp, uint64_t A, uint64_t B)
{
    uint64_t sum = 0;
    for (unsigned i = 0; i <= exp; i++)
    {
        sum += pow(A, i);
    }

    if (sum == 0) {
        return B;
    } else {
        return B*sum;
    }
}

double RandomizeArrayShared(unsigned seed, unsigned* V, size_t n, unsigned min, unsigned max)
{
    uint64_t A = 6364136223846793005;
    uint64_t B = 1;
    unsigned T;
    uint64_t findA, findB;
    uint64_t Sum = 0;

#pragma omp parallel shared(T, V, findA, findB)
    {
        unsigned t = (unsigned) omp_get_thread_num();
#pragma omp single
        {
            T = (unsigned) omp_get_num_threads();
            findA = pow(A, T);
            findB = getB(T - 1, A, B);
        }
        uint64_t prev = seed;
        uint64_t elem;
        for (unsigned i = t; i < n; i += T)
        {
            if (i == t)
            {
                elem = pow(A, i + 1) * prev + getB(i, A, B);
            } else
            {
                elem = findA * prev + findB;
            }
            V[i] = (elem % (max - min + 1)) + min;
            prev = elem;
        }
    }

    for (unsigned i = 0; i < n; i++)
    {
        Sum += V[i];
    }

    return (double)Sum/(double)n;
}

experiment_result RandomizerExperiment (randomize_function f)
{
    size_t ArrayLength = 500000;
    unsigned Array[ArrayLength];
    unsigned seed = SEED;

    double t0, t1, result;

    t0 = omp_get_wtime();
    result = f(seed, (unsigned *)&Array, ArrayLength, 1, 255);
    t1 = omp_get_wtime();

    return {result, t1 - t0};
}

void ShowExperimentResultRand(randomize_function f)
{
    double T1;
    printf("%10s. %10s %10sms %10s\n", "Threads", "Result", "Time", "Acceleration");
    for(unsigned T = 1; T <=omp_get_num_procs(); T++)
    {
        experiment_result Experiment;
        set_num_threads(T);
        Experiment = RandomizerExperiment(f);
        if (T == 1) {
            T1 = Experiment.TimeMS;
        }
        printf("%10d. %10g %10gms %10g\n", T, Experiment.Result, Experiment.TimeMS, T1/Experiment.TimeMS);
    }
    printf("\n");
}

int main() {
    ShowExperimentResultRand(RandomizeArrayShared);
}
