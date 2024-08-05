#include <stdio.h>
#include <cmath>
#include <fstream>
#include <vector>
#include "cuda_runtime.h"

#define BS 256 // 32 64 128 256 1024

#define TYPE float
#define TYPE3 float3

//#define TYPE double
//#define TYPE3 double3

#define T 1.

#define G 6.67e-11
#define tau 0.01

std::string input_file_name = "input64000.txt";
std::string output_file_name = "Trash3.txt";

__device__ void minus_r(TYPE3 a, TYPE* const& b2,
                        TYPE3& res) {
    res.x = a.x - b2[0];
    res.y = a.y - b2[1];
    res.z = a.z - b2[2];
}

__device__ TYPE abs_vec(TYPE3 vec) {
    TYPE result = vec.x * vec.x + vec.y * vec.y + vec.z * vec.z;
    return __dsqrt_rd(result);
}

__global__ void RK2_EulerIt(int N, TYPE* bodies_M, TYPE* bodies_inp_R,
                            TYPE* bodies_inp_V, TYPE* bodies_out_R,
                            TYPE* bodies_out_V, int type) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int locIdx = threadIdx.x; //номер нити в блоке
    int locIdx3 = 3 * locIdx;  // х в лок памяти

    __shared__ TYPE sharedM[BS]; // единая память одля одного блока
    __shared__ TYPE sharedR[3 * BS];

    TYPE3 R;
    TYPE3 V;

    R.x = bodies_inp_R[3 * i];
    R.y = bodies_inp_R[3 * i + 1];
    R.z = bodies_inp_R[3 * i + 2];

    V.x = bodies_inp_V[3 * i];
    V.y = bodies_inp_V[3 * i + 1];
    V.z = bodies_inp_V[3 * i + 2];

    TYPE3 a;
    a.x = 0.;
    a.y = 0.;
    a.z = 0.;

    TYPE buffer = 0.0;

    for (int j = 0; j < N; j += BS) {

        sharedM[locIdx] = bodies_M[j + locIdx];

        sharedR[locIdx3] = bodies_inp_R[3 * (j + locIdx)]; // копируем в шаред здачения из хоста
        sharedR[locIdx3 + 1] = bodies_inp_R[3 * (j + locIdx) + 1];
        sharedR[locIdx3 + 2] = bodies_inp_R[3 * (j + locIdx) + 2];

        __syncthreads();

        for (int k = 0; k < BS; k++) {
            if ((i != (j + k)) && (j + k < N)) {
                TYPE3 r_n;
                minus_r(R, sharedR + 3 * k, r_n);
                TYPE r_n_abs3 = abs_vec(r_n);
                r_n_abs3 = r_n_abs3 * r_n_abs3 * r_n_abs3;

                buffer = G *  sharedM[k] / r_n_abs3;

                a.x = a.x - buffer * r_n.x ;
                a.y = a.y - buffer * r_n.y;
                a.z = a.z - buffer * r_n.z ;


            }
        }
        __syncthreads();
    }
    if (i < N) {
        if (type == 0) {
            bodies_out_V[3 * i] = V.x + tau * a.x * 0.5;
            bodies_out_R[3 * i] = R.x + tau * V.x * 0.5;

            bodies_out_V[3 * i + 1] = V.y + tau * a.y * 0.5;
            bodies_out_R[3 * i + 1] = R.y + tau * V.y * 0.5;

            bodies_out_V[3 * i + 2] = V.z + tau * a.z * 0.5;
            bodies_out_R[3 * i + 2] = R.z + tau * V.z * 0.5;

        } else if (type == 1) {
            bodies_out_V[3 * i] = bodies_out_V[3 * i] + tau * a.x;
            bodies_out_R[3 * i] = bodies_out_R[3 * i] + tau * V.x;

            bodies_out_V[3 * i + 1] = bodies_out_V[3 * i + 1] + tau * a.y;
            bodies_out_R[3 * i + 1] = bodies_out_R[3 * i + 1] + tau * V.y;

            bodies_out_V[3 * i + 2] = bodies_out_V[3 * i + 2] + tau * a.z;
            bodies_out_R[3 * i + 2] = bodies_out_R[3 * i + 2] + tau * V.z;
        }
    }
}

void RK2_GPU(int N, TYPE* bodies_M, TYPE* bodies_R, TYPE* bodies_V) {
    std::ofstream out_f("ExTraj1.txt");
    TYPE *dev_M, *dev_R, *dev_V;
    TYPE *half_it_R, *half_it_V;

    dim3 blocks = ((N + BS - 1) / BS);  // структура для размера блока
    dim3 threads(BS); // размерность потока

    if (cudaSuccess != cudaMalloc(&dev_M, N * sizeof(TYPE)))
        printf("Error in cudaMalloc for dev_M\n");
    if (cudaSuccess != cudaMalloc(&dev_R, 3 * N * sizeof(TYPE)))
        printf("Error in cudaMalloc for dev_R\n");
    if (cudaSuccess != cudaMalloc(&dev_V, 3 * N * sizeof(TYPE)))
        printf("Error in cudaMalloc for dev_V\n");
    if (cudaSuccess != cudaMalloc(&half_it_R, 3 * N * sizeof(TYPE)))
        printf("Error in cudaMalloc for dev_R\n");
    if (cudaSuccess != cudaMalloc(&half_it_V, 3 * N * sizeof(TYPE)))
        printf("Error in cudaMalloc for dev_V\n");

    if (cudaSuccess !=
        cudaMemcpy(dev_M, bodies_M, N * sizeof(TYPE), cudaMemcpyHostToDevice))
        printf("Error in cudaMemcpy for dev_M\n");
    if (cudaSuccess !=
        cudaMemcpy(dev_R, bodies_R, 3 * N * sizeof(TYPE), cudaMemcpyHostToDevice))
        printf("Error in cudaMemcpy for dev_R\n");
    if (cudaSuccess !=
        cudaMemcpy(dev_V, bodies_V, 3 * N * sizeof(TYPE), cudaMemcpyHostToDevice))
        printf("Error in cudaMemcpy for dev_V\n");

    cudaEvent_t start, finish; // для времени
    cudaEventCreate(&start);
    cudaEventCreate(&finish);
    cudaEventRecord(start);
    cudaEventSynchronize(start);

    int timesteps = round(T / tau);
    out_f << 0 << " " << bodies_R[0] << " " << bodies_R[1] << " " << bodies_R[2]
          << std::endl;
    for (int i = 1; i <= timesteps; ++i) {
        RK2_EulerIt <<<blocks, threads >>> (N, dev_M, dev_R, dev_V, half_it_R,
                                         half_it_V, 0);
        RK2_EulerIt<<<blocks, threads>>>(N, dev_M, half_it_R, half_it_V, dev_R,
                                         dev_V, 1);
        if (cudaSuccess != cudaMemcpy(bodies_R, dev_R, 3 * N * sizeof(TYPE),
                                      cudaMemcpyDeviceToHost))
            printf("Error in cudaMemcpy for dev to Host\n");
        out_f << tau * i << " " << bodies_R[0] << " " << bodies_R[1] << " "
              << bodies_R[2] << std::endl;
    }

    cudaEventRecord(finish);
    cudaEventSynchronize(finish);

    float dt;
    cudaEventElapsedTime(&dt, start, finish);
    cudaEventDestroy(start);
    cudaEventDestroy(finish);

    cudaFree(dev_M);
    cudaFree(dev_R);
    cudaFree(dev_V);

    if(sizeof ( TYPE ) == 4 ) {
       printf("mytype is float\n");
    }
    if (sizeof ( TYPE ) == 8 ) {
        printf("mytype is double\n");
    }
    printf("block size = %d,\nN = %d\n", BS, N);
    printf("GPU time = %f\n", dt / 1000.0);
}

int main(int argc, char** argv) {
    int N = 0;
    std::ifstream in_f(input_file_name);
    in_f >> N;

    TYPE* bodies_M = new TYPE[N];
    TYPE* bodies_R = new TYPE[3 * N];
    TYPE* bodies_V = new TYPE[3 * N];

    for (int i = 0; i < N; ++i) {
        in_f >> bodies_M[i] >> bodies_R[3 * i] >> bodies_R[3 * i + 1] >>
             bodies_R[3 * i + 2] >> bodies_V[3 * i] >> bodies_V[3 * i + 1] >>
             bodies_V[3 * i + 2];
    }
    RK2_GPU(N, bodies_M, bodies_R, bodies_V);

    delete[] bodies_M;
    delete[] bodies_R;
    delete[] bodies_V;
    return 0;
}