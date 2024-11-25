#include <cuda_runtime.h>
#include <cmath>
#include <iostream>
#include <atomic>
#include <thread>
#include <chrono>
#include <nvml.h>

// Variáveis atômicas para controle de execução
std::atomic<bool> running_3d{ false };
std::atomic<bool> running_mem{ false };
std::atomic<bool> running_stressCopy{ false };

// Encapsular todas as funções exportadas em extern "C"
extern "C" {

    __declspec(dllexport) void startStress3D();
    __declspec(dllexport) void stopStress3D();

    __declspec(dllexport) void startStressMemory();
    __declspec(dllexport) void stopStressMemory();

    __declspec(dllexport) void startStressCopy();
    __declspec(dllexport) void stopStressCopy();
}

cudaDeviceProp getCudaDeviceProp(int device) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    return prop;
}

// Kernel de estresse 3D
__global__ void stress3DKernel(float* output, int width, int height, int iterations) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int idx = x + y * width;

    if (x < width && y < height) {
        float value = 0.0f;
        for (int i = 0; i < iterations; ++i) {
            value += sinf(x * y + i) * cosf(x - y + i);
        }
        output[idx] = value;
    }
}

// Função para estressar a GPU com operações de 3D (realiza cálculos simples em paralelo)
__global__ void kernel3D(float* d_out) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < 1 << 28) {  // Aumentando o número de threads processados
        d_out[idx] = sinf((float)idx) * cosf((float)idx);  // Simples cálculo trigonométrico
    }
}

// Controle do estresse 3D
void startStress3D() {
    int device = 0;
    int targetGpuUsagePercentage = 50;
    size_t freeMem, totalMem;
    cudaMemGetInfo(&freeMem, &totalMem);

    const size_t elementSize = sizeof(float);
    size_t targetMem = static_cast<size_t>(freeMem * (targetGpuUsagePercentage / 100.0f));
    int N = targetMem / elementSize;
    float* d_out = nullptr;

    cudaDeviceProp prop = getCudaDeviceProp(device);
    int maxThreadsPerBlock = prop.maxThreadsPerBlock;

    nvmlInit();
    nvmlDevice_t nvmlDevice;
    nvmlDeviceGetHandleByIndex(device, &nvmlDevice);

    cudaMalloc((void**)&d_out, N * elementSize);

    int blocksPerGrid;
    int threadsPerGrid;

    while (true) {
        // Monitorar uso atual da GPU
        cudaMemGetInfo(&freeMem, &totalMem);

        // Monitorar uso atual da GPU
        nvmlUtilization_t utilization;
        nvmlDeviceGetUtilizationRates(nvmlDevice, &utilization);

        if (utilization.gpu < targetGpuUsagePercentage) {
            // GPU abaixo do alvo, aumentar carga
            size_t additionalMem = static_cast<size_t>(freeMem * ((targetGpuUsagePercentage - utilization.gpu) / 100.0f));
            size_t newMemUsage = std::min(additionalMem, freeMem); // Garantir que não exceda a memória livre

            blocksPerGrid = (freeMem / elementSize + maxThreadsPerBlock - 1) / maxThreadsPerBlock;
            blocksPerGrid = static_cast<size_t>(blocksPerGrid * ((targetGpuUsagePercentage - utilization.gpu) / 100.0f));
            threadsPerGrid = static_cast<size_t>(maxThreadsPerBlock * ((targetGpuUsagePercentage - utilization.gpu) / 100.0f));

            kernel3D<<<blocksPerGrid, threadsPerGrid>>>(d_out);
            cudaDeviceSynchronize();
        }
    }

    cudaFree(d_out);
    nvmlShutdown();
}

void stopStress3D() {
    running_3d = false;
}

// Kernel de estresse de memória
__global__ void stressMemoryKernel(float* d_mem, int size, int iterations) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        for (int i = 0; i < iterations; ++i) {
            d_mem[idx] = sinf(d_mem[idx]) + cosf(d_mem[idx]);
        }
    }
}

// Controle do estresse de memória
void startStressMemory() {
    size_t totalMem = getCudaDeviceProp(0).totalGlobalMem;

    float* d_a;
    float* d_b;
    cudaMalloc((void**)&d_a, totalMem);
    cudaMalloc((void**)&d_b, totalMem);

    while (true) {}

    cudaFree(d_a);
    cudaFree(d_b);
}

void stopStressMemory() {
    running_mem = false;
}

// Kernel de multiplicação de matrizes
__global__ void matrixMulKernel(float* A, float* B, float* C, int N) {
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    if (row < N && col < N) {
        float value = 0.0f;
        for (int k = 0; k < N; ++k) {
            value += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = value;
    }
}

void startStressCopy() {
    int device = 0;
    size_t totalMem = getCudaDeviceProp(device).totalGlobalMem;

    void* d_data;

    while (true) {
        // Limitação da quantidade de memória alocada, chega no máximo em 89%
        cudaMalloc(&d_data, totalMem);
        cudaFree(d_data);
    }
}

void stopStressCopy() {
    running_stressCopy = false;
}
