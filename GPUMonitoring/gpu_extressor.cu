#include <cuda_runtime.h>
#include <cmath>
#include <iostream>
#include <atomic>
#include <thread>
#include <chrono>
#include <nvml.h>

std::atomic<bool> running_3d{ false };
std::atomic<bool> running_mem{ false };
std::atomic<bool> running_stressCopy{ false };

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

__global__ void kernel3D(float* d_out) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    d_out[idx] = sinf((float)idx) * cosf((float)idx);
}

void startStress3D() {
    int device = 0;
    int targetGpuUsagePercentage = 50;
    size_t freeMem, totalMem;
    cudaMemGetInfo(&freeMem, &totalMem);

    const size_t elementSize = sizeof(float);
    size_t targetMem = static_cast<size_t>(freeMem * (targetGpuUsagePercentage / 100.0f));
    float* d_out;

    cudaDeviceProp prop = getCudaDeviceProp(device);
    int maxThreadsPerBlock = prop.maxThreadsPerBlock;

    nvmlInit();
    nvmlDevice_t nvmlDevice;
    nvmlUtilization_t utilization;
    nvmlDeviceGetHandleByIndex(device, &nvmlDevice);

    cudaMalloc(&d_out, targetMem);

    int blocksPerGrid;
    int threadsPerGrid;

    while (true) {
        cudaMemGetInfo(&freeMem, &totalMem);
        nvmlDeviceGetUtilizationRates(nvmlDevice, &utilization);

        if (utilization.gpu < targetGpuUsagePercentage) {
            size_t additionalMem = static_cast<size_t>(freeMem * ((targetGpuUsagePercentage - utilization.gpu) / 100.0f));
            size_t newMemUsage = std::min(additionalMem, freeMem);

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

void startStressMemory() {
    size_t totalMem = getCudaDeviceProp(0).totalGlobalMem;

    void* d_a;
    cudaMalloc(&d_a, totalMem);

    while (true) {}

    cudaFree(d_a);
}

void stopStressMemory() {
    running_mem = false;
}

void startStressCopy() {
    int device = 0;
    size_t totalMem = getCudaDeviceProp(device).totalGlobalMem;

    void* d_data;

    while (true) {
        // Obs: Limitação da quantidade de memória alocada, chega no máximo em 89%
        cudaMalloc(&d_data, totalMem);
        cudaFree(d_data);
    }
}

void stopStressCopy() {
    running_stressCopy = false;
}
