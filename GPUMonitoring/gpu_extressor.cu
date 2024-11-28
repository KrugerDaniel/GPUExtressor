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

    __declspec(dllexport) void startStress3D(int device, int targetGpuUsagePercentage);
    __declspec(dllexport) void stopStress3D();

    __declspec(dllexport) void startStressMemory(int device);
    __declspec(dllexport) void stopStressMemory();

    __declspec(dllexport) void startStressCopy(int device);
    __declspec(dllexport) void stopStressCopy();
}

cudaDeviceProp getCudaDeviceProp(int device) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    return prop;
}

__global__ void kernel3D(float* d_out, int maxIdx) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < maxIdx) {
        d_out[idx] = sinf((float)idx) * cosf((float)idx);
    }
}

void startStress3D(int device, int targetGpuUsagePercentage) {
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

    int threadsPerBlock = maxThreadsPerBlock;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    running_3d = true;

    std::thread stressThread([=]() mutable {
        float integral = 0.0f;

        while (running_3d) {
            nvmlUtilization_t utilization;
            nvmlDeviceGetUtilizationRates(nvmlDevice, &utilization);

            float currentGpuUsage = static_cast<float>(utilization.gpu);
            float error = static_cast<float>(targetGpuUsagePercentage) - currentGpuUsage;

            integral += error;
            float adjustment = error * 0.1f + integral * 0.0002f;

            int dynamicBlocks = static_cast<int>(blocksPerGrid * (targetGpuUsagePercentage / 100.0f) + adjustment);

            kernel3D<<<dynamicBlocks, threadsPerBlock>>>(d_out, N);
            cudaDeviceSynchronize();

            if (targetGpuUsagePercentage != 100) {
                int sleepTime = std::max(20, 60 - static_cast<int>(std::abs(error) * 5));
                std::this_thread::sleep_for(std::chrono::milliseconds(sleepTime));
            }
        }

        cudaFree(d_out);
        nvmlShutdown();
    });


    stressThread.detach();
}

void stopStress3D() {
    running_3d = false;
}

void startStressMemory(int device) {
    size_t totalMem = getCudaDeviceProp(device).totalGlobalMem;

    float* d_a;
    cudaMalloc((void**)&d_a, totalMem);

    while (true) {}

    cudaFree(d_a);
}

void stopStressMemory() {
    running_mem = false;
}

void startStressCopy(int device) {
    size_t totalMem = getCudaDeviceProp(device).totalGlobalMem;

    void* d_data;

    while (true) {
        cudaMalloc(&d_data, totalMem);
        cudaFree(d_data);
    }
}

void stopStressCopy() {
    running_stressCopy = false;
}
