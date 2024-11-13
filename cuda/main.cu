#include <stdio.h>
#include <iostream>

__global__ void vectorAdd(float *A, float *B, float *C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

void printGPUInfo() {
    int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
    
    if (error_id != cudaSuccess) {
        std::cout << "Erro ao obter o número de dispositivos CUDA: " 
                  << cudaGetErrorString(error_id) << std::endl;
        return;
    }

    if (deviceCount == 0) {
        std::cout << "Nenhuma GPU CUDA disponível." << std::endl;
    } else {
        std::cout << "Número de GPUs CUDA detectadas: " << deviceCount << std::endl;
        for (int i = 0; i < deviceCount; i++) {
            cudaDeviceProp deviceProp;
            cudaGetDeviceProperties(&deviceProp, i);
            std::cout << "GPU " << i << ": " << deviceProp.name << std::endl;
            std::cout << "  Memória total: " << deviceProp.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
            std::cout << "  Multiprocessadores: " << deviceProp.multiProcessorCount << std::endl;
            std::cout << "  Threads por bloco: " << deviceProp.maxThreadsPerBlock << std::endl;
            std::cout << "  Blocos por grid: " << deviceProp.maxGridSize[0] << std::endl;
        }
    }
}

int getOptimalBlockSize(int deviceID) {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceID);

    // Ajusta o tamanho do bloco com base na capacidade da GPU
    int maxThreadsPerBlock = deviceProp.maxThreadsPerBlock;
    
    // Por simplicidade, vamos usar o máximo permitido por bloco (ou menos se for necessário)
    int blockSize = (maxThreadsPerBlock > 256) ? 256 : maxThreadsPerBlock; 
    return blockSize;
}

void selectGPU(int deviceID) {
    cudaSetDevice(deviceID);
}

int main() {
    // Selecionar a GPU e configurar tamanhos de bloco
    printGPUInfo();
    int deviceID = 0;
    selectGPU(deviceID);
    int blockSize = getOptimalBlockSize(deviceID);

    int N = 1 << 20;
    int numBlocks = (N + blockSize - 1) / blockSize;

    // (Código de alocação de memória e cópia como antes)
    
    // Lançar o kernel usando blockSize ajustado
    vectorAdd<<<numBlocks, blockSize>>>(d_A, d_B, d_C, N);
    
    // (Código de cópia e verificação de resultados como antes)
    
    return 0;
}

// command to create exe -> nvcc main.cu -o main
// execute -> .\main.exe 
