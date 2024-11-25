#include <iostream>
#include <cuda_runtime.h>
#include <cuda.h>
#include <vector>
#include <atomic>
#include <thread>
#include <nvml.h>

// Comando mágico : nvcc gpu_extressor.cu -o gpu_extressor -L"C:\Program Files\NVIDIA Corporation\NVSMI" -lnvml

// Variável global para controlar quando parar o estresse
std::atomic<bool> stopStress(false);

cudaDeviceProp getCudaDeviceProp(int device) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    return prop;
}

// Função para obter informações da GPU
void getGPUInfo() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0) {
        std::cerr << "Nenhuma GPU CUDA disponível." << std::endl;
        return;
    }

    for (int device = 0; device < deviceCount; ++device) {
        cudaDeviceProp prop = getCudaDeviceProp(device);
        std::cout << "Informações da GPU " << device << ":\n";
        std::cout << "  Nome: " << prop.name << std::endl;
        std::cout << "  Memória Global: " << prop.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
        std::cout << "  Arquitetura: " << prop.major << "." << prop.minor << std::endl;
        std::cout << "  Núcleos de Processamento: " << prop.multiProcessorCount << std::endl;
        std::cout << "  Frequência: " << prop.clockRate / 1000 << " MHz" << std::endl;
        std::cout << std::endl;
    }
}

// Função para estressar a GPU com operações de cópia de memória
void stressMemory() {
    size_t totalMem = getCudaDeviceProp(0).totalGlobalMem;

    float* d_a;
    float* d_b;
    cudaMalloc((void**)&d_a, totalMem);
    cudaMalloc((void**)&d_b, totalMem);

    while (!stopStress) {}

    cudaFree(d_a);
    cudaFree(d_b);
}

// Função para estressar a GPU com operações de memória (alocação e desalocação repetidas)
void stressCopy(int device) {
    size_t totalMem = getCudaDeviceProp(device).totalGlobalMem;
    
    void* d_data;

    while (!stopStress) {
        // Limitação da quantidade de memória alocada, chega no máximo em 89%
        cudaMalloc(&d_data, totalMem);
        cudaFree(d_data);
    }
}

// Função para estressar a GPU com operações de 3D (realiza cálculos simples em paralelo)
__global__ void kernel3D(float* d_out) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < 1 << 28) {  // Aumentando o número de threads processados
        d_out[idx] = sinf((float)idx) * cosf((float)idx);  // Simples cálculo trigonométrico
    }
}

void stress3D(int device, int targetGpuUsagePercentage) {
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

    while (!stopStress) {
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

// Função para estressar a GPU com operações de decodificação de vídeo
void stressVideoDecode() {
    const int N = 1 << 28;  // Tamanho dos dados para simular a decodificação
    uint8_t* d_data;

    // Aloca memória na GPU
    cudaMalloc((void**)&d_data, N * sizeof(uint8_t));

    while (!stopStress) {
        // Simula o processamento de vídeo decodificado (operando com os dados)
        for (int i = 0; i < N; ++i) {
            d_data[i] = (uint8_t)(i % 255);
        }
    }

    cudaFree(d_data);
}

// Função para estressar a GPU com operações de codificação de vídeo
void stressVideoEncode() {
    const int N = 1 << 28;  // Tamanho dos dados para simular a codificação
    uint8_t* d_data;

    // Aloca memória na GPU
    cudaMalloc((void**)&d_data, N * sizeof(uint8_t));

    while (!stopStress) {
        // Simula o processamento de vídeo codificado (operando com os dados)
        for (int i = 0; i < N; ++i) {
            d_data[i] = (uint8_t)(i % 255);
        }
    }

    cudaFree(d_data);
}

// Função para estressar a GPU com todas as operações
void stressGPU(int level, int device) {
    cudaSetDevice(device); // Seleciona a GPU correta

    // Exemplo de adaptação do estresse com base na placa
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    std::cout << "Usando a GPU: " << prop.name << std::endl;

    switch (level) {
    case 1:
        std::cout << "Estressando a GPU com operações 3D..." << std::endl;
        stress3D(0, 50);
        break;
    case 2:
        std::cout << "Estressando a GPU com cópia de memória..." << std::endl;
        stressCopy(0);
        break;
    case 3:
        std::cout << "Estressando a GPU com decodificação de vídeo..." << std::endl;
        stressVideoDecode();
        break;
    case 4:
        std::cout << "Estressando a GPU com codificação de vídeo..." << std::endl;
        stressVideoEncode();
        break;
    case 5:
        std::cout << "Estressando a GPU com operações de memória..." << std::endl;
        stressMemory();
        break;
    case 6:
        std::cout << "Estressando a GPU com todas as operações..." << std::endl;
        stress3D(0,50);
        stressCopy(0);
        stressVideoDecode();
        stressVideoEncode();
        stressMemory();
        break;
    default:
        std::cout << "Número inválido. Escolha um número entre 1 e 6." << std::endl;
    }
}

// Função para permitir ao usuário parar o estresse com uma tecla
void stopStressInput() {
    char stop;
    std::cout << "Digite 'q' para parar o estresse: ";
    std::cin >> stop;
    if (stop == 'q') {
        stopStress = true;
    }
}

int main(int argc, char* argv[]) {
    // Exibir as informações da GPU
    getGPUInfo();

    int level = std::atoi(argv[1]);

    if (level < 1 || level > 6) {
        /*std::cout << "Selecione o tipo de teste a ser feito:\n";
        std::cout << "1 - Estressando a GPU com operações 3D\n";
        std::cout << "2 - Estressando a GPU com cópia de memória\n";
        std::cout << "3 - Estressando a GPU com decodificação de vídeo\n";
        std::cout << "4 - Estressando a GPU com codificação de vídeo\n";
        std::cout << "5 - Estressando a GPU com operações de memória\n";
        std::cout << "6 - Estressando a GPU com todas as operações\n";
        std::cout << "Escolha: ";

        std::cin >> level;*/
        std::cout << "Entre com valores de teste válidos!";
        return 1;
    }

    // Estressar a GPU em um nível específico (por exemplo, nível 6)
    std::thread stressThread(stressGPU, level, 0); // Modificar o número da GPU conforme necessário

    // Aguardar a entrada do usuário para parar
    std::thread stopThread(stopStressInput);
    stopThread.join();

    // Finalizar o estresse
    stopStress = true;
    stressThread.join();

    return 0;
}
