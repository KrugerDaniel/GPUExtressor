import time
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlShutdown, nvmlDeviceGetUtilizationRates

"""
Intallation: 
pip install nvidia-ml-py3

Troubleshooting:
In case of pynvml.NVMLError_LibraryNotFound: NVML Shared Library Not Found -> copy mvnvml.dll from C:\\Windows\\System32 to C:\\Program Files\\NVIDIA Corporation\\NVSMI
"""
def monitor_gpu(interval=1):
    nvmlInit()  # Inicializa a NVML
    handle = nvmlDeviceGetHandleByIndex(0)  # Obtem a primeira GPU

    try:
        while True:
            mem_info = nvmlDeviceGetMemoryInfo(handle)
            utilization = nvmlDeviceGetUtilizationRates(handle)

            print(f"Memória usada: {mem_info.used / (1024 * 1024):.2f} MB / {mem_info.total / (1024 * 1024):.2f} MB")
            print(f"Uso de GPU: {utilization.gpu}%")
            print("-" * 40)

            time.sleep(interval)  # Espera pelo intervalo especificado antes de verificar novamente
    except KeyboardInterrupt:
        print("Monitoramento interrompido pelo usuário.")
    finally:
        nvmlShutdown()  # Desliga a NVML ao finalizar o script

# Inicia o monitoramento com intervalo de 1 segundo
monitor_gpu()