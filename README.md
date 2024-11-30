# GPUExtressor
É uma ferramenta projetada para avaliar a performance de GPUs por meio de testes de estresse em diferentes níveis. Essa aplicação utiliza uma interface intuitiva desenvolvida em Python para configurar os testes e executa códigos CUDA no backend para estressar o hardware de maneira controlada e eficiente.

## Objetivo
O projeto ajuda desenvolvedores, entusiastas e profissionais a validar a estabilidade, performance e limites de suas GPUs, seja para uso pessoal, benchmarking ou análise de capacidade para workloads intensivos.

## Funcionalidades
1. Estresse do Processamento CUDA:
   * Níveis disponíveis:
     * Baixo: Teste leve, ideal para simulações rápidas.
     * Médio: Trabalho moderado para cargas típicas.
     * Alto: Teste intenso, para cenários de uso avançado.
     * <strong>FODEO:</strong> Teste extremo, explorando o máximo de performance da GPU.
   * Estressa o hardware com cálculos massivos, utilizando kernels CUDA.
2. Estresse de Transferência de Dados:
   * Mede a performance e estabilidade na transferência de dados ao alocar e desalocar informações na memória.
3. Alocação Máxima de Memória:
   * Estressa a capacidade de memória dedicada da GPU.
4. Interface Gráfica Amigável:
   * Desenvolvida com a biblioteca PyQt
   * Permite configurar níveis de estresse e monitorar o progresso em tempo real.
   * Exibe estatísticas como utilização de uso da gpu, memória, temperatura e do gerenciador de memória.
5. Execução com o CUDA:
   * Códigos em CUDA são compilados e executados em paralelo.

<hr>

## Tecnologias utilizadas
* Frontend (exibição de dados)
  * Python
* Backend CUDA:
  * C++
* Monitoramento:
  * NVIDIA Management Library (NVML) para leitura de métricas da GPU.

## Como usar?
### Pré requisitos
* Possuir um placa da nvidia compatível com o CUDA Toolkit
* CUDA Toolkit
* Python 3.10
* 
### Execução
* Execute o arquivo princiapl python

ou alternativamente...
* Código para compilar o arquivo CUDA:
```bash
nvcc -o gpu_extressor.dll -shared -Xcompiler -fPIC gpu_extressor.cu -lnvml
```

<hr>

## Monitoramento
Durante os testes, são exibidas métricas como:
* **Utilização da GPU**: Percentual de uso do processamento CUDA.
* **Temperatura**: Monitoramento em tempo real.
* **Uso de Memória**: Memória alocada vs disponível.

# Aviso
Use com cautela! O **FODEO** pode levar sua GPU ao limite, causando superaquecimento ou instabilidade. Certifique-se de que sua GPU está devidamente resfriada e monitorada antes de executar testes extremos.
