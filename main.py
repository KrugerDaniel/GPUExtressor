import ctypes
import subprocess
import time
import psutil
from GPUtil import GPUtil
from PyQt5.QtWidgets import QApplication, QVBoxLayout, QPushButton, QWidget, QLabel, QHBoxLayout, QComboBox
from PyQt5.QtCore import QTimer, Qt
from multiprocessing import Process


# comando para compilar cuda -> nvcc -o gpu_extressor.dll -shared -Xcompiler -fPIC gpu_extressor.cu -lnvml


def get_gpu_usage():
    command = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory,temperature.gpu",
        "--format=csv,noheader,nounits"
    ]

    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    if result.returncode == 0:
        gpu_usage = []
        lines = result.stdout.strip().split("\n")
        for line in lines:
            values = line.split(", ")
            gpu_data = {
                'name': values[0],
                'memory_total': int(values[1]),
                'memory_used': int(values[2]),
                'memory_free': int(values[3]),
                'utilization_gpu': int(values[4]),
                'utilization_memory': int(values[5]),
                'temperature': int(values[6])
            }
            gpu_usage.append(gpu_data)
        return gpu_usage
    else:
        print("Erro ao executar nvidia-smi:", result.stderr)
        return None


def get_cpu_usage():
    cpu_usage = psutil.cpu_percent(interval=1)
    return cpu_usage


def stress_3d(device, target_gpu_usage_percentage):
    cuda = ctypes.CDLL("./gpu_extressor.dll")
    cuda.startStress3D(device, target_gpu_usage_percentage)
    while True:
        time.sleep(1)


def stress_memory(device):
    cuda = ctypes.CDLL("./gpu_extressor.dll")
    cuda.startStressMemory(device)
    while True:
        time.sleep(1)


def stress_copy(device):
    cuda = ctypes.CDLL("./gpu_extressor.dll")
    cuda.startStressCopy(device)
    while True:
        time.sleep(1)


class StressTestApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Estresse CUDA")
        self.setGeometry(100, 100, 1500, 400)

        layout = QVBoxLayout()

        gpus = GPUtil.getGPUs()
        if gpus:
            gpu_name = gpus[0].name
        else:
            gpu_name = "GPU não detectada"

        self.gpu_name_label = QLabel(gpu_name)
        self.gpu_name_label.setAlignment(Qt.AlignCenter)
        self.gpu_name_label.setStyleSheet("font-size: 16px; font-weight: bold; margin-bottom: 10px;")
        layout.addWidget(self.gpu_name_label)

        self.cards_layout = QVBoxLayout()

        row_1 = QHBoxLayout()
        row_2 = QHBoxLayout()

        self.memory_card = self.create_gpu_card("Memória Usada", "0 MB")
        self.memory_total_card = self.create_gpu_card("Memória Total", "0 MB")
        self.memory_free_card = self.create_gpu_card("Memória Livre", "0 MB")
        self.utilization_card = self.create_gpu_card("Uso da GPU", "0 %")
        self.utilization_memory_card = self.create_gpu_card("Gerenciador de memória", "0 %")
        self.temperature_card = self.create_gpu_card("Temperatura", "0 °C")
        self.cpu_usage_card = self.create_gpu_card("Uso da CPU", "0 %")

        row_1.addWidget(self.cpu_usage_card)
        row_1.addWidget(self.utilization_card)
        row_1.addWidget(self.utilization_memory_card)
        row_1.addWidget(self.temperature_card)

        row_2.addWidget(self.memory_card)
        row_2.addWidget(self.memory_total_card)
        row_2.addWidget(self.memory_free_card)

        self.cards_layout.addLayout(row_1)
        self.cards_layout.addLayout(row_2)

        combo_layout = QHBoxLayout()
        combo_label = QLabel("Selecione o Nível de Estresse:")
        combo_label.setStyleSheet("font-size: 18px; margin-left: 10px")
        self.stress_level_combo = QComboBox()
        self.stress_level_combo.addItems(["Baixo", "Médio", "Alto", "FODEO"])
        self.stress_level_combo.setStyleSheet("font-size: 16px; padding: 5px; cursor: pointer")
        self.stress_level_combo.setCursor(Qt.PointingHandCursor)
        combo_layout.addWidget(combo_label)
        combo_layout.addWidget(self.stress_level_combo)
        combo_layout.addStretch()

        button_layout = QHBoxLayout()
        self.btn_3d_start = QPushButton("Iniciar Estresse 3D")
        self.btn_3d_stop = QPushButton("Parar Estresse 3D")
        self.btn_mem_start = QPushButton("Iniciar Estresse Memória")
        self.btn_mem_stop = QPushButton("Parar Estresse Memória")
        self.btn_copy_start = QPushButton("Iniciar Operações de Cópia")
        self.btn_copy_stop = QPushButton("Parar Operações de Cópia")
        button_layout.addWidget(self.btn_3d_start)
        button_layout.addWidget(self.btn_3d_stop)
        button_layout.addWidget(self.btn_mem_start)
        button_layout.addWidget(self.btn_mem_stop)
        button_layout.addWidget(self.btn_copy_start)
        button_layout.addWidget(self.btn_copy_stop)
        self.style_buttons()

        layout.addLayout(self.cards_layout)
        layout.addLayout(combo_layout)
        layout.addLayout(button_layout)

        self.setLayout(layout)

        self.btn_3d_stop.setEnabled(False)
        self.btn_mem_stop.setEnabled(False)
        self.btn_copy_stop.setEnabled(False)

        self.btn_3d_start.clicked.connect(self.start_stress_3d)
        self.btn_3d_stop.clicked.connect(self.stop_stress_3d)
        self.btn_mem_start.clicked.connect(self.start_stress_mem)
        self.btn_mem_stop.clicked.connect(self.stop_stress_mem)
        self.btn_copy_start.clicked.connect(self.start_stress_copy)
        self.btn_copy_stop.clicked.connect(self.stop_stress_copy)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_metrics)
        self.timer.start(1000)  # Atualiza a cada 1 segundo

        self.process_3d = None
        self.process_mem = None
        self.process_copy = None

    def style_buttons(self):
        style = """
            QPushButton {
                font-size: 14px;
                padding: 10px;
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 5px;
                margin: 5px;
                width: 16%;
            }
            QPushButton:hover {
                cursor: pointer;
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #aaa;
            }
        """
        self.setStyleSheet(style)
        buttons = [
            self.btn_3d_start, self.btn_3d_stop,
            self.btn_mem_start, self.btn_mem_stop,
            self.btn_copy_start, self.btn_copy_stop
        ]
        for btn in buttons:
            btn.setStyleSheet("font-size: 14px; padding: 5px; cursor: pointer")
            btn.setCursor(Qt.PointingHandCursor)

    def create_gpu_card(self, title, value):
        card = QWidget()
        card_layout = QVBoxLayout()

        title_label = QLabel(title)
        value_label = QLabel(value)

        title_label.setAlignment(Qt.AlignCenter)
        value_label.setAlignment(Qt.AlignCenter)

        title_label.setStyleSheet("font-size: 20px; margin-top: 10px; padding: 0px; max-height: 40%; border: none")
        value_label.setStyleSheet("font-size: 25px; font-weight: 700;")

        card_layout.addWidget(title_label)
        card_layout.addWidget(value_label)
        card.setLayout(card_layout)

        # Estilo do card
        card.setStyleSheet(""" 
            QWidget {
                background-color: #f0f0f0;
                border: 2px solid #ccc;
                border-radius: 10px;
                padding: 10px;
                margin: 10px;
                width: 150px;
                max-height: 150px;
                text-align: center;
            }
        """)
        return card

    def update_metrics(self):
        gpu_usage = get_gpu_usage()
        cpu_usage = get_cpu_usage()

        if gpu_usage:
            memory_used = gpu_usage[0]['memory_used']
            memory_total = gpu_usage[0]['memory_total']
            memory_free = gpu_usage[0]['memory_free']
            utilization = gpu_usage[0]['utilization_gpu']
            temperature = gpu_usage[0]['temperature']
            utilization_memory = gpu_usage[0]['utilization_memory']

            self.memory_card.layout().itemAt(1).widget().setText(f"{memory_used} MB")
            self.memory_total_card.layout().itemAt(1).widget().setText(f"{memory_total} MB")
            self.memory_free_card.layout().itemAt(1).widget().setText(f"{memory_free} MB")
            self.utilization_card.layout().itemAt(1).widget().setText(f"{utilization} %")
            self.temperature_card.layout().itemAt(1).widget().setText(f"{temperature} °C")
            self.utilization_memory_card.layout().itemAt(1).widget().setText(f"{utilization_memory} %")

        self.cpu_usage_card.layout().itemAt(1).widget().setText(f"{cpu_usage} %")

    def start_stress_3d(self):
        selected_level = self.stress_level_combo.currentText()
        if selected_level == 'Baixo':
            target_gpu_usage_percentage = 20
        elif selected_level == 'Médio':
            target_gpu_usage_percentage = 50
        elif selected_level == 'Alto':
            target_gpu_usage_percentage = 95
        else:
            target_gpu_usage_percentage = 100
        self.process_3d = Process(target=stress_3d, args=(0, target_gpu_usage_percentage))
        self.process_3d.start()
        self.btn_3d_start.setEnabled(False)
        self.btn_3d_stop.setEnabled(True)

    def stop_stress_3d(self):
        if self.process_3d:
            self.process_3d.terminate()
            self.process_3d = None
        self.btn_3d_start.setEnabled(True)
        self.btn_3d_stop.setEnabled(False)

    def start_stress_mem(self):
        self.process_mem = Process(target=stress_memory, args=(0,))
        self.process_mem.start()
        self.btn_mem_start.setEnabled(False)
        self.btn_mem_stop.setEnabled(True)

    def stop_stress_mem(self):
        if self.process_mem:
            self.process_mem.terminate()
            self.process_mem = None
        self.btn_mem_start.setEnabled(True)
        self.btn_mem_stop.setEnabled(False)

    def start_stress_copy(self):
        self.process_copy = Process(target=stress_copy, args=(0,))
        self.process_copy.start()
        self.btn_copy_start.setEnabled(False)
        self.btn_copy_stop.setEnabled(True)

    def stop_stress_copy(self):
        if self.process_copy:
            self.process_copy.terminate()
            self.process_copy = None
        self.btn_copy_start.setEnabled(True)
        self.btn_copy_stop.setEnabled(False)


if __name__ == '__main__':
    app = QApplication([])
    window = StressTestApp()
    window.show()
    app.exec_()
