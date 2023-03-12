# Importação de bibliotecas que serão utilizadas para a realização dos comandos
import os
import socket
import multiprocessing as mp
from panpython.system import System
from panpython.task.htc import HtcMesh
import pickle

# Identificação do caminho para o arquivo executável do software PANDAT™
pandat = r"C:\Program Files\CompuTherm LLC\Pandat 2022\bin\Pandat.exe"

# Configuração das pastas de trabalho
# Criação das pastas onde os resultados serão salvos
# Associação dos documentos necessários para a realização dos cálculos
dir_name = os.path.abspath(__file__)
file_name = os.path.basename(__file__)
task_path = dir_name[:-len(file_name)]

dump_path = os.path.join(task_path, "output")
batch_file = os.path.join(task_path, "resource", "CrMnNbTiVZr.pbfx")
config_file = os.path.join(task_path, "resource", "Input_point.JSON")

# Definição do número de thread (o número de fluxos que são executados paralelamente)
# Caso o número indicado seja superior a capacidade do sistema
# O mesmo será automaticamente mudado para o máximo possível
thread_number: int = 8
print('>> Max thread  number on machine: ', socket.gethostname(), ' is ', mp.cpu_count())
if thread_number > mp.cpu_count():
    thread_number = mp.cpu_count()
print('>> Thread number is set to ', thread_number)

# Inicialização do comando para cálculo de alto rendimento
m_system = System(pandat=pandat, dump_path=dump_path)
m_system.add_task(task_instance=HtcMesh(batch_file=batch_file, config_file=config_file, thread_num=thread_number))

# Inicialização do comando para cálculo de alto rendimento para o modo multi-thread
if __name__ == '__main__':
    mp.freeze_support()
    m_system.run()
