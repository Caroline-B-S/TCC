# Importação de bibliotecas que serão utilizadas para a realização dos comandos
from panpython.sdk.stat.htc_result_post_processing import PandatHTCResultProcessor
import os

# Configuração das pastas de trabalho
# Criação das pastas onde os resultados serão salvos
dir_name = os.path.abspath(__file__)
script_name = os.path.basename(__file__)
task_path = dir_name[:-len(script_name)]

# Indicação do caminho e do nome dos documentos necessários para a realização dos cálculos
cal_type = 'point'
htc_result_parent_path = 'output/intermediate'
file_name = 'Default.dat.table'

# Configuração da pasta e nome do documento de saída após realização dos cálculos
dump_path = os.path.join(task_path, 'output')
file_output = 'merged.csv'

# Inicialização do comando para cálculo de pós processamento dos dados obtidos via HTC
if __name__ == "__main__":
    # file_name = PandatHTCResultProcessor.get_table_group(cal_type, htc_result_parent_path)[0]

    m_processor = PandatHTCResultProcessor(parent_path=htc_result_parent_path,
                                           file_name=file_name,
                                           cal_type=cal_type)
    m_processor.save_to_csv(output_path=dump_path, output_file=file_output)

