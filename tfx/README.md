# Introdução ao MLOps com TFX

source: <https://www.tensorflow.org/tfx/guide/understanding_tfx_pipelines?hl=pt-br>

Def.: MLOps é a prática de aplicar DevOps para automatizar, gerenciar e auditar fluxos de trabalho
de machine learning.

- Fluxos:
  - Preparar, analizar e transformar dados;
  - Treinar e avaliar um modelo
  - Implantar modelos treinados para produção
  - Rastrear artefatos de ML e suas dependências

Vantagens de implementar o fluxo de trabalho com pipelines TFX:

- Automatizar o processo de ML >> treinar, avaliar e implementar o modelo regularmente
- Uso de computação distribuída para processamento de grandes volumes de dados e cargas de trabalho
- Aumento na velocidade de experimentação ao permitir a execução de um mesmo pipeline em diferentes conjuntos de hiperparâmetros.

## Partes do TFX

### Artefato

- São as saídas das etapas em um pipeline do TFX.
- Podem alimentar as etapas subsequentes.
- Com isso o TFX permite que passemos dados entre as etapas do fluxo de trabalho.
- Devem ser fortemente tipados com um tipo de artefato registrado no repositório de metadados de ML.

### Componentes

- É uma implementação de uma tarefa de ML que podemos usar como uma etapa no pipeline do TFX.
- Compostos por:
  - Especificação de componente >> define os artefatos de entrada e saída do componente e os parâmetros necessários do componente.
  - Executor >> implementa o código para realizar uma etapa em seu fluxo de trabalho de ML.
  - Interface de componente >> empacota a especificação e o executor do componente para uso em um pipeline.

### Pipeline

- Implementação portátil de um fluxo de trabalho de ML que pode ser executado em vários orquestradores: Ariflow, Beam, Kubeflow Pipelines.
- Composto de instâncias de componentes e parâmetros de entrada.

Abaixo temos um diagrama de um pipline do TFX.

![fluxo](https://www.tensorflow.org/static/tfx/guide/images/tfx_pipeline_graph.svg?hl=pt-br)