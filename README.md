# Relatório Técnico - Deep Learning (Mosquito detection - YOLO)

## 1. Capa
* **Equipe:** Grupo G
* **Integrantes:**
  * Gabriel Abreu Cunha De Alencar - 2315097
  * Igor Gomes Ximenes -2217665
  * Kalil Smith Pinto Palheta - 2223857

* **Repositório:**
  * [Repositório do projeto](https://github.com/Kalil-S/Projeto-Yolo-Mosquito)

* **(OBS):** Caso a visualização do arquivo (`main.ipynb`) no github não funcionar, informando essa mensagem de alerta:
> [!WARNING]
> Unable to render code block

 Clique neste link, para conseguir visualizar: [Repositório no nbviewer](https://nbviewer.org/github/Kalil-S/Projeto-Yolo-Mosquito/tree/main/) ou [Link do Google Colab](https://colab.research.google.com/drive/1BX3l0i1hHpmp_wwaftm5JwycmrLyopva?usp=sharing)

---
## 1. Introdução e Motivação
Este relatório detalha a metodologia científica, estatística e computacional aplicada ao desenvolvimento de um modelo preditivo de Visão Computacional baseado na arquitetura YOLOv8. O problema central consiste em realizar a detecção e classificação taxonômica de espécies de mosquitos das famílias *Culicidae*, com foco nas variantes endêmicas e transmissoras de arboviroses no Brasil, como o *Aedes aegypti*, *Aedes albopictus* e o gênero *Culex*.

A escolha deste tema justifica-se pela extrema relevância na saúde pública, servindo como base inicial para sistemas de monitoramento epidemiológico automatizado. Do ponto de vista técnico, atende perfeitamente ao requisito da disciplina por ser uma classe "inédita", não listada nas 80 categorias clássicas pré-treinadas do conjunto de dados COCO (*Common Objects in Context*), exigindo um fluxo rigoroso de treinamento supervisionado.

---

## 2. Análise Exploratória de Dados (EDA)
A fase de exploração permitiu desvelar a estrutura latente dos dados, suportando as decisões subsequentes de modelagem. Os dados foram obtidos publicamente na plataforma Roboflow Universe. O volume bruto consolidou-se em 928 imagens de validação anotadas com caixas delimitadoras (*bounding boxes*).

A análise da distribuição de frequências revelou um panorama de alta complexidade analítica decorrente de um desbalanceamento severo de classes e de inconsistências de ontologia (rotulação). A tabela abaixo expõe a distribuição final de instâncias contidas na validação do modelo:

| Classe Taxonômica / Label Original | Instâncias | % de Representatividade |
| :--- | :---: | :---: |
| **Total Geral (all)** | **928** | **100,00%** |
| culex | 211 | 22,74% |
| albopictus | 206 | 22,20% |
| - Mosquito Detection - 2023-10-18 (Ruído) | 200 | 21,55% |
| non_aedes (Classe Genérica) | 158 | 17,02% |
| mosquito (Classe Genérica) | 97 | 10,45% |
| culiseta | 31 | 3,34% |
| japonicus-koreicus | 16 | 1,72% |
| anopheles | 6 | 0,65% |
| **aegypti (Alvo Principal)** | **3** | **0,32%** |

> **Nota Crítica:** A classe prioritária do ponto de vista epidemiológico, *Aedes aegypti*, representou apenas 0,32% do volume total, dispondo de meras 3 instâncias. A presença de classes parasitas e genéricas gerou forte entropia visual, sobrepondo definições específicas e penalizando a capacidade latente de abstração da rede convolucional.

---

## 3. Metodologia Computacional
O *pipeline* de engenharia foi executado no ecossistema do Google Colab, utilizando aceleração por hardware baseada numa GPU dedicada NVIDIA Tesla T4. Adotou-se a *framework* oficial Ultralytics para inicialização e controlo fino da rede neural.

### Arquitetura Escolhida e Hiperparâmetros
A escolha recaiu sobre o modelo **YOLOv8 Nano (`yolov8n.pt`)**. Com aproximadamente 3,2 milhões de parâmetros e suporte nativo a operações matemáticas otimizadas em ponto flutuante de 16 bits (FP16), o modelo foi selecionado de forma estratégica visando a sua futura embarcação em dispositivos móveis no ecossistema final do projeto ("Mosquito Watcher"). 

O treinamento foi conduzido com as seguintes diretrizes operacionais:
* **Resolução de Entrada (`imgsz`):** Fixada em 640x640 *pixels* para balancear o custo de convoluções com a necessidade de reter os micro detalhes morfológicos necessários para a diferenciação entomológica.
* **Tamanho do Lote (`batch size`):** Configurado em 16 instâncias por iteração.
* **Épocas (`epochs`):** Limitado a 25 iterações completas sobre o *dataset* para evitar *overfitting* em classes super amostradas.
* **Otimizador:** Auto-selecionado pela Ultralytics (AdamW adaptativo) com o *stripping* de pesos redundantes ativo ao término do ciclo, compactando o ficheiro final `best.pt` para apenas 6,2 MB.

---

## 4. Análise de Resultados e Métricas do Modelo
Ao concluir as 25 épocas regulamentares, a biblioteca gerou e salvou as métricas consolidadas dentro do diretório de execução. A tabela a seguir expõe de forma matemática o desempenho obtido pelo modelo na fase de validação:

| Classe Avaliada | Imagens | Instâncias | Precisão (Box P) | Recall (Box R) | mAP @0.5 | mAP @0.5:0.95 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Média Geral (all)** | 928 | 928 | 0,627 | 0,541 | **0,472** | 0,365 |
| - Mosquito Detection | 200 | 200 | 0,541 | 0,640 | **0,713** | 0,476 |
| 🔴 **Aedes aegypti** | 3 | 3 | 1,000 | 0,000 | **0,0108** | 0,0050 |
| 🟢 **Aedes albopictus** | 206 | 206 | 0,574 | 0,947 | **0,799** | 0,799 |
| Anopheles | 6 | 6 | 1,000 | 0,000 | **0,0502** | 0,0413 |
| 🟢 **Culex** | 211 | 211 | 0,568 | 0,947 | **0,785** | 0,785 |
| Culiseta | 31 | 31 | 0,285 | 0,419 | **0,329** | 0,272 |
| Japonicus-koreicus | 16 | 16 | 0,488 | 0,250 | **0,326** | 0,286 |
| mosquito (classe lixo) | 97 | 97 | 0,236 | 0,670 | **0,242** | 0,188 |
| non_aedes | 158 | 158 | 0,949 | 1,000 | **0,995** | 0,848 |

A interpretação matemática destes dados revela o comportamento probabilístico típico de redes convolucionais expostas ao desequilíbrio de amostragem. As classes majoritárias (*Albopictus* e *Culex*) alcançaram métricas excelentes de Recall, ambas estabilizando em 0,947 (94,7%) com mAP@0.5 expressivo de 0,799 e 0,785, respetivamente. Isto indica que para estas duas espécies o modelo aprendeu as assinaturas de contorno de forma robusta.

Por outro lado, **o modelo falhou completamente ao tentar predizer o *Aedes aegypti***, registrando um Recall absoluto de 0,000 e mAP@0.5 residual de 0,0108. Estatisticamente, a rede foi induzida ao viés probabilístico de "Chute Seguro": diante da incerteza, classificar o objeto como *Culex* ou *Albopictus* confere menor penalização à função de custo de entropia do otimizador do que classificar como *Aegypti*, cuja probabilidade a priori no conjunto era inferior a 1%.

<img src="https://github.com/Kalil-S/Projeto-Yolo-Mosquito/blob/main/Plots/Matriz%20de%20confus%C3%A3o.png">

<img src="https://github.com/Kalil-S/Projeto-Yolo-Mosquito/blob/main/Plots/Gr%C3%A1fico%20de%20treinamento.png">

---

# 5. Inferência Prática e Desvio de Domínio
A equipa consolidou um conjunto de testes externos composto por imagens reais obtidas em cenários domésticos pelos próprios alunos e imagens colhidas da internet para validação. Os resultados comportaram-se de duas maneiras distintas:

* **Sucesso Prático no Gênero Culex:** Nas imagens capturadas diretamente via câmara de *smartphone*, o modelo demonstrou alta estabilidade. Ele localizou o inseto contra superfícies complexas (paredes domésticas com texturas e sombras) e desenhou perfeitamente as *bounding boxes*, provando o funcionamento prático do YOLO.
* **Fenômeno de Desvio de Domínio (*Domain Shift*):** Ao testar fotos reais de *Aedes aegypti* e *Anopheles* capturadas fora de condições controladas, o modelo classificou-as erroneamente como *Albopictus*, *Culex*, *Culiseta* e mosquito. Biologicamente similares, as espécies diferem por um micro-padrão no tórax e outras por falta de imagens oferecidas de classificação do *dataset*. A compressão do YOLO aliada à diferença de iluminação do ambiente real (*Domain Shift*) forçou a rede a colapsar para a classe majoritária mais próxima.

<img src=https://github.com/Kalil-S/Projeto-Yolo-Mosquito/blob/main/Fotos_Resultados/Mosquito%20tirado%20por%20mim3.jpg>

<img src=https://github.com/Kalil-S/Projeto-Yolo-Mosquito/blob/main/Fotos_Resultados/Mosquito%20tirado%20por%20mim6.jpg>

<img src="https://github.com/Kalil-S/Projeto-Yolo-Mosquito/blob/main/Fotos_Resultados/Mosquito%20tirado%20por%20mim4.jpg">

---

## 6. Conclusão e Trabalhos Futuros
O experimento técnico atingiu com sucesso os objetivos, provando que a arquitetura YOLOv8 é plenamente viável para detecção de mosquitos no mundo real, demonstrando alto vigor ao isolar a espécie *Culex*. Contudo, do ponto de vista estrito da Engenharia de *Machine Learning*, os erros de predição serviram para chancelar a máxima científica de que um algoritmo é tão bom quanto o dado que o alimenta (*Garbage In, Garbage Out*). A falha no reconhecimento do *Aedes aegypti* não decorreu de limitações no código ou hiperparâmetros, mas sim da fragilidade estrutural do *dataset* público utilizado.

Para viabilizar a transição bem-sucedida deste experimento rumo ao projeto definitivo **"Mosquito Watcher"**, a equipa estabelece o seguinte roteiro de melhorias obrigatórias:

1. **Saneamento Ontológico Completo:** Expurgar do *dataset* original todas as classes geradoras de ruído e entropia (como *labels* genéricas e *strings* de *timestamp* corrompidas).
2. **Injeção de Dados Balanceados:** Executar uma fusão de dados (*Merge*) coletando imagens exclusivas de *Aedes aegypti* em alta definição de repositórios médicos internacionais, igualando a amostragem em patamares de paridade com o gênero *Culex* (Mínimo de 1000 imagens por classe).
3. **Pipeline Customizado de Data Augmentation via Código:** Implementado em Python (utilizando bibliotecas como `Albumentations`) transformações de imagem focadas na realidade urbana: injeção de ruído Gaussiano (para simular câmaras de *smartphones* de baixa qualidade), cortes aleatórios (*crop*) focados no tórax do inseto e simulações agressivas de subexposição e sombras para treinar a rede a enxergar mosquitos à noite.

---

## 7. Referências Bibliográficas

* **MOSQUITO**. *Mosquito Dataset*. Roboflow Universe, fev. 2024. Tipo: Open Source Dataset. Disponível em: [https://universe.roboflow.com/mosquito-zcdzw/mosquito-9s7ph](https://universe.roboflow.com/mosquito-zcdzw/mosquito-9s7ph). Acesso em: 06 jun. 2026.
* **ULTRALYTICS**. *YOLOv8 Documentation and Framework Architecture*. Disponível em: [https://docs.ultralytics.com/](https://docs.ultralytics.com/). Acesso em: 14 jun. 2026.
