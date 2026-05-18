```markdown
# 🚗 Car Price Prediction Model | Projeto de Inteligência Artificial

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Latest-orange.svg)](https://scikit-learn.org/)
[![Status](https://img.shields.io/badge/Status-Conclu%C3%ADdo-brightgreen.svg)]()

Este repositório contém um projeto de ponta a ponta de Inteligência Artificial voltado para a **previsão de preços de automóveis**. Utilizando técnicas de Aprendizado de Máquina Supervisionado e Engenharia de Recursos (*Feature Engineering*), o modelo foi desenvolvido para simular um cenário real de negócios, auxiliando na tomada de decisões estratégicas de precificação automotiva com base em dados de mercado.

Ideal para recrutadores e profissionais de tecnologia que buscam avaliar minhas competências em **Ciência de Dados, Machine Learning e Engenharia de Software**.

---

## 🎯 Resultados de Destaque (Métricas de Performance)
Para garantir a precisão e confiabilidade das previsões, o modelo foi validado exaustivamente e alcançou métricas de altíssimo nível:

* **Coeficiente de Determinação ($R^2$ Score):** **95.81%**
    * *Significado prático:* O modelo é capaz de explicar e prever mais de 95% da variabilidade dos preços dos veículos.
* **Erro Absoluto Médio (MAE):** **2.066,42**
    * *Significado prático:* Em média, as previsões de preço do modelo variam apenas este valor em relação ao preço real de mercado, demonstrando alta assertividade e estabilidade.

---

## 🛠️ Tecnologias, Frameworks e Conceitos Aplicados
* **Linguagem:** Python 3.10+
* **Machine Learning:** Scikit-Learn (`DecisionTreeRegressor`)
* **Manipulação e Análise de Dados:** Pandas, NumPy
* **Conceitos de IA aplicados:** Aprendizado Supervisionado, Modelos de Regressão, Seleção de Recursos (*Feature Selection*), Avaliação de Métricas (MAE, $R^2$).

---

## 🧠 Arquitetura e Ciclo de Desenvolvimento

O pipeline do projeto foi estruturado seguindo as melhores práticas do mercado de dados:

1.  **Análise Exploratória de Dados (EDA):** Entendimento inicial do comportamento dos dados, estudo de correlações e identificação de padrões relevantes que impactam o preço final de um carro.
2.  **Pré-processamento & Feature Engineering:** Limpeza de dados, tratamento de variáveis categóricas (*Encoding*) e seleção inteligente de atributos (*Feature Selection*) para evitar *overfitting* e otimizar o tempo de processamento.
3.  **Treinamento do Modelo:** Implementação e ajuste do algoritmo **Decision Tree Regressor**, balanceando a profundidade da árvore para garantir máxima capacidade de generalização com novos dados.
4.  **Validação Robustecida:** Avaliação do modelo utilizando dados de teste nunca vistos pelo algoritmo, garantindo os excelentes índices de $R^2$ e MAE apresentados.
