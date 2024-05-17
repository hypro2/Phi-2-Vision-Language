# 통합 언어 및 비전 모델

이 프로젝트는 고급 언어 및 비전 모델의 개발과 통합을 포괄하며 기계 학습에서 포괄적인 접근 방식을 보여줍니다. 이 프로젝트는 처음부터 완전히 사용자 정의 언어 모델인 Phi 2를 학습하고, 강인한 다중 모달 이해를 위한 비전-텍스트 정렬 기술로 그것을 더 발전시키는 것을 포함합니다.

## 개요

프로젝트는 두 가지 주요 부분으로 나뉩니다:

### 1부: Phi 2 언어 모델 학습
Phi 2 언어 모델의 학습에 중점을 둡니다. 두 가지 다른 방법을 사용하여 이루어집니다.

#### 처음부터 학습
이 접근 방식에서는 Phi 2 언어 모델이 완전히 처음부터 개발됩니다. 새로운 아키텍처가 설계되었으며, MLP 이후의 전문가 혼합이 새롭게 추가되었습니다. 이 버전의 Phi 2는 프로젝트 구텐베르크 데이터셋의 포괄적이고 다양한 텍스트를 학습하였으며, 이를 통해 언어 모델의 능력을 기초 수준에서 구축하고 개선할 수 있는 독특한 기회를 제공했습니다.

#### LIT-GPT 구현
이 구현은 LIT-GPT 프레임워크를 사용하여 Phi 2 모델을 학습하는 것을 포함합니다. 이 학습은 Redpajama 데이터셋에서 진행되었으며, 모델이 학습할 특정 맥락과 언어적 패턴을 제공했습니다. 이 방법은 처음부터 학습하는 것과 비교 연구로서의 역할을 하여, 다른 학습 방법에 따른 모델 성능에 대한 통찰을 제공했습니다.

### 2부: 사전 학습 및 파인튜닝
개선된 언어 및 비전 기능을 위한 사전 학습 및 파인튜닝.

#### 1: 사전 학습을 위한 비전-텍스트 정렬

사전 학습 단계에서는 얼어붙은 사전 학습된 Phi-2 언어 모델과 CLIP 모델을 사용했으며, 사용자 지정 프로젝션 계층과 결합되었습니다. 이 설정은 비전-텍스트 정렬을 위해 특별히 설계되었으며, 프로젝션 계층을 학습하여 CLIP 임베딩을 Phi 2 모델이 예상하는 임베딩과 일치시킵니다. 학습은 COCO 2017 데이터셋의 약 40,000개의 이미지 및 캡션 하위 집합을 사용하여 진행되었으며, 모델이 시각적 및 텍스트 정보를 이해하고 정렬하는 능력을 개발하는 데 중점을 두었습니다.

#### 2: 지시문 따르기 파인튜닝

파인튜닝 단계는 모델의 지시 따르기 능력을 향상시키는 것을 목표로 합니다. 이를 위해 얼어붙은 CLIP 모델, 얼어붙지 않은 사전 학습된 Phi-2, 및 사전 학습 단계의 얼어붙지 않은 프로젝션 계층을 사용했습니다. 학습은 Instruct150K 데이터셋의 약 40,000개의 이미지 하위 집합을 사용했습니다. 이 단계에서는 데이터셋 준비를 위한 두 가지 주요 방법을 탐구했습니다: 교사 강제를 사용한 자기 회귀 토큰 예측 및 표준 언어 모델 학습 방법. 이러한 방법은 지시문 따르기와 같은 특정 작업을 위해 모델의 성능을 최적화하는 데 중요한 역할을 했으며, 계산 효율성과 학습의 심도 사이의 균형을 이루는 데 기여했습니다.

## 참고 자료

다음 자료 및 논문들이 이 프로젝트의 개념화와 개발에 크게 기여했습니다:

1. "시각적 지시 튜닝을 통한 개선된 기준선" - 이 논문은 시각 및 언어 모델 통합을 위한 파인튜닝 과정에 대한 통찰을 제공합니다. [자세히 읽기](https://arxiv.org/abs/2310.03744)
2. "LLaVA: 대규모 언어 및 비전 어시스턴트" - 이 연구는 대규모 언어 모델과 비전 기능의 통합을 탐구하여 이 프로젝트에 대한 기본 지식을 제공합니다. [자세히 읽기](https://arxiv.org/abs/2304.08485)
3. OpenAI CLIP - OpenAI가 제공하는 자연어 지도를 통해 시각적 개념을 학습하는 최첨


# Integrated Language and Vision Models

This project encompasses the development and integration of advanced language and vision models, demonstrating a comprehensive approach in machine learning. It involves training a custom language model, Phi 2, from scratch, and further enhancing it with vision-text alignment techniques for robust multimodal understanding.

## Overview

The project is divided into two main parts:

### Part 1: Phi 2 Language Model Training
Focusing on training the Phi 2 language model using two distinct methods - 

## Training from Scratch
In this approach, the Phi 2 language model is developed entirely from the ground up. A custom architecture was designed, including a novel addition of a mixture of experts after the MLP. This version of Phi 2 was trained on the comprehensive and diverse texts of the Project Gutenberg dataset, offering a unique opportunity to build and refine the language model's capabilities from a foundational level.

## LIT-GPT Implementation
This implementation involves training the Phi 2 model using the LIT-GPT framework, a well-established method for language model development. The training was conducted on the Redpajama dataset, providing a specific context and set of linguistic patterns for the model to learn. This approach served as a comparative study to the from-scratch training, offering insights into the model's performance under different training methodologies.

### Part 2: Pretraining and Finetuning
Pretraining and finetuning of the model for enhanced language and vision capabilities.

## 1: Vision-Text Alignment for Pretraining

In the pretraining stage, the project utilized a frozen pretrained Phi-2 language model and a frozen pretrained CLIP model, combined with a custom projection layer. This setup was specifically designed for vision-text alignment, training the projection layer to align CLIP embeddings with those expected by the Phi 2 model. The training employed a subset of 40,000 images and captions from the COCO 2017 dataset, focusing on developing the model's capability to understand and align visual and textual information.

## 2: Instruction Following Fine-Tunin

The fine-tuning stage aimed at enhancing the model's ability to follow instructions. This involved using a frozen CLIP model, an unfrozen pretrained Phi-2, and the unfrozen projection layer from the pretraining stage. The training utilized a subset of about 40,000 images from the Instruct150K dataset. This stage explored two key methods for dataset preparation: Autoregressive Token Prediction with Teacher Forcing and the Standard Language Model Training Method. These methods were instrumental in optimizing the model's performance for specific tasks such as instruction following, striking a balance between computational efficiency and the depth of learning.


## References

The following resources and papers have significantly contributed to the conceptualization and development of this project:

1. "Improved Baselines with Visual Instruction Tuning" - This paper provides insights into the fine-tuning processes for visual and language model integrations. [Read More](https://arxiv.org/abs/2310.03744)
2. "LLaVA: Large Language and Vision Assistant" - This research explores the integration of large language models with vision capabilities, offering foundational knowledge for this project. [Read More](https://arxiv.org/abs/2304.08485)
3. OpenAI CLIP - A cutting-edge model by OpenAI for learning visual concepts from natural language supervision. [Learn More](https://openai.com/clip/)
4. "Mixture of Experts Explained" - This Hugging Face blog post provides an overview of the Mixture of Experts architecture, a key concept in the development of the Phi 2 model. [Read More](https://huggingface.co/blog/moe)
5. Lightning AI - Lit-GPT - A GitHub repository offering an implementation framework for language models, used in part of this project. [Explore Repository](https://github.com/Lightning-AI/lit-gpt)
6. Rohan Shravan ERAv1 [The School Of AI](https://www.theschoolof.ai)
