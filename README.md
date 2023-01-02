# BDC-정리-README

진행사항: 진행
태그: PROJECT

## generation-code

- story cloze dataset의 4 sentences 데이터를 이용해서 다음문장을 생성할 때 사용했던 코드
- train은 finetuning을 진행하는 코드
- few_shot은 args의 few_shot 개수에 맞게끔 프롬프트를 조절해서 넣어줄 수 있게끔 짜놓은 코드
- data는 데이터 디렉토리, 스토리 클로즈 데이터셋과 ROCStories 데이터셋이 들어있음
- generation results : 실제로 knowledge distilation 에 사용한 생성된 데이터. T5는 finetuning한 결과를 활용하였고, GPT는 fewshot으로 생성한 결과를 사용하였음

데이터 생성과 관련하여 정리해놓은 내용

[Story Cloze data generation (T5, GPT) - 민주 (공유)](https://www.notion.so/Story-Cloze-data-generation-T5-GPT-a6d83e524554497fb82c84ea16af4637)

## t5_finetuning_factory

- baseline 모델과 pretrain 되었던 모델을 4가지 task의 데이터셋으로 finetuning하는 코드
- train 은 finetuning하는 코드이고, generation 때와는 다르게 각 에포크마다 test set (val set)을 generation한 것을 확인할 수 있게하고, 에포크별로 저장된 모델을 저장하도록 되어 있음
- train코드와 test 코드를 실행할 때 입력
    
    ```python
    CUDA_VISIBLE_DEVICES=1 python T5_train.py --model t5-base --task emocap --max_src_len 64 --max_trg_len 16 --epoch 8 --batch 8 --sampling 1
    ```
    
    ```python
    CUDA_VISIBLE_DEVICES=0 python T5_test.py --model t5-base_last_cnn_0.1/model_files_2 --task cnn --max_src_len 1536 --max_trg_len 256 --epoch 0 --batch 16 --sampling 1
    ```
    

- CUDA_VISIBLE_DEVICES : GPU 번호를 입력
- model : baseline 모델 / pretrain된 모델 / finetuning된 모델 (test 시에)
- task : cnn / slurp / story_cloze / emocap 중 하나
- max_src_len : src len
- max_trg_len : trg len
- epoch : train epoch. cnn은 2에퐄 정도가 가장 좋았고, slurp과 emocap의 경우 10에퐄 정도 충분히 돌린 후 확인하는 것이 좋았음
- sampling : 100% 데이터셋을 모두 finetuning에 쓸 것인지 10%만 뽑아서 할 것인지 결정

## T5-Finetuning-PyTorch

- train / test 에 참고한 예제코드

## T5-pretrain

- knowledge distilation 결과를 확인하기 위해서 GPT에서 생성된 데이터를 T5에 pretrain한 코드
- pytorch lightning으로 학습
- 코드 원본 깃헙: [https://github.com/manueldeprada/Pretraining-T5-PyTorch-Lightning](https://github.com/manueldeprada/Pretraining-T5-PyTorch-Lightning)