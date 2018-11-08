# Adversatial-Audio-Attack
Model的部分是使用DeepSpeech，需要先安裝以下套件：

## Clone the Mozilla DeepSpeech repository into a folder called DeepSpeech：
git clone https://github.com/mozilla/DeepSpeech.git
cd DeepSpeech
git checkout tags/v0.1.1

## DeepSpeech Model：
wget https://github.com/mozilla/DeepSpeech/releases/download/v0.1.0/deepspeech-0.1.0-models.tar.gz
tar -xzf deepspeech-0.1.0-models.tar.gz
  
  
## 執行：
python classify.py XXX.wav
