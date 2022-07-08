# beto-emoji
Fine-tunning BETO for emoji-prediction

## HuggingFace

🤗 [huggingface.co/ccarvajal/beto-emoji](https://huggingface.co/ccarvajal/beto-emoji)

## Installation

It requires the installation of [pytorch](https://pytorch.org/get-started/locally/), which depends on the system and whether there's a GPU. The library [transformers](https://huggingface.co/docs/transformers/index). For the rest, run 

``` pip install -r requirements.txt ```

## Repository
Details with training and a use example are shown in [github.com/camilocarvajalreyes/beto-emoji](https://github.com/camilocarvajalreyes/beto-emoji). A deeper analysis of this and other models on the full dataset can be found in [github.com/furrutiav/data-mining-2022](https://github.com/furrutiav/data-mining-2022). We have used this model for a project for [CC5205 Data Mining](https://github.com/dccuchile/CC5205) course.

## Notebooks
- **[Fine-tunning](https://github.com/camilocarvajalreyes/beto-emoji/blob/main/finetuning.ipynb)**
- [Classification](https://github.com/camilocarvajalreyes/beto-emoji/blob/main/classifier_example_and_results.ipynb)
- [Visualisation with bertviz](https://github.com/camilocarvajalreyes/beto-emoji/blob/main/attention_visualisation.ipynb)

## Reproducibility
The Multilingual Emoji Prediction dataset (Barbieri et al. 2010) consists of tweets in English and Spanish that originally had a single emoji, which is later used as a tag. Test and trial sets can be downloaded [here](https://github.com/fvancesco/Semeval2018-Task2-Emoji-Detection/blob/master/dataset/Semeval2018-Task2-EmojiPrediction.zip?raw=true), but the train set needs to be downloaded using a [twitter crawler](https://github.com/fra82/twitter-crawler/blob/master/semeval2018task2TwitterCrawlerHOWTO.md). The goal is to predict that single emoji that was originally in the tweet using the text in it (out of a fixed set of possible emojis, 20 for English and 19 for Spanish).

Training parameters:
```python
training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01
)
