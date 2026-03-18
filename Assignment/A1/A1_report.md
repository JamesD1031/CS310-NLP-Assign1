# Assignment 1 Report

## 1. Data processing choices

The dataset is stored in JSONL format, with one example per line:

```json
{"sentence": "...", "choices": ["0", "1"], "label": [0], "id": "..."}
```

I loaded both `train.jsonl` and `test.jsonl` with Python's standard `json` module and converted each label from a one-element list such as `[0]` or `[1]` into a scalar integer `0` or `1`.

For training and validation, I performed a stratified split on `train.jsonl`, using 90% for training and 10% for validation. The provided `test.jsonl` file is labeled, so I used it as the final held-out test set.

## 2. Basic vs improved tokenizer

### Basic tokenizer

- Treat each Chinese character as one token.
- Discard non-Chinese content such as English letters, digit sequences, and punctuation.
- Example:
  - Input: `刑警小段男说：那好，那先把你的BP机拿出来，给我们看看`
  - Output: `['刑', '警', '小', '段', '男', '说', '那', '好', '那', '先', '把', '你', '的', '机', '拿', '出', '来', '给', '我', '们', '看', '看']`

### Improved tokenizer

- Use regex to preserve:
  - consecutive English words
  - consecutive digits
  - punctuation
- Use `jieba` on Chinese spans to segment multi-character Chinese words.
- Example:
  - Input: `刑警小段男说：那好，那先把你的BP机拿出来，给我们看看`
  - Output: `['刑警', '小段', '男', '说', '：', '那好', '，', '那先', '把', '你', '的', 'bp', '机拿', '出来', '，', '给', '我们', '看看']`

## 3. Vocabulary size comparison

Vocabulary was built from the full training set.

| Tokenizer | Vocabulary size | Chinese unit | Keeps English / digits / punctuation |
| --- | ---: | --- | --- |
| Basic | `2687` | Single Chinese character | No |
| Improved | `13840` | `jieba` multi-character Chinese words | Yes |

Difference: `+11153`

The improved tokenizer produces a much larger vocabulary because it keeps punctuation, English words, and digit sequences, while also introducing many multi-character Chinese tokens from `jieba`.

## 4. Model architecture

The model is a bag-of-words neural classifier implemented with `torch.nn`.

- Embedding layer: `nn.EmbeddingBag(vocab_size, 128, mode="mean")`
- Fully connected classifier:
  - `Linear(128, 128) -> ReLU -> Dropout(0.2)`
  - `Linear(128, 64) -> ReLU -> Dropout(0.2)`
  - `Linear(64, 2)`

This satisfies the assignment requirement of using `EmbeddingBag` and having at least two hidden layers in the fully connected part.

## 5. Training setup

- Random seed: `310`
- Optimizer: `Adam`
- Learning rate: `1e-3`
- Scheduler: `ReduceLROnPlateau`
- Batch size: `64`
- Epochs: `12`
- Loss: weighted cross-entropy

Because the training data is imbalanced, I used class weights in the cross-entropy loss to avoid collapsing toward the majority non-humor class.

## 6. Final test metrics

The final notebook uses the improved tokenizer and keeps the model checkpoint with the best validation F1 from the train/validation split.

- Test accuracy: `0.5883`
- Test precision: `0.3533`
- Test recall: `0.6941`
- Test F1: `0.4683`

Confusion counts on the test set:

- TP = `118`
- TN = `265`
- FP = `216`
- FN = `52`

## 7. Brief discussion

The improved tokenizer clearly increases vocabulary coverage, but the final performance is still moderate. This is reasonable because the model is still a bag-of-words model and does not capture word order, speaker context, or deeper discourse cues, all of which matter for humor detection.

The model tends to predict the positive class more aggressively, which improves recall but lowers precision. The weighted loss helps recover positive examples, but it also increases false positives. A stronger model would likely need contextual representations rather than a pure bag-of-words architecture.
