# Prompting via Activation Maximization

Complete code for my prompting via activation maximization experiment.

We load an off-the-shelf, generatively pretrained Large Language model, weave a randomly initialized "primer" into the context window and target the _primer_ for optimization.

This resembles tuning, except model weights are left untouched, there is no LoRA and the trained primer is provided wholly in-band.

In this way, we apply backpropagation and stochastic gradient descent to the problem of prompt engineering.

## Instructions

I've written this experiment against the Yelp Review Polarity dataset.

1. Activate a Python virtual environment; install with `requirements.txt`.
2. Download the [Yelp Review Polarity](https://www.kaggle.com/datasets/irustandi/yelp-review-polarity) dataset. Extract as `yelp_review_polarity`.
3. Run `prep.sh` to encode the test and train datasets. (Open this file to configure dataset size.)
4. (Optional) Run `python prompt.long.py` or `python prompt.short.py` to run the plain-text prompts.
5. Run `python experiment.py` to train a primer, and run it against the test set.
6. (Optional) Find main configuration at the top of `experiment.py`

## My Tests

All tests conducted on an Nvidia 3090, 24gb.

### Text Prompts

| Prompt         | Tokens | Score | Correct / Total |
|----------------|---|-------|-----------------|
| long prompt | 83 | 57%   | 21,948 / 38,000 |
| short prompt   | 35 | 56%   | 21,550 / 38,000 |

### Synthetic Prompts

| Samples | Tokens | LR Schedule                               | Score | Correct / Total | Notes                                                        |
|---------|---------------|-------------------------------------------|-------|-----------------|--------------------------------------------------------------|
| 5,600   | 16            | 1e-5                                      | 57%   | 21,916 / 38,000 | –                                                            |
| 5,600   | 16            | 1e-4, 1e-5, 1e-6                          | 94%   | 2,269 / 2,400   | first multi-epoch; degrading LR; reduced test set            |
| 5,600   | 8             | 1e-4, 1e-5, 1e-6                          | 91%   | 2,197 / 2,400   | try a smaller primer                                         |
| 5,600   | 4             | 1e-4, 1e-5, 1e-6                          | 86%   | 2,078 / 2,400   | –                                                            |
| 5,600   | 4             | 1e-4, 1e-5, 1e-6, 5e-7, 1e-7              | 87.5% | 2,100 / 2,400   | –                                                            |
| 58,000  | 4             | 1e-4, 1e-5, 1e-6, 5e-7, 1e-7              | 95.9% | 36,442 / 38,000 | –                                                            |

