# Better Text Generation with MLE-GAN

To reproduce the results from the paper follow the following instructions:

##### Train the baseline MLE model
* python ptb_word_lm.py --data_path=data/ --save_path=mle/ --train_lm --total_epochs=13 --save_embeddings
* Test set perplexity will be printed at the end.

##### Train the MLE-GAN model
* python ptb_word_lm.py --data_path=data/ --save_path=gan/ --train_lm --train_gan --load_embeddings --lm_lr=2e-3 --g_lr=2e-3 --d_lr=5e-4 --gan_steps=40 --toal_epochs=26
* Test set perplexity will be printed at the end.

#### Test MLE model and save evaluation metrics
* python ptb_word_lm.py --data_path=data/ --save_path=mle/ --sample_mode --npy_suffix=mle
* When asked 'Enter your sample prefix:' enter the word 'the' (without apostrophes).
* When asked 'Sample size:' enter the number 19.
* %-in-test-n, BLUE-n and distinct-n (for n=1,2,3,4) will be printed at the end along with 100 examples of sentences.

#### Test MLE-GAN model and save evaluation metrics
* python ptb_word_lm.py --data_path=data/ --save_path=gan/ --sample_mode --npy_suffix=gan
* When asked 'Enter your sample prefix:' enter the word 'the' (without apostrophes).
* When asked 'Sample size:' enter the number 19.
* %-in-test-n, BLUE-n and distinct-n (for n=1,2,3,4) will be printed at the end along with 100 examples of sentences.

#### Print p-value for %-in-test-n and BLEU-n
* python compute_pvalue.py
