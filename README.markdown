# Handwriting synthesis

This is an implementation of ["Generating Sequences With Recurrent Neural Networks" paper by Alex Graves](https://arxiv.org/abs/1308.0850)

### Generated samples

#### Unconditional samples
Setting: ``bias=0.2, seed=range(5)``
![](asset/un1.png)
![](asset/un2.png)
![](asset/un3.png)
![](asset/un4.png)
![](asset/un5.png)

#### Conditional samples 
Setting: ``bias=2, seed=range(5)``
_Strength means blessed with an enemy_ : ![](asset/co_0.png)
_words are wind!_ : ![](asset/co_1.png)
_Bridged by a lightwave_ : ![](asset/co_2.png)
_kimi no na ma_ : ![](asset/co_3.png)
_Bonjour! On y va?_ : ![](asset/co_4.png)


### Usage
- Place data (``sentences.txt`` and strokes-py3.npy``) in ``data/`` directory.
- Use `train.py` script to train the model.
- Runs are saved in `runs/` directory.
- To generate above samples, use `models/dummy.py`
    ```python
    from models.dummy import generate_unconditionally, generate_conditionally
    for i in range(5):
        x = generate_unconditionally(i)
        plot_stroke(x,save_name=f'asset/un_{i+1}')

    texts = ['Strength means blessed with an enemy!', 'words are wind!', 'Bridged by a lightwave ', 'kimi no na ma', 'Bonjour! On y va?']

    for i in range(5):
        x = generate_conditionally(random_seed=i, text=texts[i])
        plot_stroke(x, save_name=f'asset/co_{i}')
```