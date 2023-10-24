<!-- MIT License

Copyright (c) 2023  Sehyun Choi and The HuggingFace Inc. team

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE. -->

# RetNet

## Overview

The RetNet model proposed in
[Retentive Network: A Successor to Transformer for Large Language Models](https://arxiv.org/abs/2307.08621).
For detail about the architecture, please read the original paper and this
[blog post](https://medium.com/@choisehyun98/the-rise-of-rnn-review-of-retentive-network-a080a9a1ad1d).

***Abstract:***

*"In this work, we propose Retentive Network (RetNet) as a foundation architecture*
*for large language models, simultaneously achieving training parallelism,*
*low-cost inference, and good performance. We theoretically derive the connection*
*between recurrence and attention. Then we propose the retention mechanism for*
*sequence modeling, which supports three computation paradigms, i.e., parallel,*
*recurrent, and chunkwise recurrent. Specifically, the parallel representation*
*allows for training parallelism. The recurrent representation enables low-cost*
*O(1) inference, which improves decoding throughput, latency, and GPU memory*
*without sacrificing performance. The chunkwise recurrent representation*
*facilitates efficient long-sequence modeling with linear complexity, where each*
*chunk is encoded parallelly while recurrently summarizing the chunks.*
*Experimental results on language modeling show that RetNet achieves favorable*
*scaling results, parallel training, low-cost deployment, and efficient*
*inference. The intriguing properties make RetNet a strong successor to*
*Transformer for large language models. Code will be available at*
*[this https URL](https://www.github.com/microsoft/torchscale)."*


## Model Details

During training, use `forward_mode='parallel'` or `forward_mode='chunkwise'` if the sequence length
gets longer, such as 8k or 16k. For inference, the prompt tokens should be processed with
`forward_mode='parallel'` and the completions with `forward_mode='recurrent'` for best performance.
This is automatically handled when called `.generate()`.

This model was contributed by [syncdoth](<https://huggingface.co/syncdoth).
The original code can be found [here](https://github.com/syncdoth/retnet).


## License

RetNet models are released under the MIT license.

## Usage

New RetNetModels can be instantiated as follows:

```python
>>> from transformers import RetNetConfig, RetNetForCausalLM, AutoTokenizer
>>> device = "cuda" # the device to load the model onto

>>> tokenizer = AutoTokenizer.from_pretrained("openlm-research/open_llama_3b_v2")  # or any tokenizer of your choice
>>> tokenizer.pad_token_id = tokenizer.unk_token_id  # specific to open_llama models
>>> config = RetNetConfig(vocab_size=len(tokenizer),
...                       pad_token_id=tokenizer.pad_token_id,
...                       eos_token_id=tokenizer.eos_token_id,
...                       bos_token_id=tokenizer.bos_token_id)
>>> model = RetNetForCausalLM(config)

>>> prompt = "The model is not trained yet; the output"

>>> model_inputs = tokenizer([prompt], return_tensors="pt").to(device)
>>> model.to(device)

>>> generated_ids = model.generate(**model_inputs, max_new_tokens=100, do_sample=True)
>>> tokenizer.batch_decode(generated_ids)[0]
"will not make sense."
```

## RetNetConfig

[[autodoc]] RetNetConfig

## RetNetModel

[[autodoc]] RetNetModel
    - forward


## RetNetForCausalLM

[[autodoc]] RetNetForCausalLM
    - forward

## RetNetForSequenceClassification

[[autodoc]] transformers.RetNetForSequenceClassification
    - forward
