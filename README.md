# Image-Text Aggregation for Open-Vocabulary Semantic Segmentation (ITA)

## Brief

This is the implementation of paper: Shengyang Cheng, Jianyong Huang, Xiaodong Wang, Lei Huang*, Zhiqiang Wei, Image–text aggregation for open-vocabulary semantic segmentation, Neurocomputing, 2025.

## The Framework of the Proposed ITA
<table border=0 >
	<tbody>
    <tr>
		<tr>
			<td width="40%" > <img src="./img/me_1.png"> </td>
		</tr>
	</tbody>
</table>



## Datasets
See [Preparing Datesets for ITA](dataset.md).
 ### Environment

See [installation instruction](INSTALL.md).

### CLIP

See [Preparing vision-language model](laionCLIP-convnext_large_d_320.laion2B-s29B-b131K-ft-soup/vision-language-model.md).

### Training/Resume Training

```python
# Use train_net.py to train the semantic segmentation task.
python train_net.py --num-gpus 8 --config-file configs/coco/semantic-segmentation/ita/ita_convnext_large_eval_ade20k.yaml
```

### Test/Evaluation

```python
# Evaluate our ITA
python train_net.py --config-file configs/coco/semantic-segmentation/ita/ita_convnext_large_eval_ade20k.yaml --eval-only MODEL.WEIGHTS /path/to/checkpoint_file
# Evaluate our ITA in other open-vocabulary semantic segmentation datasets.
ita_convnext_large_eval_ade20k.yaml 
# Replace to
ita_convnext_large_eval_ade20k.yaml
ita_convnext_large_eval_a847.yaml
ita_convnext_large_eval_pc59.yaml
ita_convnext_large_eval_pas20.yaml
ita_convnext_large_eval_pas21.yaml
```

## Acknowledgements

Our work is based on the following theoretical works:

- [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)
- [Convolutions Die Hard: Open-Vocabulary Segmentation with Single Frozen Convolutional CLIP](https://arxiv.org/abs/2308.02487)
- [Masked-attention Mask Transformer for Universal Image Segmentation](https://arxiv.org/abs/2112.01527)

  ## Citation

If you find the code in this repository useful for your research consider citing it.

```
@article{Cheng2025ITA,
title={Image-Text Aggregation for Open-Vocabulary Semantic Segmentation},
journal = {Neurocomputing}​，
author={Shengyang Cheng, Jianyong Huang, Xiaodong Wang, Lei Huang, Zhiqiang Wei},
year={2025}
}
```

