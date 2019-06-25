## Inverse Autoregressive Flows 

A Pytorch implementation of [this paper](https://arxiv.org/abs/1606.04934)

Main work taken from the [official implementation](https://github.com/openai/iaf)

Running the following command gets ~ 3.35 BPD (which is still some ways from the 3.11 in the original paper)

```
python main.py --batch_size 128 --depth 1 --n_blocks 20 --free_bits 0.001 --lr 0.002 --z_size 32 --h_size 160 --iaf 1
```

As comparison, the baseline (IAF disabled) gets ~ 3.55 BPD. You can test with the following command 
```
python main.py --batch_size 128 --depth 1 --n_blocks 20 --free_bits 0.001 --lr 0.002 --z_size 32 --h_size 160 --iaf 0
```


### Images
Here are some test set reconstructions, and samples 
<p align="center">
<img src="https://github.com/pclucas14/iaf-vae/blob/master/images/test_99.png">
<img src="https://github.com/pclucas14/iaf-vae/blob/master/images/sample_999.png">
</p>
Judging by how good the reconstructions are,  maybe the free bits constraint is too loose. More investigation required.

### Contribute
I'm having trouble closing the performance gap with the offical code, as Conda does not support anymore the required tensorflow version to run it. Therefore, all contributions / comments / remarks are highly welcomed. 


