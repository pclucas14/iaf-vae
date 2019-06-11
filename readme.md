## Inverse Autoregressive Flows 

A Pytorch implementation of [this paper](https://arxiv.org/abs/1606.04934)

Main work taken from the [official implementation](https://github.com/openai/iaf)

Running the following command gets <3.8 BPD (which is still some ways from the 3.17 in the original paper)

```
python main.py --batch_size 128 --depth 1 --n_blocks 20 --free_bits 0.128 --lr 0.002 --z_size 32 --h_size 160 --iaf 1
```

### Images
Here are some test set reconstructions, and samples 
<p align="center">
<img src="https://github.com/pclucas14/iaf-vae/images/samples_99.png">
<img src="https://github.com/pclucas14/iaf-vae/images/test_recon_99.png">
</p>
Judging by how good the reconstructions are, and how bad the samples are, maybe the free bits constraint is too loose. More investigation required.

### Contribute
I'm having trouble closing the performance gap with the offical code, as Conda does not support anymore the required tensorflow version to run it. Therefore, all contributions / comments / remarks are highly welcomed. 


