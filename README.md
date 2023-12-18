# HexPlane nerfstudio integration
HexPlane model implementation based on nerfstudio

> As it is not completed yet and requires few (but important) additional implementations, it'll be welcomed if you want to contribute to this repository. (Please refer to **"Roadmap"** section at bottom of the README)
>
> Please feel free to send an e-mail to me or leave a comment/issue here. 

## Description

## Installation

**0. Install nerfstudio dependencies and nerfstudio**

Refer to the [nerfstudio installation document](https://docs.nerf.studio/en/latest/quickstart/installation.html) to install nerfstudio dependencies and nerfstudio.

**1. Clone this repo**

```
git clone https://github.com/wowo0709/hexplane-nerfstudio.git
```

**2. Install this repo as a python package**

Navigate to this folder and run below command.

```
python -m pip install -e .
```

**3. Run `ns-install-cli`** 

This needs to be rerun when the CLI changes, for example if nerfstudio is updated. 

```
ns-install-cli
```

**4. Check the installation**

With below command, use should see a list of "subcommands" with `hexplane` included among them. 

```
ns-train -h
```

With below command, use should see lists of parameters that we can change while using `hexplane`. 

```
ns-train hexplane -h
```



 



## Using HexPlane-nerfstudio

**1. Download/Prepare the dataset**

For DNeRF dataset, you can use below command. (Refer to [nerfstudio document](https://docs.nerf.studio/en/latest/reference/cli/ns_download_data.html) for further details)

It will download the DNeRF dataset under the `data` folder. 

```
ns-download-data dnerf
```

If you encounter an error, simply download the dataset [here](https://www.dropbox.com/s/0bf6fl0ye2vz3vr/data.zip?dl=0) (Thanks to [D-NeRF repository](https://github.com/albertpumarola/D-NeRF))

**2. Run `hexplane`**

Run hexplane model with below command. (Refer to [nerfstudio document](https://docs.nerf.studio/en/latest/quickstart/first_nerf.html) for further details)

```
ns-train hexplane --data <data_folder>
```




## Results on DNeRF dataset
**Qualitative results**


**Quantitative results**


**Ablations**



## Roadmap

Expected future updates to this repository: 

 - [ ] Change beta value of Adam optimizer according to the paper
 - [ ] Implement TV regularization
 - [ ] Include other dataset (such as Plenoptic video dataset which was used in the paper)
 - [ ] Support depth loss

## References

- [Nerfstudio repository](https://github.com/nerfstudio-project/nerfstudio)
- [HexPlane repository (official)](https://github.com/Caoang327/HexPlane)
