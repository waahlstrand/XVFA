# 🩻 XVFA 
The **XVFA** (*e**x**plainable **v**ertebral **f**racture **a**ssessment*) from the paper [*Explainable vertebral fracture analysis with uncertainty estimation using differentiable rule-based classification*]() is an automated deep learning model for the detection and classification of vertebral fractures in lateral spine radiographs. The model is based on a differentiable rule-based classification approach that allows for the generation of explanations for the model's predictions. The model also provides uncertainty estimates for its predictions, which can be used to assess the reliability of the model's predictions.

The paper was accepted at the [MICCAI 2024](https://www.miccai2024.org/) conference, but a preprint version is available on [arXiv](https://arxiv.org/abs/2407.02926).

## Installation
The **XVFA** model is implemented in Python using the PyTorch and Lightning libraries. Clone this repository to your local machine and install the required dependencies using the following command:
```bash
pip install -r requirements.txt
```

### Installing DETR-DINO
The project depends on the DETR-DINO repository.

<details>

   1. Clone the repo in the ``models/backbones`` folder
   ```sh
   cd models/backbones
   git clone https://github.com/IDEA-Research/DINO.git
   cd DINO
   ```

   2. Install other needed packages
   ```sh
   pip install -r requirements.txt
   ```

   4. Compiling CUDA operators
   ```sh
   cd models/dino/ops
   python setup.py build install
   # unit test (should see all checking is True)
   python test.py
   cd ../../..
   ```
</details>


## Training
The model can easily be trained on new data using the Lightning framework, by providing a Lightning datamodule adhering to the `BaseDataModule` class in `xvfa/datamodules/base_datamodule.py`. See the [Lightning documentation](https://pytorch-lightning.readthedocs.io/en/latest/datamodules.html) for more information on how to create a custom datamodule.

All models may also be trained in a conventional PyTorch training loop, although examples of this are not provided in this repository.

## Cite this work
If you use the **XVFA** model in your research, please cite the following paper:
```
@article{,
  title={Explainable vertebral fracture analysis with uncertainty estimation using differentiable rule-based classification},
  author={V. Wåhlstrand Skärström, L. Johansson, J. Alvén, M. Lorentzon and I. Häggström},
  journal={Lecture Notes in Computer Science, Medical Image Computing and Computer-Assisted Intervention – MICCAI 2024},
  year={2024},
  volume={},
  number={},
  pages={},
  doi={}
}
```
