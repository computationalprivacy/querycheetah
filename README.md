# QueryCheetah: Fast Discovery of Attribute Inference Attacks against Query-Based Systems

Source code repository for the ACM CCS 2024 paper [QueryCheetah: Fast Discovery of Attribute Inference Attacks against Query-Based
Systems](https://arxiv.org/abs/2409.01992) by Bozhidar Stevanoski, Ana-Maria Cretu, and Yves-Alexandre de Montjoye.

We introduce QueryCheetah, the first efficient and scalable tool for automatically finding privacy vulnerabilities in query-based systems (QBS). Our source code allows anyone to thoroughly evaluate the record-specific privacy risk of providing access to a dataset via a QBS. 
The code is structured as follows:
- `main.py` is the entry-point running QueryCheetah  
- `src` contains all accompanying utility methods needed for QueryCheetah
- `defense` contains the implementation of the QBS whose privacy guarantees you are evaluating.

While in the paper we instantiated QueryCheetah against Diffix, the most heavily studied and developed real-world QBS, we cannot release our Diffix implementation here, for intellectual property reasons. We instead release our generic class implementing a QBS together with an easily extendable implementation of a simple QBS, called `ExampleQBS`, which returns exact query answers (i.e., does not implement any defense).

The method and the code are easily extendable to other QBSs supporting the same syntax: (1) the new QBS needs to be implemented in `defense/qbs.py` as a class inheriting from the abstract `QBS` class and (2) the factory method `_get_qbs` in `src/utils.py` needs to be updated to return objects of the new class.

# 1. Python environment
The source code uses commonly used Python libraries. To be able to run the code, first please install the dependencies
in the `requirements.txt` file as follows:
```
conda create --name querycheetah_env python=3.9
conda activate querycheetah_env
pip install -r requirements.txt
```

# 2. Dataset pre-processing

First, run the notebook `notebooks/dataset_preprocessing.ipynb`.
It contains (1) instructions where to find the datasets used in the paper and (2) code to preprocess them once downloaded.


# 3. Running QueryCheetah

To search for attacks, run `main.py` by specifying as command-line arguments, such as:

- `only_limited_syntax`: an indicator if QueryCheetah will search only withing the limited syntax;
- `use_mitigations`: an indicator if the attacked QBS implements mitigations;
- `attack_type`: an indicator if QueryCheetah will search for attribute inference or membership inference attacks;
- `only_post_hoc`: an indicator if the attacks discovered against a QBS without mitigations are applied to a QBS that implements
  mitigations;
- `dataset_name`: the name of the dataset;
- `qbs`: the name of the QBS.

For example, to search for attribute inference attacks in the limited syntax against the ExampleQBS that does not implement
mitigations, run:

```
python3 main.py --only_limited_syntax=1 --use_mitigations=0  --attack_type="aia" --only_post_hoc=0 --dataset_name="adult" --qbs="example"
```

# How to cite

If you re-use QueryCheetah, please cite our paper:

```
@inproceedings{stevanoski2024querycheetah,
  title={QueryCheetah: Fast Automated Discovery of Attribute Inference Attacks Against Query-Based Systems},
  author={Stevanoski, Bozhidar, Ana-Maria Cretu, and Yves-Alexandre de Montjoye},
  booktitle={Proceedings of the 2024 ACM SIGSAC Conference on Computer and Communications Security},
  year={2024}
}
```
