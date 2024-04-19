# Sus-Net

### Setup

1. Creating python virtual environment

```bash
conda env create -f environment.yml
conda activate sus_net
```

2. Setup pre-commit hooks

```bash
pre-commit install
```

3. Have fun!

### Possible Experiments:

1. Dummy state (that we pad when sequence length is greater than state history) all zeros vs start state repeated
