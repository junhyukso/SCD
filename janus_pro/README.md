# Speculative Coupled Decoding for Training-Free Lossless Acceleration of Autoregressive Visual Generation

This is the official repository for integrating SCD with the Janus Pro model, supporting inference in three modes: SJD (original), SJD + Maximal Coupling, and SJD + Gumbel Coupling.

## Installing the Dependencies
``` bash
pip install -r requirements.txt
```
</br>

## Running Text-to-Image

#### Run vanila AR model
```bash
python test_ar.py
```

#### Run with SJD
```bash
python test_scd.py --coupling_mode sjd
```

#### Run with Maximal Coupling
```bash
python test_scd.py --coupling_mode maximal
```

#### Run with Gumbel Coupling
```bash
python test_scd.py --coupling_mode gumbel
```

</br>

## Acknowledgements
This implementation is based on [tyshiwo1/Accelerating-T2I-AR-with-SJD](https://github.com/tyshiwo1/Accelerating-T2I-AR-with-SJD/), [deepseek-ai/Janus](https://github.com/deepseek-ai/Janus).