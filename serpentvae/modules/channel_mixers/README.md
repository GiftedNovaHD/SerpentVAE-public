# SerpentVAE Channel Mixers

This directory contains the channel mixers for the SerpentVAE. 

File Structure:

```sh
channel_mixers/
├── channel_mixer_mlp.py
├── channel_mixer_block.py
├── __init__.py
└── README.md
```


## Available Sequence Mixers
- Mamba2
  <details>
  <summary>Mamba2 arguments</summary>
  - `d_model`: The dimension of the model
  - `d_state`: The size of the state
  - `d_conv`: The length of the convolution
  - `expand`: The expansion factor
  - `headdim`: The dimension of the head
  </details>

- Mamba1
  <details>
  <summary>Mamba1 arguments</summary>
  - `d_model`: The dimension of the model
  - `d_state`: The size of the state
  - `d_conv`: The length of the convolution
  - `expand`: The expansion factor
  </details>

- MultiLatentAttention
  - Work in progress


## Adding a new sequence mixer
- Create a new file in the `sequence_mixers` folder with the name of the sequence mixer (if needed).
- Otherwise just import the sequence mixer in `block.py` and add it to the `mixer_lst` list.
- Configure the initialization of the new sequence mixer in `create_block()` in `block.py`.
- Add the sequence mixer and its arguments to the `README.md` file.
