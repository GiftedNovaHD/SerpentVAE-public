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

## Available Channel Mixers
- MLP
  <details>
  <summary>MLP arguments</summary>

  - `mlp_inner_dim`: The inner dimension of the MLP
  </details>

## Adding a new Channel Mixer
- Create a new file in the `channel_mixers` folder with the name of the channel mixer (if needed).
- Otherwise just import the channel mixer in `block.py` and add it to the `mixer_lst` list.
- Configure the initialization of the new channel mixer in `create_channel_mixer_block()` in `channel_mixer_block.py`.
- Add the channel mixer and its arguments to the `README.md` file (for documentation purposes).
