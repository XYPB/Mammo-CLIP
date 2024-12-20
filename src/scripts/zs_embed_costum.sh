#!/bin/sh

python ./src/codebase/eval_zero_shot_clip.py --config-name zs_clip_embed.yaml hydra.run.dir="./src/codebase/outputs/zs_pred" model.clip_check_point="./src/codebase/outputs/Mammo-CLIP_ckpt/b5-model-best-epoch-7.tar"

python ./src/codebase/eval_zero_shot_clip.py --config-name zs_clip_embed.yaml hydra.run.dir="./src/codebase/outputs/zs_pred" model.clip_check_point="./src/codebase/outputs/Mammo-CLIP_ckpt/b2-model-best-epoch-10.tar"