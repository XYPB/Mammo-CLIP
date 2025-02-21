#!/bin/sh

python ./src/codebase/eval_zero_shot_clip.py --config-name zs_clip_embed_birads_screen.yaml hydra.run.dir="./src/codebase/outputs/b5_birads_screen_zs_pred" model.clip_check_point="./src/codebase/outputs/Mammo-CLIP_ckpt/b5-model-best-epoch-7.tar"

python ./src/codebase/eval_zero_shot_clip.py --config-name zs_clip_embed_birads_screen.yaml hydra.run.dir="./src/codebase/outputs/b2_birads_screen_zs_pred" model.clip_check_point="./src/codebase/outputs/Mammo-CLIP_ckpt/b2-model-best-epoch-10.tar"

python ./src/codebase/eval_zero_shot_clip.py --config-name zs_clip_embed_birads.yaml hydra.run.dir="./src/codebase/outputs/b5_birads_zs_pred" model.clip_check_point="./src/codebase/outputs/Mammo-CLIP_ckpt/b5-model-best-epoch-7.tar"

python ./src/codebase/eval_zero_shot_clip.py --config-name zs_clip_embed_birads.yaml hydra.run.dir="./src/codebase/outputs/b2_birads_zs_pred" model.clip_check_point="./src/codebase/outputs/Mammo-CLIP_ckpt/b2-model-best-epoch-10.tar"

python ./src/codebase/eval_zero_shot_clip.py --config-name zs_clip_embed_density.yaml hydra.run.dir="./src/codebase/outputs/b5_density_zs_pred" model.clip_check_point="./src/codebase/outputs/Mammo-CLIP_ckpt/b5-model-best-epoch-7.tar"

python ./src/codebase/eval_zero_shot_clip.py --config-name zs_clip_embed_density.yaml hydra.run.dir="./src/codebase/outputs/b2_density_zs_pred" model.clip_check_point="./src/codebase/outputs/Mammo-CLIP_ckpt/b2-model-best-epoch-10.tar"