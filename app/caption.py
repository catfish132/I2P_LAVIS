"""
 # Copyright (c) 2022, salesforce.com, inc.
 # All rights reserved.
 # SPDX-License-Identifier: BSD-3-Clause
 # For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import sys
import os

# 获取当前工作目录
current_dir = os.getcwd()
# print('cwd:' + current_dir)
# 将当前工作目录添加到 sys.path 的开头
sys.path.insert(0, current_dir)

import streamlit as st
import torch.types

from app import device, load_demo_image
from app.utils import load_model_cache
from lavis.models import load_model_and_preprocess
from lavis.processors import load_processor
from PIL import Image

def clean(strs):
    # 将输入字符串按逗号分割成单词列表
    words = strs.split(',')

    # 使用集合去除重复单词，并保持它们的顺序
    unique_words = list(dict.fromkeys(words))

    # 将去重后的单词再次合并成字符串
    result = ','.join(unique_words)

    return result

def app():
    # ===== layout =====
    device = torch.device("cuda")
    st.markdown(
        "<h1 style='text-align: center;'>Image Description Generation</h1>",
        unsafe_allow_html=True,
    )

    instructions = """Try the provided image or upload your own:"""
    file = st.file_uploader(instructions)

    col1, col2 = st.columns(2)

    if file:
        raw_img = Image.open(file).convert("RGB")
    else:
        raw_img = load_demo_image()

    col1.header("Image")

    w, h = raw_img.size
    scaling_factor = 720 / w
    resized_image = raw_img.resize((int(w * scaling_factor), int(h * scaling_factor)))
    col1.image(resized_image, use_column_width=True)
    col2.header("Description")
    cap_button = st.button("Generate")
    model, vis_processors = load_model()

    if cap_button:
        # img = vis_processors(raw_img).unsqueeze(0).to(device)
        img = vis_processors["eval"](raw_img).unsqueeze(0).to(device)
        out = model.generate({"image": img},
                             use_nucleus_sampling=False,
                             num_beams=3,
                             max_length=100,
                             min_length=10)
        # captions = generate_caption(
        #     model=model, image=img, use_nucleus_sampling=not use_beam
        # )
        output = clean(out[0])
        col2.write(output, use_column_width=True)


@st.cache_resource
def load_model():
    model, vis_processors, _ = load_model_and_preprocess(name="blip_caption", model_type="blip_img2tag", is_eval=True,
                                                         device=device)
    # model_type选择
    # 原版模型：large_coco
    # 长度修改之后：blip_img2tag
    model.load_checkpoint(
        '/teams/ai_model_1667305326/WujieAITeam/private/jyd/img2prompt/LAVIS/lavis/output/minicoco_enhanced_maxlen/20230719061/checkpoint_best.pth'
    )
    # 3vj_room:"/teams/ai_model_1667305326/WujieAITeam/private/jyd/img2prompt/LAVIS/lavis/output/3vj_Room/20230707100/checkpoint_best.pth"
    # 3vj_room_maxlen:"/teams/ai_model_1667305326/WujieAITeam/private/jyd/img2prompt/LAVIS/lavis/output/minicoco_enhanced_maxlen/20230719061/checkpoint_best.pth"
    # minicoco:'/teams/ai_model_1667305326/WujieAITeam/private/jyd/img2prompt/LAVIS/lavis/output/minicoco/20230712072/checkpoint_best.pth'
    # huaban_room: /teams/ai_model_1667305326/WujieAITeam/private/jyd/img2prompt/LAVIS/lavis/output/Huaban_Room/checkpoint_best.pth
    # minicoco_enhanced_ram: /teams/ai_model_1667305326/WujieAITeam/private/jyd/img2prompt/LAVIS/lavis/output/minicoco_enhanced/checkpoint_best.pth
    # minicoco_enhanced_maxlen: /teams/ai_model_1667305326/WujieAITeam/private/jyd/img2prompt/LAVIS/lavis/output/minicoco_enhanced_maxlen/20230719061/checkpoint_best.pth
    model.to(device)
    return model, vis_processors


# def generate_caption(
#         model, image, use_nucleus_sampling=False, num_beams=3, max_length=40, min_length=5
# ):
#     samples = {"image": image}
#
#     captions = []
#     if use_nucleus_sampling:
#         for _ in range(5):
#             caption = model.generate(
#                 samples,
#                 use_nucleus_sampling=True,
#                 max_length=max_length,
#                 min_length=min_length,
#                 top_p=0.9,
#             )
#             captions.append(caption[0])
#     else:
#         caption = model.generate(
#             samples,
#             use_nucleus_sampling=False,
#             num_beams=num_beams,
#             max_length=max_length,
#             min_length=min_length,
#         )
#         captions.append(caption[0])
#
#     return captions


app()
