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
import streamlit.components.v1 as components


def clean(strs):
    """
        用于后处理生成的tags
    """
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
        "<h1 style='text-align: center;'>Image to Tags Generation</h1>",
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.write("推荐使用maxlen后缀的模型，他们可以生成更长的tags")
        model_type = st.selectbox(
            "model type:",
            ["minicoco_enhanced_maxlen", "3vj_room_maxlen", "img2prompt_maxlen", "3vj_room", "minicoco", "huaban_room",
             "minicoco_enhanced_ram"]
        )
        st.write("beam_search 会影响推理速度,提升推理质量，选择1即不使用")
        beam_search = st.selectbox(
            "beam search", [1, 2, 3, 4, 5]
        )
        st.write("最大生成长度,默认100，img2prompt可选择200")
        max_l = st.slider('MAX SEQ LENGTH', value=100, max_value=200, min_value=0)
        st.write("最小生成长度，默认10")
        min_l = st.slider('MIN SEQ LENGTH', value=10, max_value=200, min_value=0)

    instructions = """Try the provided image or upload your own:"""
    file = st.file_uploader(instructions)

    col1, col2 = st.columns(2)
    button_bar, oringal_out = st.columns(2)

    # embed streamlit docs in a streamlit app  跨域无法嵌套
    # st.write("CLIP-Interrogator")
    # components.iframe("https://huggingface.co/spaces/pharma/CLIP-Interrogator", width=1000, height=500, scrolling=True)
    # st.write("deepdanbooru")
    # components.iframe("http://dev.kanotype.net:8003/deepdanbooru/", width=1000, height=500, scrolling=True)  # http://dev.kanotype.net:8003/deepdanbooru/
    if file:
        raw_img = Image.open(file).convert("RGB")
    else:
        raw_img = load_demo_image()

    col1.header("Image")

    w, h = raw_img.size
    scaling_factor = 720 / w
    resized_image = raw_img.resize((int(w * scaling_factor), int(h * scaling_factor)))
    col1.image(resized_image, use_column_width=True)
    col2.header("Cleaned Tags")

    with button_bar:
        st.header("Button Bar")
        cap_button = st.button("Generate")
        st.markdown(
            "<h8> 如果修改了侧边栏的模型类型，请点击'reload_model',以重新加载ckpt</h8>",
            unsafe_allow_html=True,
        )
        reload_model = st.button("reload_model")
    with oringal_out:
        st.header("The original output without clean")

    model, vis_processors = init_model()

    if reload_model:
        model = load(model, model_type)
    if cap_button:
        img = vis_processors["eval"](raw_img).unsqueeze(0).to(device)
        out = model.generate({"image": img},
                             use_nucleus_sampling=False,
                             num_beams=beam_search,
                             max_length=max_l,
                             min_length=min_l)
        output = clean(out[0])
        col2.write(output, use_column_width=True)
        oringal_out.write(out[0], use_column_width=True)


@st.cache_resource
def init_model():
    model, vis_processors, _ = load_model_and_preprocess(name="blip_caption", model_type="blip_img2tag", is_eval=True,
                                                         device=device)
    # model_type选择
    # 原版模型：large_coco 原版只能生成长度有限的tags
    # 长度修改之后：blip_img2tag
    model = load(model)
    model.to(device)
    return model, vis_processors


def load(model, model_type="minicoco_enhanced_maxlen"):
    # 默认使用通用域模型
    model_map = {  # 模型和路径的映射
        "3vj_room": "/teams/ai_model_1667305326/WujieAITeam/private/jyd/img2prompt/LAVIS/lavis/output/3vj_Room/20230707100/checkpoint_best.pth",
        "3vj_room_maxlen": "/teams/ai_model_1667305326/WujieAITeam/private/jyd/img2prompt/LAVIS/lavis/output/3vj_Room_maxlen/20230720014/checkpoint_best.pth",
        "minicoco": '/teams/ai_model_1667305326/WujieAITeam/private/jyd/img2prompt/LAVIS/lavis/output/minicoco/20230712072/checkpoint_best.pth',
        "huaban_room": "/teams/ai_model_1667305326/WujieAITeam/private/jyd/img2prompt/LAVIS/lavis/output/Huaban_Room/checkpoint_best.pth",
        "minicoco_enhanced_ram": "/teams/ai_model_1667305326/WujieAITeam/private/jyd/img2prompt/LAVIS/lavis/output/minicoco_enhanced/checkpoint_best.pth",
        "minicoco_enhanced_maxlen": "/teams/ai_model_1667305326/WujieAITeam/private/jyd/img2prompt/LAVIS/lavis/output/minicoco_enhanced_maxlen/20230720025/checkpoint_best.pth",
        "img2prompt_maxlen": "/teams/ai_model_1667305326/WujieAITeam/private/jyd/img2prompt/LAVIS/lavis/output/I2P_maxlen/20230720073/checkpoint_best.pth"
    }
    model.load_checkpoint(model_map[model_type])
    return model


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
