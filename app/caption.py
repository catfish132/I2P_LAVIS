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
sys.path.append("/teams/ai_model_1667305326/WujieAITeam/private/jyd/CV/recognize-anything/")
sys.path.append("/teams/ai_model_1667305326/WujieAITeam/private/jyd/CV/FastSAM-main/")
import streamlit as st
import torch.types

from app import device, load_demo_image
from app.utils import load_model_cache
from lavis.models import load_model_and_preprocess
from PIL import Image
import streamlit.components.v1 as components
# RAM
from ram.models import ram
from ram import inference_ram as inference
from ram import get_transform
# clip_interrogator
from PIL import Image
from clip_interrogator import Config, Interrogator


# SAM


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
        st.write('Use the Clip interrogator, this will cost a lot of time')
        use_ci = st.selectbox("USE CI", [False, True])

    instructions = """Try the provided image or upload your own:"""
    file = st.file_uploader(instructions)

    col1, col2 = st.columns(2)
    CI_out, ram_bar = st.columns(2)
    button_bar, oringal_out = st.columns(2)
    ram_bar.header('RAM Result')
    CI_out.header('CLIP Interrogator Result')

    if file:
        raw_img = Image.open(file).convert("RGB")
    else:
        raw_img = load_demo_image()

    col1.header("Image")

    w, h = raw_img.size
    scaling_factor = 720 / w
    resized_image = raw_img.resize((int(w * scaling_factor), int(h * scaling_factor)))
    col1.image(resized_image)
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

    blip_model, vis_processors = init_model()
    ram_model, ram_transform = load_RAM()
    ci = load_CI()

    if reload_model:
        blip_model = load(blip_model, model_type)
    if cap_button:
        img = vis_processors["eval"](raw_img).unsqueeze(0).to(device)
        out = blip_model.generate({"image": img},
                                  use_nucleus_sampling=False,
                                  num_beams=beam_search,
                                  max_length=max_l,
                                  min_length=min_l)
        blip_output = clean(out[0])
        col2.write(blip_output)
        oringal_out.write(out[0])
        # inference RAM
        ram_image = ram_transform(raw_img).unsqueeze(0).to(device)

        ram_res = inference(ram_image, ram_model)
        ram_bar.write(ram_res[0])
        # interrogator
        if use_ci:
            ci_res = ci.interrogate(raw_img)
            CI_out.write(ci_res)


@st.cache_resource
def load_RAM():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = get_transform(image_size=384)

    #######load model
    model = ram(
        pretrained="/teams/ai_model_1667305326/WujieAITeam/private/jyd/CV/recognize-anything/pretrained/ram_swin_large_14m.pth",
        image_size=384,
        vit='swin_l')
    model.eval()

    model = model.to(device)
    return model, transform

@st.cache_resource
def load_CI():
    return Interrogator(Config(clip_model_name="ViT-L-14/openai"))

# @st.cache_resource
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

app()
