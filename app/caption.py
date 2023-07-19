"""
 # Copyright (c) 2022, salesforce.com, inc.
 # All rights reserved.
 # SPDX-License-Identifier: BSD-3-Clause
 # For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import streamlit as st
import torch.types

from app import device, load_demo_image
from app.utils import load_model_cache
from lavis.models import load_model_and_preprocess
from lavis.processors import load_processor
from PIL import Image


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
        out = model.generate({"image": img})
        # captions = generate_caption(
        #     model=model, image=img, use_nucleus_sampling=not use_beam
        # )

        col2.write(out[0], use_column_width=True)


@st.cache_resource
def load_model():
    model, vis_processors, _ = load_model_and_preprocess(name="blip_caption", model_type="large_coco", is_eval=True,
                                                         device=device)
    # vis_processor = load_processor("blip_image_eval").build(image_size=384)
    # if model_type.startswith("BLIP"):
    #     blip_type = model_type.split("_")[1].lower()
    #     model = load_model_cache(
    #         "blip_caption",
    #         model_type=f"{blip_type}_coco",
    #         is_eval=True,
    #         device=device,
    #         # checkpoint="/teams/ai_model_1667305326/WujieAITeam/private/jyd/img2prompt/LAVIS/lavis/output/3vj_Room/20230707100/checkpoint_best.pth"
    #     )
    model.load_checkpoint(
        '/teams/ai_model_1667305326/WujieAITeam/private/jyd/img2prompt/LAVIS/lavis/output/minicoco_enhanced/20230718020/checkpoint_best.pth'
    )
    # 3vj_room:"/teams/ai_model_1667305326/WujieAITeam/private/jyd/img2prompt/LAVIS/lavis/output/3vj_Room/20230707100/checkpoint_best.pth"
    # minicoco:'/teams/ai_model_1667305326/WujieAITeam/private/jyd/img2prompt/LAVIS/lavis/output/minicoco/20230712072/checkpoint_best.pth'
    # huaban_room: /teams/ai_model_1667305326/WujieAITeam/private/jyd/img2prompt/LAVIS/lavis/output/Huaban_Room/checkpoint_best.pth
    # minicoco_enhanced_ram: /teams/ai_model_1667305326/WujieAITeam/private/jyd/img2prompt/LAVIS/lavis/output/minicoco_enhanced/checkpoint_best.pth
    model.to(device)
    return model, vis_processors


def generate_caption(
        model, image, use_nucleus_sampling=False, num_beams=3, max_length=40, min_length=5
):
    samples = {"image": image}

    captions = []
    if use_nucleus_sampling:
        for _ in range(5):
            caption = model.generate(
                samples,
                use_nucleus_sampling=True,
                max_length=max_length,
                min_length=min_length,
                top_p=0.9,
            )
            captions.append(caption[0])
    else:
        caption = model.generate(
            samples,
            use_nucleus_sampling=False,
            num_beams=num_beams,
            max_length=max_length,
            min_length=min_length,
        )
        captions.append(caption[0])

    return captions


app()
