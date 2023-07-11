"""
 # Copyright (c) 2022, salesforce.com, inc.
 # All rights reserved.
 # SPDX-License-Identifier: BSD-3-Clause
 # For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from PIL import Image
import requests

import streamlit as st
import torch


@st.cache_data()
def load_demo_image():
    img_url = (
        "//3vj-render.3vjia.com//UpFile_Render/C00000022/DesignSchemeRenderFile/201708/3/05629590/4ae9c0895e214857916b7ceb93740e9c.jpg"
    )
    raw_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")
    return raw_image
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cache_root = "/export/home/.cache/lavis/"
