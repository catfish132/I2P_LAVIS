# LAVIS官方教程
https://opensource.salesforce.com/LAVIS/latest/tutorial.processors.html
按照教程中关于数据集的指导，一步一步走就可以实现。
[官方README](README_origin.md)
# Image2Prompt/Tags效果
[Image2Tags项目介绍.pdf](Image2Tags项目介绍.pdf)
# 使用
定义了四个自定义的任务（都衍生自caption）
1. i2p_caption(image2prompt) : lavis/projects/blip/i2p_cap_ft_iter.yaml
   1. 使用来自线上的数据，9w+图文对，文本标注：用chatGPT处理用户输入的tags获得的prompt。图片：利用prompt生成的图片。
   2. 数据集路径："/teams/ai_model_1667305326/WujieAITeam/private/jyd/dataset/image2prompt/"
   3. 模型路径："/teams/ai_model_1667305326/WujieAITeam/private/jyd/img2prompt/LAVIS/lavis/output/I2P/checkpoint_best.pth"
   4. 注意：prompt格式地文本带有特殊符号，BLIP processor会清理这些格式，如果需要进一步优化，得注册一个新的processor
2. room_caption: lavis/projects/blip/room_cap_ft_iter.yaml
   1. 使用来自三维家的数据集，1200张图片
   2. 数据集路径：/teams/ai_model_1667305326/WujieAITeam/private/jyd/dataset/Room/
   3. 模型路径："/teams/ai_model_1667305326/WujieAITeam/private/jyd/img2prompt/LAVIS/lavis/output/3vj_Room/checkpoint_best.pth"
3. huaban_room_caption: lavis/projects/blip/coco_cap_ft_iter.yaml
   1. 使用来自花瓣网的数据集，6900张图片
   2. 数据集路径："/teams/ai_model_1667305326/WujieAITeam/private/jyd/dataset/huaban/"
   3. 模型路径："/teams/ai_model_1667305326/WujieAITeam/private/jyd/img2prompt/LAVIS/lavis/output/Huaban_Room/checkpoint_best.pth"
4. minicoco_caption: lavis/projects/blip/minicoco_cap_ft_iter.yaml
   1. 从COCO测试集中挑选了3000张图片
   2. 数据集路径："/teams/ai_model_1667305326/WujieAITeam/private/jyd/dataset/minicoco/"
   3. 模型路径："/teams/ai_model_1667305326/WujieAITeam/private/jyd/img2prompt/LAVIS/lavis/output/minicoco/checkpoint_best.pth"
5. minicoco_enhanced: lavis/projects/blip/minicoco_enhanced_cap_ft_iter.yaml
   1. 在4的基础上，添加ram推理的tags
   2. 数据集使用enhanced_train和enhanced_test
6. run python train.py --cfg-path <yaml_path>  可以参考run_scripts/blip/train/train_caption_coco_large_iters.sh
7. BLIP训练需要至少40GB显存，用A6000或者A40
8. MiniGPT4的项目框架跟LAVIS是基本一致的,其使用BLIP2的框架，建议只微调第二阶段，用A100.

# 数据集说明
1. train.json:训练集的标注
2. test.json:测试集的标注
3. test_cache.json:验证时，计算指标依赖的缓存文件，在dataset初始化时生成，如果需要计算指标，则必须提供此文件
4. test_n2id_cache.json:用来寻找LAVIS标定的id和标注中的id之间的对应关系
5. tags: 散装的tags文本
6. captions: 散装的captions文本
7. image: 图片路径


# 制作新的数据集
1. 脚本在generate_dataset/下
2. annotation.py 用以从csv和url中下载并做制作
3. get_dataset.py用以从本地图片tags对中制作数据集
4. 利用Mini GPT-4获取tag/caption的伪标签，脚本在teams/ai_model_1667305326/WujieAITeam/private/ztn/MiniGPT-4/minigpt4_caption_tag.py, 大概15s/it
5. MiniGPT-4的部署，可以将teams/ai_model_1667305326/WujieAITeam/private/jyd/minigpt4_env.zip 拷贝到/root/miniconda3/envs/下，解压后可以直接使用

# 注册新的数据集
1. 在lavis/configs/datasets中新建数据集配置文件
2. 在lavis/datasets/datasets中新建dataset类文件
3. 在lavis/datasets/builders 中新建builder，直接参照caption的就可以
4. 在lavis/tasks中新建task类文件。主要是coco验证方法不适用其他数据集，调用pycocoevalcap中的方法，（只是用BLEU方法就可以，METEOR会报错）
5. 在运行yaml中修改数据集和任务等

# 验证指标
1. 调用nlg_metrics.py中的compute_scores函数
2. 调用Meteor会产生多线程相关错误
3. 调用Rough依赖于Java环境


# Streamlit 前端界面
参考 http://cw.hubwiz.com/card/c/streamlit-manual/
建议使用缓存修饰器，以加速推理
命令：streamlit run app/caption.py --server.fileWatcherType none
部署需要至少12GB显存的机器
加载需要较长时间
![img_1.png](img_1.png)

# 调整生成句子长度
1. 需要修改task层面的最大长度。直接修改运行cfg即可
2. 需要修改model层面的最大长度。需要修改模型配置文件，例如lavis/configs/models/blip_caption_large_coco.yaml中model下面添加max_txt_len: 300。也可以自定义个一个模型配置文件，记得修改运行cfg中的model—type 和lavis/models/blip_models/blip_caption.py中的映射
3. 调整句子长度会导致显存消耗增加

# TODO
1. i2p任务需要对文本进一步预处理，prompt中带有各种括号和特殊字符，是正常caption中没有的,需要自定义一个文本预处理方法
2. 还没有广泛调参
3. 开放域image2tags任务数据量太少，泛化性能不足，需要扩大数据集。
4. RAM增强可以丰富tags，同时也会引入同义词噪声，如何更好的融合两种数据
5. 扩大数据渠道，比如网页上的文本
6. DEMO可以添加ground SAM模型，进一步可视化比较我们的模型
