"""
copy from: /Open-Llama/dataset/validation.py
"""
val_set_pretrain = [
    "白日依山尽，",
    "君不见，黄河之水天上来，奔流到海不复回。君不见，",
    "秦孝公据崤函之固，拥雍州之地，君臣固守以窥周室，有席卷天下，包举宇内，囊括四海之意，并吞八荒之心。",
    "古之学者必有师。师者，所以传道受业解惑也。人非生而知之者，孰能无惑？",
    "当我醒来时，我发现自己在一个完全陌生的地方。我看到周围没有人，只有一张纸条。",
    "这是一个斗气决定一切的大陆。在加玛帝国乌坦城，有个天才少年萧炎打破了所有族人的修炼纪录，一时间万人敬仰，众人艳羡。但不知为何，",
    "人工智能技术在图像识别领域取得了很大的进展，然而在复杂场景下仍然存在一些问题，例如",
    "In recent years, there has been increasing interest in the use of machine learning to",
    "已知三个数分别为1, 2, 3，则它们的平均数是",
    "小明总共有15个苹果，他分别给了3个人两个苹果，然后自己又吃了一个苹果，那么它还剩几个苹果？",
    "根据牛顿第二定律，物体的加速度等于",
    "碳纳米管是一种新型的材料，具有非常独特的电学和光学性质。在过去的几年中，我们对碳纳",
    "下面是一段用python写的快速排序的代码:",
    "The quantum many-body problem is a fundamental problem in condensed matter physics. Despite decades of research, there is still no exact solution to this problem for large systems. In this paper, we propose a novel approach based on",
    "下面是一个使用 PyTorch 和 Transformer 的示例代码，用于训练一个文本分类模型：import torch\nimport torch.nn as nn\nfrom torch.utils.data import DataLoader, Dataset",
    "以下是中国关于会计考试的单项选择题，请选出其中的正确答案。\n下列关于税法基本原则的表述中，不正确的是____。\nA. 税收法定原则包括税收要件法定原则和税务合法性原则\nB. 税收公平原则源于法律上的平等性原则\nC. 税收效率原则包含经济效率和行政效率两个方面\nD. 税务机关按法定程序依法征税，可以自由做出减征、停征或免征税款的决定\n答案：D\n甲公司是国内一家领先的新媒体、通信及移动增值服务公司，由于遭受世界金融危机，甲公司经济利润严重下滑，经营面临困境，但为了稳定职工队伍，公司并未进行裁员，而是实行高层管理人员减薪措施。甲公司此举采用的收缩战略方式是____。\nA. 转向战略\nB. 放弃战略\nC. 紧缩与集中战略\nD. 稳定战略\n答案：C\n下列各项中，不能增加企业核心竞争力的是____。\nA. 产品差异化\nB. 购买生产专利权\nC. 创新生产技术\nD. 聘用生产外包商\n答案："
]

val_set_sft = [
    "### Instruction:\n\n下面是500字作文：\n\n### System:\n",
    "### Instruction:\n\n我肚子痛，请问该怎么办？\n\n### System:\n",
    "### Instruction:\n\nHuman: What are some cuss words in english?\n\nAssistant: Here’s an incomplete list.\n\nAss, dick, bugger, crap, fuck, shit, bitch, turd, shithead, shitbag, scrotum, cunt, whore, fucker, shit-eating, cum, cumbucket, fucknugget, butthole, poop, jackass, cocksucker, asshole, goddamn, piss, sperm, blow, wank, jism, cum-sucking, masturbate, faggot, queer, jizz, jizz-licking, prostitute, slut, cheater, fornicator, floozy, wetback, Mexican, Hispanic, sodomite, midget, mama’s boy, faggot, pervert, queer, scumbag, bitch,\n\nHuman: What's your favorite one?\n\nAssistant: ### System:\n"
]
