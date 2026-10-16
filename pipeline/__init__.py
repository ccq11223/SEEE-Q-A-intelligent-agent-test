"""课程辅导智能体 · 离线建库流水线。

对应方案《课程辅导智能体实施方案》§4.1 的七个步骤：

    1 资料登记与分级  → manifest.py
    2 格式规范化      → convert.py
    3 结构化解析      → ai_parser.py
    4 人工抽查        → sampler.py
    5 清洗与切块      → chunker.py
    6 向量化入库      → embedder.py / store.py
    7 往年题处理      → questions.py（阶段 1 后段实现）
"""

__version__ = "0.1.0"
