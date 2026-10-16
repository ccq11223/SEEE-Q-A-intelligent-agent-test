# SEEE 课程辅导智能体

基于 HUST-SEEE 课程资料构建知识库，为电气工程专业学生提供核心专业课的辅助答疑。

设计文档见 [课程辅导智能体实施方案.md](docs/课程辅导智能体实施方案.md)（v1.0）。

## 当前进度

| 阶段 | 内容 | 状态 |
| --- | --- | --- |
| 阶段 0 | 技术验证：提问 → 检索 → 带页码引用的回答 | 代码就绪，待真实密钥实测 |
| 阶段 1 | 建库流水线：课件 → 结构化知识库 | **已实现**（本仓库主体） |
| 阶段 2 | 在线答疑最小闭环（学生端页面） | 尚未开始 |
| 阶段 3 | 诊断模块与学情报告 | 尚未开始 |
| 阶段 4 | 维护者后台与调教面板 | 尚未开始 |

## 快速上手

```powershell
# 1. 装依赖
pip install -r requirements.txt

# 2. 配密钥（照 docs/云平台与账号准备.md 开通账号）
Copy-Item .env.example .env

# 3. 自检：会告诉你还差什么
python -m pipeline doctor
python -m pipeline keys            # 密钥体检：确认有效、没填错行（免费接口）

# 4. 登记资料：扫一遍 data/raw 生成草稿，再人工确认类型与可信级别
python -m pipeline register --dry-run    # 先预览，不写文件
python -m pipeline register              # 写入 config/manifest.csv
python -m pipeline manifest              # 复核

# 5. 先跑几页试水，肉眼看看公式还原得怎么样
python -m pipeline run --doc slide-02 --pages 1-5 --dry-run

# 6. 全量建库
python -m pipeline run --all

# 7. 提问验收：看能不能命中正确页码
python -m pipeline ask "节点电压法的适用条件是什么？"
```

完整操作步骤：

- [操作手册.md](docs/操作手册.md) —— **照着做**的逐步清单（在哪里点、粘什么、看到什么算成功）
- [知识库构建指南.md](docs/知识库构建指南.md) —— 为什么这么做、验收标准、FAQ

## 目录结构

```
agent-code/
├── app/                 在线服务（阶段 2 开始填充）
├── pipeline/            离线建库流水线 ← 本阶段主体
│   ├── manifest.py        步骤 1 资料登记与分级
│   ├── register.py        步骤 1 辅助：扫描原始资料目录，生成台账草稿
│   ├── convert.py         步骤 2 课件 → 逐页图片
│   ├── ai_parser.py       步骤 3 视觉模型逐页结构化解析
│   ├── sampler.py         步骤 4 人工抽查
│   ├── chunker.py         步骤 5 清洗与切块
│   ├── embedder.py        步骤 6 向量化
│   ├── store.py           步骤 6 写入知识库（postgres / local 两种后端）
│   ├── dsn.py             数据库连接串体检（口令里的特殊字符会被悄悄解析错）
│   ├── keycheck.py        密钥体检（走各家免费的 /models 接口）
│   ├── retrieval.py       检索与「有没有依据」判定
│   ├── answer.py          答疑决策链（阶段 2 在线服务复用同一套）
│   └── cli.py             命令行入口
├── config/              提示词、模型、检索参数、资料台账（纳入 git）
├── db/migrations/       数据库建表脚本
├── tests/               单元测试 + 离线端到端回归
├── docs/                方案与操作文档
└── data/                原始资料与中间产物（不入 git）
```

## 几个关键设计

**离线建库、在线只读。** 学生提问时绝不重新解析课件，解析这类重活只做一次。

**每一步都落盘、可单独重跑。** `data/work/<资料编号>/` 下依次是渲染结果、逐页 Markdown、
切块、向量。解析到一半断了，接着跑就行，不会重复花钱。

**页码是硬约束。** 逐页解析时写入 `<!-- page: N -->` 标记，切块时继承页码，
回答时校验页码是否真实存在于检索结果里——校验不过就重新生成，再不过就转人工。

**解析结果有缓存。** 缓存键包含图片内容、提示词版本、模型名。改提示词会自动重跑，
没改的部分不会再调用模型。

**两种存储后端。** `local` 只写 JSONL，用来在没申请到云账号前跑通全流程；
`postgres` 写云端 Supabase。上层检索与答疑代码完全一样。

## 测试

```powershell
python -m unittest discover -s tests -t .
```

135 项测试，全部离线：唯一的假件是「视觉模型」和「向量模型」，
渲染、切块、存储、检索走真实代码。

## 文档

- [课程辅导智能体实施方案.md](docs/课程辅导智能体实施方案.md) —— 总体设计（v1.0）
- [操作手册.md](docs/操作手册.md) —— 逐步操作清单（先看这份）
- [云平台与账号准备.md](docs/云平台与账号准备.md) —— 选哪家云、开哪些账号、多少钱
- [知识库构建指南.md](docs/知识库构建指南.md) —— 建库要做什么、怎么做、怎么验收
- [db/README.md](db/README.md) —— 数据库建表步骤
