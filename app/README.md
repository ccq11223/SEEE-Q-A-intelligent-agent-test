# 在线答疑服务（阶段 2）

学生端：打开链接 → 填一个匿名代号 → 提问 → 得到带页码引用的回答；
资料没覆盖时如实说明并建议线下确认。

```
app/
├── main.py        FastAPI 应用入口（create_app / app）
├── api.py         接口：/api/config、/api/health、/api/ask
├── services.py    依赖组装（复用 pipeline 的检索与决策链）
├── ratelimit.py   每 IP 每小时提问上限
├── config.py      读取 config/app.yaml
└── web/index.html 学生端页面（单文件、手机优先、无构建步骤）
```

**关键约定：在线服务不重新实现任何建库逻辑**，直接复用
`pipeline/retrieval.py` 与 `pipeline/answer.py`，
保证「自检时怎么判、学生用时也怎么判」。

启动与对外发布见 [docs/在线服务使用指南.md](../docs/在线服务使用指南.md)。
