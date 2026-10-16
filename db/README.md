# 数据库迁移

云端使用 Supabase 的 PostgreSQL（已内置 pgvector 扩展），迁移脚本按序号手工执行。

## 执行方式

**方式 A：Supabase SQL Editor（推荐首次使用）**

1. 打开项目 → 左侧 `SQL Editor` → `New query`；
2. 粘贴 `001_init.sql` 全文 → `Run`；
3. 再粘贴 `002_seed.sql` → `Run`；
4. 到 `Table Editor` 确认能看到 12 张表。

**方式 B：命令行（有 psql 时）**

```bash
psql "$SUPABASE_DB_URL" -f db/migrations/001_init.sql
psql "$SUPABASE_DB_URL" -f db/migrations/002_seed.sql
```

（Windows PowerShell 里 `$SUPABASE_DB_URL` 不通用，直接写完整连接串即可。）

## 约定

- 迁移脚本只做「新增」，不做破坏性修改；后续变更新增 `003_xxx.sql`，不回头改 001。
- `chunks` 是唯一需要 pgvector 的表；`embedding` 维度必须与 `config/models.yaml` 里
  `embedding.dim` 一致（默认 1024，对应 BGE-M3）。**换向量模型必须同时改这两处并重建向量列。**
- 流水线使用数据库直连串（`SUPABASE_DB_URL`）写入；将来在线服务用 Supabase 的
  公开 API Key + 行级安全策略（RLS）只读访问，两套凭据不要混用。
