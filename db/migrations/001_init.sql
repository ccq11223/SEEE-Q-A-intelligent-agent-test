-- =====================================================================
-- 学业指导中心 · 课程辅导智能体  初始化迁移
-- 对应方案《课程辅导智能体实施方案》§5 数据模型草案
-- 执行环境：PostgreSQL 15+（Supabase 免费层满足），需要 pgvector 扩展
-- =====================================================================

create extension if not exists vector;

-- ---------------------------------------------------------------------
-- 1. 课程（v1.0 只服务一门课，但架构按 course_id 参数化，为将来多课程预留）
-- ---------------------------------------------------------------------
create table if not exists courses (
    id          text primary key,
    name        text        not null,
    term        text,
    created_at  timestamptz not null default now()
);

-- ---------------------------------------------------------------------
-- 2. 资料台账
--    trust_level：1=权威可直接引用 2=可用须标注 3=待确认不得直接引用
--    doc_key 为流水线的稳定标识（对应 config/manifest.csv），用于幂等重跑
-- ---------------------------------------------------------------------
create table if not exists documents (
    id                bigserial primary key,
    course_id         text        not null references courses(id) on delete cascade,
    doc_key           text        not null,
    filename          text        not null,
    doc_type          text        not null,          -- textbook/slide/exam_scope/example/past_exam/web
    trust_level       smallint    not null check (trust_level between 1 and 3),
    title             text,
    term              text,
    source_note       text,
    status            text        not null default 'registered',
                                  -- registered/converted/parsed/chunked/embedded/loaded/failed
    page_count        integer,
    chunk_count       integer     not null default 0,
    parse_error_count integer     not null default 0,
    content_hash      text,
    created_at        timestamptz not null default now(),
    updated_at        timestamptz not null default now(),
    unique (course_id, doc_key)
);

-- ---------------------------------------------------------------------
-- 3. 知识库内容块（检索的基本单位）
-- ---------------------------------------------------------------------
create table if not exists chunks (
    id           bigserial primary key,
    course_id    text        not null references courses(id) on delete cascade,
    document_id  bigint      not null references documents(id) on delete cascade,
    chunk_key    text        not null,               -- 稳定标识，形如 slide-ch01#0007
    page_no      integer,                            -- 起始页码
    page_end     integer,                            -- 结束页码（跨页块才有值）
    chapter      text,
    section_title text,
    content      text        not null,
    embedding    vector(1024),
    trust_level  smallint    not null check (trust_level between 1 and 3),
    char_count   integer,
    token_count  integer,
    content_hash text,
    created_at   timestamptz not null default now(),
    unique (document_id, chunk_key)
);

create index if not exists chunks_course_trust_idx on chunks (course_id, trust_level);
create index if not exists chunks_document_idx     on chunks (document_id);

-- 向量索引：HNSW 支持增量插入，适合「一次建库、后续增量补充」的形态
create index if not exists chunks_embedding_hnsw
    on chunks using hnsw (embedding vector_cosine_ops);

-- ---------------------------------------------------------------------
-- 4. 往年题与解析
-- ---------------------------------------------------------------------
create table if not exists questions (
    id               bigserial primary key,
    course_id        text        not null references courses(id) on delete cascade,
    q_key            text        not null,           -- 稳定标识，形如 2023#一.3
    year             integer,
    q_no             text        not null,
    stem             text        not null,
    options          jsonb,
    answer           text,
    explanation      text,
    chapter          text,
    source_chunk_ids bigint[],
    reference_pages  integer[],
    has_kb_evidence  boolean     not null default false,  -- 找不到课件依据的题 → 知识缺口
    review_status    text        not null default 'pending',
                                 -- pending/approved/rejected
    created_at       timestamptz not null default now(),
    updated_at       timestamptz not null default now(),
    unique (course_id, q_key)
);

create index if not exists questions_course_year_idx on questions (course_id, year);

-- ---------------------------------------------------------------------
-- 5-6. 诊断题组与题组内单题
-- ---------------------------------------------------------------------
create table if not exists diagnostic_sets (
    id         bigserial primary key,
    course_id  text        not null references courses(id) on delete cascade,
    set_key    text        not null,
    name       text        not null,
    active     boolean     not null default true,
    created_at timestamptz not null default now(),
    unique (course_id, set_key)
);

create table if not exists diagnostic_items (
    id            bigserial primary key,
    set_id        bigint      not null references diagnostic_sets(id) on delete cascade,
    order_no      integer     not null,
    item_type     text        not null check (item_type in ('concept', 'calc', 'synthesis')),
    stem          text        not null,
    options       jsonb,
    answer        text        not null,
    wrong_patterns jsonb,                     -- [{"code":"A","desc":"...","chapter":"..."}]
    chapter       text,
    reference_pages integer[],
    unique (set_id, order_no)
);

-- ---------------------------------------------------------------------
-- 7-8. 学生匿名会话与诊断结果（不存姓名、学号）
-- ---------------------------------------------------------------------
create table if not exists student_sessions (
    id         bigserial primary key,
    course_id  text        not null references courses(id) on delete cascade,
    anon_code  text        not null,
    created_at timestamptz not null default now(),
    unique (course_id, anon_code)
);

create table if not exists diagnostics (
    id           bigserial primary key,
    session_id   bigint      not null references student_sessions(id) on delete cascade,
    set_id       bigint      not null references diagnostic_sets(id),
    answers      jsonb       not null,
    result       jsonb       not null,               -- 四要素报告的结构化版本
    report_token text        not null unique,
    passcode_hash text,                              -- 4 位口令，只存哈希
    created_at   timestamptz not null default now()
);

-- ---------------------------------------------------------------------
-- 9. 问答日志（质量改进的数据源）
-- ---------------------------------------------------------------------
create table if not exists qa_logs (
    id                  bigserial primary key,
    course_id           text        not null references courses(id) on delete cascade,
    session_id          bigint      references student_sessions(id) on delete set null,
    question            text        not null,
    retrieved_chunk_ids bigint[],
    answer              text,
    mode                text        not null check (mode in ('kb', 'web', 'handoff')),
    citations           integer[],
    citation_valid      boolean,
    top_score           real,
    model               text,
    latency_ms          integer,
    created_at          timestamptz not null default now()
);

create index if not exists qa_logs_course_created_idx on qa_logs (course_id, created_at desc);

-- ---------------------------------------------------------------------
-- 10. 知识缺口清单
-- ---------------------------------------------------------------------
create table if not exists gaps (
    id               bigserial primary key,
    course_id        text        not null references courses(id) on delete cascade,
    question         text        not null,
    hit_count        integer     not null default 1,
    status           text        not null default 'open',
                                 -- open/researched/approved/rejected
    web_candidate    text,
    approved_chunk_id bigint     references chunks(id) on delete set null,
    first_seen_at    timestamptz not null default now(),
    last_seen_at     timestamptz not null default now()
);

create index if not exists gaps_course_status_idx on gaps (course_id, status, hit_count desc);

-- ---------------------------------------------------------------------
-- 11. 错题台账
-- ---------------------------------------------------------------------
create table if not exists error_ledger (
    id             bigserial primary key,
    course_id      text        not null references courses(id) on delete cascade,
    question       text        not null,
    ai_output      text,
    correct_answer text,
    reporter       text,
    fixed          boolean     not null default false,
    created_at     timestamptz not null default now()
);

-- ---------------------------------------------------------------------
-- 12. 解析质量问题登记（方案 §4.1 步骤 4 人工抽查的落点）
-- ---------------------------------------------------------------------
create table if not exists parse_errors (
    id          bigserial primary key,
    document_id bigint      not null references documents(id) on delete cascade,
    page_no     integer,
    error_type  text        not null,   -- formula_lost/figure_lost/garbled/table_broken/other
    detail      text,
    reporter    text,
    resolved    boolean     not null default false,
    created_at  timestamptz not null default now()
);

create index if not exists parse_errors_document_idx on parse_errors (document_id, resolved);

-- =====================================================================
-- 检索函数：向量召回 + 可信级别过滤
-- 在线服务与流水线自检共用，保证「检索口径」只有一处定义
-- =====================================================================
create or replace function match_chunks(
    query_embedding vector(1024),
    p_course_id     text,
    p_max_trust     smallint default 3,
    p_limit         integer  default 20
)
returns table (
    id            bigint,
    document_id   bigint,
    chunk_key     text,
    page_no       integer,
    page_end      integer,
    chapter       text,
    section_title text,
    content       text,
    trust_level   smallint,
    score         double precision
)
language sql
stable
as $$
    select c.id,
           c.document_id,
           c.chunk_key,
           c.page_no,
           c.page_end,
           c.chapter,
           c.section_title,
           c.content,
           c.trust_level,
           1 - (c.embedding <=> query_embedding) as score
      from chunks c
     where c.course_id = p_course_id
       and c.trust_level <= p_max_trust
       and c.embedding is not null
     order by c.embedding <=> query_embedding
     limit p_limit;
$$;
