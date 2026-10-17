-- qa_logs.mode 增加 general（通用知识作答）这一档。
-- 原来的约束只允许 kb / web / handoff；现在课件没覆盖时不再直接拒绝，
-- 而是先用通用知识作答并标注，所以需要新增一种模式。

alter table qa_logs drop constraint if exists qa_logs_mode_check;

alter table qa_logs
    add constraint qa_logs_mode_check
    check (mode in ('kb', 'web', 'general', 'handoff'));
