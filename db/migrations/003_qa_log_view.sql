-- 便于后台排查的只读视图：一眼看到每次问答走了哪条分支
create or replace view qa_log_view as
select l.id,
       l.created_at,
       l.course_id,
       l.question,
       l.mode,
       l.citations,
       l.citation_valid,
       round(l.top_score::numeric, 3) as top_score,
       l.model,
       l.latency_ms,
       l.answer
  from qa_logs l
 order by l.created_at desc;
