-- 初始课程数据。新增课程时追加 insert，不要改动已有行。
insert into courses (id, name, term)
values ('circuit-theory-1', '电路理论', '2026秋')
on conflict (id) do nothing;
