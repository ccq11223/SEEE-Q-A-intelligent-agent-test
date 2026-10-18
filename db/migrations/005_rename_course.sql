-- 课程范围从「电路理论（上）」调整为整本《电路理论》。
-- 课程 id 保持不变，只改显示名称，避免影响已有内容块和问答日志。

update courses
set name = '电路理论'
where id = 'circuit-theory-1';
