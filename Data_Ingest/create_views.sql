CREATE VIEW session_duration_date AS
SELECT
    [CH1].[user_id],
    [CH1].[session_id],
    CAST([CH1].[created_at] AS DATE) AS [date],
    DATEDIFF(SECOND, [CH1].[created_at], [CH1].[updated_at]) AS [sess_duration]
FROM [dbo].[chat_sessions] AS [CH1];
GO

CREATE VIEW [dbo].[ContentIssues] AS
SELECT
    [session_id],
    [trace_id],
    [user_id],
    [agent_id],
    [message_type],
    [content],
    [content_filter_results],
    -- Assuming content_filter_results is a JSON object, we will extract specific fields
    JSON_VALUE([content_filter_results], '$.hate.filtered') AS hate_filtered,
    JSON_VALUE([content_filter_results], '$.hate.severity') AS hate_severity,
    JSON_VALUE([content_filter_results], '$.jailbreak.filtered') AS jailbreak_filtered,
    JSON_VALUE([content_filter_results], '$.jailbreak.detected') AS jailbreak_detected,
    JSON_VALUE([content_filter_results], '$.self_harm.filtered') AS self_harm_filtered,
    JSON_VALUE([content_filter_results], '$.self_harm.severity') AS self_harm_severity,
    JSON_VALUE([content_filter_results], '$.sexual.filtered') AS sexual_filtered,
    JSON_VALUE([content_filter_results], '$.sexual.severity') AS sexual_severity,
    JSON_VALUE([content_filter_results], '$.violence.filtered') AS violence_filtered,
    JSON_VALUE([content_filter_results], '$.violence.severity') AS violence_severity
FROM
    [dbo].[chat_history] as ch
WHERE [ch].[message_type] = 'ai';
GO

CREATE VIEW UserAsks AS
SELECT [CH1].[user_id], [CH1].[trace_id], [CH1].[content], [CH1].[session_id], [CH2].[response_time_seconds], [CH1].[date]
FROM
(SELECT
    [user_id],
    [trace_id],
    [content],
    [session_id],
    CAST([trace_end] AS DATE) AS [date]
FROM
    [dbo].[chat_history] as ch
WHERE [ch].[message_type] = 'human' ) AS CH1
JOIN (
SELECT
    trace_id,
    CASE
        WHEN [response_time_ms] IS NOT NULL THEN [response_time_ms] / 1000.0
        ELSE NULL
    END AS response_time_seconds
    FROM
    [chat_history]) AS CH2 ON [CH1].trace_id = CH2.trace_id
WHERE [CH2].[response_time_seconds] IS NOT NULL;
GO