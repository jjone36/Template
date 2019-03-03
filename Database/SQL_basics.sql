-- https://classroom.udacity.com/courses/ud198
------------ Basic
SELECT names
FROM accounts
WHERE name LIKE 'c%'

SELECT names
FROM accounts
WHERE names IN ('Walmart', 'Apple')

SELECT names
FROM accounts
WHERE names NOT IN ('Walmart'),
      names NOT LIKE '%google%'

SELECT *
FROM orders
WHERE standart_qty > 1000 AND poster_qty = 0;

SELECT name
FROM accounts
WHERE name NOT LIKE 'C%' AND name LIKE '%s'

SELECT *
FROM web_events
WHERE channel IN ('organic', 'adwords') OR occurred_at BETWEEN '2016-01-01' AND '2017-01-01'
ORDER BY occurred_at DESC;

------------ join_
SELECT orders.*, accounts.*
FROM accounts
JOIN orders
ON accounts.id = orders.account_id;

SELECT w.account_id, w.occurred_at, w.channel, a.name
FROM web_events w
JOIN accounts a
ON w.account_id = a.id
WHERE a.name = 'Walmart';

SELECT r.name, s.name, a.name
FROM region r
JOIN sales_reps s
ON r.id = s.region_id
JOIN accounts a
ON s.id = a.sales_rep_id
ORDER BY a.name;

SELECT r.name, s.name, a.name
FROM sales_reps s
JOIN region r
ON r.id = s.region_id
JOIN accounts a
ON a.sales_rep_id = s.id
WHERE a.name LIKE 's%' AND r.name = 'Midwest'
ORDER BY a.name;

SELECT o.name region, a.name account, o.total_amt_usd/(a.total + .01) unit price
FROM order o
JOIN accounts a
ON a.id = o.account_id
JOIN sales_reps s
ON o.sales_rep_id = s.id
JOIN region r
ON r.id = s.region_id
WHERE o.standart_qty > 100 AND poster_qty > 50
ORDER BY unit price;

SELECT DISTINCT w.channel, a.name
FROM web_events w
JOIN account a
ON w.account_id = a.id
WHERE w.account_id = '1001';

------------ aggregation
SELECT SUM(poster_qty) AS total_poster_sales
FROM orders;

SELECT MIN(occurred_at)
FROM orders;

SELECT a.name, SUM(total_amt_usd) total_sales
FROM orders o
JOIN accounts a
ON a.id = o.account_id
GROUP BY a.name;

SELECT s.id, s.name, COUNT(*) num_accounts
FROM accounts a
JOIN sales_reps s
ON s.id = a.sales_rep_id
GROUP BY s.id, s.name
ORDER BY num_accounts;

SELECT DISTINCT id, name
FROM sales_reps;

SELECT s.id, s.name, COUNT(*) num_accounts
FROM accounts a
JOIN sales_reps s
ON s.id = a.sales_rep_id
GROUP BY s.id, s.name
HAVING COUNT(*) > 5
ORDER BY num_accounts;


-- percentile_disc(~~) WITHIN GROUP (ORDER BY ~~)
SELECT sector, avg(assets) AS mean,
       percentile_disc(.5) WITHIN GROUP (ORDER BY assets) AS median
  FROM fortune500
 GROUP BY sector
 ORDER BY mean;


-- CASE WHEN ~~ THEN ~~ ELSE ~~ END
SELECT a.name, sum(o.total_amt_usd) AS total_sales,
       CASE WHEN sum(o.total_amt_usd) > 200000 THEN 'Level 1'
       WHEN sum(o.total_amt_usd) BETWEEN 100000 AND 2000000 THEN 'Level 2'
       ELSE 'Level 3' END AS level
FROM orders o
JOIN accounts a
ON a.id = o.account_id
WHERE date_part('year', occurred_at) BETWEEN 2016 AND 2017
GROUP BY a.name
ORDER BY 2 DESC;

SELECT s.name AS sales_rep_name, count(*) AS nums, sum(o.total),
       CASE WHEN count(*) > 200 OR sum(o.total) > 750000 THEN 'top'
            WHEN count(*) > 150 OR sum(o.totaORl) > 500000 THEN 'middle'
            ELSE 'low' END AS groups
FROM sales_reps s
JOIN accounts a
ON s.id = a.sales_rep_id
JOIN orders o
ON o.account_id = a.id
GROUP BY s.name
ORDER BY 4 DESC;


------------ times
SELECT DATE_PART('year', occurred_at) ord_year,  SUM(total_amt_usd) total_spent
FROM orders
GROUP BY 1
ORDER BY 2 DESC;

SELECT DATE_TRUNC('month', o.occurred_at) ord_date, SUM(o.gloss_amt_usd) tot_spent
FROM orders o
JOIN accounts a
ON a.id = o.account_id
WHERE a.name = 'Walmart'
GROUP BY 1
ORDER BY 2 DESC
LIMIT 1;

-------------------------------------------------------------------------------
------------ Sub-query
SELECT channel, avg(num)ORDER
FROM
    (SELECT date_trunc('day', occurred_at) AS day, channel, count(*) AS num
    FROM web_events
GROUP BY 1, 2) AS sub
GROUP BY channel
ORDER BY 2;

SELECT avg(standard_qty) avg_standard, sum(total_amt_usd)
FROM orders
WHERE date_trunc('month', occurred_at) =
    (SELECT date_trunc('month', occurred_at)
    FROM orders
    ORDER BY 1
    LIMIT 1);


------------ with_
WITH t1 AS (SELECT s.name sales_name, r.name region_name, sum(o.total_amt_usd) sum
FROM orders o
JOIN accounts a
ON o.account_id = a.id
JOIN sales_reps s
ON s.id = a.sales_rep_id
JOIN region r
ON r.id = s.region_id
GROUP BY 1, 2
ORDER BY 3),

    t2 AS (SELECT region_name, max(sum) max
    FROM t1
    GROUP BY region_name)

SELECT t1.sales_name, t2.region_name, t2.max
FROM t1
JOIN t2
ON t1.region_name = t2.region_name
WHERE t1.sum = t2.max;
-- https://classroom.udacity.com/courses/ud198/lessons/b50a9cfd-566a-4b42-bf4f-70081b557c0b/concepts/a4ea6477-dbb6-4890-ac82-ad19f60cc3c5


------------ Temporary Tables
DROP TABLE IF EXISTS top_compaines;

CREATE TEMP TABLE top_compaines AS
SELECT rank, title
FROM fortune500
WHERE rank <= 10;
    -- same
INSERT INTO top_compaines
SELECT rank, title
FROM fortune500
WHERE rank BETWEEN 11 AND 20;
    --
SELECT *
FROM top_compaines;

-------------------------------------------------------------------------------
-- LEFT / RIGHT / UPPER / LOWER
WITH t1 AS (SELECT name,
        CASE WHEN left(name, 1) IN ('a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U')
        THEN 1 ELSE 0 END AS vowel, count(*) num
FROM accounts
GROUP BY 1, 2)

SELECT vowel, sum(vowel), sum(num)
FROM t1
GROUP BY vowel;

--
WITH t1 AS (SELECT replace(name, ' ', '') AS company,
        left(primary_poc, strpos(primary_poc, ' ')) AS first,
        left(primary_poc, -strpos(primary_poc, ' ')) AS last
        FROM accounts)

SELECT concat(first, '.', last, '@', company, '.com') AS address
FROM t1

-- CONCAT / REPLACE / SUBSTR / COALESCE
SELECT date,
   concat(SUBSTR(date, 7, 4), '-', left(date, 2), '-', substr(date, 4, 2))::date AS date_2
FROM sf_crime_data

-------------------------------------------------------------------------------
