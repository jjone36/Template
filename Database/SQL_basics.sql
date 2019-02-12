https://classroom.udacity.com/courses/ud198
########################### Basic
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

########################### join_
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

########################### aggregation
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

########################### times
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

########################### case_statements
"CASE WHEN ~~ THEN ~~ ELSE ~~ END"
select a.name, sum(o.total_amt_usd) as total_sales,
       case when sum(o.total_amt_usd) > 200000 then 'Level 1'
       when sum(o.total_amt_usd) between 100000 and 2000000 then 'Level 2'
       else 'Level 3' end as level
from orders o
join accounts a
on a.id = o.account_id
where date_part('year', occurred_at) between 2016 and 2017
group by a.name
order by 2 desc;

select s.name as sales_rep_name, count(*) as nums, sum(o.total),
       case when count(*) > 200 or sum(o.total) > 750000 then 'top'
            when count(*) > 150 or sum(o.total) > 500000 then 'middle'
            else 'low' end as groups
from sales_reps s
join accounts a
on s.id = a.sales_rep_id
join orders o
on o.account_id = a.id
group by s.name
order by 4 desc;

########################### Sub-query
select channel, avg(num)
from
    (select date_trunc('day', occurred_at) as day, channel, count(*) as num
    from web_events
group by 1, 2) as sub
group by channel
order by 2;

select avg(standard_qty) avg_standard, sum(total_amt_usd)
from orders
where date_trunc('month', occurred_at) =
    (select date_trunc('month', occurred_at)
    from orders
    order by 1
    limit 1);

########################### with_
'https://classroom.udacity.com/courses/ud198/lessons/b50a9cfd-566a-4b42-bf4f-70081b557c0b/concepts/a4ea6477-dbb6-4890-ac82-ad19f60cc3c5'
with t1 as (select s.name sales_name, r.name region_name, sum(o.total_amt_usd) sum
from orders o
join accounts a
on o.account_id = a.id
join sales_reps s
on s.id = a.sales_rep_id
join region r
on r.id = s.region_id
group by 1, 2
order by 3),

    t2 as (select region_name, max(sum) max
    from t1
    group by region_name)

select t1.sales_name, t2.region_name, t2.max
from t1
join t2
on t1.region_name = t2.region_name
where t1.sum = t2.max;

#################################################################################
## LEFT / RIGHT / UPPER / LOWER / CONCAT / REPLACE / SUBSTR / COALESCE
with t1 as (SELECT name,
        CASE WHEN left(name, 1) in ('a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U')
        THEN 1 ELSE 0 END AS vowel, count(*) num
FROM accounts
group by 1, 2)

select vowel, sum(vowel), sum(num)
from t1
group by vowel;


with t1 as (select replace(name, ' ', '') as company,
        left(primary_poc, strpos(primary_poc, ' ')) as first,
        left(primary_poc, -strpos(primary_poc, ' ')) as last
        from accounts)

select concat(first, '.', last, '@', company, '.com') as address
from t1


select date,
   concat(SUBSTR(date, 7, 4), '-', left(date, 2), '-', substr(date, 4, 2))::DATE date_2
from sf_crime_data
