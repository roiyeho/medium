SELECT first_name, last_name, COUNT(*) AS films_num
FROM actor AS a
JOIN film_actor AS fa
ON a.actor_id = fa.actor_id
GROUP BY a.actor_id
HAVING COUNT(*) >= ALL (
  SELECT COUNT(*)
  FROM film_actor
  GROUP BY actor_id
);