SELECT DISTINCT CONCAT(a1.first_name, ' ', a1.last_name) AS first_actor, 
CONCAT(a2.first_name, ' ', a2.last_name) AS second_actor
FROM actor AS a1
JOIN film_actor AS fa1
ON a1.actor_id = fa1.actor_id
JOIN film_actor AS fa2
ON fa1.film_id = fa2.film_id
JOIN actor AS a2
ON a2.actor_id = fa2.actor_id
WHERE a1.actor_id < a2.actor_id;