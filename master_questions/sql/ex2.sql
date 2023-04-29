SELECT first_name, last_name
FROM actor AS a
JOIN film_actor AS fa
ON a.actor_id = fa.actor_id
JOIN film AS f
ON f.film_id = fa.film_id
WHERE f.title = 'Academy Dinosaur';