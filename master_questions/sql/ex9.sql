SELECT title, COUNT(*) AS times_rented
FROM film AS f
JOIN inventory AS i
ON f.film_id = i.film_id
JOIN rental AS r
ON i.inventory_id = r.inventory_id
GROUP BY f.film_id
ORDER BY times_rented DESC
LIMIT 5;