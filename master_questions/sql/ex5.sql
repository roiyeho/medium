SELECT i.store_id, c.name AS category, COUNT(*) AS films_num
FROM category AS c
JOIN film_category AS fc
ON c.category_id = fc.category_id
JOIN inventory AS i
ON i.film_id = fc.film_id
GROUP BY i.store_id, c.category_id
ORDER BY i.store_id, c.name;