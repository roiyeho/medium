SELECT COUNT(*) AS copies_in_inventory
FROM inventory AS i
JOIN film AS f
ON i.film_id = f.film_id
WHERE f.title = 'Hunchback Impossible';