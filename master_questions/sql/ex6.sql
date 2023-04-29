SELECT i.store_id, SUM(amount) AS total_revenue
FROM payment AS p
JOIN rental AS r
ON p.rental_id = r.rental_id
JOIN inventory AS i
ON i.inventory_id = r.inventory_id
GROUP BY i.store_id;