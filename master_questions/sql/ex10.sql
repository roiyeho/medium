SELECT EXISTS (
  SELECT *
  FROM inventory AS i
  JOIN film AS f
  ON i.film_id = f.film_id
  WHERE f.title = 'Academy Dinosaur'
    AND i.store_id = 1
    AND NOT EXISTS (
      SELECT *      
      FROM rental AS r
      WHERE i.inventory_id = r.inventory_id      
        AND r.return_date IS NULL
  )
) AS 'available';