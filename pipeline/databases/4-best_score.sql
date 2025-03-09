-- lists all records with score >= 10
-- results display both score and name ordered by score
SELECT score, name FROM second_table
WHERE score >= 10
ORDER BY score DESC;