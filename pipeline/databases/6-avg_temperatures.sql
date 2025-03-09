-- Displays average temp(fahrenheit) by city
-- order by temp DESC
SELECT city, AVG(value) AS avg_temp
FROM temperatures
GROUP BY city
ORDER BY avg_temp DESC;