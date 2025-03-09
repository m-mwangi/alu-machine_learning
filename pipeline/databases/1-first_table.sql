-- creates table first_table in database
-- if table exists, script not fail
CREATE TABLE IF NOT EXISTS first_table (
	id INT,
	name VARCHAR(256));