-- create a stored procedure that adds a new correction for a student
-- The stored procedure is called  addBonus  and takes three parameters:  
-- user_id - Users.id value, 
-- project_name new or already existing projects if no projects.name exists,
-- and score.

-- The procedure should insert a new row into the  bonus  table with the  user_id ,  project_id , and  score  columns set to the corresponding parameters.

-- If the project does not exist, it should be created in the  projects  table before the row is inserted into the  bonus  table.

-- solution
DELIMITER //
CREATE PROCEDURE AddBonus(IN user_id INT, IN project_name VARCHAR(255), IN score INT)
BEGIN
	    DECLARE project_id INT;
	    SELECT id INTO project_id FROM projects WHERE name = project_name;
	    IF project_id IS NULL THEN
		        INSERT INTO projects (name) VALUES (project_name);
			        SET project_id = LAST_INSERT_ID();
				    END IF;
				    INSERT INTO corrections (user_id, project_id, score) VALUES (user_id, project_id, score);
			END //
			DELIMITER ;