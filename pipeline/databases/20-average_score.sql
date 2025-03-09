-- create a stored procedure ComputeAverageScore that takes:
-- a user_id as a parameter and returns the average score of all the corrections for that user.

-- DELIMITER //
-- CREATE PROCEDURE ComputeAverageScore(IN user_id_param INT, OUT average_score)
-- BEGIN
--     SELECT AVG(score) INTO average_score FROM corrections WHERE user_id = user_id_param;
-- END //
-- DELIMITER ;

-- DECIMAL(5, 2) means that the number will have 5 digits in total, 2 of which will be after the decimal point.

DELIMITER //

CREATE PROCEDURE ComputeAverageScoreForUser(IN user_id_param INT)
BEGIN
	    DECLARE average_score DECIMAL(5, 2);
	    DECLARE total_score INT;
	    DECLARE total_corrections INT;

	    SELECT SUM(score) INTO total_score FROM corrections WHERE user_id = user_id_param;
	    SELECT COUNT(*) INTO total_corrections FROM corrections WHERE user_id = user_id_param;
	    IF total_corrections = 0 THEN
		        SET average_score = 0;
			    ELSE
				        SET average_score = total_score / total_corrections;
					    END IF;

					    UPDATE users 
					    SET average_score = average_score 
					    WHERE id = user_id_param;
				END //

				DELIMITER ;