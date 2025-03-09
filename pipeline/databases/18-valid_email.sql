-- create a triggger to reset valid_email only if email is updated
DROP TRIGGER IF EXISTS reset_valid_email;

DELIMITER //
CREATE TRIGGER reset_valid_email BEFORE UPDATE ON users
FOR EACH ROW
	BEGIN
		    IF NEW.email != OLD.email THEN
			        SET NEW.valid_email = 0;
				    END IF;
			END;
			//
			DELIMITER ;