-- Creates function SafeDiv that divides
-- and returns firts by second number or returns 0 
-- if second number is equal to 0
DELIMITER //

CREATE FUNCTION SafeDiv(a INT, b INT) RETURNS FLOAT

BEGIN
    IF b = 0 THEN
        RETURN 0;
    ELSE
        RETURN a / b;
    END IF;
END //