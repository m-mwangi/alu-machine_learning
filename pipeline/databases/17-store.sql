-- create a trigger that decreases quantity of an item after adding a new order
-- quantity can be negative
DROP TRIGGER IF EXISTS decrease_quantity;
DELIMITER //
CREATE TRIGGER decrease_quantity AFTER INSERT ON orders
FOR EACH ROW
BEGIN
        UPDATE items
        SET quantity = quantity - NEW.number
        WHERE name = NEW.item_name;
END;
//
DELIMITER ;