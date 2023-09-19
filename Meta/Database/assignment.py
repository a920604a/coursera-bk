# Task 1
import mysql.connector as connector
from mysql.connector.pooling import MySQLConnectionPool
from mysql.connector import Error

dbconfig={"database":"little_lemon_db", "user":"root", "password":""}
try:
	pool = MySQLConnectionPool(pool_name = "lemon_pl",
                           pool_size = 2, #default is 5
                           **dbconfig)
	print("The connection pool is created with a name: ",pool.pool_name)
	print("The pool size is:",pool.pool_size)

except Error as er:
    print("Error code:", er.errno)
    print("Error message:", er.msg)


# Task 2
#guest 1
connection1 = pool.get_connection()
c1 = connection1.cursor()
stmt = """
INSERT INTO Bookings 
    (TableNo, GuestFirstName, GuestLastName, BookingSlot, EmployeeID)
VALUES
    (8,'Anees','Java','18:00:00',6);"""
c1.execute(stmt)
connection1.commit()
#guest 2
connection2 = pool.get_connection()
c2 = connection2.cursor()
stmt = """
INSERT INTO Bookings 
    (TableNo, GuestFirstName, GuestLastName, BookingSlot, EmployeeID)
VALUES
    (5, 'Bald','Vin','19:00:00',6);"""
c2.execute(stmt)
connection2.commit()

try:
    connection3 = pool.get_connection()
    print("The guest is connected")
except:
    print("Adding new connection in the pool.")
    connection=connector.connect(**dbconfig)
    
    pool.add_connection(cnx=connection)
    print("A new connection is added in the pool.\n")
    
    connection3 = pool.get_connection()
    print("'connection3' is added in the pool.")

#guest 3
c3=connection3.cursor()
stmt="""INSERT INTO Bookings 
    (TableNo, GuestFirstName, GuestLastName, BookingSlot, EmployeeID)
VALUES
    (12, 'Jay','Kon','19:30:00',6);"""
c3.execute(stmt)
connection3.commit()

# Task 3
connection = pool.get_connection()
cursor=connection.cursor()
print("""The cursor object "cursor" is created.""")
# The name and EmployeeID of the Little Lemon manager.
stmt = """
SELECT 
    Name, EmployeeID 
FROM 
    Employees 
WHERE 
    Role = 'Manager'
"""
cursor.execute(stmt)
results=cursor.fetchall()
columns=cursor.column_names
print(columns)
for result in results:
    print(result)
# The name and role of the employee who receives the highest salary.
stmt = """
SELECT 
    Name, Role 
FROM 
    Employees 
ORDER BY 
    Annual_Salary DESC LIMIT 1
"""
cursor.execute(stmt)
results=cursor.fetchall()
columns=cursor.column_names
print(columns)
for result in results:
    print(result)
stmt = """
SELECT 
    COUNT(BookingID) AS n_booking_between_18_20_hrs
FROM 
    Bookings 
WHERE 
    BookingSlot 
BETWEEN '18:00:00' AND '20:00:00';
"""
cursor.execute(stmt)
results=cursor.fetchall()
columns=cursor.column_names
print(columns)
for result in results:
    print(result)
stmt="""
SELECT 
    Bookings.BookingID AS ID,  
    CONCAT(GuestFirstName,' ',GuestLastName) AS GuestName, 
    Role AS Employee
FROM Bookings 
    LEFT JOIN Employees 
    ON Employees.EmployeeID=Bookings.EmployeeID
    WHERE Employees.Role = "Receptionist"
    ORDER BY BookingSlot DESC;
"""
cursor.execute(stmt)
print("The following guests are waiting to be seated:")
results=cursor.fetchall()
columns=cursor.column_names
print(columns)
for result in results:
    print(result)

# Task 4
# Create a stored procedure named BasicSalesReport. 
cursor.execute("DROP PROCEDURE IF EXISTS BasicSalesReport;")
stmt="""
CREATE PROCEDURE BasicSalesReport()
BEGIN
    SELECT 
        SUM(BillAmount) AS Total_Sale,
        AVG(BillAmount) AS Average_Sale,
        MIN(BillAmount) AS Min_Bill_Paid,
        MAX(BillAmount) AS Max_Bill_Paid
    FROM Orders;
END
"""
cursor.execute(stmt)
cursor.callproc("BasicSalesReport")

# Retrieve records in "dataset"
results = next(cursor.stored_results())
results = results.fetchall()

# Retrieve column names using list comprehension in a for loop 
for column_id in cursor.stored_results():
    cols = [column[0] for column in column_id.description]
    
print("Today's sales report:")
for result in results:
    print("\t",cols[0],":",result[0])
    print("\t",cols[1],":",result[1])
    print("\t",cols[2],":",result[2])
    print("\t",cols[3],":",result[3])


# Task 5
# display the next 10 upcoming bookings from the Bookings
cursor.execute("DROP PROCEDURE IF EXISTS UpcomingBookings;")
stmt="""
CREATE PROCEDURE UpcomingBookings()
BEGIN
SELECT 
	CONCAT("BookingSlot"," ", b.BookingSlot) as bookingSlot,
	CONCAT("Guest_name:"," ", CONCAT(b.GuestFirstName," ",b.GuestLastName)) as Guest_name,
	CONCAT("Asign to:"," ", CONCAT(e.Name, " [", e.Role, "]")) AS assign_to
from Bookings b
INNER JOIN 
Employees e
ON e.EmployeeID=b.EmployeeID
ORDER BY b.BookingSlot ASC LIMIT 3;
END
"""
# cursor.execute(stmt)
buffered_cursor = connection.cursor(buffered = True)
buffered_cursor.execute(stmt)
buffered_cursor.callproc("UpcomingBookings")
results = next(buffered_cursor.stored_results())
results = results.fetchall()

for result in results:
    print("\n",result[0])
    print("\t",result[1])
    print("\t",result[2])
connection.close()